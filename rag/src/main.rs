use std::{collections::HashMap, fmt::Debug};

use anyhow::Result;
use clap::Parser;
use lib::{
    d_bert::{generate_embeddings, generate_embeddings_v3, get_prompt_embeddings, tokenize_chunks_get_embeddings, Args},
    insert_many,
};
pub use qdrant_client::prelude::Value as QdrantValue;
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        point_id::PointIdOptions, CreateCollection, Distance, SearchPoints, VectorParams,
        VectorsConfig,
    },
};
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde_json::json;

//GPU pipeline post : https://fgiesen.wordpress.com/2011/07/01/a-trip-through-the-graphics-pipeline-2011-part-1/

/// Represents a found point in the vector database.
#[derive(Debug)]
pub struct FoundPoint {
    pub id: u64,
    pub score: f32,
    pub payload: Option<HashMap<String, QdrantValue>>, // assuming Value is from serde_json
}
pub type Value = QdrantValue;

#[tokio::main]
async fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    dotenv::dotenv().ok();

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let text = std::fs::read_to_string(std::path::Path::new(&args.file))
        .unwrap()
        .lines()
        .filter(|&ln| !ln.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    // let collection = std::path::Path::new(&args.file)
    //     .file_stem()
    //     .and_then(|name| name.to_str())
    //     .ok_or_else(|| anyhow::anyhow!("Failed to extract file stem"))?
    //     .to_string();
    const COLLECTION_NAME: &'static str = "Rag-demo";

    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let _=client.delete_collection(COLLECTION_NAME);
    client
        .create_collection(&CreateCollection {
            collection_name: COLLECTION_NAME.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                    VectorParams {
                        size: 512,
                        distance: Distance::Cosine.into(),
                        
                        ..Default::default()
                    },
                )),
            }),
            ..Default::default()
        })
        .await?;

    let coll = client.list_collections().await?;
    dbg!(coll);

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let prompt_embedding = get_prompt_embeddings(&model, &mut tokenizer, &args)?;
    //println!("p embed\n: {prompt_embedding}");

    //v2  Generate embeddings for the document chunks
    //let embeddings = tokenize_chunks_get_embeddings(text, model, &mut tokenizer).await?;
    
    //v3
    let embeddings = generate_embeddings_v3(text, model, tokenizer).await?;

    //v1
    //let embeddings = generate_embeddings(chunks, model, tokenizer).await?;

    println!("generated embeddings {:?}", embeddings.shape());

    // Store generated embeddings
    insert_many(&COLLECTION_NAME, embeddings.to_vec2()?, &client).await?;

    //search qdrant for similarities
    println!("Constructing Search request ...........");

    let limit: u64 = 4;
    let search_request = SearchPoints {
        collection_name: COLLECTION_NAME.into(),
        vector: prompt_embedding.to_vec1::<f32>()?,
        //filter, https://qdrant.tech/documentation/quick-start/#add-a-filter
        limit: limit,
        with_payload: Some(true.into()),
        ..Default::default()
    };

    let response = client.search_points(&search_request).await?;

    println!("Qdrant query response ln 102: {:?}", response);

    // dbg!(&response);

    // let result: Vec<FoundPoint> = response
    //     .result
    //     .into_iter()
    //     .filter_map(|scored_point| {
    //         let id = match scored_point.id {
    //             Some(point_id) => {
    //                 match point_id.point_id_options {
    //                     Some(PointIdOptions::Num(id)) => id,
    //                     _ => return None, // Ignore other variants or if it's None
    //                 }
    //             }
    //             None => return None, // Skip this point if it doesn't have an ID
    //         };
    //         let score = scored_point.score;
    //         let payload = scored_point.payload;

    //         Some(FoundPoint {
    //             id,
    //             score,
    //             payload: Some(payload),
    //         })
    //     })
    //     .collect();

    //     for r in result{
    //         println!("{:?}",r);
    //     }

    let prompt_for_model = r#"
    {{#chat}}

        {{#system}}
        You are a highly advanced assistant. You receive a prompt from a user and relevant excerpts extracted from a PDF. You then answer truthfully to the best of your ability. If you do not know the answer, your response is I don't know.
        {{/system}}

        {{#user}}
        {{user_prompt}}
        {{/user}}

        {{#system}}
        Based on the retrieved information from the PDF, here are the relevant excerpts:
        
        {{#each payloads}}
        {{this}}
        {{/each}}

        Please provide a comprehensive answer to the user's question, integrating insights from these excerpts and your general knowledge.
        {{/system}}

    {{/chat}}
    "#;

    //for mistral api
    // let context = json!({
    //     "user_prompt": &args.prompt,
    //     "payloads": result
    //         .iter()
    //         .filter_map(|found_point| {
    //             found_point.payload.as_ref().map(|payload| {
    //                 serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string())
    //             })
    //         })
    //         .collect::<Vec<String>>()
    // });

    //TODO: Figure out how to manually parse above r# formated text (or use handlebars) context

    //TODO: Figure out how to execute() or simplify  (-> MapReduce [MapReducePipeline in orca ] )
    //     let pipe = LLMPipeline::new(&mistral)
    //     .load_template("query", prompt_for_model)?
    //     .load_context(&OrcaContext::new(context)?)?
    //     .load_memory(Buffer::new());

    //      let res = pipe.execute("query").await?;

    // //GROQ message API : https://console.groq.com/docs/text-chat#required-parameters
    // let api_key = dotenv::var("GROQ_API_KEY").expect("GROQ_API_KEY not set");
    // let client = Client::new();

    // let mut headers = HeaderMap::new();
    // headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    // headers.insert(
    //     AUTHORIZATION,
    //     HeaderValue::from_str(&format!("Bearer {}", api_key))?,
    // );

    // //TODO : UPDATE 'GDRANT RESPONSES' with the Vec<Qdrant relevant top N responses>
    // let body = json!({
    //     "messages": [{"role":"system","content":"You are a highly advanced assistant. You receive a prompt from a user and relevant excerpts extracted from a text document. You then answer truthfully to the best of your ability. If you do not know the answer, your response is I don't know."},
    //     {"role": "user", "content": &args.prompt},
    //     {"role":"system","content":"Based on the retrieved information from the document, here are the relevant excerpts:{{GDRANT RESPONSES}}Please provide a comprehensive answer to the user's question, integrating insights from these excerpts and your general knowledge."}],
    //     "model": "mixtral-8x7b-32768"
    // });

    // match client
    //     .post("https://api.groq.com/openai/v1/chat/completions")
    //     .headers(headers)
    //     .json(&body)
    //     .send()
    //     .await
    // {
    //     Ok(res) => {
    //         //Got the response
    //         dbg!(res);
    //     }
    //     Err(er) => {
    //         eprintln!("Error while Requesting GROQ API {er}")
    //     }
    // }

    Ok(())
}
//tokio streaming response example
//https://www.shuttle.rs/blog/2024/02/28/rag-llm-rust (above wrapping it all up... section)

//cd rag
//cargo run --release --bin rag -- --prompt "tell me something about GPU pipeline" --file gpu_pipeline_1.txt

// QDRANT CMD's
// docker pull qdrant/qdrant

//Run without a persisted storage volume
//podman run -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant

//Run with persisted storage volume (windows)
// podman run --name rag_demo -p 6333:6333 -p 6334:6334 -v c:/qdrant_storage:/qdrant/storage qdrant/qdrant

//with grpc
//podman run --name rag_demo -p 6333:6333 -p 6334:6334 -v c:/qdrant_storage:/qdrant/storage  -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant
