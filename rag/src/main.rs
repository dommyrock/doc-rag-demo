use anyhow::Result;
use clap::Parser;
use lib::{d_bert::{generate_embeddings, get_prompt_embeddings, Args}, insert_many};
use qdrant_client::{client::QdrantClient, qdrant::{CreateCollection, Distance, VectorParams, VectorsConfig}};

//GPU pipeline post : https://fgiesen.wordpress.com/2011/07/01/a-trip-through-the-graphics-pipeline-2011-part-1/


#[tokio::main]
async fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // let collection = std::path::Path::new(&args.file)
    //     .file_stem()
    //     .and_then(|name| name.to_str())
    //     .ok_or_else(|| anyhow::anyhow!("Failed to extract file stem"))?
    //     .to_string();
    const COLLECTION_NAME:&'static str = "Rag-demo";

    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    client.create_collection(&CreateCollection {
        collection_name: COLLECTION_NAME.into(),
        vectors_config: Some(VectorsConfig {
            config: Some(qdrant_client::qdrant::vectors_config::Config::Params(VectorParams {
                size: 384,
                distance: Distance::Cosine.into(),
                ..Default::default()
            })),
        }),
        ..Default::default()
    })
    .await?;

    let chunks = lib::split_into_chunks(&args.file);
    for c in chunks.iter() {
        let p = format!("chunk LEN :: [{}] \n {}\n", c.len(), c);
        println!("{p}");
    }

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let prompt_embedding = get_prompt_embeddings(&model, &mut tokenizer, args)?;
    //println!("p embed\n: {prompt_embedding}");

    // Generate embeddings for the document chunks
    let embeddings = generate_embeddings(chunks,model,tokenizer).await?;

    // Store generated embeddings 
    insert_many(&COLLECTION_NAME, embeddings.to_vec2()?, client).await?;

    //search qdrant for similarities
    //TODO: let result = client.search(&collection, query_embedding.to_vec()?.clone(), 1, None).await?;

    Ok(())
}

// QDRANT CMD's
// pull
//

//Run without a persisted storage volume
//podman run -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant

//Run with persisted storage volume (windows)
// podman run --name rag_demo -p 6333:6333 -p 6334:6334 -v c:/qdrant_storage:/qdrant/storage qdrant/qdrant

//with grpc
//podman run --name rag_demo -p 6333:6333 -p 6334:6334 -v c:/qdrant_storage:/qdrant/storage  -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant