#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::distilbert::DTYPE;
use candle_transformers::models::distilbert::{Config, DistilBertModel};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{Model, PaddingParams, PaddingStrategy, Tokenizer};

use crate::device;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[clap(long)]
    /// The path to the document to vectorize
    pub file: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    pub model_id: Option<String>,

    #[arg(long)]
    pub revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    pub prompt: String,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    pub use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    pub n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    pub normalize_embeddings: bool,
}

impl Args {
    pub fn build_model_and_tokenizer(&self) -> Result<(DistilBertModel, Tokenizer)> {
        let device = device(self.cpu)?;

        let default_model = "distilbert-base-uncased".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let model = DistilBertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn get_mask(size: usize, device: &Device) -> Tensor {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device).unwrap()
}

/// # Usage example
/// ```
/// fn main() -> Result<()> {
///     use tracing_chrome::ChromeLayerBuilder;
///     use tracing_subscriber::prelude::*;
///
///     let args = Args::parse();
///     let _guard = if args.tracing {
///         println!("tracing...");
///         let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
///         tracing_subscriber::registry().with(chrome_layer).init();
///         Some(guard)
///     } else {
///         None
///     };
///     //load model
///     let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
///     let prompt_embedding = get_prompt_embeddings(&model, &mut tokenizer, args)?;
///     println!("{embeddings}");
///
///     Ok(())
/// }
/// ````
pub fn get_prompt_embeddings(
    bert_model: &DistilBertModel,
    mut tokenizer: Tokenizer,
    args: &Args,
) -> Result<Tensor, E> {
    let device = &bert_model.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let mask = get_mask(tokens.len(), device);

    let embeddings = bert_model.forward(&token_ids, &mask)?;
    let embeddings = if args.normalize_embeddings {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };

    Ok(embeddings)
}

pub fn generate_embedding(
    bert_model: &DistilBertModel,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<Tensor, E> {
    let device = &bert_model.device;

    let tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("----------------PROMPT Embeddings------------");
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    println!("token_ids shape: {:?}", token_ids.shape());
    let token_type_ids = token_ids.zeros_like()?;
    println!("running inference {:?}", token_ids.shape());
    let start = std::time::Instant::now();
    let embedding = bert_model.forward(&token_ids, &token_type_ids)?;
    println!("embedding shape: {:?}", embedding.shape());
    println!("Embedding took {:?} to generate", start.elapsed());
    println!("----------------PROMPT Embeddings------------\n");
    Ok(embedding)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub async fn generate_embeddings(
    txt: String,
    bert_model: &DistilBertModel,
    tokenizer: &mut Tokenizer,
) -> Result<(Tensor, Vec<String>), E> {
    let device = &bert_model.device;
    //You could extract summarization first, than split that to chunks if needed
    let doc_chunks = split_text_into_chunks(300, 80, &txt);

    //println!("Doc chunks:\n {:?}",doc_chunks);

    //Add padding and special tokens
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        //you can usually find special tokens by looking at model card > vocab.txt
        let sep = tokenizer.get_model().token_to_id("[SEP]").unwrap(); //[101]
        let cls = tokenizer.get_model().token_to_id("[CLS]").unwrap(); //[102]
        tokenizer.with_post_processor(BertProcessing::new(
            (String::from("SEP"), sep),
            (String::from("CLS"), cls),
        ));
    }

    //Encode all the sentences in parallel
    let tokens = tokenizer
        .encode_batch(
            doc_chunks.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            true,
        )
        .map_err(E::msg)?;

    //DEMO:
    println!("Vocab size : {}\n", tokenizer.get_vocab_size(true));
    // dbg!(&tokens); //quite big
    println!("Tokens: {:?}\n", tokens.first().unwrap().get_tokens());

    let token_ids = tokens
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let tokens = tokens.get_ids().to_vec();
            let tensor = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            Ok((i, tensor))
        })
        .collect::<Result<Vec<_>>>()?;

    //DEMO
    // dbg!(&token_ids); //quite big
    println!("IDS: {:?}\n", tokens.first().unwrap().get_ids());

    let embeddings = vec![Tensor::ones((2, 3), candle_core::DType::F32, device)?; token_ids.len()];

    // Wrap the embeddings vector in an Arc<Mutex<_>> for thread-safe access
    let embeddings_arc = Arc::new(Mutex::new(embeddings));

    println!("Computing embeddings");
    let start = std::time::Instant::now();

    token_ids.par_iter().try_for_each_with(
        embeddings_arc.clone(),
        |embeddings_arc, (i, token_ids)| {
            let token_type_ids = token_ids.zeros_like()?;
            let embedding = bert_model.forward(token_ids, &token_type_ids)?.squeeze(0)?;

            if *i == 0 {
                println!(
                    "Embedding shape {:?}\nEmbedding {}\n",
                    embedding.dims(),
                    embedding.get(1)? //last dimension
                );
            }

            // Lock the mutex and write the embedding to the correct index
            //.lock() returns a MutexGuard --> which provides a mutable reference to the data
            let mut embeddings = embeddings_arc
                .lock()
                .map_err(|e| anyhow!("Mutex error: {}", e))?;
            embeddings[*i] = embedding;

            Ok::<(), anyhow::Error>(())
        },
    )?;

    println!("Done computing embeddings");
    println!("Embeddings took {:?} to generate\n", start.elapsed());

    // Retrieve the final ordered embeddings
    let embeddings_arc = Arc::try_unwrap(embeddings_arc)
        .map_err(|_| anyhow!("Arc unwrap failed"))?
        .into_inner()
        .map_err(|e| anyhow!("Mutex error: {}", e))?;

    let stacked_embeddings = Tensor::stack(&embeddings_arc, 0)?;
    Ok((stacked_embeddings, doc_chunks))
}

///SEE Chunking section  : https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/
pub fn split_text_into_chunks(chunk_size: usize, overlap: usize, txt: &str) -> Vec<String> {
    let words: Vec<&str> = txt.split_whitespace().collect();
    let mut chunks: Vec<String> = vec![];

    let mut i = 0;
    while i < words.len() {
        let end = usize::min(i + chunk_size, words.len());
        let chunk: Vec<&str> = words[i..end].to_vec();
        chunks.push(chunk.join(" "));
        i = if end < words.len() {
            end - overlap
        } else {
            end
        };
    }
    chunks
}

///Custom implementation (didn't quite work, since i ended up with 4dim , but Qdrant Expected 2 when inserting :/ )
pub async fn tokenize_chunks_get_embeddings(
    text: String,
    bert_model: DistilBertModel,
    tokenizer: &mut Tokenizer,
) -> Result<Vec<Tensor>> {
    let device = &bert_model.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer
        .encode(text, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

    let chunks = chunk_with_overlap(&token_ids, 510, 100, device)?;

    // Parallelize the chunks into Vec<Embeddings>
    let embeddings: Result<Vec<Tensor>> = chunks
        .par_iter()
        .map(|(token_ids, mask)| {
            let embeddings = bert_model.forward(token_ids, mask)?.squeeze(0)?;
            let normalized_embeddings = normalize_l2(&embeddings)?;
            //println!("Bart Embedding shape: {:?}", normalized_embeddings.shape());
            Ok(normalized_embeddings)
        })
        .collect();

    for (t, mask) in chunks {
        let t_len: usize = t.dims().iter().product();
        let mask_l: usize = mask.dims().iter().product();
        println!("Tensor  len : [{t_len}]   - mask len[{mask_l}]");
        println!("{t}");
        println!("{mask}");
        println!("Shape: {:?}\n", t.shape());
    }

    println!("Embeddings: \n\n{:?} \n", embeddings);

    //SHAPE
    //[1, 512, 768], this means that each chunk of 512 tokens is represented by a 768-dimensional embedding.
    let embeddings = embeddings.expect("Should conatain embeddings for document chunks.");
    // let stacked_embeddings = Tensor::stack(&embeddings, 0)?;

    //Stacked SHAPE
    //[9, 1, 512, 768], this means that each chunk of 512 tokens is represented by a 768-dimensional embedding.
    // Ok(stacked_embeddings)

    Ok(embeddings)
}

//The recommended order of operations is:
//
//1. Tokenize the input sequence.
//2. Add the special tokens (e.g. "CLS" at the beginning and "SEP" at the end).
//3. Pad the resulting sequence to the desired length.

fn chunk_with_overlap(
    tensor: &Tensor,
    chunk_size: usize,
    overlap: usize,
    device: &Device,
) -> Result<Vec<(Tensor, Tensor)>, candle_core::Error> {
    let total_elements: usize = tensor.dims().iter().product();
    let mut start = 0;
    let mut chunks: Vec<(Tensor, Tensor)> = Vec::new();

    while start < total_elements {
        let end = std::cmp::min(start + chunk_size, total_elements);
        if let Ok(chunk) = tensor.narrow(D::Minus1, start, end - start) {
            //special start /end tokens
            let st_start = Tensor::new([101 as u32].as_slice(), device)?.unsqueeze(0)?;
            let st_end = Tensor::new([102 as u32].as_slice(), device)?.unsqueeze(0)?;
            let mut cat = Tensor::cat(&[st_start, chunk, st_end], D::Minus1)?;
            let mut mask = get_ones_mask(cat.dims().iter().product(), device)?;

            //Add padding where len < 512
            let padding = 512 - cat.dims().iter().product::<usize>();

            cat = cat.pad_with_zeros(D::Minus1, 0, padding)?;
            mask = mask.pad_with_zeros(D::Minus1, 0, padding)?;

            chunks.push((cat, mask));

            start += chunk_size - overlap;
        }
    }
    Ok(chunks)
}

fn get_ones_mask(size: usize, device: &Device) -> Result<Tensor, candle_core::Error> {
    let mask = Tensor::ones(&[1, size], candle_core::DType::U8, device);
    mask
}

//There is also special tokes api
//https://github.com/huggingface/tokenizers/blob/main/tokenizers/tests/common/mod.rs#L41

/* Dimensionality
* OpenAI's embeddings have higher dimensionality (1536) compared to DistilBERT (768)

"distilbert-base-uncased" is a distilled version of the BERT model, which produces embeddings of size 768 for each token in the input sequence. So, for a 512-token chunk of text, the output embeddings would have a shape of `[512, 768]` (assuming you're using the last-layer hidden states as the embeddings).

On the other hand, OpenAI's embeddings model (`text-embedding-ada-002`) produces embeddings of size 1536 for each input token. So, for a 512-token chunk of text, the output embeddings would have a shape of `[512, 1536]`.


In the context of retrieval-augmented generation, using the last-layer hidden states as the embeddings is a common choice, as it provides a compact and informative representation of the input text that can be used to find similar documents or sentences. However, you could also experiment with using earlier layers or a combination of layers to see if that works better for your use case.
*/
