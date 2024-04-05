#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::distilbert::DTYPE;
use candle_transformers::models::distilbert::{Config, DistilBertModel};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokenizers::Tokenizer;

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
    tokenizer: &mut Tokenizer,
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

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub async fn generate_embeddings(
    doc_chunks: Vec<String>,
    bert_model: DistilBertModel,
    tokenizer: Tokenizer,
) -> Result<Tensor, E> {
    let device = &bert_model.device;

    //Encode all the sentences in parallel
    let tokens = tokenizer
        .encode_batch(
            doc_chunks.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            true,
        )
        .map_err(E::msg)?;

    let token_ids = tokens
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let tokens = tokens.get_ids().to_vec();
            let tensor = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            Ok((i, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
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

            // Lock the mutex and write the embedding to the correct index
            let mut embeddings = embeddings_arc
                .lock()
                .map_err(|e| anyhow!("Mutex error: {}", e))?;
            embeddings[*i] = embedding;

            Ok::<(), anyhow::Error>(())
        },
    )?;
    println!("Done computing embeddings");
    println!("Embeddings took {:?} to generate", start.elapsed());

    // Retrieve the final ordered embeddings
    let embeddings_arc = Arc::try_unwrap(embeddings_arc)
        .map_err(|_| anyhow!("Arc unwrap failed"))?
        .into_inner()
        .map_err(|e| anyhow!("Mutex error: {}", e))?;

    let stacked_embeddings = Tensor::stack(&embeddings_arc, 0)?;

    Ok(stacked_embeddings)
}
