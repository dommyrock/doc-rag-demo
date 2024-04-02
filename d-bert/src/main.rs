//TLDR TODO
/*
1 load Bert embeddings model

(jina-bert 8k ctx)
https://github.com/huggingface/candle/blob/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/jina-bert/main.rs

(bert)
https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/bert

(distil-bert)
https://github.com/huggingface/candle/tree/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/distilbert

2 load mistral model (or phi2 or gemma)
https://github.com/huggingface/candle/blob/b23436bf90b99eb17aed36aaa219875d3c962a7e/candle-examples/examples/mistral/main.rs


3 read file .... passs inputs
*/

/*
Conclusion : 

    DistilBert provides 95% perf of the Bert while being 60% faster so i'll use it instead
    https://huggingface.co/docs/transformers/model_doc/distilbert

installation : https://huggingface.github.io/candle/guide/installation.html

[Make sure you add candle-core with cuda featue enabled]
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"

Other deps: 
cargo add --git https://github.com/huggingface/candle.git candle-nn
cargo add tokenizers -features "http","onig"

Make sure candle-core,candle-transformers,candle-nn are all the same version !!!

Check most recent stable varsion here:
    https://github.com/huggingface/candle/blob/main/Cargo.toml

Curent gpu: 
nvidia-smi

Compute cap :
nvidia-smi --query-gpu=compute_cap --format=csv


Example fresh model addition:
    https://github.com/huggingface/candle/blob/main/candle-examples/examples/moondream/main.rs 
    (check commit messages to se quantization process)
 */


 #[cfg(feature = "mkl")]
 extern crate intel_mkl_src;
 
 #[cfg(feature = "accelerate")]
 extern crate accelerate_src;
 use candle_transformers::models::distilbert::{Config, DistilBertModel};
 use candle_transformers::models::distilbert::DTYPE;
 use anyhow::{Error as E, Result};
 use candle_core::{Device, Tensor};
 use candle_nn::VarBuilder;
 use clap::Parser;
 use hf_hub::{api::sync::Api, Repo, RepoType};
 use tokenizers::Tokenizer;
 
 

 #[derive(Parser, Debug)]
 #[command(author, version, about, long_about = None)]
 struct Args {
     /// Run on CPU rather than on GPU.
     #[arg(long)]
     cpu: bool,
 
     /// Enable tracing (generates a trace-timestamp.json file).
     #[arg(long)]
     tracing: bool,
 
     /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
     #[arg(long)]
     model_id: Option<String>,
 
     #[arg(long)]
     revision: Option<String>,
 
     /// When set, compute embeddings for this prompt.
     #[arg(long)]
     prompt: String,
 
     /// Use the pytorch weights rather than the safetensors ones
     #[arg(long)]
     use_pth: bool,
 
     /// The number of times to run the prompt.
     #[arg(long, default_value = "1")]
     n: usize,
 
     /// L2 normalization for embeddings.
     #[arg(long, default_value = "true")]
     normalize_embeddings: bool,
 }
 
 impl Args {
     fn build_model_and_tokenizer(&self) -> Result<(DistilBertModel, Tokenizer)> {
        //  let device = candle_examples::device(self.cpu)?;
        let device = Device::new_cuda(0)?;
        
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
             VarBuilder::from_pth(&weights_filename,DTYPE, &device)?
         } else {
             unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename],DTYPE, &device)? }
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
 
 fn main() -> Result<()> {
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
     let (bert_model, mut tokenizer) = args.build_model_and_tokenizer()?;
     let device = &bert_model.device;
 
     let tokenizer = tokenizer
         .with_padding(None)
         .with_truncation(None)
         .map_err(E::msg)?;
     let tokens = tokenizer
         .encode(args.prompt, true)
         .map_err(E::msg)?
         .get_ids()
         .to_vec();
     let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
     let mask = get_mask(tokens.len(), device);
 
     println!("token_ids: {:?}", token_ids.to_vec2::<u32>());
     println!("mask: {:?}", mask.to_vec2::<u8>());
 
     let ys = bert_model.forward(&token_ids, &mask)?;
     println!("{ys}");
 
     Ok(())
 }
 
 pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
     Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
 }