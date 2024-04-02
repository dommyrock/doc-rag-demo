pub mod token_output_stream;
use candle_core as candle;
//use candle::utils::{cuda_is_available, metal_is_available};
use candle::Result;

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
   repo: &hf_hub::api::sync::ApiRepo,
   json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
   let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
   let json_file = std::fs::File::open(json_file)?;
   let json: serde_json::Value =
       serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
   let weight_map = match json.get("weight_map") {
       None => candle::bail!("no weight map in {json_file:?}"),
       Some(serde_json::Value::Object(map)) => map,
       Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
   };
   let mut safetensors_files = std::collections::HashSet::new();
   for value in weight_map.values() {
       if let Some(file) = value.as_str() {
           safetensors_files.insert(file.to_string());
       }
   }
   let safetensors_files = safetensors_files
       .iter()
       .map(|v| repo.get(v).map_err(candle::Error::wrap))
       .collect::<Result<Vec<_>>>()?;
   Ok(safetensors_files)
}
