pub mod d_bert;
pub mod token_output_stream;

use std::collections::HashMap;

use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Error, Result};
use candle_core as candle;
use qdrant_client::client::Payload;
pub use qdrant_client::prelude::Value as QdrantValue;
use qdrant_client::qdrant::PointStruct;

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

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn split_into_chunks(pth: &str) -> Vec<String> {
    let chunk_size = 400;
    let overlap = 80;

    let text = std::fs::read_to_string(std::path::Path::new(&pth))
        .unwrap()
        .lines()
        .filter(|&ln| ln.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    text.chars()
        .collect::<Vec<char>>()
        .windows(chunk_size)
        .step_by(chunk_size - overlap)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

///Inserts embeddings to the Qdrant vector store
pub async fn insert_many(
    collection_name: &str,
    vectors: Vec<Vec<f32>>,
    client: &qdrant_client::client::QdrantClient,
    payload: Vec<String>,
    source_file: &str,
) -> anyhow::Result<()> {
    println!("Vectors received: {}", vectors.len());

    let points_result: anyhow::Result<Vec<PointStruct>> = vectors
        .into_iter()
        .zip(payload)
        .enumerate()
        .map(|(id, (vector, doc_chunk))| {
            let payload = to_payload(doc_chunk,source_file)?;
            Ok(PointStruct::new(id as u64, vector, payload))
        })
        .collect();

    let points = points_result?;

    client
        .upsert_points_blocking(collection_name, None, points, None)
        .await?;
    Ok(())
}

fn to_payload(p: String, source: &str) -> Result<Payload> {
    match serde_json::to_value(p) {
        Ok(value) => {
            if let serde_json::Value::Object(map) = value {
                let converted_map: HashMap<String, QdrantValue> = map
                    .into_iter()
                    .map(|(k, v)| (k, QdrantValue::from(v)))
                    .collect();
                Ok(Payload::new_from_hashmap(converted_map))
            } else {
                // If the value is not an object, wrap it in a map with a generic key.
                let mut map = HashMap::new();
                map.insert("txt_chunk".to_string(), QdrantValue::from(value));
                map.insert("source".to_string(), QdrantValue::from(source));
                Ok(Payload::new_from_hashmap(map))
            }
        }
        Err(e) => {
            eprintln!("Error serializing doc chunks to json...");
            Err(Error::msg(e))
        }
    }
}
