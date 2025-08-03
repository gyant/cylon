use anyhow::{bail, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::DType;
use candle_core::Device;
use serde_json::Value;
use std::fs::File;
use std::path::{Path, PathBuf};

pub fn device() -> Result<Device> {
    if cuda_is_available() {
        println!("Running on CUDA device...");
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        println!("Running on Metal device...");
        Ok(Device::new_metal(0)?)
    } else {
        println!("Running on CPU...");
        Ok(Device::Cpu)
    }
}

pub fn parse_dtype(dtype: &Option<String>) -> Result<DType> {
    match dtype.as_deref() {
        Some("f16") => Ok(DType::F16),
        Some("bf16") => Ok(DType::BF16),
        Some("f32") => Ok(DType::F32),
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => Ok(DType::F16),
    }
}

pub fn load_safetensor_model_files(model_path: &Path) -> Result<Vec<PathBuf>> {
    let model_index_file = File::open(model_path.join("model.safetensors.index.json"))?;
    let model_index_json: Value = serde_json::from_reader(&model_index_file).map_err(candle_core::Error::wrap)?;
    let model_weight_map = match model_index_json.get("weight_map") {
        None => bail!("no weight map in {model_index_file:?}"),
        Some(Value::Object(map)) => map,
        Some(_) => bail!("weight map in {model_index_file:?} is not a map"),
    };

    let mut safetensors_files = std::collections::HashSet::new();
    for value in model_weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }

    let safetensors_files: Vec<_> = safetensors_files
        .iter()
        .map(|v| model_path.join(v))
        .collect();

    Ok(safetensors_files)
}