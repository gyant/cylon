pub mod utils;
pub mod llama;
pub mod qwen;
pub mod model_wrapper;

pub use llama::LlamaModel;
pub use qwen::QwenModel;
pub use model_wrapper::ModelWrapper;

use anyhow::{bail, Error as E, Result};
use cylon_config::CylonConfig;
use cylon_inference_engine::TextGenerator;

/// Factory function to create models based on configuration
pub fn create_model(config: &CylonConfig) -> Result<Box<dyn TextGenerator>, E> {
    match config.model_family.as_str() {
        "llama" => Ok(Box::new(LlamaModel::new(config)?)),
        "qwen" => Ok(Box::new(QwenModel::new(config)?)),
        
        // Future model implementations would go here:
        // "gpt2" => Ok(Box::new(Gpt2Model::new(config)?)),
        
        _ => bail!("Unsupported model family: {}", config.model_family),
    }
}