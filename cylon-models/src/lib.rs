pub mod utils;

#[cfg(feature = "llama")]
pub mod llama;

#[cfg(feature = "llama")]
pub use llama::LlamaModel;

use anyhow::{bail, Error as E, Result};
use cylon_config::CylonConfig;
use cylon_inference_engine::TextGenerator;

/// Factory function to create models based on configuration
pub fn create_model(config: &CylonConfig) -> Result<Box<dyn TextGenerator>, E> {
    match config.model_family.as_str() {
        #[cfg(feature = "llama")]
        "llama" => Ok(Box::new(LlamaModel::new(config)?)),
        
        // Future model implementations would go here:
        // #[cfg(feature = "gpt2")]
        // "gpt2" => Ok(Box::new(Gpt2Model::new(config)?)),
        
        _ => bail!("Unsupported model family: {}", config.model_family),
    }
}