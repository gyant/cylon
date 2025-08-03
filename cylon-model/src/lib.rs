pub mod eos;
pub mod generation;
pub mod cylon_llama;
pub mod utils;

use cylon_config::CylonConfig;
use anyhow::{bail, Error as E, Result};
use generation::TextGenerator;
use eos::EosTokenHandler;
use cylon_llama::LlamaModel;

#[derive(Debug)]
pub struct Model {
    generator: Box<dyn TextGenerator>,
}

impl Model {
    pub fn new(config: &CylonConfig) -> Result<Model, E> {
        let generator: Box<dyn TextGenerator> = match config.model_family.as_str() {
            "llama" => Box::new(LlamaModel::new(config)?),
            // Add other models here, e.g., "qwen" => Box::new(qwen::QwenModel::new(config)?),
            _ => bail!("Unsupported model family: {}", config.model_family),
        };

        Ok(Model { generator })
    }

    pub fn inference(
        &self,
        prompt: &Vec<String>,
        max_tokens: usize,
    ) -> Result<String, E> {
        self.generator.inference(prompt, max_tokens)
    }
}