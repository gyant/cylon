use crate::utils::{load_safetensor_model_files, parse_dtype, device};
use cylon_inference_engine::{TextGenerator, EosTokenHandler, ModelInference, InferenceEngine, InferenceConfig};
use anyhow::{bail, Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama;
use llama::{LlamaConfig, LlamaEosToks};
use minijinja::{context, Environment};
use serde::Deserialize;
use serde_json::{from_str, Value};
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use cylon_config::CylonConfig;

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    bos_token: String,
    chat_template: String,
}

#[derive(Debug)]
pub struct LlamaModel {
    model: llama::Llama,
    config: llama::Config,
    tokenizer: Tokenizer,
    tokenizer_config: TokenizerConfig,
    device: Device,
    dtype: DType,
    eos_handler: EosTokenHandler,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: Option<u64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    enable_kv_cache: bool,
}

impl LlamaModel {
    pub fn new(config: &CylonConfig) -> Result<Self> {
        let device = device()?;
        info!("Using device: {:?}", device);
        let dtype = parse_dtype(&config.dtype)?;
        info!("Using dtype: {:?}", dtype);

        let model_dir = Path::new(&config.model_path);

        if !model_dir.exists() {
            bail!("Model directory does not exist: {}", model_dir.display());
        } else if !model_dir.is_dir() {
            bail!("Model path is not a directory: {}", model_dir.display());
        }

        let safetensors_files = load_safetensor_model_files(&model_dir)
            .with_context(|| format!("Failed to load safetensors files at {}", model_dir.display()))?;

        let model_config_file = File::open(&model_dir.join("config.json"))
            .with_context(|| format!("Failed to open model config file at {}", model_dir.join("config.json").display()))?;

        let llama_config: LlamaConfig = serde_json::from_reader(&model_config_file)?;
        
        // Disable flash attention on Metal since it's CUDA-only
        let use_flash_attn = match device {
            Device::Metal(_) => {
                if config.use_flash_attn {
                    warn!("Flash attention is not supported on Metal, disabling");
                }
                false
            },
            Device::Cuda(_) => config.use_flash_attn,
            _ => false,
        };

        let llama_config = llama_config.into_config(use_flash_attn);

        let eos_handler: EosTokenHandler = match &llama_config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => EosTokenHandler::Single(*id),
            Some(LlamaEosToks::Multiple(ids)) => EosTokenHandler::Multiple(ids.clone()),
            None => EosTokenHandler::None,
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };

        let model = llama::Llama::load(vb, &llama_config)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).map_err(E::msg)?;

        let tokenizer_config_file = File::open(&model_dir.join("tokenizer_config.json"))?;
        let tokenizer_config: TokenizerConfig = serde_json::from_reader(&tokenizer_config_file)?;

        Ok(LlamaModel {
            model,
            config: llama_config,
            tokenizer,
            tokenizer_config,
            device,
            eos_handler,
            dtype,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            seed: Some(config.seed),
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
            enable_kv_cache: config.enable_kv_cache,
        })
    }

    fn inference_config(&self) -> InferenceConfig {
        InferenceConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
        }
    }
}

impl ModelInference for LlamaModel {
    type Cache = llama::Cache;

    fn create_cache(&self, enable_kv_cache: bool, dtype: DType, device: &Device) -> Result<Self::Cache> {
        llama::Cache::new(enable_kv_cache, dtype, &self.config, device).map_err(E::from)
    }

    fn forward(&self, input: &Tensor, context_index: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        self.model.forward(input, context_index, cache).map_err(E::from)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn use_kv_cache(&self) -> bool {
        self.enable_kv_cache
    }

    fn eos_handler(&self) -> &EosTokenHandler {
        &self.eos_handler
    }
}

impl TextGenerator for LlamaModel {
    fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<String, E> {
        let tokens = self.tokenize(prompt.as_str())?;
        let config = self.inference_config();
        
        let generated_tokens = InferenceEngine::generate(self, tokens, max_tokens, &config)?;
        let generated_text = self.decode(&generated_tokens)?;

        Ok(generated_text)
    }

    fn inference(
        &self,
        prompt: &Vec<String>,
        max_tokens: usize,
    ) -> Result<String, E> {
        let rendered = self.render(prompt)?;

        self.generate(rendered, max_tokens)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, E> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, E> {
        self.tokenizer.decode(tokens, true).map_err(E::msg)
    }

    fn render(&self, prompt: &Vec<String>) -> Result<String, E> {
        let mut template_env = Environment::new();
        let template_key = "prompt";
        template_env.add_template(template_key, self.tokenizer_config.chat_template.as_str())?;

        let messages: Vec<Value> = prompt
            .iter()
            .map(|s| from_str(s).expect("Failed to parse JSON"))
            .collect();

        let template = template_env.get_template(template_key)?;

        let rendered = template.render(context! {
            messages => messages,
            bos_token => self.tokenizer_config.bos_token.as_str(),
            add_generation_prompt => true,
        })?;

        Ok(rendered)
    }
}