use crate::utils::{load_safetensor_model_files, parse_dtype, device};
use cylon_inference_engine::{TextGenerator, EosTokenHandler, ModelInference, InferenceEngine, InferenceConfig};
use anyhow::{bail, Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use minijinja::{context, Environment};
use serde::Deserialize;
use serde_json::{from_str, Value};
use std::fs::File;
use std::path::Path;
use std::sync::RwLock;
use tokenizers::Tokenizer;
use cylon_config::CylonConfig;

// Using the Qwen2 implementation from candle-transformers
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2ModelForCausalLM};

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

// Simplified to use only Qwen2 for now - can extend later for MoE variants

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    bos_token: Option<String>,  // BOS token can be null in Qwen models
    chat_template: String,
}

#[derive(Debug)]
pub struct QwenModel {
    model: RwLock<Qwen2ModelForCausalLM>, // Use RwLock for thread-safe interior mutability
    config: Qwen2Config,
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

impl QwenModel {
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

        let qwen_config: Qwen2Config = serde_json::from_reader(&model_config_file)?;
        info!("Loaded Qwen2 config: vocab_size={}, hidden_size={}, num_layers={}", 
              qwen_config.vocab_size, qwen_config.hidden_size, qwen_config.num_hidden_layers);

        // Create EOS token handler - Qwen2.5 uses <|im_end|> token (151645) as EOS
        // The Qwen2Config doesn't have eos_token_id field, so we use the standard one
        let eos_handler = EosTokenHandler::Single(151645); // <|im_end|> token for Qwen2.5

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };

        let model = Qwen2ModelForCausalLM::new(&qwen_config, vb)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).map_err(E::msg)?;

        let tokenizer_config_file = File::open(&model_dir.join("tokenizer_config.json"))?;
        let tokenizer_config: TokenizerConfig = serde_json::from_reader(&tokenizer_config_file)?;

        Ok(QwenModel {
            model: RwLock::new(model),
            config: qwen_config,
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

impl ModelInference for QwenModel {
    type Cache = (); // Qwen2 handles its own internal KV cache

    fn create_cache(&self, _enable_kv_cache: bool, _dtype: DType, _device: &Device) -> Result<Self::Cache> {
        // Qwen2 manages its own internal cache, so we just return unit type
        Ok(())
    }

    fn forward(&self, input: &Tensor, context_index: usize, _cache: &mut Self::Cache) -> Result<Tensor> {
        // For Qwen2, the model handles its own KV caching internally
        // The context_index from inference engine is the position where new tokens start,
        // but Qwen2 expects seqlen_offset to be the total cached sequence length

        // When KV cache is disabled, we need to clear the internal cache before each forward pass
        // to prevent shape mismatches from accumulated cache state
        if !self.use_kv_cache() {
            self.clear_kv_cache()?;
        }

        // If this is the first call (context_index = 0), seqlen_offset = 0
        // If this is a continuation (context_index > 0), seqlen_offset = context_index
        let seqlen_offset = context_index;

        let logits = self.model.write().unwrap().forward(input, seqlen_offset).map_err(E::from)?;

        // Qwen2 ModelForCausalLM returns [batch, 1, vocab_size], but inference engine expects [batch, vocab_size]
        // So we squeeze the middle dimension
        logits.squeeze(1).map_err(E::from)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn use_kv_cache(&self) -> bool {
        // Disable KV cache for Qwen models - the internal cache management
        // conflicts with the inference engine's token-by-token generation loop
        false
    }

    fn eos_handler(&self) -> &EosTokenHandler {
        &self.eos_handler
    }

    fn supports_persistent_cache(&self) -> bool {
        // Qwen2 uses internal cache, so persistence is handled differently
        false
    }

    fn clear_kv_cache(&self) -> Result<()> {
        // Clear the internal KV cache of the Qwen2 model
        self.model.write().unwrap().clear_kv_cache();
        Ok(())
    }
}

impl TextGenerator for QwenModel {
    fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<String, E> {
        // Clear KV cache before each new generation to avoid shape mismatches
        self.clear_kv_cache().map_err(E::from)?;
        
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
        debug!("Rendered prompt: {}", rendered);
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

        let bos_token = self.tokenizer_config.bos_token.as_deref().unwrap_or("");
        let rendered = template.render(context! {
            messages => messages,
            bos_token => bos_token,
            add_generation_prompt => true,
        })?;

        Ok(rendered)
    }
}