use crate::config::CylonConfig;
use anyhow::{bail, Error as E, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama;
use llama::{LlamaConfig, LlamaEosToks};
use std::fs::File;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

trait TextGenerator: std::fmt::Debug {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, E>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, E>;
    fn decode(&self, tokens: &[u32]) -> Result<String, E>;
}

trait EosTokenHandler: std::fmt::Debug {
    fn is_eos_token(&self, token_id: u32) -> bool;
}

#[derive(Debug)]
struct SingleEosToken(u32);
impl EosTokenHandler for SingleEosToken {
    fn is_eos_token(&self, token_id: u32) -> bool {
        token_id == self.0
    }
}

#[derive(Debug)]
struct MultipleEosTokens(Vec<u32>);
impl EosTokenHandler for MultipleEosTokens {
    fn is_eos_token(&self, token_id: u32) -> bool {
        self.0.contains(&token_id)
    }
}

#[derive(Debug)]
struct NoEosToken;
impl EosTokenHandler for NoEosToken {
    fn is_eos_token(&self, _token_id: u32) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct LlamaModel {
    model: llama::Llama,
    config: llama::Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    eos_handler: Box<dyn EosTokenHandler>,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: Option<u64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    enable_kv_cache: bool,
}

impl TextGenerator for LlamaModel {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, E> {
        let mut tokens = self.tokenize(prompt)?;

        let mut cache =
            llama::Cache::new(self.enable_kv_cache, self.dtype, &self.config, &self.device)?;

        let mut logits_processor = {
            let sampling = if self.temperature <= 0. {
                Sampling::ArgMax
            } else {
                let temperature = self.temperature;
                match (self.top_k, self.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(self.seed.unwrap(), sampling)
        };

        let start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut token_generated = 0;

        let mut generated_tokens = Vec::new();

        for index in 0..max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);
            generated_tokens.push(next_token);

            if self.eos_handler.is_eos_token(next_token) {
                break;
            }
        }

        let generation_time = start_gen.elapsed();
        let tokens_per_second = token_generated as f64 / generation_time.as_secs_f64();

        println!(
            "{} tokens generated ({} token/s)",
            token_generated, tokens_per_second
        );

        let generated_text = self.decode(&generated_tokens)?;

        Ok(generated_text)
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
}

#[derive(Debug)]
pub struct Model {
    generator: Box<dyn TextGenerator>,
}

impl Model {
    pub fn new(config: &CylonConfig) -> Result<Model, E> {
        let generator: Box<dyn TextGenerator> = match config.model_family.as_str() {
            "llama" => {
                let model = Self::load_llama_model(config)?;
                Box::new(model)
            }
            // "qwen" => Box::new(QwenModel::new(config)?),
            // "mistral" => Box::new(MistralModel::new(config)?),
            _ => bail!("Unsupported model family: {}", config.model_family),
        };

        Ok(Model { generator })
    }

    fn load_llama_model(config: &CylonConfig) -> Result<LlamaModel, E> {
        let device = device()?;
        let dtype = parse_dtype(&config.dtype)?;

        let model_dir = Path::new(&config.model_path);

        let safetensors_files = load_safetensor_model_files(&model_dir)?;

        let model_config_file = File::open(&model_dir.join("config.json"))?;
        let llama_config: LlamaConfig = serde_json::from_reader(&model_config_file)?;
        let llama_config = llama_config.into_config(config.use_flash_attn);

        let eos_handler: Box<dyn EosTokenHandler> = match &llama_config.eos_token_id {
            Some(eos_tokens) => match eos_tokens {
                LlamaEosToks::Single(id) => Box::new(SingleEosToken(*id)),
                LlamaEosToks::Multiple(ids) => Box::new(MultipleEosTokens(ids.clone())),
            },
            None => Box::new(NoEosToken),
        };

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };

        let model = llama::Llama::load(vb, &llama_config)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).map_err(E::msg)?;

        Ok(LlamaModel {
            model,
            config: llama_config,
            tokenizer,
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

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, E> {
        self.generator.generate(prompt, max_tokens)
    }
}

fn device() -> Result<Device> {
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

fn parse_dtype(dtype: &Option<String>) -> Result<DType, E> {
    match dtype.as_deref() {
        Some("f16") => Ok(DType::F16),
        Some("bf16") => Ok(DType::BF16),
        Some("f32") => Ok(DType::F32),
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => Ok(DType::F16),
    }
}

fn load_safetensor_model_files(model_path: &Path) -> Result<Vec<PathBuf>, E> {
    let model_index_file = File::open(model_path.join("model.safetensors.index.json"))?;
    let model_index_json: serde_json::Value =
        serde_json::from_reader(&model_index_file).map_err(candle_core::Error::wrap)?;
    let model_weight_map = match model_index_json.get("weight_map") {
        None => bail!("no weight map in {model_index_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
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
