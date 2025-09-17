use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use crate::EosTokenHandler;

#[allow(unused_imports)]
use tracing::{info, debug};

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl InferenceConfig {
    pub fn create_logits_processor(&self) -> LogitsProcessor {
        let sampling = if self.temperature <= 0. {
            debug!("Using ArgMax sampling (greedy)");
            Sampling::ArgMax
        } else {
            let temperature = self.temperature;
            let sampling = match (self.top_k, self.top_p) {
                (None, None) => {
                    debug!("Using All sampling with temperature: {}", temperature);
                    Sampling::All { temperature }
                },
                (Some(k), None) => {
                    debug!("Using TopK sampling: k={}, temperature={}", k, temperature);
                    Sampling::TopK { k, temperature }
                },
                (None, Some(p)) => {
                    debug!("Using TopP sampling: p={}, temperature={}", p, temperature);
                    Sampling::TopP { p, temperature }
                },
                (Some(k), Some(p)) => {
                    debug!("Using TopKThenTopP sampling: k={}, p={}, temperature={}", k, p, temperature);
                    Sampling::TopKThenTopP { k, p, temperature }
                },
            };
            sampling
        };
        LogitsProcessor::from_sampling(self.seed.unwrap_or(42), sampling)
    }
}

pub trait ModelInference: Send + Sync {
    type Cache;
    
    fn create_cache(&self, enable_kv_cache: bool, dtype: DType, device: &Device) -> Result<Self::Cache>;
    fn forward(&self, input: &Tensor, context_index: usize, cache: &mut Self::Cache) -> Result<Tensor>;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn use_kv_cache(&self) -> bool;
    fn eos_handler(&self) -> &EosTokenHandler;
    fn clear_kv_cache(&self) -> Result<()>;
    fn supports_persistent_cache(&self) -> bool;
}

pub struct InferenceEngine;

impl InferenceEngine {
    pub fn generate<M: ModelInference>(
        model: &M,
        mut tokens: Vec<u32>,
        max_tokens: usize,
        config: &InferenceConfig,
    ) -> Result<Vec<u32>> {
        let mut cache = model.create_cache(model.use_kv_cache(), model.dtype(), model.device())?;
        let mut logits_processor = config.create_logits_processor();

        debug!("Starting generation with {} initial tokens, KV cache: {}", 
              tokens.len(), model.use_kv_cache());
        
        let mut token_generated = 0;
        let mut generated_tokens = Vec::new();
        let _initial_tokens_len = tokens.len();
                
        let prefill_start = std::time::Instant::now();
        let mut generation_start: Option<std::time::Instant> = None;

        for index in 0..max_tokens {
            let (context_size, context_index) = if model.use_kv_cache() && index > 0 {
                (1, tokens.len() - 1)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            
            if index < 3 || index % 50 == 0 {
                debug!("Token {}: context_size={}, context_index={}, total_tokens={}, ctxt_len={}", 
                       index, context_size, context_index, tokens.len(), ctxt.len());
            }
            
            let total_iter_start = std::time::Instant::now();
            let forward_start = std::time::Instant::now();
            
            let input = Tensor::new(ctxt, model.device())?.unsqueeze(0)?;
            let logits = model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let forward_time = forward_start.elapsed();
            
            let penalty_start = std::time::Instant::now();
            let logits = if config.repeat_penalty != 1. {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repeat_penalty,
                    &tokens[start_at..],
                )?
            } else {
                logits
            };
            let penalty_time = penalty_start.elapsed();
            
            let sample_start = std::time::Instant::now();
            let next_token = logits_processor.sample(&logits)?;
            let sample_time = sample_start.elapsed();
            let total_iter_time = total_iter_start.elapsed();
            
            if index < 5 || index % 50 == 0 {
                debug!("Token {}: total={:?}, forward={:?}, penalty={:?}, sample={:?}", 
                       index, total_iter_time, forward_time, penalty_time, sample_time);
            }
            token_generated += 1;
            tokens.push(next_token);
            generated_tokens.push(next_token);
            
            // Start generation timer after first token (exclude prefill)
            if generation_start.is_none() {
                let prefill_time = prefill_start.elapsed();
                debug!("Prefill completed in {:?} for {} tokens", prefill_time, tokens.len() - 1);
                generation_start = Some(std::time::Instant::now());
            }

            if model.eos_handler().is_eos_token(next_token) {
                break;
            }
        }

        let total_time = prefill_start.elapsed();
        let generation_time = generation_start.map(|s| s.elapsed()).unwrap_or_default();
        
        let total_tokens_per_second = token_generated as f64 / total_time.as_secs_f64();
        let generation_tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            (token_generated - 1) as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };

        debug!(
            "{} tokens generated | Total: {:.2} tok/s | Generation only: {:.2} tok/s | Prefill: {:?} | Generation: {:?}",
            token_generated, total_tokens_per_second, generation_tokens_per_second, 
            total_time - generation_time, generation_time
        );

        Ok(generated_tokens)
    }
}