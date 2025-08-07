use anyhow::{Context, Error as E, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use serde_yaml;
use std::fs;
use std::path::Path;

/*
  TODO: Introduce argument relationships based on model_type / model_family
  #[arg(long, default_value = "generic")]
  model_family: String,
  /// Argument X (required if model_family is "llama")
  #[arg(long, required_if_eq("model_family", "llama"))]
  x: Option<String>,
*/

/// Supported queue types for job processing
#[derive(ValueEnum, Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QueueType {
    /// Local in-memory queue (default)
    Local,
    /// Redis-based distributed queue
    Redis,
    /// Kafka-based distributed queue
    Kafka,
}

impl std::fmt::Display for QueueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueType::Local => write!(f, "local"),
            QueueType::Redis => write!(f, "redis"),
            QueueType::Kafka => write!(f, "kafka"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CliArgs {
    #[arg(long, env = "CYLON_DEBUG", default_value_t = false)]
    debug: bool,

    #[arg(long, env = "CYLON_LISTEN_ADDRESS", default_value = "127.0.0.1")]
    listen_address: String,

    #[arg(long, env = "CYLON_LISTEN_PORT", default_value = "8080")]
    listen_port: String,

    #[arg(long, env = "CYLON_QUEUE_DISABLED", default_value_t = false)]
    queue_disabled: bool,

    // TODO: Add support for different queue types
    #[arg(long, env = "CYLON_QUEUE_TYPE", default_value_t = QueueType::Local)]
    queue_type: QueueType,

    #[arg(long, env = "CYLON_QUEUE_BUFFER_SIZE", default_value_t = 100)]
    queue_buffer_size: usize,

    #[arg(long, env = "CYLON_RESULT_CACHE_TTL", default_value_t = 3600)]
    result_cache_ttl: i64,

    #[arg(long, env = "CYLON_MODEL_FAMILY", default_value = "llama")]
    model_family: String,

    #[arg(
        long,
        env = "CYLON_MODEL_PATH",
        default_value = "/data/models/my-model"
    )]
    model_path: String,

    #[arg(long, env = "CYLON_CONFIG_FILE")]
    config_file: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long, env = "CYLON_TEMPERATURE", default_value_t = 0.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, env = "CYLON_TOP_P")]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long, env = "CYLON_TOP_K")]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, env = "CYLON_SEED", default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, env = "CYLON_SAMPLE_LEN", default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long, env = "CYLON_ENABLE_KV_CACHE", default_value_t = true)]
    enable_kv_cache: bool,

    /// The system prompt.
    #[arg(long, env = "CYLON_SYSTEM_PROMPT", default_value_t = String::from("You are a helpful assistant."))]
    system_prompt: String,

    /// Use different dtype than f16
    #[arg(long, env = "CYLON_DTYPE", default_value = "f16")]
    dtype: Option<String>,

    #[arg(long, env = "CYLON_USE_FLASH_ATTN", default_value_t = false)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, env = "CYLON_REPEAT_PENALTY", default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, env = "CYLON_REPEAT_LAST_N", default_value_t = 128)]
    repeat_last_n: usize,
}

#[derive(Debug, Deserialize)]
pub struct CylonConfig {
    pub debug: bool,
    pub listen_address: String,
    pub listen_port: String,
    pub queue_disabled: bool,
    pub queue_type: QueueType,
    pub queue_buffer_size: usize,
    pub result_cache_ttl: i64,
    pub model_family: String,
    pub model_path: String,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub sample_len: usize,
    pub enable_kv_cache: bool,
    pub system_prompt: String,
    pub dtype: Option<String>,
    pub use_flash_attn: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl CylonConfig {
    pub fn new() -> Result<CylonConfig, E> {
        let args = CliArgs::parse();

        let yaml_config = if let Some(config_file) = args.config_file {
            let config_path = Path::new(&config_file);
            let content = fs::read_to_string(config_path).with_context(|| {
                format!("Config file not found: {}", config_path.to_string_lossy())
            })?;
            serde_yaml::from_str(&content).with_context(|| "Failed to deserialize YAML config")?
        } else {
            CylonConfig {
                debug: args.debug,
                listen_address: args.listen_address,
                listen_port: args.listen_port,
                queue_disabled: args.queue_disabled,
                queue_type: args.queue_type,
                queue_buffer_size: args.queue_buffer_size,
                result_cache_ttl: args.result_cache_ttl,
                model_family: args.model_family,
                model_path: args.model_path,
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                seed: args.seed,
                sample_len: args.sample_len,
                enable_kv_cache: args.enable_kv_cache,
                system_prompt: args.system_prompt,
                dtype: args.dtype,
                use_flash_attn: args.use_flash_attn,
                repeat_penalty: args.repeat_penalty,
                repeat_last_n: args.repeat_last_n,
            }
        };

        Ok(yaml_config)
    }
}
