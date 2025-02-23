//use candle_core::{Device, Tensor};
//
//fn main() -> Result<(), Box<dyn std::error::Error>> {
//    //let device = Device::Cpu;
//    let device = Device::new_cuda(0)?;
//
//    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
//    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
//
//    let c = a.matmul(&b)?;
//    println!("{c}");
//    Ok(())
//}
mod tokenoutputstream;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};
use std::fs::File;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Tensor};

use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::Write;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

use tokenoutputstream::TokenOutputStream;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.95")]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long, default_value = "40")]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long, default_value_t = false)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    model_path: String,

    /// Use different dtype than f16
    //#[arg(long)]
    //dtype: Option<String>,

    // TODO: Figure out feature compilation for this
    #[arg(long, default_value_t = false)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let device = device().unwrap();

    let dtype = DType::F16;

    let model_dir = std::path::Path::new(&args.model_path);
    let model_index_file = File::open(model_dir.join("model.safetensors.index.json"))?;
    let model_index_json: serde_json::Value =
        serde_json::from_reader(&model_index_file).map_err(candle_core::Error::wrap)?;
    let model_weight_map = match model_index_json.get("weight_map") {
        None => anyhow::bail!("no weight map in {model_index_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {model_index_file:?} is not a map"),
    };

    let mut safetensors_files = std::collections::HashSet::new();
    for value in model_weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }

    let safetensors_files: Vec<_> = safetensors_files
        .iter()
        .inspect(|x| println!("{x:?}"))
        .map(|v| model_dir.join(v))
        .collect();

    println!("{safetensors_files:?}");

    let model_config_file = File::open(model_dir.join("config.json"))?;
    let model_tokenizer_file = model_dir.join("tokenizer.json");

    let config: LlamaConfig = serde_json::from_reader(&model_config_file)?;
    let config = config.into_config(args.use_flash_attn);

    println!("{config:?}");

    let mut cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

    // There is a self alternative VarBuilder::from_safetensors but it's reported to take up more
    // RAM and load slower. Will need to experiment.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };

    let llama = Llama::load(vb, &config)?;

    let tokenizer = Tokenizer::from_file(&model_tokenizer_file).map_err(E::msg)?;

    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });

    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    println!("{prompt}");

    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
