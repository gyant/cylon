mod config;
mod model;

use anyhow::{Error as E, Result};
use config::CylonConfig;
use std::sync::Arc;

fn main() -> Result<()> {
    let config = CylonConfig::new()?;

    let model = Arc::new(model::Model::new(&config)?);

    let system_prompt = format!(
        "{{\"role\":\"system\",\"content\":\"{}\"}}",
        config.system_prompt
    );

    let user_prompt = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", config.prompt);

    let prompt = vec![system_prompt.as_str(), user_prompt.as_str()];

    let response = model.generate(prompt, config.sample_len)?;

    println!("{response}");

    Ok(())
}
