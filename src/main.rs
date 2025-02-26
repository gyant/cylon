mod config;
mod model;

use anyhow::{Error as E, Result};
use config::CylonConfig;
use std::sync::Arc;

fn main() -> Result<()> {
    let config = CylonConfig::new()?;

    let model = Arc::new(model::Model::new(&config)?);

    let response = model.generate(config.prompt.as_str(), config.sample_len)?;

    println!("{response}");

    Ok(())
}
