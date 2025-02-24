mod config;
mod model;

use anyhow::{Error as E, Result};
use config::CylonConfig;

fn main() -> Result<()> {
    let config = CylonConfig::new()?;

    println!("CONFIG:\n{config:?}");

    let mut model = model::Model::new(&config)?;

    let response = model.generate(config.prompt.as_str(), config.sample_len)?;

    println!("{response}");

    Ok(())
}
