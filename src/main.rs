mod config;
mod model;

use anyhow::{Error as E, Result};
use config::CylonConfig;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() -> Result<()> {
    let config = CylonConfig::new()?;

    let model = Arc::new(model::Model::new(&config)?);

    let system_prompt = format!(
        "{{\"role\":\"system\",\"content\":\"{}\"}}",
        config.system_prompt
    );

    // let user_prompt = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", config.prompt);

    // let prompt = vec![system_prompt.as_str(), user_prompt.as_str()];

    // let response = model.generate(prompt, config.sample_len)?;

    // println!("{response}");

    let prompt1 = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", "Tell me about cats");
    let prompt2 = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", "Tell me about dogs");
    let prompt3 = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", "Tell me about birds");
    let prompt4 = format!("{{\"role\":\"user\",\"content\":\"{}\"}}", "Tell me about snakes");

    let full_prompt1 = vec![system_prompt.clone(), prompt1];
    let full_prompt2 = vec![system_prompt.clone(), prompt2];
    let full_prompt3 = vec![system_prompt.clone(), prompt3];
    let full_prompt4 = vec![system_prompt.clone(), prompt4];

    println!("Starting concurrent generation...");
    let start = Instant::now();

    // Clone Arc references for each thread
    let model1 = model.clone();
    let model2 = model.clone();
    let model3 = model.clone();
    let model4 = model.clone();
    let sample_len = config.sample_len;

    // Spawn two threads
    let handle1 = thread::spawn(move || {
        model1.generate(full_prompt1, sample_len)
    });

    let handle2 = thread::spawn(move || {
        model2.generate(full_prompt2, sample_len)
    });

    let handle3 = thread::spawn(move || {
        model3.generate(full_prompt3, sample_len)
    });

    let handle4 = thread::spawn(move || {
        model4.generate(full_prompt4, sample_len)
    });

    // Wait for both to complete
    let result1 = handle1.join().expect("Thread 1 panicked")?;
    let result2 = handle2.join().expect("Thread 2 panicked")?;
    let result3 = handle3.join().expect("Thread 3 panicked")?;
    let result4 = handle4.join().expect("Thread 4 panicked")?;

    let duration = start.elapsed();
    println!("Both generations completed in {:?}", duration);

    // Print results
    println!("\n=== RESULT 1 (CATS) ===");
    println!("{}", result1);
    
    println!("\n=== RESULT 2 (DOGS) ===");
    println!("{}", result2);

    println!("\n=== RESULT 3 (BIRDS) ===");
    println!("{}", result3);

    println!("\n=== RESULT 4 (SNAKES) ===");
    println!("{}", result4);

    Ok(())
}
