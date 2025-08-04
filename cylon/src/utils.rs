use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// TODO: Set a dynamic LOG_LEVEL environment variable to support error / warn / info / debug / trace
pub fn init_logging(debug: bool) {
    let base_level = if debug { "debug" } else { "info" };
    
    let filter = EnvFilter::new(base_level)
        .add_directive("tokenizers::tokenizer::serialization=error".parse().unwrap());

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().json()) // JSON output
        .init();
}