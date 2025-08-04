pub mod cylon_proto {
    tonic::include_proto!("cylon");
}

mod prompt_queue;
mod result_cache;
mod queue_processor;
mod api;

use anyhow::Result;
use cylon_config::CylonConfig;
use cylon_proto::{InferenceRunRequest, InferenceRunReply};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;
use cylon_models::{create_model};
use prompt_queue::PromptQueue;
use result_cache::ResultCache;

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

#[derive(Debug)]
pub struct Cylon {
    model: Arc<Mutex<Box<dyn cylon_inference_engine::TextGenerator>>>,
    system_prompt: String,
    sample_len: usize,
    queue: Arc<Mutex<PromptQueue>>,
    processing: Arc<Mutex<bool>>,
    results: Arc<ResultCache<String, InferenceRunReply>>,
    queue_disabled: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Prompt {
    pub role: String,
    pub content: String,
}

impl Cylon {
    pub fn new(config: &CylonConfig) -> anyhow::Result<Self> {
        let model = Arc::new(Mutex::new(create_model(config)?));
        
        let system_prompt = Prompt {
            role: String::from("system"),
            content: config.system_prompt.clone(),
        };
        let system_prompt = serde_json::to_string(&system_prompt)?;

        let queue = Arc::new(Mutex::new(PromptQueue::new(config.queue_buffer_size)));
        let processing = Arc::new(Mutex::new(false));
        let results = Arc::new(ResultCache::new(config.result_cache_ttl));

        // Start background cleanup task for expired results (every 5 minutes)
        ResultCache::start_cleanup_task(Arc::clone(&results), 300);

        Ok(Cylon {
            model,
            system_prompt,
            sample_len: config.sample_len,
            queue,
            processing,
            results,
            queue_disabled: config.queue_disabled,
        })
    }

    // Delegate to shared inference logic
    async fn process_inference_request(&self, req: InferenceRunRequest) -> Result<String, Status> {
        process_inference_request_shared(&self.model, &self.system_prompt, self.sample_len, req).await
    }
}

/// Shared inference processing logic used by both immediate and queued requests
async fn process_inference_request_shared(
    model: &Arc<Mutex<Box<dyn cylon_inference_engine::TextGenerator>>>,
    system_prompt: &str,
    sample_len: usize,
    req: InferenceRunRequest,
) -> Result<String, Status> {
    let mut prompt_vec: Vec<String> = vec![system_prompt.to_string()];
    
    for msg in req.messages {
        let p = Prompt {
            role: msg.role,
            content: msg.content,
        };
        let json = serde_json::to_string(&p)
            .map_err(|e| Status::internal(format!("Failed to serialize message: {}", e)))?;
        prompt_vec.push(json);
    }
    let prompt = Arc::new(prompt_vec);

    let response = tokio::task::spawn_blocking({
        let model = Arc::clone(model);
        let prompt = Arc::clone(&prompt);
        move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let model_guard = model.lock().await;
                model_guard.inference(&prompt, sample_len)
            })
        }
    })
    .await
    .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
    .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

    Ok(response)
}