use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

use crate::cylon_proto::{InferenceRunRequest, InferenceRunReply, Message};
use crate::prompt_queue::PromptQueue;
use crate::result_cache::ResultCache;
use cylon_inference_engine::TextGenerator;

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

// Helper struct for queue processing in background tasks
pub struct QueueProcessor {
    pub queue: Arc<Mutex<PromptQueue>>,
    pub processing: Arc<Mutex<bool>>,
    pub results: Arc<ResultCache<String, InferenceRunReply>>,
    pub model: Arc<Mutex<Box<dyn TextGenerator>>>,
    pub system_prompt: String,
    pub sample_len: usize,
}

impl QueueProcessor {
    pub async fn process_queue(&self) {
        loop {
            let mut queue = self.queue.lock().await;
            if let Some(queued_request) = queue.dequeue().await {
                let job_id = queued_request.job_id.clone();
                let request = queued_request.request;
                drop(queue); // Release queue lock
                
                debug!("Processing queued request with job_id: {}", job_id);
                
                // Process the queued request
                if let Ok(response) = self.process_inference_request(request).await {
                    // Store the result using DashMap
                    self.results.insert(job_id.clone(), InferenceRunReply {
                        response: Some(Message {
                            role: "assistant".to_string(),
                            content: response,
                        }),
                        status: "COMPLETED".to_string(),
                        uuid: job_id.clone(),
                    });
                    
                    debug!("Completed queued request: {}", job_id);
                } else {
                    error!("Failed to process queued request: {}", job_id);
                    
                    // Store error result using DashMap
                    self.results.insert(job_id.clone(), InferenceRunReply {
                        response: None,
                        status: "ERROR".to_string(),
                        uuid: job_id,
                    });
                }
                // Continue processing next item in queue
            } else {
                // No more items in queue, reset processing flag and exit
                let mut processing = self.processing.lock().await;
                *processing = false;
                debug!("Queue empty, reset processing flag to false");
                break;
            }
        }
    }

    async fn process_inference_request(&self, req: InferenceRunRequest) -> Result<String, Status> {
        crate::process_inference_request_shared(&self.model, &self.system_prompt, self.sample_len, req).await
    }
}