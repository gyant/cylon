use std::sync::Arc;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::cylon_proto::cylon_api_server::CylonApi;
use crate::cylon_proto::{InferenceRunReply, InferenceRunRequest, InferenceStatusRequest, InferenceStatusReply, InferenceResultRequest, InferenceResultResponse, Message};
use crate::queue_processor::QueueProcessor;
use crate::Cylon;

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

#[tonic::async_trait]
impl CylonApi for Cylon {
    async fn inference_run(
        &self,
        request: Request<InferenceRunRequest>,
    ) -> Result<Response<InferenceRunReply>, Status> {
        // Extract client IP from request
        let client_ip = request.remote_addr()
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        
        info!("Got a request for inference from client IP: {}", client_ip);

        debug!("Request: {:?}", request);

        let req = request.into_inner();
        let job_id = Uuid::new_v4().to_string();

        // If queue is disabled, process all requests immediately and sequentially
        if self.queue_disabled {
            debug!("Queue disabled - processing request immediately and sequentially");
            
            // Wait for any current processing to complete, then process this request
            let _processing_guard = self.processing.lock().await;
            
            let response = self.process_inference_request(req).await?;
            
            let reply = InferenceRunReply { 
                response: Some(Message{ role: "assistant".to_string(), content: response }), 
                status: "OK".to_string(), 
                uuid: job_id 
            };
            
            return Ok(Response::new(reply));
        }
        
        // Check if we're currently processing a request
        let mut processing = self.processing.lock().await;
        let is_processing = *processing;
        debug!("Processing flag is: {}", is_processing);
        
        if !is_processing {
            // No inference running - process this request immediately
            *processing = true;
            drop(processing); // Release the processing lock
            
            let response = self.process_inference_request(req).await?;

            let reply = InferenceRunReply { 
                response: Some(Message{ role: "assistant".to_string(), content: response }), 
                status: "OK".to_string(), 
                uuid: job_id 
            };

            // Spawn a task to process queued items after this one completes
            let queue = Arc::clone(&self.queue);
            let processing = Arc::clone(&self.processing);
            let results = Arc::clone(&self.results);
            let model = Arc::clone(&self.model);
            let system_prompt = self.system_prompt.clone();
            let sample_len = self.sample_len;
            
            tokio::spawn(async move {
                // Create a temporary Cylon-like struct for queue processing
                let processor = QueueProcessor {
                    queue,
                    processing,
                    results,
                    model,
                    system_prompt,
                    sample_len,
                };
                processor.process_queue().await;
            });

            Ok(Response::new(reply))
        } else {
            // Currently processing - enqueue this request and return QUEUED status
            drop(processing); // Release the processing lock
            
            let mut queue = self.queue.lock().await;
            queue.enqueue(job_id.clone(), req).await
                .map_err(|e| Status::internal(format!("Failed to enqueue request: {}", e)))?;
            drop(queue);
            
            // Store the job as QUEUED status using DashMap
            self.results.insert(job_id.clone(), InferenceRunReply {
                response: None,
                status: "QUEUED".to_string(),
                uuid: job_id.clone(),
            });
            
            let reply = InferenceRunReply { 
                response: None, 
                status: "QUEUED".to_string(), 
                uuid: job_id 
            };

            Ok(Response::new(reply))
        }
    }

    async fn inference_status(
        &self,
        request: Request<InferenceStatusRequest>,
    ) -> Result<Response<InferenceStatusReply>, Status> {
        let req = request.into_inner();
        let job_id = req.uuid;
        
        if let Some(result) = self.results.get(&job_id) {
            Ok(Response::new(InferenceStatusReply { status: result.status.clone() }))
        } else {
            Err(Status::not_found(format!("Job ID {} not found", job_id)))
        }
    }

    async fn inference_result(
        &self,
        request: Request<InferenceResultRequest>,
    ) -> Result<Response<InferenceResultResponse>, Status> {
        let req = request.into_inner();
        let job_id = req.uuid;
        
        if let Some(result) = self.results.get(&job_id) {
            Ok(Response::new(InferenceResultResponse { 
                response: result.response.clone() 
            }))
        } else {
            Err(Status::not_found(format!("Job ID {} not found", job_id)))
        }
    }
}