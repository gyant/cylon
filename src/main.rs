mod config;
mod model;

use anyhow::Result;
use config::CylonConfig;
use cylon::agent_server::{Agent, AgentServer};
use cylon::{InferenceReply, InferenceRequest};
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};

pub mod cylon {
    tonic::include_proto!("cylon");
}

#[derive(Debug)]
pub struct CylonAgent {
    model: Arc<model::Model>,
    system_prompt: String,
    sample_len: usize,
}

#[tonic::async_trait]
impl Agent for CylonAgent {
    async fn run_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceReply>, Status> {
        println!("Got a request: {:?}", request);

        let user_prompt = format!(
            "{{\"role\":\"user\",\"content\":\"{}\"}}",
            request.into_inner().prompt
        );

        let prompt = vec![self.system_prompt.clone(), user_prompt];

        let response = tokio::task::spawn_blocking({
            let model = Arc::clone(&self.model);
            let sample_len = self.sample_len;
            move || model.generate(prompt, sample_len)
        })
        .await
        .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let reply = InferenceReply { response };

        Ok(Response::new(reply))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CylonConfig::new()?;

    let model = Arc::new(model::Model::new(&config)?);

    let system_prompt = format!(
        "{{\"role\":\"system\",\"content\":\"{}\"}}",
        config.system_prompt
    );

    let addr = format!("{}:{}", config.listen_address, config.listen_port).parse()?;
    let agent = CylonAgent {
        model: Arc::clone(&model),
        system_prompt,
        sample_len: config.sample_len,
    };

    println!("Server listening: {}", addr);

    Server::builder()
        .add_service(AgentServer::new(agent))
        .serve(addr)
        .await?;

    Ok(())
}
