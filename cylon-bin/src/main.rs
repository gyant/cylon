use anyhow::Result;
use cylon_config::CylonConfig;
use cylon_proto::agent_server::{Agent, AgentServer};
use cylon_proto::{InferenceReply, InferenceRequest};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use cylon_model::Model;

pub mod cylon_proto {
    tonic::include_proto!("cylon_proto");
}

#[derive(Debug)]
pub struct CylonAgent {
    model: Arc<Model>,
    system_prompt: String,
    sample_len: usize,
}

#[derive(Serialize, Deserialize)]
struct Prompt {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct AgentAction {
    action: String,
    //action_input: {
    //    location: Option<String>,
    //},
}

#[tonic::async_trait]
impl Agent for CylonAgent {
    async fn run_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceReply>, Status> {
        println!("Got a request: {:?}", request);

        let user_prompt = Prompt {
            role: String::from("user"),
            content: request.into_inner().prompt,
        };

        let user_prompt = serde_json::to_string(&user_prompt)
            .map_err(|e| Status::internal(format!("Failed to parse prompt: {}", e)))?;

        let prompt = Arc::new(vec![self.system_prompt.clone(), user_prompt]);

        let response = tokio::task::spawn_blocking({
            let model = Arc::clone(&self.model);
            let prompt = Arc::clone(&prompt);
            let sample_len = self.sample_len;
            move || model.standard_inference(&prompt, sample_len, None)
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

    let model = Arc::new(Model::new(&config)?);

    let system_prompt = Prompt {
        role: String::from("system"),
        content: config.system_prompt.clone(),
    };

    let system_prompt = serde_json::to_string(&system_prompt)?;

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
