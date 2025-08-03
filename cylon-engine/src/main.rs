use anyhow::Result;
use cylon_config::CylonConfig;
use cylon_proto::engine_server::{Engine, EngineServer};
use cylon_proto::{InferenceReply, InferenceRequest};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use cylon_model::Model;

pub mod cylon_proto {
    tonic::include_proto!("cylon_engine");
}

#[derive(Debug)]
pub struct CylonEngine {
    model: Arc<Model>,
    system_prompt: String,
    sample_len: usize,
}

#[derive(Serialize, Deserialize)]
struct Prompt {
    role: String,
    content: String,
}

#[tonic::async_trait]
impl Engine for CylonEngine {
    async fn run_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceReply>, Status> {
        println!("Got a request: {:?}", request);

        let req = request.into_inner();
        let mut prompt_vec: Vec<String> = vec![self.system_prompt.clone()];
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
            let model = Arc::clone(&self.model);
            let prompt = Arc::clone(&prompt);
            let sample_len = self.sample_len;
            move || model.inference(&prompt, sample_len)
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
    println!("Starting Cylon Engine");
    let config = CylonConfig::new()?;

    println!("Loading model");
    let model = Arc::new(Model::new(&config)?);

    println!("Loading system prompt");
    let system_prompt = Prompt {
        role: String::from("system"),
        content: config.system_prompt.clone(),
    };

    let system_prompt = serde_json::to_string(&system_prompt)?;

    let addr = format!("{}:{}", config.listen_address, config.listen_port).parse()?;
    let agent = CylonEngine {
        model: Arc::clone(&model),
        system_prompt,
        sample_len: config.sample_len,
    };

    println!("Server listening: {}", addr);

    Server::builder()
        .add_service(EngineServer::new(agent))
        .serve(addr)
        .await?;

    Ok(())
}
