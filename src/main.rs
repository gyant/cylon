mod config;
mod model;

use anyhow::Result;
use config::CylonConfig;
use cylon::agent_server::{Agent, AgentServer};
use cylon::{AgentReply, AgentRequest};
use cylon::{InferenceReply, InferenceRequest};
use serde::{Deserialize, Serialize};
use serde_json;
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

#[derive(Serialize, Deserialize)]
struct Prompt {
    role: String,
    content: String,
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

        let prompt = vec![self.system_prompt.clone(), user_prompt];

        let response = tokio::task::spawn_blocking({
            let model = Arc::clone(&self.model);
            let sample_len = self.sample_len;
            move || model.generate(prompt, sample_len, None)
        })
        .await
        .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let reply = InferenceReply { response };

        Ok(Response::new(reply))
    }

    async fn run_agent(
        &self,
        request: Request<AgentRequest>,
    ) -> Result<Response<AgentReply>, Status> {
        println!("Got a request: {:?}", request);

        let agent_system_prompt = String::from(
            r#"
Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use : 

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer.  
"#,
        );

        let agent_prompt = Prompt {
            role: String::from("system"),
            content: agent_system_prompt,
        };

        let agent_prompt = serde_json::to_string(&agent_prompt).map_err(|e| {
            Status::internal(format!(
                "Problem converting agent_system_prompt to Prompt: {}",
                e
            ))
        })?;

        let user_prompt = Prompt {
            role: String::from("user"),
            content: request.into_inner().prompt,
        };

        let user_prompt = serde_json::to_string(&user_prompt)
            .map_err(|e| Status::internal(format!("Failed to parse prompt: {}", e)))?;

        let prompt = vec![agent_prompt, user_prompt];

        let observation = tokio::task::spawn_blocking({
            let model = Arc::clone(&self.model);
            let sample_len = self.sample_len;
            move || model.generate(prompt, sample_len, Some("Observation: "))
        })
        .await
        .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let reply = AgentReply {
            response: observation,
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CylonConfig::new()?;

    let model = Arc::new(model::Model::new(&config)?);

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
