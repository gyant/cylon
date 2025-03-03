mod config;
mod model;
mod tools;
mod utils;

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
Thought: you should always think about one action to take AND INCLUDE THE THOUGHT IN OUTPUT. Only one action at a time in this format:

Action:

$JSON_BLOB (inside markdown cell)

ENSURE ACTION PREFIX IS INCLUDED AND WRAP JSON IN MARKDOWN CELL.

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

        let prompt = Arc::new(vec![agent_prompt, user_prompt]);

        let mut final_found = false;
        let mut agent_prompt = Arc::new(String::from(""));

        while !final_found {
            let agent_output = tokio::task::spawn_blocking({
                let model = Arc::clone(&self.model);
                let sample_len = self.sample_len;
                let prompt = Arc::clone(&prompt);
                let agent_prompt = Arc::clone(&agent_prompt);
                let stop = vec!["Observation:", "Final Answer:"];
                move || model.agent_inference(&prompt, &agent_prompt, sample_len, Some(&stop))
            })
            .await
            .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

            agent_prompt = Arc::new((*agent_prompt).clone() + &agent_output);

            let action: AgentAction;
            if agent_output.contains("Observation:") {
                match utils::get_last_json(&agent_output) {
                    Some(json) => {
                        println!("FOUND JSON: {}", json);
                        action = serde_json::from_value(json).map_err(|e| {
                            Status::internal(format!("Failed to deserialize JSON: {}", e))
                        })?;
                    }
                    None => {
                        println!("NO JSON FOUND");
                        action = AgentAction {
                            action: String::from("parse_error"),
                        }
                    }
                };

                let actioned_output: String;

                match action.action.as_str() {
                    "get_weather" => {
                        // TODO: Get output from Agent response
                        actioned_output = tools::get_weather("Casper");
                    }
                    "parse_error" => {
                        actioned_output = String::from("JSON parse error. Please try again and ensure your Action JSON is wrapped in ```");
                    }
                    _ => {
                        let reply = AgentReply {
                            response: String::from(
                                "I'm sorry, this is not an action I currently support.",
                            ),
                        };
                        return Ok(Response::new(reply));
                    }
                }

                agent_prompt = Arc::new((*agent_prompt).clone() + " " + &actioned_output);
                println!("AGENT PROMPT: {}", &agent_prompt);
            } else if agent_output.contains("Final Answer:") {
                agent_prompt = Arc::new((*agent_prompt).clone() + &agent_output);
                println!("FINAL PROMPT: {}", &agent_prompt);
                final_found = true;
            }
        }

        let final_answer = tokio::task::spawn_blocking({
            let model = Arc::clone(&self.model);
            let prompt = Arc::clone(&prompt);
            let agent_prompt = Arc::clone(&agent_prompt);
            let sample_len = self.sample_len;
            move || model.agent_inference(&prompt, &agent_prompt, sample_len, None)
        })
        .await
        .map_err(|e| Status::internal(format!("Task failed: {}", e)))?
        .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let reply = AgentReply {
            response: final_answer,
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
