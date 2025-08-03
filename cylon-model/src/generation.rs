use anyhow::Error as E;
use anyhow::Result;

pub trait TextGenerator: std::fmt::Debug + Send + Sync {
    fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<String, E>;
    fn inference(
        &self,
        prompt: &Vec<String>,
        max_tokens: usize,
    ) -> Result<String, E>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, E>;
    fn decode(&self, tokens: &[u32]) -> Result<String, E>;
    fn render(&self, prompt: &Vec<String>) -> Result<String, E>;
}