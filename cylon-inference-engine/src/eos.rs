#[derive(Debug, Clone)]
pub enum EosTokenHandler {
    Single(u32),
    Multiple(Vec<u32>),
    None,
}

impl EosTokenHandler {
    pub fn is_eos_token(&self, token_id: u32) -> bool {
        match self {
            EosTokenHandler::Single(id) => token_id == *id,
            EosTokenHandler::Multiple(ids) => ids.contains(&token_id),
            EosTokenHandler::None => false,
        }
    }
}