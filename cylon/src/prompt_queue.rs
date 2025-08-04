use tokio::sync::mpsc::{self, Receiver, Sender};
use crate::cylon_proto::InferenceRunRequest;

#[derive(Debug, Clone)]
pub struct QueuedRequest {
    pub job_id: String,
    pub request: InferenceRunRequest,
}

#[derive(Debug)]
pub struct PromptQueue {
    sender: Sender<QueuedRequest>,
    receiver: Receiver<QueuedRequest>,
    queue_len: usize,
}

impl PromptQueue {
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer_size);  // Bounded to prevent overload
        PromptQueue { sender, receiver, queue_len: 0 }
    }

    pub async fn enqueue(&mut self, job_id: String, req: InferenceRunRequest) -> Result<(), String> {
        let queued_req = QueuedRequest { job_id, request: req };
        self.sender.send(queued_req).await.map_err(|e| format!("Queue full: {}", e))?;
        self.queue_len += 1;
        Ok(())
    }

    pub async fn dequeue(&mut self) -> Option<QueuedRequest> {
        // Use try_recv to check if there's an item immediately available
        match self.receiver.try_recv() {
            Ok(item) => {
                self.queue_len = self.queue_len.saturating_sub(1);
                Some(item)
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => None,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => None,
        }
    }


}