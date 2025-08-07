use dashmap::DashMap;
use chrono::{DateTime, Utc, Duration};
use std::hash::Hash;
use std::sync::Arc;
use tokio::time;

#[derive(Debug)]
pub struct ResultCache<K: Eq + Hash + Clone, V> {
    cache: DashMap<K, (V, DateTime<Utc>)>,
    ttl: Duration,
}

impl<K: Eq + Hash + Clone, V: Clone> ResultCache<K, V> {
    pub fn new(ttl_seconds: i64) -> Self {
        ResultCache {
            cache: DashMap::new(),
            ttl: Duration::seconds(ttl_seconds),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(entry) = self.cache.get(key) {
            let (value, timestamp) = entry.value();
            if Utc::now() - *timestamp < self.ttl {
                return Some(value.clone());
            } else {
                drop(entry);
                self.cache.remove(key);
            }
        }
        None
    }

    pub fn insert(&self, key: K, value: V) {
        self.cache.insert(key, (value, Utc::now()));
    }

    /// Remove all expired entries from the cache
    pub fn cleanup_expired(&self) {
        let now = Utc::now();
        self.cache.retain(|_, (_, timestamp)| {
            now - *timestamp < self.ttl
        });
    }

    /// Get the current number of entries in the cache
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Start a background task that periodically cleans up expired entries
    /// 
    /// # Arguments
    /// * `cache` - Arc reference to the cache to clean up
    /// * `cleanup_interval_secs` - How often to run cleanup (default: 300 seconds = 5 minutes)
    pub fn start_cleanup_task(cache: Arc<Self>, cleanup_interval_secs: u64) 
    where
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        tokio::spawn(async move {
            let mut interval = time::interval(time::Duration::from_secs(cleanup_interval_secs));
            
            loop {
                interval.tick().await;
                
                let before_count = cache.len();
                cache.cleanup_expired();
                let after_count = cache.len();
                
                if before_count != after_count {
                    tracing::debug!(
                        "ResultCache cleanup: removed {} expired entries ({} -> {} entries)",
                        before_count - after_count,
                        before_count,
                        after_count
                    );
                }
            }
        });
    }
}
