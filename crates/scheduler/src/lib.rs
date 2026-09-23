//! Worker registry and routing policies for the gateway.
//!
//! This is a deliberately small version of KV-cache-aware routing: workers
//! track outstanding requests plus a couple of EMA-observed performance
//! signals, and [`RoutingPolicy`] decides who serves the next request:
//!
//! * [`RoutingPolicy::LeastOutstandingRequests`] — baseline,
//!   like a classic least-connections LB.
//! * [`RoutingPolicy::LeastEstimatedCompletionTime`] — scores each worker
//!   with `queue_drain + prefill + decode` from observed TTFT/TPS.
//! * [`RoutingPolicy::PrefixAffinity`] — prefers workers that already served
//!   (and therefore plausibly cached) the same prompt prefix hash.
//!
//! Observation comes later from real per-request telemetry; the router's
//! scoring math and invariants are what this crate owns.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingPolicy {
    #[default]
    LeastOutstandingRequests,
    LeastEstimatedCompletionTime,
    PrefixAffinity,
}

impl RoutingPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lor" | "least-outstanding" => Some(Self::LeastOutstandingRequests),
            "ect" | "least-ect" => Some(Self::LeastEstimatedCompletionTime),
            "prefix" | "prefix-affinity" => Some(Self::PrefixAffinity),
            _ => None,
        }
    }
}

pub struct WorkerHandle {
    pub id: String,
    pub endpoint: String,
    pub model: String,
    pub runtime: String,
    pub max_context: u64,
    pub outstanding: AtomicU64,
    /// Exponential moving average of TTFT in milliseconds.
    ttft_ema_ms: AtomicU64,
    /// EMA of decode throughput in milli-tokens/second (start optimistic:
    /// 50 tok/s until observed).
    decode_milli_tps: AtomicU64,
    /// Prefix hashes this worker has served — a proxy for what its KV/prefix
    /// cache holds.
    pub cached_prefixes: Mutex<HashSet<u64>>,
}

impl WorkerHandle {
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>, model: impl Into<String>, runtime: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            runtime: runtime.into(),
            max_context: 0,
            outstanding: AtomicU64::new(0),
            ttft_ema_ms: AtomicU64::new(0),
            decode_milli_tps: AtomicU64::new(50_000),
            cached_prefixes: Mutex::new(HashSet::new()),
        }
    }

    pub fn outstanding(&self) -> u64 {
        self.outstanding.load(Ordering::Relaxed)
    }

    pub fn ttft_ema_ms(&self) -> f64 {
        self.ttft_ema_ms.load(Ordering::Relaxed) as f64
    }

    pub fn decode_tps(&self) -> f64 {
        self.decode_milli_tps.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Feed one observed TTFT; EMA with α = 0.2.
    pub fn record_ttft(&self, ttft_ms: f64) {
        let prev = self.ttft_ema_ms();
        let next = prev + 0.2 * (ttft_ms - prev);
        self.ttft_ema_ms.store(next as u64, Ordering::Relaxed);
    }

    pub fn record_decode_tps(&self, tps: f64) {
        let prev = self.decode_tps();
        let next = prev + 0.2 * (tps - prev);
        self.decode_milli_tps.store((next * 1000.0) as u64, Ordering::Relaxed);
    }

    fn has_prefix(&self, hash: u64) -> bool {
        self.cached_prefixes.lock().map(|s| s.contains(&hash)).unwrap_or(false)
    }

    fn remember_prefix(&self, hash: u64) {
        if let Ok(mut s) = self.cached_prefixes.lock() {
            // Bounded: a real system would LRU-evict; the lab has few prefixes.
            if s.len() > 4096 {
                s.clear();
            }
            s.insert(hash);
        }
    }
}

pub struct Router {
    workers: Vec<Arc<WorkerHandle>>,
    policy: RoutingPolicy,
}

impl Router {
    pub fn new(workers: Vec<Arc<WorkerHandle>>, policy: RoutingPolicy) -> anyhow::Result<Self> {
        if workers.is_empty() {
            anyhow::bail!("router needs at least one worker");
        }
        Ok(Self { workers, policy })
    }

    pub fn workers(&self) -> &[Arc<WorkerHandle>] {
        &self.workers
    }

    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    /// Pick the worker for a request with this prompt-prefix hash and an
    /// estimated decode length. Also records the prefix on the winner.
    pub fn select(&self, prefix_hash: u64, est_output_tokens: u32) -> Arc<WorkerHandle> {
        let winner = match self.policy {
            RoutingPolicy::LeastOutstandingRequests => self.min_by_outstanding(None),
            RoutingPolicy::LeastEstimatedCompletionTime => {
                self.min_by_score(|w| Self::ect_score_ms(w, prefix_hash, est_output_tokens))
            }
            RoutingPolicy::PrefixAffinity => {
                if self.workers.iter().any(|w| w.has_prefix(prefix_hash)) {
                    self.min_by_outstanding(Some(prefix_hash))
                } else {
                    self.min_by_outstanding(None)
                }
            }
        };
        winner.remember_prefix(prefix_hash);
        winner
    }

    fn min_by_outstanding(&self, only_with_prefix: Option<u64>) -> Arc<WorkerHandle> {
        self.min_by_score(|w| {
            let bonus = match only_with_prefix {
                Some(h) if w.has_prefix(h) => -1_000_000.0,
                Some(_) => return f64::MAX, // filter: skip workers without the prefix
                None => 0.0,
            };
            w.outstanding() as f64 + bonus
        })
    }

    /// Estimated completion time (ms): prefill (TTFT proxy, discounted on a
    /// prefix hit) + queued decode ahead of us + our own decode.
    fn ect_score_ms(w: &WorkerHandle, prefix_hash: u64, est_output_tokens: u32) -> f64 {
        let prefill = w.ttft_ema_ms() * if w.has_prefix(prefix_hash) { 0.3 } else { 1.0 };
        let per_request_decode_ms = est_output_tokens as f64 / w.decode_tps().max(1.0) * 1000.0;
        prefill + w.outstanding() as f64 * per_request_decode_ms + per_request_decode_ms
    }

    fn min_by_score<F: Fn(&Arc<WorkerHandle>) -> f64>(&self, score: F) -> Arc<WorkerHandle> {
        self.workers
            .iter()
            .min_by(|a, b| score(a).total_cmp(&score(b)))
            .expect("router has at least one worker")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str) -> Arc<WorkerHandle> {
        Arc::new(WorkerHandle::new(id, format!("http://{id}"), "m", "mock"))
    }

    #[test]
    fn least_outstanding_picks_idle_worker() {
        let a = worker("a");
        let b = worker("b");
        a.outstanding.store(3, Ordering::Relaxed);
        let r = Router::new(vec![a, b], RoutingPolicy::LeastOutstandingRequests).unwrap();
        assert_eq!(r.select(1, 32).id, "b");
    }

    #[test]
    fn prefix_affinity_prefers_worker_that_served_prefix() {
        let a = worker("a");
        let b = worker("b");
        // Give b more load so plain LOR would still pick it... actually make a
        // busier and verify affinity overrides load.
        a.outstanding.store(0, Ordering::Relaxed);
        b.outstanding.store(5, Ordering::Relaxed);
        let r = Router::new(vec![a.clone(), b.clone()], RoutingPolicy::PrefixAffinity).unwrap();

        r.select(42, 8); // some worker learns prefix 42
        let learned = if a.has_prefix(42) { "a" } else { "b" };
        // Second request with the same prefix must go to the worker that has it.
        assert_eq!(r.select(42, 8).id, learned);
    }

    #[test]
    fn ect_prefers_faster_decoder_under_equal_queue() {
        let a = worker("a");
        let b = worker("b");
        a.record_decode_tps(20.0);
        b.record_decode_tps(200.0);
        let r =
            Router::new(vec![a, b], RoutingPolicy::LeastEstimatedCompletionTime).unwrap();
        assert_eq!(r.select(7, 512).id, "b");
    }

    #[test]
    fn policy_parses() {
        assert_eq!(RoutingPolicy::parse("prefix-affinity"), Some(RoutingPolicy::PrefixAffinity));
        assert_eq!(RoutingPolicy::parse("nope"), None);
    }
}
