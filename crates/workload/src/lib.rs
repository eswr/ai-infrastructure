//! Scenario definitions and synthetic prompt/arrival generation.
//!
//! A [`Scenario`] fully describes one benchmark run: endpoint, concurrency
//! cap, request count, target prompt/output sizes, streaming mode, shared
//! prefix size (for prefix-cache experiments) and the arrival process.
//! Everything is seeded, so a scenario reruns identically.

use inference_lab_protocol::{ChatCompletionRequest, Message};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Arrival {
    /// All requests released immediately.
    Burst,
    /// Fixed inter-arrival interval.
    Constant { interval_s: f64 },
    /// Exponentially-distributed inter-arrival (Poisson process).
    Poisson { mean_interval_s: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    /// e.g. http://127.0.0.1:8082
    pub endpoint: String,
    pub model: String,
    /// Max in-flight requests (closed-loop cap).
    pub concurrency: usize,
    pub num_requests: usize,
    /// Target prompt length in tokens (approximate; see build_prompt).
    pub input_tokens: usize,
    /// `max_tokens` sent to the server.
    pub output_tokens: u32,
    pub stream: bool,
    /// Tokens of prompt shared verbatim by every request (prefix-cache
    /// experiments). 0 disables.
    #[serde(default)]
    pub shared_prefix_tokens: usize,
    pub arrival: Arrival,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_timeout() -> f64 {
    300.0
}

fn default_seed() -> u64 {
    42
}

impl Scenario {
    pub fn load_toml(path: &str) -> anyhow::Result<Scenario> {
        let raw = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }

    /// Release offset of each request from run start.
    pub fn arrival_offsets(&self) -> Vec<Duration> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut out = Vec::with_capacity(self.num_requests);
        let mut t = 0.0f64;
        for _ in 0..self.num_requests {
            let inc = match &self.arrival {
                Arrival::Burst => 0.0,
                Arrival::Constant { interval_s } => *interval_s,
                Arrival::Poisson { mean_interval_s } => {
                    rng.random::<f64>().ln() * -mean_interval_s
                }
            };
            t += inc;
            out.push(Duration::from_secs_f64(t));
        }
        out
    }

    /// Chat request for scenario request `idx`, with a deterministic,
    /// per-request-unique prompt (plus the shared prefix, if configured).
    pub fn request(&self, idx: usize) -> ChatCompletionRequest {
        let messages = build_messages(self, idx);
        ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            stream: self.stream,
            max_tokens: self.output_tokens,
            temperature: 1.0,
            top_p: None,
            include_usage: true,
        }
    }
}

/// Prompt synthesis: ~1 token ≈ 0.75 words for English prose, so we emit
/// `tokens * 4 / 3` words. The first `shared_prefix_tokens` worth of words
/// are identical across all requests so prefix caching has something to hit;
/// the remainder is unique per request (seeded by (seed, idx)).
pub fn build_messages(scenario: &Scenario, idx: usize) -> Vec<Message> {
    const VOCAB: &[&str] = &[
        "inference", "scheduler", "kernel", "latency", "throughput", "batching",
        "prefill", "decode", "memory", "cache", "router", "gateway", "streaming",
        "quantization", "tokens", "attention", "worker", "queue", "pressure",
        "system", "buffer", "engine", "pipeline", "server", "request", "context",
        "window", "model", "runtime", "metal", "unified", "bandwidth", "compute",
        "granularity", "eviction", "admission", "backlog", "capacity", "elastic",
        "signal", "processor", "graph", "operator", "gradient", "shard", "fabric",
    ];

    let words_for = |tokens: usize| tokens * 4 / 3;
    let prefix_words = words_for(scenario.shared_prefix_tokens);
    let total_words = words_for(scenario.input_tokens.max(scenario.shared_prefix_tokens));

    // Shared prefix: seeded only by the scenario seed.
    let prefix = if prefix_words > 0 {
        let mut rng = StdRng::seed_from_u64(scenario.seed ^ 0x5EED_0000);
        (0..prefix_words).map(|_| VOCAB[rng.random_range(0..VOCAB.len())]).collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // Per-request tail: seeded by (seed, idx).
    let mut rng = StdRng::seed_from_u64(scenario.seed.wrapping_add((idx as u64) << 32));
    let tail_len = total_words.saturating_sub(prefix.len());
    let tail: Vec<&str> = (0..tail_len)
        .map(|_| VOCAB[rng.random_range(0..VOCAB.len())])
        .collect();

    let mut body = String::with_capacity(total_words * 10);
    body.push_str("You are processing a synthetic inference benchmark. ");
    body.push_str("Summarize the following sequence of terms in one sentence:\n");
    for w in prefix.iter().chain(tail.iter()) {
        body.push_str(w);
        body.push(' ');
    }

    vec![
        Message {
            role: "system".into(),
            content: "You are a concise assistant serving the inference-lab benchmark suite.".into(),
        },
        Message { role: "user".into(), content: body },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> Scenario {
        Scenario {
            name: "test".into(),
            endpoint: "http://localhost:1".into(),
            model: "m".into(),
            concurrency: 4,
            num_requests: 10,
            input_tokens: 300,
            output_tokens: 16,
            stream: true,
            shared_prefix_tokens: 100,
            arrival: Arrival::Poisson { mean_interval_s: 0.1 },
            timeout_s: 10.0,
            seed: 7,
        }
    }

    #[test]
    fn prompts_are_deterministic_and_share_prefix() {
        let s = scenario();
        let a0 = build_messages(&s, 0);
        let a1 = build_messages(&s, 1);
        let a0_again = build_messages(&s, 0);
        assert_eq!(a0, a0_again, "same idx must be identical");
        assert_ne!(a0[1].content, a1[1].content, "different idx must differ");
        // Shared prefix: common leading words beyond the fixed instruction.
        let user0: Vec<&str> = a0[1].content.split_whitespace().collect();
        let user1: Vec<&str> = a1[1].content.split_whitespace().collect();
        let shared = user0
            .iter()
            .zip(user1.iter())
            .take_while(|(x, y)| x == y)
            .count();
        assert!(shared > 100, "expected a long shared prefix, got {shared} words");
    }

    #[test]
    fn poisson_offsets_are_monotonic_and_positive() {
        let s = scenario();
        let offs = s.arrival_offsets();
        assert_eq!(offs.len(), s.num_requests);
        assert!(offs.windows(2).all(|w| w[0] <= w[1]));
        assert!(offs.iter().all(|o| o.as_secs_f64() >= 0.0));
    }

    #[test]
    fn burst_releases_everything_at_zero() {
        let mut s = scenario();
        s.arrival = Arrival::Burst;
        assert!(s.arrival_offsets().iter().all(|o| o.is_zero()));
    }
}
