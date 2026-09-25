//! Functional probing and the lifecycle control loop.
//!
//! The phase-4A runtime study produced three failure signatures that a
//! `/health` poll cannot tell apart:
//!
//! ```text
//! 1. silent wedge        /health 200, generation probe hangs   (mlx-lm)
//! 2. connection failure  connect refused / stream reset
//! 3. semantic failure    HTTP 200, valid payload, 0 tokens      (vllm-metal)
//! ```
//!
//! This module owns the two probe loops that feed the workers' lifecycle
//! state machines (see `inference-lab-worker-lifecycle`):
//!
//! * [`transport_poller`] — `/health` at `HEALTH_INTERVAL_S`; feeds the
//!   transport dimension only. Never promotes a worker past WARMING.
//! * [`functional_prober`] — a tiny *real generation* at
//!   `FUNCTIONAL_PROBE_INTERVAL_S` (fast) while the worker is not READY and
//!   at `FUNCTIONAL_PROBE_INTERVAL_READY_S` (slow) while it is. Acceptance
//!   is [`judge_probe`]'s contract: HTTP success + valid payload +
//!   `completion_tokens > 0` + non-empty output. A `/health` 200 is never
//!   sufficient to declare READY — that is the whole lesson.
//!
//! Probes are cheap by construction (max 2 tokens, temperature 0) but they
//! are real GPU work: benchmark runs that must stay quiet set
//! `FUNCTIONAL_PROBE_INTERVAL_READY_S=0` to disable steady-state probing.

use crate::AppState;
use inference_lab_scheduler::WorkerHandle;
use inference_lab_worker_lifecycle::{
    Dimension, FailureReason, HealthObservation, WorkerState,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Tunables for the probe loop, all env-overridable.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Probe cadence for workers not yet READY (plus DEGRADED — the
    /// attention state). Fast on purpose: this cadence is the recovery
    /// latency after a restart.
    pub probe_interval: Duration,
    /// Steady-state cadence for READY workers: the silent-wedge detector.
    /// `None` (0) disables it — the setting for measured benchmark runs.
    pub probe_interval_ready: Option<Duration>,
    /// Per-probe request timeout. Firing this while `/health` still passes
    /// is the immediate-quarantine wedge signature.
    pub probe_timeout: Duration,
    pub probe_max_tokens: u32,
}

impl LifecycleConfig {
    pub fn from_env() -> Self {
        let f64v = |k: &str, d: f64| -> f64 {
            std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d)
        };
        Self {
            probe_interval: Duration::from_secs_f64(f64v("FUNCTIONAL_PROBE_INTERVAL_S", 5.0).max(0.5)),
            probe_interval_ready: match f64v("FUNCTIONAL_PROBE_INTERVAL_READY_S", 30.0) {
                s if s <= 0.0 => None,
                s => Some(Duration::from_secs_f64(s)),
            },
            probe_timeout: Duration::from_secs_f64(f64v("FUNCTIONAL_PROBE_TIMEOUT_S", 10.0).max(0.5)),
            probe_max_tokens: f64v("FUNCTIONAL_PROBE_MAX_TOKENS", 2.0).max(1.0) as u32,
        }
    }
}

/// The probe request readiness is defined against: as small as a real
/// inference can be, deterministic, and impossible to satisfy without
/// actually generating a token.
pub fn probe_payload(model: &str, max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with: OK"}],
        "stream": false,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

/// Judge a probe answer. Acceptance, in order:
/// HTTP success ∧ parseable completion payload ∧ `completion_tokens > 0`
/// ∧ non-empty output. Anything less is a specific failure, not "unhealthy".
/// Thinking models (Qwen3.5) spend their first tokens on reasoning —
/// `message.reasoning_content` is real generated output and counts.
pub fn judge_probe(http_ok: bool, status: u16, body: &str) -> HealthObservation {
    if !http_ok {
        return HealthObservation::failed(Dimension::Functional, FailureReason::MalformedResponse)
            .with_detail(format!("http {status}"));
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(body) else {
        return HealthObservation::failed(Dimension::Functional, FailureReason::MalformedResponse)
            .with_detail("body is not a JSON completion");
    };
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    let reasoning = v["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or_default();
    // usage is the honest token count; a runtime that omits it gets judged
    // on its words instead.
    let words = content.split_whitespace().count() + reasoning.split_whitespace().count();
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(words as u64);
    if completion_tokens == 0 {
        return HealthObservation::failed(Dimension::Functional, FailureReason::EmptyCompletion)
            .with_detail("completion carried no tokens");
    }
    if content.trim().is_empty() && reasoning.trim().is_empty() {
        return HealthObservation::failed(Dimension::Functional, FailureReason::EmptyCompletion)
            .with_detail("token count > 0 but output is empty");
    }
    HealthObservation::up(Dimension::Functional)
}

/// One functional probe: `probe_payload` with the configured timeout,
/// classified into a transport- or functional-dimension observation.
pub async fn run_functional_probe(
    client: &reqwest::Client,
    worker: &WorkerHandle,
    cfg: &LifecycleConfig,
) -> (HealthObservation, Duration) {
    let url = format!("{}/v1/chat/completions", worker.endpoint.trim_end_matches('/'));
    let payload = probe_payload(&worker.model, cfg.probe_max_tokens);
    let t0 = Instant::now();
    let obs = match client.post(&url).json(&payload).timeout(cfg.probe_timeout).send().await {
        Err(e) if e.is_timeout() => HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FunctionalProbeTimeout,
        )
        .with_detail(format!("probe exceeded {}s", cfg.probe_timeout.as_secs_f64())),
        // Connect-refused is transport truth, not a wedge: the escalation
        // class differs (accumulating vs immediate) and must not be blurred.
        Err(e) if e.is_connect() => {
            HealthObservation::failed(Dimension::Transport, FailureReason::ConnectFailed)
                .with_detail(e.to_string())
        }
        Err(e) => HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FunctionalProbeTimeout,
        )
        .with_detail(e.to_string()),
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            judge_probe(status.is_success(), status.as_u16(), &body)
        }
    };
    (obs, t0.elapsed())
}

/// Background loop: probe every worker on the cadence its state deserves,
/// feed observations into the lifecycle, publish transitions to metrics.
pub async fn functional_prober(st: std::sync::Arc<AppState>, cfg: LifecycleConfig) {
    let mut last_probe: HashMap<String, Instant> = HashMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        for w in st.router.workers() {
            let cadence = match w.state() {
                WorkerState::Ready => match cfg.probe_interval_ready {
                    Some(iv) => iv,
                    None => continue, // steady-state probing off (benchmarks)
                },
                WorkerState::Draining | WorkerState::Stopped => continue,
                // STARTING/WARMING/RECOVERING/UNHEALTHY fast, DEGRADED under
                // observation.
                _ => cfg.probe_interval,
            };
            if !last_probe.get(&w.id).map(|t| t.elapsed() >= cadence).unwrap_or(true) {
                continue;
            }
            last_probe.insert(w.id.clone(), Instant::now());
            let (obs, dur) = run_functional_probe(&st.client, w, &cfg).await;
            st.metrics
                .worker_functional_probe_seconds
                .with_label_values(&[w.id.as_str()])
                .observe(dur.as_secs_f64());
            if !obs.ok {
                st.metrics
                    .worker_functional_probe_failures_total
                    .with_label_values(&[
                        w.id.as_str(),
                        obs.reason.map(|r| r.as_str()).unwrap_or("unknown"),
                    ])
                    .inc();
            }
            let transitions = w.observe(&obs);
            st.metrics.record_transitions(w, &transitions);
        }
    }
}

/// Background `/health` poll: transport-dimension evidence only. A pass can
/// lift quarantine into RECOVERING and move STARTING → WARMING, but it can
/// never make a worker READY — that requires a real generation.
pub async fn transport_poller(st: std::sync::Arc<AppState>, interval: Duration) {
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        for w in st.router.workers() {
            let url = format!("{}/health", w.endpoint.trim_end_matches('/'));
            let obs = match st.client.get(&url).timeout(Duration::from_secs(2)).send().await {
                Ok(r) if r.status().is_success() => HealthObservation::up(Dimension::Transport),
                Ok(r) => HealthObservation::failed(Dimension::Transport, FailureReason::ConnectFailed)
                    .with_detail(format!("http {}", r.status())),
                Err(_) => HealthObservation::failed(Dimension::Transport, FailureReason::ConnectFailed),
            };
            let transitions = w.observe(&obs);
            st.metrics.record_transitions(w, &transitions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_accepts_only_real_generations() {
        let good = r#"{"choices":[{"message":{"role":"assistant","content":"OK"}}],"usage":{"prompt_tokens":9,"completion_tokens":1}}"#;
        assert!(judge_probe(true, 200, good).ok);

        // usage missing but real words present: judged on content.
        let no_usage = r#"{"choices":[{"message":{"content":"OK then"}}]}"#;
        assert!(judge_probe(true, 200, no_usage).ok);

        // Thinking models (Qwen3.5 via llama.cpp) spend max_tokens=2 on
        // reasoning: content empty, reasoning non-empty, usage honest.
        // That is a healthy generation and must pass the READY gate.
        let reasoning = r#"{"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"Thinking Process"}}],"usage":{"prompt_tokens":14,"completion_tokens":2}}"#;
        assert!(judge_probe(true, 200, reasoning).ok);
    }

    #[test]
    fn judge_rejects_the_empty_completion_signature() {
        let empty = r#"{"choices":[{"message":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":9,"completion_tokens":0}}"#;
        let obs = judge_probe(true, 200, empty);
        assert!(!obs.ok);
        assert_eq!(obs.reason, Some(FailureReason::EmptyCompletion));

        // zero tokens but non-empty content is still a semantic failure:
        // the payload lies about usage, the words are the only truth... and
        // here there are none.
        let lying = r#"{"choices":[{"message":{"content":""}}]}"#;
        assert_eq!(
            judge_probe(true, 200, lying).reason,
            Some(FailureReason::EmptyCompletion)
        );
    }

    #[test]
    fn judge_rejects_malformed_and_http_failures() {
        let obs = judge_probe(true, 200, "<html>not json</html>");
        assert_eq!(obs.reason, Some(FailureReason::MalformedResponse));

        let obs = judge_probe(false, 503, r#"{"error":"nope"}"#);
        assert!(!obs.ok);
        assert_eq!(obs.reason, Some(FailureReason::MalformedResponse));
        assert_eq!(obs.dimension, Dimension::Functional);
    }

    #[test]
    fn probe_payload_is_minimal_and_deterministic() {
        let p = probe_payload("m", 2);
        assert_eq!(p["max_tokens"], 2);
        assert_eq!(p["temperature"], 0.0);
        assert_eq!(p["stream"], false);
        assert!(p["messages"][0]["content"].as_str().unwrap().len() < 32);
    }
}
