//! Worker registry, adaptive service-time model and routing policies.
//!
//! Phase-3 upgrade: instead of scoring workers by raw request counts, each
//! worker carries an adaptive **service-time model**
//!
//! ```text
//! service_time = fixed_overhead
//!             + prefill_ms_per_token × input_tokens
//!             + decode_ms_per_token × output_tokens
//! ```
//!
//! whose parameters are exponential moving averages updated from gateway
//! observations (`worker_first_token_ms` and streaming-phase throughput).
//! Seeds come from the phase-2 study of this machine (MLX-LM, Qwen3.5-9B-4bit:
//! TTFT(c=1) ≈ 0.75 s + 5.35 ms/token, ~76 ms/token decode) — overridable
//! per deployment via [`ServiceDefaults`].
//!
//! [`RoutingPolicy`] decides who serves the next request:
//!
//! * [`RoutingPolicy::LeastOutstandingRequests`] — baseline (classic
//!   least-connections).
//! * [`RoutingPolicy::LeastEstimatedCompletionTime`] — `remaining active work
//!   + service_time(request)`, the "smallest expected finishing time" rule.
//! * [`RoutingPolicy::PrefixAffinity`] — prefers workers that already served
//!   (and therefore plausibly cached) the same prompt prefix.
//!
//! Workers also carry a full lifecycle (see `inference-lab-worker-lifecycle`):
//! a per-worker state machine over four health dimensions (process /
//! transport / functional / semantic) with per-reason escalation, recovery
//! hysteresis and generation epochs. Only READY and DEGRADED workers receive
//! traffic; a functional probe (a tiny real generation) is required before
//! a worker is READY — `/health` alone never suffices, the phase-4A silent
//! wedge saw to that.

use inference_lab_worker_lifecycle::{
    FailureReason, HealthObservation, LifecyclePolicy, Transition, WorkerCondition, WorkerLifecycle,
    WorkerState,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// EWMA update weight for service-model parameters. Small on purpose: the
/// prefill regression solves a 2×2 system every update and needs a long,
/// stable memory to separate fixed overhead from per-token cost.
const EWMA_ALPHA: f64 = 0.05;

// ---------------------------------------------------------------------------
// Routing policy
// ---------------------------------------------------------------------------

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

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LeastOutstandingRequests => "lor",
            Self::LeastEstimatedCompletionTime => "ect",
            Self::PrefixAffinity => "prefix-affinity",
        }
    }
}

/// What the router needs to know about the request being placed.
#[derive(Debug, Clone, Copy)]
pub struct RequestShape {
    /// Prompt-prefix hash (prefix-cache affinity + prefill discount).
    pub prefix_hash: u64,
    /// Estimated input tokens (calibrated estimate, not tokenizer truth).
    pub input_tokens: u64,
    /// Requested output tokens (`max_tokens`).
    pub output_tokens: u64,
}

/// Result of worker selection.
pub enum SelectOutcome {
    Worker(Arc<WorkerHandle>),
    /// Every registered worker is unhealthy.
    AllUnhealthy,
}

/// Seed values for the adaptive service model. Defaults are the phase-2
/// regression for MLX-LM Qwen3.5-9B-4bit on this lab's M5; a different
/// runtime/model should be seeded with its own first measurements.
#[derive(Debug, Clone, Copy)]
pub struct ServiceDefaults {
    pub fixed_ms: f64,
    pub prefill_ms_per_tok: f64,
    pub decode_ms_per_tok: f64,
}

impl Default for ServiceDefaults {
    fn default() -> Self {
        Self {
            fixed_ms: 750.0,
            prefill_ms_per_tok: 5.35,
            decode_ms_per_tok: 76.0,
        }
    }
}

/// Clamp band for each parameter; observations outside it are noise (or a
/// pathological runtime) and get pinned to the edge, never NaN/below zero.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}

struct Ewma {
    value: Mutex<f64>,
}

impl Ewma {
    fn seeded(v: f64) -> Self {
        Self { value: Mutex::new(v) }
    }

    fn get(&self) -> f64 {
        *self.value.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One EWMA step; returns the new value.
    fn update(&self, obs: f64) -> f64 {
        let mut v = self.value.lock().unwrap_or_else(|e| e.into_inner());
        *v += EWMA_ALPHA * (obs - *v);
        *v
    }
}

/// Exponentially-weighted online least squares for
/// `y = fixed + rate·t` (prefill time vs input tokens). Solving the normal
/// equations each update is exact and unconditionally stable, unlike
/// per-parameter gradient steps which oscillate on this problem.
struct OnlineRateModel {
    fixed: Mutex<f64>,
    rate: Mutex<f64>,
    // EWMA moments: E[1], E[t], E[t²], E[y], E[t·y]
    m_n: Ewma,
    m_t: Ewma,
    m_tt: Ewma,
    m_y: Ewma,
    m_ty: Ewma,
    /// Minimum weighted variance of t before the split is identifiable.
    min_t_variance: f64,
}

impl OnlineRateModel {
    fn seeded(fixed: f64, rate: f64) -> Self {
        Self {
            fixed: Mutex::new(fixed),
            rate: Mutex::new(rate),
            m_n: Ewma::seeded(1.0),
            m_t: Ewma::seeded(0.0),
            m_tt: Ewma::seeded(0.0),
            m_y: Ewma::seeded(0.0),
            m_ty: Ewma::seeded(0.0),
            min_t_variance: 1.0,
        }
    }

    fn fixed(&self) -> f64 {
        *self.fixed.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn rate(&self) -> f64 {
        *self.rate.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Feed one (t, y) observation; refits fixed/rate when the observed
    /// token counts have enough spread to identify both parameters.
    fn observe(&self, t: f64, y: f64) {
        self.m_n.update(1.0);
        self.m_t.update(t);
        self.m_tt.update(t * t);
        self.m_y.update(y);
        self.m_ty.update(t * y);

        let (n, mt, mtt, my, mty) =
            (self.m_n.get(), self.m_t.get(), self.m_tt.get(), self.m_y.get(), self.m_ty.get());
        let var_t = mtt - mt * mt / n;
        if var_t < self.min_t_variance * n {
            // Not enough spread in prompt sizes yet — keep the seeds.
            return;
        }
        let rate = clamp((mty - mt * my / n) / var_t, 0.1, 50.0);
        let fixed = clamp((my - rate * mt) / n, 0.0, 10_000.0);
        *self.rate.lock().unwrap_or_else(|e| e.into_inner()) = rate;
        *self.fixed.lock().unwrap_or_else(|e| e.into_inner()) = fixed;
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct WorkerHandle {
    pub id: String,
    pub endpoint: String,
    pub model: String,
    pub runtime: String,
    pub max_context: u64,
    /// Outstanding requests (raw count — LOR baseline and gauges).
    outstanding: AtomicU64,
    /// Lifecycle state machine: four health dimensions, per-reason
    /// escalation, recovery hysteresis, generation epochs. READY/DEGRADED
    /// serve traffic; everything else is quarantined.
    lifecycle: Mutex<WorkerLifecycle>,

    // Adaptive service model: prefill regression + decode EWMA.
    prefill_model: OnlineRateModel,
    decode_ms_per_tok: Ewma,

    /// Active dispatches: (dispatched_at, estimated total ms). Drives
    /// remaining-work ECT scoring and gauge export.
    active_estimates: Mutex<Vec<(Instant, f64)>>,

    /// Prefix hashes this worker has served — a proxy for what its KV/prefix
    /// cache holds.
    pub cached_prefixes: Mutex<HashSet<u64>>,
}

impl WorkerHandle {
    pub fn new(
        id: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        runtime: impl Into<String>,
    ) -> Self {
        Self::with_defaults(id, endpoint, model, runtime, ServiceDefaults::default())
    }

    pub fn with_defaults(
        id: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        runtime: impl Into<String>,
        defaults: ServiceDefaults,
    ) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            runtime: runtime.into(),
            max_context: 0,
            outstanding: AtomicU64::new(0),
            lifecycle: Mutex::new(WorkerLifecycle::new(LifecyclePolicy::default())),
            prefill_model: OnlineRateModel::seeded(defaults.fixed_ms, defaults.prefill_ms_per_tok),
            decode_ms_per_tok: Ewma::seeded(defaults.decode_ms_per_tok),
            active_estimates: Mutex::new(Vec::new()),
            cached_prefixes: Mutex::new(HashSet::new()),
        }
    }

    pub fn outstanding(&self) -> u64 {
        self.outstanding.load(Ordering::Relaxed)
    }

    pub fn inc_outstanding(&self) {
        self.outstanding.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement outstanding work. When a DRAINING worker's last request
    /// finishes, the drain completes: returns the DRAINING → STOPPED
    /// transition (normally empty).
    pub fn dec_outstanding(&self) -> Vec<Transition> {
        if self.outstanding.fetch_sub(1, Ordering::Relaxed) == 1 {
            let mut lc = self.lock_lifecycle();
            lc.notify_drained()
        } else {
            vec![]
        }
    }

    // -- service model ------------------------------------------------------

    pub fn fixed_overhead_ms(&self) -> f64 {
        self.prefill_model.fixed()
    }

    pub fn prefill_ms_per_tok(&self) -> f64 {
        self.prefill_model.rate()
    }

    pub fn decode_ms_per_tok(&self) -> f64 {
        self.decode_ms_per_tok.get()
    }

    /// Estimated total service time in ms for a request on this worker.
    /// A known prefix discounts the prefill term (cache hit).
    pub fn service_time_ms(&self, input_tokens: u64, output_tokens: u64, prefix_hit: bool) -> f64 {
        let prefill = self.prefill_ms_per_tok()
            * input_tokens as f64
            * if prefix_hit { 0.3 } else { 1.0 };
        let decode = self.decode_ms_per_tok.get() * output_tokens as f64;
        self.fixed_overhead_ms() + prefill + decode
    }

    /// Feed one `worker_first_token_ms` observation (response headers →
    /// first streamed token; queue wait excluded by construction). The
    /// (fixed, ms/token) split refits once observed prompt sizes vary.
    pub fn record_prefill(&self, input_tokens: u64, worker_first_token_ms: f64) {
        self.prefill_model
            .observe(input_tokens.max(1) as f64, worker_first_token_ms);
    }

    /// Feed one streaming observation: ms spent decoding `output_tokens`.
    pub fn record_decode(&self, output_tokens: u64, streaming_ms: f64) {
        let toks = output_tokens.max(1) as f64;
        let obs = streaming_ms / toks;
        self.decode_ms_per_tok.update(clamp(obs, 1.0, 2_000.0));
    }

    // -- active-work tracking (ECT) ------------------------------------------

    /// Register a dispatch with its estimated total service time. Returns
    /// the start instant to hand back to [`WorkerHandle::end_work`].
    pub fn begin_work(&self, est_ms: f64) -> Instant {
        let started = Instant::now();
        if let Ok(mut v) = self.active_estimates.lock() {
            v.push((started, est_ms));
        }
        started
    }

    /// Remove a finished dispatch (idempotent: first matching entry goes).
    pub fn end_work(&self, started: Instant, est_ms: f64) {
        if let Ok(mut v) = self.active_estimates.lock() {
            if let Some(pos) = v.iter().position(|(s, e)| *s == started && *e == est_ms) {
                v.swap_remove(pos);
            }
        }
    }

    /// Sum of estimated remaining work across active dispatches, ms.
    pub fn remaining_work_ms(&self) -> f64 {
        self.active_estimates
            .lock()
            .map(|v| {
                v.iter()
                    .map(|(start, est)| (est - start.elapsed().as_secs_f64() * 1000.0).max(0.0))
                    .sum()
            })
            .unwrap_or(0.0)
    }

    // -- lifecycle ------------------------------------------------------------

    fn lock_lifecycle(&self) -> std::sync::MutexGuard<'_, WorkerLifecycle> {
        self.lifecycle.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Feed one raw health observation into the worker's lifecycle.
    /// Returns the transitions it caused (callers turn them into metrics).
    pub fn observe(&self, obs: &HealthObservation) -> Vec<Transition> {
        self.lock_lifecycle().observe(obs)
    }

    /// Record a failure on `dimension` with `reason` (escalation class is
    /// the policy's business, not the caller's). Returns transitions.
    pub fn record_failure(
        &self,
        dimension: inference_lab_worker_lifecycle::Dimension,
        reason: FailureReason,
    ) -> Vec<Transition> {
        self.observe(&HealthObservation::failed(dimension, reason))
    }

    /// Transport evidence: the endpoint answered (dispatch connect or /health).
    pub fn record_transport_success(&self) -> Vec<Transition> {
        self.observe(&HealthObservation::up(inference_lab_worker_lifecycle::Dimension::Transport))
    }

    /// A real request completed with ≥1 generated token — the strongest
    /// evidence (semantic ⇒ functional ⇒ transport).
    pub fn request_served(&self) -> Vec<Transition> {
        self.lock_lifecycle().request_served()
    }

    pub fn state(&self) -> WorkerState {
        self.lock_lifecycle().state()
    }

    pub fn condition(&self) -> WorkerCondition {
        self.lock_lifecycle().condition()
    }

    pub fn generation(&self) -> u64 {
        self.lock_lifecycle().generation()
    }

    /// Total transitions ever emitted (history itself is bounded).
    pub fn total_transitions(&self) -> u64 {
        self.lock_lifecycle().total_transitions()
    }

    /// Stop assigning new work to this worker; in-flight requests finish.
    pub fn drain(&self) -> Vec<Transition> {
        self.lock_lifecycle().drain()
    }

    /// True when this worker receives new traffic (READY or DEGRADED).
    /// Kept as the router-wide name for "route to me".
    pub fn is_healthy(&self) -> bool {
        self.lock_lifecycle().serves_traffic()
    }

    // -- prefixes ----------------------------------------------------------------

    pub fn has_prefix(&self, hash: u64) -> bool {
        self.cached_prefixes
            .lock()
            .map(|s| s.contains(&hash))
            .unwrap_or(false)
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

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

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

    /// Healthy workers only (empty slice means "nobody home").
    pub fn healthy_workers(&self) -> Vec<Arc<WorkerHandle>> {
        self.workers.iter().filter(|w| w.is_healthy()).cloned().collect()
    }

    /// Pick the worker for a request. Also records the prefix on the winner.
    pub fn select(&self, req: RequestShape) -> SelectOutcome {
        let candidates = self.healthy_workers();
        if candidates.is_empty() {
            return SelectOutcome::AllUnhealthy;
        }
        let winner = match self.policy {
            RoutingPolicy::LeastOutstandingRequests => {
                min_by(&candidates, |w| w.outstanding() as f64)
            }
            RoutingPolicy::LeastEstimatedCompletionTime => {
                // smallest expected finishing time: work already on the worker
                // plus this request's own service time (prefill discounted on
                // a prefix hit — plausibly cached).
                min_by(&candidates, |w| {
                    let prefix_hit = w.has_prefix(req.prefix_hash);
                    w.remaining_work_ms()
                        + w.service_time_ms(req.input_tokens, req.output_tokens, prefix_hit)
                })
            }
            RoutingPolicy::PrefixAffinity => {
                let with_prefix: Vec<_> = candidates
                    .iter()
                    .filter(|w| w.has_prefix(req.prefix_hash))
                    .cloned()
                    .collect();
                if with_prefix.is_empty() {
                    min_by(&candidates, |w| w.outstanding() as f64)
                } else {
                    min_by(&with_prefix, |w| w.outstanding() as f64)
                }
            }
        };
        winner.remember_prefix(req.prefix_hash);
        SelectOutcome::Worker(winner)
    }
}

fn min_by<F: Fn(&Arc<WorkerHandle>) -> f64>(workers: &[Arc<WorkerHandle>], score: F) -> Arc<WorkerHandle> {
    workers
        .iter()
        .min_by(|a, b| score(a).total_cmp(&score(b)))
        .expect("candidate list is non-empty")
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker that has passed transport + functional probes (READY).
    fn worker(id: &str) -> Arc<WorkerHandle> {
        let w = Arc::new(WorkerHandle::new(id, format!("http://{id}"), "m", "mock"));
        w.record_transport_success();
        w.request_served();
        w
    }

    fn shape(prefix: u64, in_tok: u64, out_tok: u64) -> RequestShape {
        RequestShape { prefix_hash: prefix, input_tokens: in_tok, output_tokens: out_tok }
    }

    #[test]
    fn least_outstanding_picks_idle_worker() {
        let a = worker("a");
        let b = worker("b");
        a.outstanding.store(3, Ordering::Relaxed);
        let r = Router::new(vec![a, b], RoutingPolicy::LeastOutstandingRequests).unwrap();
        match r.select(shape(1, 128, 64)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, "b"),
            SelectOutcome::AllUnhealthy => panic!("expected a worker"),
        }
    }

    #[test]
    fn prefix_affinity_prefers_worker_that_served_prefix() {
        let a = worker("a");
        let b = worker("b");
        b.outstanding.store(5, Ordering::Relaxed);
        let r = Router::new(vec![a.clone(), b.clone()], RoutingPolicy::PrefixAffinity).unwrap();

        r.select(shape(42, 8, 8)); // some worker learns prefix 42
        let learned = if a.has_prefix(42) { "a" } else { "b" };
        match r.select(shape(42, 8, 8)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, learned),
            SelectOutcome::AllUnhealthy => panic!("expected a worker"),
        }
    }

    #[test]
    fn ect_prefers_less_remaining_work_over_fewer_requests() {
        // a: 3 active short dispatches; b: 1 active long dispatch.
        // LOR would pick b (fewer requests); ECT must pick a because b's
        // remaining work dominates.
        let a = worker("a");
        let b = worker("b");
        a.begin_work(100.0);
        a.begin_work(100.0);
        a.begin_work(100.0); // ~300 ms remaining
        b.begin_work(60_000.0); // ~60 s remaining
        a.outstanding.store(3, Ordering::Relaxed);
        b.outstanding.store(1, Ordering::Relaxed);

        let lor = Router::new(vec![a.clone(), b.clone()], RoutingPolicy::LeastOutstandingRequests)
            .unwrap();
        match lor.select(shape(7, 128, 64)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, "b"),
            SelectOutcome::AllUnhealthy => panic!(),
        }

        let ect = Router::new(
            vec![a.clone(), b.clone()],
            RoutingPolicy::LeastEstimatedCompletionTime,
        )
        .unwrap();
        match ect.select(shape(7, 128, 64)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, "a"),
            SelectOutcome::AllUnhealthy => panic!(),
        }
    }

    #[test]
    fn service_model_adapts_to_observations() {
        let w = WorkerHandle::with_defaults(
            "w",
            "http://w",
            "m",
            "mlx",
            ServiceDefaults { fixed_ms: 750.0, prefill_ms_per_tok: 5.35, decode_ms_per_tok: 76.0 },
        );
        // Synthetic runtime: TTFT = 500ms + 10ms/token, varied prompt sizes so
        // the (fixed, rate) split is identifiable.
        for i in 0..200 {
            let toks = 500 + (i % 7) * 350;
            w.record_prefill(toks as u64, 500.0 + 10.0 * toks as f64);
        }
        assert!((w.prefill_ms_per_tok() - 10.0).abs() < 1.0, "{}", w.prefill_ms_per_tok());
        assert!((w.fixed_overhead_ms() - 500.0).abs() < 400.0, "{}", w.fixed_overhead_ms());
        for _ in 0..200 {
            w.record_decode(100, 5_000.0); // 50 ms/token
        }
        assert!((w.decode_ms_per_tok() - 50.0).abs() < 0.5, "{}", w.decode_ms_per_tok());
    }

    #[test]
    fn unhealthy_worker_gets_no_traffic_until_recovery() {
        // `a` is raw (STARTING): prove warming is required before traffic.
        let a = Arc::new(WorkerHandle::new("a", "http://a", "m", "mock"));
        let b = worker("b");
        let r = Router::new(vec![a.clone(), b], RoutingPolicy::LeastEstimatedCompletionTime).unwrap();
        assert!(!a.is_healthy(), "a worker that never passed a probe gets no traffic");
        a.record_transport_success();
        a.request_served();
        assert!(a.is_healthy());

        // Three consecutive connect failures quarantine it (1 → degraded,
        // 3 → unhealthy — the same escalation the gateway's probes drive).
        a.record_failure(inference_lab_worker_lifecycle::Dimension::Transport, FailureReason::ConnectFailed);
        assert!(a.is_healthy(), "one strike must not quarantine");
        a.record_failure(inference_lab_worker_lifecycle::Dimension::Transport, FailureReason::ConnectFailed);
        a.record_failure(inference_lab_worker_lifecycle::Dimension::Transport, FailureReason::ConnectFailed);
        assert_eq!(a.state(), WorkerState::Unhealthy);
        match r.select(shape(1, 8, 8)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, "b"),
            SelectOutcome::AllUnhealthy => panic!(),
        }
        // Recovery is not one probe either: full ladder, then routable.
        a.record_transport_success();
        a.request_served();
        a.request_served();
        assert_eq!(a.state(), WorkerState::Ready);
        match r.select(shape(2, 8, 8)) {
            SelectOutcome::Worker(w) => assert_eq!(w.id, "a"),
            SelectOutcome::AllUnhealthy => panic!(),
        }
    }

    #[test]
    fn all_unhealthy_is_reported() {
        let a = worker("a");
        for _ in 0..3 {
            a.record_failure(
                inference_lab_worker_lifecycle::Dimension::Transport,
                FailureReason::ConnectFailed,
            );
        }
        let r = Router::new(vec![a], RoutingPolicy::LeastEstimatedCompletionTime).unwrap();
        assert!(matches!(r.select(shape(1, 8, 8)), SelectOutcome::AllUnhealthy));
    }

    #[test]
    fn end_work_removes_estimate() {
        let w = worker("w");
        let stored = w.begin_work(1234.0);
        assert!((w.remaining_work_ms() - 1234.0).abs() < 5.0);
        // A start Instant that can never match the stored one. (Two
        // consecutive Instant::now() calls can return the same tick on a
        // coarse clock, which would "match" and remove the entry — this
        // test exists to prove non-matching starts are ignored.)
        let other = stored.checked_sub(std::time::Duration::from_nanos(1)).unwrap();
        w.end_work(other, 1234.0); // no exact start match → not removed
        assert!(w.remaining_work_ms() > 0.0);
    }

    #[test]
    fn policy_parses() {
        assert_eq!(RoutingPolicy::parse("prefix-affinity"), Some(RoutingPolicy::PrefixAffinity));
        assert_eq!(RoutingPolicy::parse("nope"), None);
    }
}
