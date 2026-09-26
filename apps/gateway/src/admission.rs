//! Admission control: bound what reaches the workers using *estimated work*,
//! not request counts.
//!
//! Phase 2 showed why: at 8K tokens × concurrency 8 the MLX worker accepted
//! everything, time-sliced prefill across 8 requests, and 16/16 requests
//! timed out with zero tokens. The fix is not "fewer requests" but "bounded
//! concurrent work": each request carries an estimated cost
//!
//! ```text
//! est_work = fixed_overhead + prefill_ms/tok × input_tokens
//!                            + decode_ms/tok × output_tokens
//! ```
//!
//! (from the scheduler's adaptive service model) and the controller keeps
//!
//! ```text
//! Σ active input_tokens  ≤ max_active_input_tokens
//! Σ active output_tokens ≤ max_active_output_tokens
//! ```
//!
//! Requests that don't fit join a FIFO queue accounted in tokens and
//! estimated milliseconds — `queue_depth = 4` says nothing, `queue holds
//! 31k tokens ≈ 190 s of work` says everything. Overload is explicit: when
//! the estimated wait for a new arrival exceeds the overload threshold the
//! gateway answers `429` with `Retry-After` instead of admitting work it
//! cannot serve in reasonable time.
//!
//! Cancellation is wired through: a client that disconnects while its
//! request is still queued is removed from the queue (via the Drop guard on
//! the handler future — works for HTTP/2 stream resets; on HTTP/1.1 the
//! disconnect is only observable at response-write time), releasing the
//! queued work immediately.

use inference_lab_scheduler::ServiceClass;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;
use tokio::sync::Notify;

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Token/work budgets. `0` disables a limit. Admission is active when any
/// budget is set; with all disabled the gateway proxies unbounded (phase-2
/// behavior, kept for A/B experiments).
#[derive(Debug, Clone, Copy)]
pub struct AdmissionPolicy {
    /// Σ estimated input tokens of active requests (prefill budget).
    pub max_active_input_tokens: u64,
    /// Σ estimated output tokens of active requests (decode budget).
    pub max_active_output_tokens: u64,
    /// Reject single requests whose estimated work exceeds this (ms).
    pub max_estimated_work_ms: u64,
    /// Estimated queue wait beyond which the gateway reports DEGRADED.
    pub degraded_queue_s: f64,
    /// Estimated queue wait beyond which new arrivals get 429 + Retry-After.
    pub overloaded_queue_s: f64,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            max_active_input_tokens: 0,
            max_active_output_tokens: 0,
            max_estimated_work_ms: 0,
            degraded_queue_s: 5.0,
            overloaded_queue_s: 30.0,
        }
    }
}

impl AdmissionPolicy {
    pub fn from_env() -> Self {
        let env_u64 = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let env_f64 = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or(d)
        };
        Self {
            max_active_input_tokens: env_u64("ADMIT_MAX_ACTIVE_INPUT_TOKENS").unwrap_or(0),
            max_active_output_tokens: env_u64("ADMIT_MAX_ACTIVE_OUTPUT_TOKENS").unwrap_or(0),
            max_estimated_work_ms: env_u64("ADMIT_MAX_WORK_MS").unwrap_or(0),
            degraded_queue_s: env_f64("ADMIT_DEGRADED_QUEUE_S", 5.0),
            overloaded_queue_s: env_f64("ADMIT_OVERLOADED_QUEUE_S", 30.0),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_active_input_tokens > 0 || self.max_active_output_tokens > 0
    }

    fn fits(&self, active: &AdmissionState, size: &RequestSize) -> bool {
        fits_vals(self, active.active_input_tokens, active.active_output_tokens, size)
    }
}

/// What one request costs, in the units the controller budgets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestSize {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Estimated total service time in ms (scheduler service model).
    pub est_work_ms: f64,
    /// Requested service class: decides grant priority when capacity
    /// frees (interactive first — phase 4C).
    pub class: ServiceClass,
}

// ---------------------------------------------------------------------------
// Load state
// ---------------------------------------------------------------------------

/// Budget check against explicit running totals (used by `release` while
/// scanning the queue without mutating state mid-iteration).
fn fits_vals(policy: &AdmissionPolicy, act_in: u64, act_out: u64, size: &RequestSize) -> bool {
    if policy.max_active_input_tokens > 0
        && act_in + size.input_tokens > policy.max_active_input_tokens
    {
        return false;
    }
    if policy.max_active_output_tokens > 0
        && act_out + size.output_tokens > policy.max_active_output_tokens
    {
        return false;
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadState {
    Normal = 0,
    Degraded = 1,
    Overloaded = 2,
}

impl LoadState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Degraded => "degraded",
            Self::Overloaded => "overloaded",
        }
    }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

pub struct Queued {
    size: RequestSize,
    enqueued_at: Instant,
    /// Queue depth (requests) at enqueue time — trace attribute.
    depth_at_enqueue: usize,
    granted: AtomicBool,
    cancelled: AtomicBool,
    /// Per-entry wake signal. A stored `notify_one` permit makes the
    /// grant→wake handoff race-free.
    notify: Notify,
}

impl Queued {
    pub fn class(&self) -> ServiceClass {
        self.size.class
    }
}

pub enum AdmissionDecision {
    /// Budgets fit and nobody was waiting — dispatch now; the permit owns
    /// the granted capacity.
    Admit(AdmissionPermit),
    /// Must wait; await [`AdmissionController::wait_for_grant`].
    Queue(Arc<Queued>),
    /// Refused at the door. `retry_after_s` feeds the `Retry-After` header.
    Reject { reason: &'static str, retry_after_s: u64 },
}

impl std::fmt::Debug for AdmissionDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admit(p) => f.debug_tuple("Admit").field(&p.size).finish(),
            Self::Queue(_) => f.debug_tuple("Queue").finish(),
            Self::Reject { reason, retry_after_s } => f
                .debug_struct("Reject")
                .field("reason", reason)
                .field("retry_after_s", retry_after_s)
                .finish(),
        }
    }
}

#[derive(Default)]
struct AdmissionState {
    active_requests: u64,
    active_input_tokens: u64,
    active_output_tokens: u64,
    queue: VecDeque<Arc<Queued>>,
}

/// Indices into the queue in grant-consideration order: service class
/// priority (interactive → standard → batch), FIFO within a class.
fn priority_order(queue: &VecDeque<Arc<Queued>>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..queue.len()).collect();
    order.sort_by_key(|&i| (queue[i].size.class.queue_rank(), i));
    order
}

impl AdmissionState {
    fn queued_input_tokens(&self) -> u64 {
        self.queue.iter().map(|q| q.size.input_tokens).sum()
    }

    fn queued_output_tokens(&self) -> u64 {
        self.queue.iter().map(|q| q.size.output_tokens).sum()
    }

    fn queued_work_ms(&self) -> f64 {
        self.queue.iter().map(|q| q.size.est_work_ms).sum()
    }

    /// Heuristic wait for a new arrival: queued work drains at roughly the
    /// rate of the currently-active requests working in parallel. It is a
    /// conservative lower bound (active requests still mid-flight also hold
    /// capacity), which is what an admission decision wants to be.
    fn estimated_wait_s(&self) -> f64 {
        (self.queued_work_ms() / 1000.0) / (self.active_requests as f64).max(1.0)
    }

    fn load_state(&self, policy: &AdmissionPolicy) -> LoadState {
        let w = self.estimated_wait_s();
        if w >= policy.overloaded_queue_s {
            LoadState::Overloaded
        } else if w >= policy.degraded_queue_s {
            LoadState::Degraded
        } else {
            LoadState::Normal
        }
    }
}

pub struct AdmissionController {
    policy: AdmissionPolicy,
    state: Mutex<AdmissionState>,
}

/// Handle proving the request holds active capacity. Dropping it releases
/// the budgets and re-checks the queue — no error path can leak a slot.
pub struct AdmissionPermit {
    ctrl: Arc<AdmissionController>,
    size: RequestSize,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.ctrl.release(self.size);
    }
}

#[derive(Debug, Clone)]
pub struct AdmissionSnapshot {
    pub enabled: bool,
    pub active_requests: u64,
    pub active_input_tokens: u64,
    pub active_output_tokens: u64,
    pub queued_requests: usize,
    pub queued_input_tokens: u64,
    pub queued_output_tokens: u64,
    pub estimated_queue_seconds: f64,
    pub load_state: LoadState,
}

impl AdmissionController {
    pub fn new(policy: AdmissionPolicy) -> Arc<Self> {
        Arc::new(Self { policy, state: Mutex::new(AdmissionState::default()) })
    }

    fn lock(&self) -> MutexGuard<'_, AdmissionState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Decide what to do with an arriving request.
    pub fn decide(self: &Arc<Self>, size: RequestSize) -> AdmissionDecision {
        let mut st = self.lock();

        // A single request whose estimated work alone exceeds the cap can
        // never be served in reasonable time — refuse rather than queue.
        if self.policy.max_estimated_work_ms > 0
            && size.est_work_ms > self.policy.max_estimated_work_ms as f64
        {
            return AdmissionDecision::Reject {
                reason: "estimated_work_exceeds_cap",
                retry_after_s: 1,
            };
        }

        // Overload gate: refuse new arrivals when the queue they would join
        // is already longer (in estimated time) than the overload threshold.
        let est_wait = st.estimated_wait_s();
        if est_wait >= self.policy.overloaded_queue_s {
            return AdmissionDecision::Reject {
                reason: "overloaded",
                retry_after_s: est_wait.ceil().max(1.0) as u64,
            };
        }

        // Immediate admission when the queue is empty AND budgets fit — a
        // busy queue is never barged past. When the machine is IDLE the
        // budget is soft: admitting a request that overshoots it (a prompt
        // a hair over the cap) is bounded by one max-size request and is
        // strictly better than idling — without this, a budget sized at the
        // workload's nominal token count deadlocks on estimator jitter.
        // (The matching release-side rule force-grants the queue head when
        // nothing fits and nothing is active.)
        if !self.policy.enabled() || (st.queue.is_empty() && (self.policy.fits(&st, &size) || st.active_requests == 0))
        {
            st.active_requests += 1;
            st.active_input_tokens += size.input_tokens;
            st.active_output_tokens += size.output_tokens;
            return AdmissionDecision::Admit(AdmissionPermit { ctrl: self.clone(), size });
        }

        let depth = st.queue.len();
        let entry = Arc::new(Queued {
            size,
            enqueued_at: Instant::now(),
            depth_at_enqueue: depth,
            granted: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            notify: Notify::new(),
        });
        st.queue.push_back(entry.clone());
        AdmissionDecision::Queue(entry)
    }

    /// Wait until a queued entry is granted. Returns `None` if the entry was
    /// cancelled (client gone). Hold the returned permit for the lifetime of
    /// the request; dropping it releases capacity.
    pub async fn wait_for_grant(
        self: &Arc<Self>,
        entry: &Arc<Queued>,
    ) -> Option<AdmissionPermit> {
        loop {
            if entry.granted.load(Ordering::Acquire) {
                return Some(AdmissionPermit { ctrl: self.clone(), size: entry.size });
            }
            if entry.cancelled.load(Ordering::Acquire) {
                return None;
            }
            entry.notify.notified().await;
        }
    }

    /// Remove a queued entry because its client is gone. Idempotent.
    /// Returns the saved estimated work in ms (0 if it was never queued or
    /// already granted).
    pub fn cancel_queued(&self, entry: &Arc<Queued>) -> f64 {
        let mut st = self.lock();
        if entry.granted.load(Ordering::Acquire) || entry.cancelled.swap(true, Ordering::AcqRel) {
            return 0.0;
        }
        let before = st.queue.len();
        st.queue.retain(|q| !Arc::ptr_eq(q, entry));
        let saved = if st.queue.len() < before { entry.size.est_work_ms } else { 0.0 };
        drop(st);
        if saved > 0.0 {
            entry.notify.notify_one(); // wake the waiter so it observes the cancel
        }
        saved
    }

    /// How long an entry has been waiting (for traces/metrics at grant time).
    pub fn queue_wait(&self, entry: &Arc<Queued>) -> std::time::Duration {
        entry.enqueued_at.elapsed()
    }

    pub fn queue_depth_at_enqueue(entry: &Arc<Queued>) -> usize {
        entry.depth_at_enqueue
    }

    fn release(&self, size: RequestSize) {
        let mut st = self.lock();
        st.active_requests = st.active_requests.saturating_sub(1);
        st.active_input_tokens = st.active_input_tokens.saturating_sub(size.input_tokens);
        st.active_output_tokens = st.active_output_tokens.saturating_sub(size.output_tokens);
        // Grant waiters that now fit. FIFO scan with skip-ahead: a small
        // request may pass a too-big head (avoids head-of-line blocking)
        // but grants otherwise happen in arrival order. Running totals are
        // simulated first so the queue is only read, then applied.
        // Class-priority grant scan (phase 4C): interactive entries are
        // considered before standard before batch, FIFO within each class.
        // Priority changes the scan ORDER only — the fits checks and the
        // decide-side no-barging rule are untouched, so a class never
        // overruns budgets, it just gets first claim on freed capacity.
        if self.policy.enabled() {
            let mut act_in = st.active_input_tokens;
            let mut act_out = st.active_output_tokens;
            let mut granted = Vec::new();
            let order = priority_order(&st.queue);
            for i in order.iter() {
                let q = &st.queue[*i];
                if q.granted.load(Ordering::Acquire) || q.cancelled.load(Ordering::Acquire) {
                    continue; // stale entry awaiting cleanup
                }
                if fits_vals(&self.policy, act_in, act_out, &q.size) {
                    act_in += q.size.input_tokens;
                    act_out += q.size.output_tokens;
                    q.granted.store(true, Ordering::Release);
                    granted.push(q.clone());
                }
            }
            // Liveness rule: nothing fits and nothing is active — the queue
            // would deadlock. Force-grant the head (highest-priority
            // non-cancelled); overshoot is bounded by one max-size request
            // (e.g. a prompt a hair over the budget).
            if granted.is_empty() && st.active_requests == 0 {
                if let Some(head) = order.iter().map(|i| &st.queue[*i]).find(|q| {
                    !q.cancelled.load(Ordering::Acquire)
                }) {
                    let head = head.clone();
                    act_in += head.size.input_tokens;
                    act_out += head.size.output_tokens;
                    head.granted.store(true, Ordering::Release);
                    granted.push(head);
                }
            }
            if !granted.is_empty() {
                st.active_requests += granted.len() as u64;
                st.active_input_tokens = act_in;
                st.active_output_tokens = act_out;
                st.queue.retain(|q| !q.granted.load(Ordering::Acquire));
            }
            drop(st);
            for q in granted {
                q.notify.notify_one();
            }
        }
    }

    /// Count queued entries per service class (`[interactive, standard,
    /// batch]`) — the queue-composition gauge.
    pub fn queued_by_class(&self) -> [u64; 3] {
        let st = self.lock();
        let mut out = [0u64; 3];
        for q in st.queue.iter() {
            out[q.size.class.queue_rank() as usize] += 1;
        }
        out
    }

    pub fn snapshot(&self) -> AdmissionSnapshot {
        let st = self.lock();
        AdmissionSnapshot {
            enabled: self.policy.enabled(),
            active_requests: st.active_requests,
            active_input_tokens: st.active_input_tokens,
            active_output_tokens: st.active_output_tokens,
            queued_requests: st.queue.len(),
            queued_input_tokens: st.queued_input_tokens(),
            queued_output_tokens: st.queued_output_tokens(),
            estimated_queue_seconds: st.estimated_wait_s(),
            load_state: st.load_state(&self.policy),
        }
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Input-token estimation with runtime calibration.
///
/// Base heuristic: the lab's synthetic prose is ~1.33 words per token (see
/// `inference-lab-workload`), so `tokens ≈ words × 3/4`. When the server
/// reports true `prompt_tokens` (usage chunk), an EWMA of
/// `actual / estimated` nudges future estimates toward reality — the
/// estimator learns the workload instead of hard-coding a tokenizer.
pub struct TokenEstimator {
    /// actual/estimated EWMA, seeded at 1.0.
    ratio: Mutex<f64>,
}

impl TokenEstimator {
    pub fn new() -> Self {
        Self { ratio: Mutex::new(1.0) }
    }

    pub fn estimate(&self, req: &inference_lab_protocol::ChatCompletionRequest) -> u64 {
        let words: u64 = req
            .messages
            .iter()
            .map(|m| m.content.split_whitespace().count() as u64)
            .sum();
        let raw = (words * 3 / 4).max(1);
        let r = *self.ratio.lock().unwrap_or_else(|e| e.into_inner());
        ((raw as f64 * r).ceil() as u64).max(1)
    }

    /// Feed a server-reported `prompt_tokens` for a request estimated at
    /// `estimated`. α = 0.1, bounded so a single weird response can't skew
    /// sizing by more than ±25% per observation.
    pub fn calibrate(&self, estimated: u64, actual: u64) {
        if estimated == 0 || actual == 0 {
            return;
        }
        let obs = (actual as f64 / estimated as f64).clamp(0.25, 4.0);
        let mut r = self.ratio.lock().unwrap_or_else(|e| e.into_inner());
        *r += 0.1 * (obs - *r);
        // Keep the multiplier sane long-term.
        *r = r.clamp(0.25, 4.0);
    }

    pub fn ratio(&self) -> f64 {
        *self.ratio.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ctrl(inp: u64, out: u64) -> Arc<AdmissionController> {
        AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: inp,
            max_active_output_tokens: out,
            ..Default::default()
        })
    }

    fn size(in_tok: u64, out_tok: u64) -> RequestSize {
        RequestSize {
            input_tokens: in_tok,
            output_tokens: out_tok,
            est_work_ms: in_tok as f64 * 5.0 + out_tok as f64 * 75.0,
            class: ServiceClass::Standard,
        }
    }

    fn sized(in_tok: u64, out_tok: u64, class: ServiceClass) -> RequestSize {
        RequestSize { class, ..size(in_tok, out_tok) }
    }

    /// decide() → owned permit, panicking on anything but immediate admit.
    /// Hold the result for as long as the "request" should stay active.
    fn admit(c: &Arc<AdmissionController>, s: RequestSize) -> AdmissionPermit {
        match c.decide(s) {
            AdmissionDecision::Admit(p) => p,
            other => panic!("expected admit, got {other:?}"),
        }
    }

    fn queue_entry(c: &Arc<AdmissionController>, s: RequestSize) -> Arc<Queued> {
        match c.decide(s) {
            AdmissionDecision::Queue(q) => q,
            other => panic!("expected queueing, got {other:?}"),
        }
    }

    #[test]
    fn immediate_admit_when_budget_fits_and_queue_empty() {
        let c = ctrl(8192, 1024);
        let _p = admit(&c, size(8192, 256));
        let s = c.snapshot();
        assert_eq!(s.active_requests, 1);
        assert_eq!(s.active_input_tokens, 8192);
    }

    #[test]
    fn second_heavy_request_queues_behind_first() {
        let c = ctrl(8192, 1024);
        let _p = admit(&c, size(8192, 256));
        let q = queue_entry(&c, size(8192, 256));
        assert_eq!(AdmissionController::queue_depth_at_enqueue(&q), 0);
        let s = c.snapshot();
        assert_eq!(s.queued_requests, 1);
        assert_eq!(s.queued_input_tokens, 8192);
        assert!(s.estimated_queue_seconds > 0.0);
    }

    #[test]
    fn permit_drop_grants_next_waiter() {
        let c = ctrl(8192, 1024);
        let permit = admit(&c, size(8192, 256));
        let q = queue_entry(&c, size(8192, 256));
        drop(permit); // first request finishes → waiter must be granted
        assert!(q.granted.load(Ordering::Acquire));
        let s = c.snapshot();
        assert_eq!(s.queued_requests, 0);
        assert_eq!(s.active_requests, 1);
    }

    #[tokio::test]
    async fn wait_for_grant_resolves_after_release() {
        let c = ctrl(8192, 1024);
        let permit = admit(&c, size(8192, 256));
        let q = queue_entry(&c, size(8192, 256));
        let waiter = {
            let c = c.clone();
            let q = q.clone();
            tokio::spawn(async move { c.wait_for_grant(&q).await.is_some() })
        };
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        drop(permit);
        assert!(waiter.await.unwrap(), "waiter should be granted");
    }

    #[test]
    fn no_barging_past_the_queue() {
        // Queue non-empty → even a tiny request that would fit must queue.
        let c = AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: 12288,
            max_active_output_tokens: 1_000_000,
            degraded_queue_s: 5.0,
            overloaded_queue_s: 100_000.0, // mechanics test: keep the gate open
            max_estimated_work_ms: 0,
        });
        let _p = admit(&c, size(8192, 512));
        let _q1 = queue_entry(&c, size(8192, 512)); // 16384 > 12288 → waits
        assert!(matches!(c.decide(size(8, 8)), AdmissionDecision::Queue(_)));
    }

    #[test]
    fn small_request_can_pass_a_blocked_head() {
        // Head needs 8K tokens against a 4K budget; a 16-token request
        // behind it still gets granted on release (no head-of-line death).
        let c = AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: 4096,
            max_active_output_tokens: 1_000_000,
            degraded_queue_s: 5.0,
            overloaded_queue_s: 10_000.0, // mechanics test: keep the gate open
            max_estimated_work_ms: 0,
        });
        let permit = admit(&c, size(4096, 64));
        let big = queue_entry(&c, size(8192, 64));
        let small = queue_entry(&c, size(16, 16));
        drop(permit); // release active
        assert!(small.granted.load(Ordering::Acquire), "small waiter passes big head");
        assert!(!big.granted.load(Ordering::Acquire), "big head still waiting");
    }

    #[test]
    fn cancel_queued_removes_entry_and_reports_saved_work() {
        let c = ctrl(8192, 1024);
        let _p = admit(&c, size(8192, 256));
        let q = queue_entry(&c, size(8192, 256));
        let saved = c.cancel_queued(&q);
        assert_eq!(saved, size(8192, 256).est_work_ms);
        let s = c.snapshot();
        assert_eq!(s.queued_requests, 0);
        assert_eq!(s.queued_input_tokens, 0);
        // Idempotent.
        assert_eq!(c.cancel_queued(&q), 0.0);
    }

    #[tokio::test]
    async fn cancelled_waiter_returns_none() {
        let c = ctrl(8192, 1024);
        let _p = admit(&c, size(8192, 256));
        let q = queue_entry(&c, size(8192, 256));
        let waiter = {
            let c = c.clone();
            let q = q.clone();
            tokio::spawn(async move { c.wait_for_grant(&q).await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        c.cancel_queued(&q);
        assert!(waiter.await.unwrap().is_none(), "cancelled waiter must return None");
    }

    #[test]
    fn overload_rejects_with_retry_after() {
        let c = AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: 8192,
            max_active_output_tokens: 100, // binds after one 64-token request
            degraded_queue_s: 5.0,
            overloaded_queue_s: 30.0,
            max_estimated_work_ms: 0,
        });
        let _p = admit(&c, size(512, 64)); // ~7.4 s of estimated work each
        // Each queued request adds ~7.4 s to the estimated wait; the 6th
        // arrival sees 5 × 7.4 ≈ 36.8 s ≥ 30 s and is refused.
        for _ in 0..5 {
            assert!(matches!(c.decide(size(512, 64)), AdmissionDecision::Queue(_)));
        }
        match c.decide(size(512, 64)) {
            AdmissionDecision::Reject { reason, retry_after_s } => {
                assert_eq!(reason, "overloaded");
                assert!(retry_after_s >= 30);
            }
            other => panic!("expected reject, got {other:?}"),
        }
        assert_eq!(c.snapshot().load_state, LoadState::Overloaded);
    }

    #[test]
    fn single_request_work_cap_rejects() {
        let c = AdmissionController::new(AdmissionPolicy {
            max_estimated_work_ms: 10_000,
            ..Default::default()
        });
        match c.decide(size(8192, 4096)) {
            AdmissionDecision::Reject { reason, .. } => {
                assert_eq!(reason, "estimated_work_exceeds_cap")
            }
            other => panic!("expected reject, got {other:?}"),
        }
    }

    #[test]
    fn load_state_thresholds() {
        let c = AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: 8192,
            max_active_output_tokens: 100, // binds after one 64-token request
            degraded_queue_s: 5.0,
            overloaded_queue_s: 30.0,
            max_estimated_work_ms: 0,
        });
        assert_eq!(c.snapshot().load_state, LoadState::Normal);
        let _p = admit(&c, size(256, 64)); // ~6.1 s of estimated work
        // One queued request ≈ 6.1 s wait → degraded but not overloaded.
        let _q = queue_entry(&c, size(256, 64));
        assert_eq!(c.snapshot().load_state, LoadState::Degraded);
    }

    #[test]
    fn disabled_admission_lets_everything_through() {
        let c = AdmissionController::new(AdmissionPolicy::default());
        let mut permits = Vec::new();
        for _ in 0..100 {
            permits.push(admit(&c, size(8192, 256)));
        }
        assert_eq!(c.snapshot().active_requests, 100);
    }

    #[test]
    fn idle_machine_admits_oversize_request() {
        // The phase-3 experiment deadlock, distilled: every request
        // estimates a hair over the budget. An idle machine must start work
        // (bounded overshoot) rather than queue into a deadlock.
        let c = ctrl(8192, 1_000_000);
        let _p = admit(&c, size(8199, 64)); // oversize: admitted while idle
        let s = c.snapshot();
        assert_eq!(s.active_requests, 1);
        assert_eq!(s.active_input_tokens, 8199);
        // Busy machine: the same request queues.
        let _q = queue_entry(&c, size(8199, 64));
    }

    #[test]
    fn finishing_active_grants_oversize_head() {
        // Active finishes; the next in line doesn't fit either → release's
        // liveness rule grants it instead of deadlocking.
        let c = ctrl(4096, 1_000_000);
        let permit = admit(&c, size(4096, 64));
        let next = queue_entry(&c, size(8192, 64));
        drop(permit);
        assert!(next.granted.load(Ordering::Acquire), "oversize head granted on release");
        let s = c.snapshot();
        assert_eq!(s.active_requests, 1);
        assert_eq!(s.queued_requests, 0);
    }

    #[test]
    fn interactive_entries_are_granted_before_batch() {
        // The 4C service-class contract: freed capacity goes to the
        // highest-priority class that fits, FIFO within a class. A small
        // interactive arrival behind a too-big batch head must not wait
        // for it.
        let c = AdmissionController::new(AdmissionPolicy {
            max_active_input_tokens: 4096,
            max_active_output_tokens: 1_000_000,
            degraded_queue_s: 5.0,
            overloaded_queue_s: 10_000.0,
            max_estimated_work_ms: 0,
        });
        let permit = admit(&c, size(4096, 64));
        let batch_head = queue_entry(&c, sized(8192, 64, ServiceClass::Batch));
        let batch_small = queue_entry(&c, sized(16, 16, ServiceClass::Batch));
        let interactive = queue_entry(&c, sized(16, 16, ServiceClass::Interactive));
        drop(permit);
        assert!(interactive.granted.load(Ordering::Acquire), "interactive granted first");
        assert!(batch_small.granted.load(Ordering::Acquire), "batch FIFO within its class");
        assert!(!batch_head.granted.load(Ordering::Acquire), "oversize batch head still waits");
        // Order matters, not just outcome: the freed 4096 budget went to
        // interactive + small batch (16+16 tokens), never to the 8K head.
        let s = c.snapshot();
        assert_eq!(s.active_requests, 2);
        assert_eq!(c.queued_by_class(), [0, 0, 1]);
    }

    #[test]
    fn estimator_counts_words_and_calibrates() {
        let e = TokenEstimator::new();
        let req = inference_lab_protocol::ChatCompletionRequest::simple(
            "m",
            "one two three four five six seven eight", // 8 words → 6 tokens
        );
        assert_eq!(e.estimate(&req), 6);
        for _ in 0..60 {
            e.calibrate(6, 12); // server says twice our estimate
        }
        let e2 = TokenEstimator::new();
        assert!(e.estimate(&req) > e2.estimate(&req), "calibration should raise estimates");
    }
}
