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
//! **Phase 4B.2 — slots as schedulable cache domains.** One llama.cpp
//! server with `-np 8` behaves like eight single-request workers that
//! happen to share weights: each slot owns a KV cache and serves one
//! request at a time, queueing the rest. The gateway models that with one
//! [`WorkerHandle`] per slot (`WorkerHandle::slot = Some(n)`) so a routing
//! decision *is* a slot-assignment decision. Three slot-level policies
//! join the lineup:
//!
//! * [`RoutingPolicy::SlotLeastWork`] — ignore caches, pure load: spread
//!   by estimated work (the token-aware-LB baseline).
//! * [`RoutingPolicy::SlotPrefixOnly`] — always join the slot with the
//!   largest reusable prefix (deliberately recreates the phase-4B
//!   cache-locality hotspot).
//! * [`RoutingPolicy::SlotCacheAwareEct`] — the actual feature. For every
//!   slot it prices the TTFT an arrival would see,
//!
//!   ```text
//!   recompute_tokens = prompt_tokens − reusable_prefix_tokens(slot)
//!   prefill_cost     = fixed + recompute_tokens × prefill_ms_per_tok
//!   estimated_ttft   = remaining_work(slot) + prefill_cost
//!   ```
//!
//!   and picks the argmin. A cache hit is therefore worth exactly what it
//!   saves (`reusable × prefill_ms_per_tok`) and **only** while that
//!   saving exceeds the wait for the warm slot — the load gate whose
//!   absence made llama.cpp's own LCP slot selection serialize cold
//!   same-prefix bursts into a 253 s TTFT hotspot (phase 4B).
//!
//! Slot cache knowledge is per-slot and single-entry (a llama.cpp slot
//! cache holds one evolving sequence, so serving prefix B effectively
//! ends prefix A's residency): [`CacheInvestment`] tracks whether the
//! prefix is [`CacheInvestment::Absent`], [`CacheInvestment::Populating`]
//! (a request carrying it is in flight — joining means waiting out that
//! request) or [`CacheInvestment::Resident`] (observed reuse from the
//! runtime's own `cache_n` telemetry). The gateway updates the state from
//! its own dispatches, and honors client-pinned `id_slot` requests so
//! external warmers teach the model directly.
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

pub mod calibration;
pub use calibration::{
    bucket_index, bucket_label, CalibrationKey, CalibrationProfile, CalibrationStore,
    DispatchTicket, DecodeCalibration, PrefillCalibration, RuntimeCalibration, BUCKET_LABELS,
    BUCKET_LOWER, BUCKET_UPPER,
};

/// Clamp band for each parameter; observations outside it are noise (or a
/// pathological runtime) and get pinned to the edge, never NaN/below zero.
pub(crate) fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Routing policy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingPolicy {
    #[default]
    LeastOutstandingRequests,
    LeastEstimatedCompletionTime,
    PrefixAffinity,
    /// Single upstream, no routing opinion: forward untouched (llama.cpp
    /// keeps its own LCP slot selection). The phase-4B baseline.
    Native,
    /// Slot level: lowest estimated work, caches ignored.
    SlotLeastWork,
    /// Slot level: largest reusable prefix wins, load be damned — the
/// explicit form of the pathology `native` stumbles into.
    SlotPrefixOnly,
    /// Slot level: argmin estimated TTFT over cache savings and load —
    /// the load-gated prefix affinity feature (a.k.a. CacheAwareECT).
    SlotCacheAwareEct,
}

impl RoutingPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lor" | "least-outstanding" => Some(Self::LeastOutstandingRequests),
            "ect" | "least-ect" => Some(Self::LeastEstimatedCompletionTime),
            "prefix" | "prefix-affinity" => Some(Self::PrefixAffinity),
            "native" => Some(Self::Native),
            "least-work" => Some(Self::SlotLeastWork),
            "prefix-only" => Some(Self::SlotPrefixOnly),
            "cache-aware-ect" | "load-gated-prefix" => Some(Self::SlotCacheAwareEct),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LeastOutstandingRequests => "lor",
            Self::LeastEstimatedCompletionTime => "ect",
            Self::PrefixAffinity => "prefix-affinity",
            Self::Native => "native",
            Self::SlotLeastWork => "least-work",
            Self::SlotPrefixOnly => "prefix-only",
            Self::SlotCacheAwareEct => "cache-aware-ect",
        }
    }

    /// Slot-level policies score candidates through the cache-aware
    /// candidate view instead of the legacy worker-level formulas.
    pub fn is_slot_policy(&self) -> bool {
        matches!(
            self,
            Self::SlotLeastWork | Self::SlotPrefixOnly | Self::SlotCacheAwareEct
        )
    }
}

/// Requested service class (phase 4C). Carried by the request, consumed by
/// the admission queue's grant priority and class-labeled telemetry — the
/// cheap, testable version of a QoS story: contention *physics* (routing
/// and the marginal gate) is class-blind, but *who gets the next freed
/// capacity* is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceClass {
    /// Chat-shaped: first token is the product. Granted from queues first.
    Interactive,
    /// Default when the request says nothing.
    #[default]
    Standard,
    /// Throughput-shaped: tolerates waiting for freed capacity.
    Batch,
}

impl ServiceClass {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(Self::Interactive),
            "standard" => Some(Self::Standard),
            "batch" => Some(Self::Batch),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Standard => "standard",
            Self::Batch => "batch",
        }
    }

    /// Grant priority in the admission queue (lower grants first).
    pub fn queue_rank(&self) -> u8 {
        match self {
            Self::Interactive => 0,
            Self::Standard => 1,
            Self::Batch => 2,
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
    /// A client-supplied `id_slot` pin: route to exactly that slot when
    /// the router models slots. External cache warmers use this to teach
    /// the gateway's slot-cache model from real traffic.
    pub pinned_slot: Option<u32>,
    /// Requested service class. On this lab's single-server slot
    /// topologies the routing objective is class-independent by
    /// construction (decode rate is server-wide, so argmin-TTFT and
    /// argmin-completion reduce to the same argmin) — the class acts in
    /// the admission queue's grant priority instead.
    pub service_class: ServiceClass,
}

// ---------------------------------------------------------------------------
// Slot cache model
// ---------------------------------------------------------------------------

/// Lifecycle of a prefix inside one slot's KV cache (see the module docs:
/// the model is per-slot, single-entry — serving a different prefix ends
/// the previous one's residency, matching llama.cpp's slot reality).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheInvestment {
    /// The slot holds nothing reusable for this prefix.
    Absent,
    /// A request carrying the prefix is in flight on the slot: reuse is
    /// coming, but only after that request's whole service time. Whether
    /// to join it is precisely the load-gate question.
    Populating,
    /// Reuse observed (runtime `cache_n` telemetry): the prefix is resident.
    Resident,
}

impl CacheInvestment {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Populating => "populating",
            Self::Resident => "resident",
        }
    }
}

/// One slot's cached-prefix state as the gateway models it.
#[derive(Debug, Clone)]
pub struct SlotCacheEntry {
    pub fingerprint: u64,
    pub state: CacheInvestment,
    /// Estimated prompt size of the request that established/refreshes
    /// the entry (drives reuse estimates while POPULATING).
    pub expected_est_tokens: u64,
    /// Server-reported prompt size of the request whose completion made
    /// the entry RESIDENT — after that completion the slot HOLDS this
    /// much reusable prefix for the fingerprint.
    pub prompt_real_tokens: u64,
    /// What THIS request's own completion reported reusing (llama.cpp
    /// `timings.cache_n`). Telemetry only — a 0 here means "I didn't
    /// reuse", never "the slot has nothing": the request just prefilled
    /// its whole prompt into the slot.
    pub last_reuse_observed_real: u64,
    pub last_used: Instant,
}

/// Read-only view of a slot's cache for one routing decision, with the
/// reusable prefix translated into the request's estimated-token units.
#[derive(Debug, Clone)]
pub struct SlotCacheView {
    pub state: CacheInvestment,
    /// Reusable tokens for this request, estimated units (0 unless the
    /// entry matches the request's fingerprint).
    pub reusable_tokens: u64,
    pub last_used: Option<Instant>,
}

/// The router's estimate of what one candidate (slot) would cost this
/// request — one row of the full decision table the gateway exports to
/// traces and logs, so every selection can answer "why?".
#[derive(Debug, Clone)]
pub struct CandidateView {
    pub worker_id: String,
    pub slot: Option<u32>,
    pub cache_state: CacheInvestment,
    /// Estimated reusable prefix tokens (this request's units).
    pub reusable_tokens: u64,
    /// Estimated ms of work ahead on the candidate (queue + active
    /// prefill/decode remaining) at decision time.
    pub remaining_work_ms: f64,
    /// `fixed + (prompt − reusable) × prefill_rate × contention(bucket)`
    /// for the load level the server carried at decision time.
    pub prefill_cost_ms: f64,
    /// `decode_rate(concurrency bucket) × output_tokens` — the decode
    /// load this request would JOIN, not the solo rate.
    pub decode_cost_ms: f64,
    /// The load-gated headline: `remaining_work + prefill_cost`.
    pub estimated_ttft_ms: f64,
    /// Full own service estimate (prefill + decode) once started.
    pub estimated_service_ms: f64,
    /// `reusable × prefill_rate` — what the cache saves vs a cold slot.
    pub cache_savings_ms: f64,
    /// Concurrency buckets the pricing used: prefill counts active
    /// prefills + this request; decode counts active decodes + prefills
    /// + this request (everything prefilling now will be decoding then).
    pub prefill_bucket: &'static str,
    pub decode_bucket: &'static str,
    /// Marginal gain of this request's prefill joining at the current
    /// server load: aggregate prefill throughput after / before. Below
    /// 1.0 means this arrival pushes the machine into its measured
    /// anti-scaling region (the contention gate's hold signal).
    pub prefill_marginal_gain: f64,
    /// Same ratio for the decode stream this request would join.
    pub decode_marginal_gain: f64,
}

/// Why a routing decision landed where it landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// One candidate only (native pass-through).
    SingleWorker,
    /// Legacy least-outstanding-requests.
    LeastOutstanding,
    /// Legacy ECT (remaining work + discounted service time).
    LeastEstimatedTime,
    /// Legacy prefix-affinity.
    PrefixAffinity,
    /// Slot least-work pick.
    LeastWorkSpread,
    /// Slot prefix-only with no resident/populating match: least-work
    /// fallback.
    NoPrefixAvailable,
    /// Slot prefix-only joined the largest match.
    PrefixOnlyAffinity,
    /// Cache-aware ECT: the cache-warm slot also had the smallest
    /// estimated TTFT.
    CacheHitCheapest,
    /// Cache-aware ECT deliberately gave up available reuse: waiting for
    /// the warm slot cost more than recomputing elsewhere.
    LoadOverrodeAffinity,
    /// Same, but the deciding factor was the warm slot being mid-prefill.
    PrefillPressureOverrodeAffinity,
    /// Cache-aware ECT: reuse existed but under the configured
    /// minimum-savings floor, so it was treated as absent.
    InsufficientBenefit,
    /// The client pinned the slot (`id_slot`); the router honored it.
    ClientPinned,
}

impl DecisionReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SingleWorker => "single_worker",
            Self::LeastOutstanding => "least_outstanding",
            Self::LeastEstimatedTime => "least_estimated_time",
            Self::PrefixAffinity => "prefix_affinity",
            Self::LeastWorkSpread => "least_work_spread",
            Self::NoPrefixAvailable => "no_prefix_available",
            Self::PrefixOnlyAffinity => "prefix_only_affinity",
            Self::CacheHitCheapest => "cache_hit_cheapest",
            Self::LoadOverrodeAffinity => "load_overrode_affinity",
            Self::PrefillPressureOverrodeAffinity => "prefill_pressure_overrode_affinity",
            Self::InsufficientBenefit => "insufficient_benefit",
            Self::ClientPinned => "client_pinned",
        }
    }

    /// The Prometheus label for `scheduler_cache_affinity_bypassed_total`
    /// when this reason means "reuse was available and refused".
    pub fn bypass_label(&self) -> Option<&'static str> {
        match self {
            Self::LoadOverrodeAffinity => Some("queue_pressure"),
            Self::PrefillPressureOverrodeAffinity => Some("prefill_pressure"),
            Self::InsufficientBenefit => Some("insufficient_benefit"),
            _ => None,
        }
    }
}

/// A completed selection: the winner, its reservation, and the full
/// decision table for telemetry.
pub struct RouteDecision {
    pub worker: Arc<WorkerHandle>,
    /// Reservation ticket from selection time — the caller MUST pass it to
    /// `end_work` exactly once.
    pub work_start: DispatchTicket,
    pub est_total_ms: f64,
    pub est_prefill_ms: f64,
    pub policy: RoutingPolicy,
    pub reason: DecisionReason,
    /// Set when available reuse was deliberately refused.
    pub bypass: Option<BypassRecord>,
    /// Every candidate scored, in registration order (slot index order
    /// for slot workers).
    pub candidates: Vec<CandidateView>,
    /// Index into `candidates` of the winner.
    pub chosen: usize,
}

/// "This request deliberately missed the cache": the rejected affinity,
/// with the numbers that justified refusing it.
#[derive(Debug, Clone)]
pub struct BypassRecord {
    /// Prometheus reason label (queue_pressure | prefill_pressure |
    /// worker_degraded | insufficient_benefit).
    pub reason: &'static str,
    pub worker_id: String,
    pub slot: Option<u32>,
    /// Estimated ms the cached slot would have saved.
    pub forgone_savings_ms: f64,
    pub forgone_match_tokens: u64,
}

// ---------------------------------------------------------------------------
// Reservation outcome
// ---------------------------------------------------------------------------

/// Result of an atomic select+reserve.
pub enum SelectOutcome {
    Decided(RouteDecision),
    /// Every registered worker is unhealthy.
    AllUnhealthy,
    /// The contention gate held this arrival: the measured curve says one
    /// more prefill would lower aggregate throughput. Nothing was
    /// reserved. The caller waits (see [`Router::contention_notify`] /
    /// [`Router::contention_gate_hold`]) and re-selects.
    Held {
        active_prefills: usize,
        marginal_gain: f64,
    },
}

/// Result of a non-reserving selection (compat path).
pub enum WorkerOutcome {
    Worker(Arc<WorkerHandle>),
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

/// Static router configuration (set once at startup from env).
#[derive(Debug, Clone, Copy)]
pub struct RouterConfig {
    /// Reuse worth less than this (ms) is treated as absent by
    /// cache-aware ECT — the knob that separates "free local hit" from
    /// "not worth a sticky decision". 0 disables the floor.
    pub min_cache_savings_ms: f64,
    /// Contention gate (phase 4C): when the measured curve says one more
    /// concurrent prefill would REDUCE aggregate prefill throughput,
    /// slot-policy selection returns [`SelectOutcome::Held`] instead of
    /// reserving — the caller waits for the machine to exit its
    /// anti-scaling region (or for the hold cap) and re-selects. Requests
    /// whose every candidate offers full cache reuse never hold (they add
    /// no prefill), and client-pinned dispatches bypass the gate.
    pub contention_gate: bool,
    /// How long a held request may wait before it dispatches anyway
    /// (spills through the anti-scaling region).
    pub gate_max_hold_s: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { min_cache_savings_ms: 0.0, contention_gate: false, gate_max_hold_s: 20.0 }
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// One active dispatch's cost estimate, for remaining-work accounting.
#[derive(Debug, Clone, Copy)]
struct ActiveEstimate {
    ticket: DispatchTicket,
}

pub struct WorkerHandle {
    pub id: String,
    pub endpoint: String,
    pub model: String,
    pub runtime: String,
    pub max_context: u64,
    /// llama.cpp slot index when this handle models one slot of a shared
    /// server (`-np` cell with its own KV cache); `None` = a whole worker.
    pub slot: Option<u32>,
    /// Outstanding requests (raw count — LOR baseline and gauges).
    outstanding: AtomicU64,
    /// Lifecycle state machine: four health dimensions, per-reason
    /// escalation, recovery hysteresis, generation epochs. READY/DEGRADED
    /// serve traffic; everything else is quarantined.
    lifecycle: Mutex<WorkerLifecycle>,

    /// Shared performance model of the SERVER this handle belongs to:
    /// prefill regression, per-concurrency-bucket decode rates, prefill
    /// contention multipliers and the active-dispatch registry. All slot
    /// handles of one llama.cpp server share one GPU — so they share one
    /// calibration, and its bucketed rates are what pricing consults.
    cal: Arc<RuntimeCalibration>,

    /// Active dispatches with their estimates. Drives remaining-work ECT
    /// scoring, the prefill-pressure signal and gauge export.
    active_estimates: Mutex<Vec<ActiveEstimate>>,

    /// Prefix hashes this worker has served — a proxy for what its
    /// KV/prefix cache holds (coarse, worker-granular; slot workers carry
    /// the richer single-entry model below).
    pub cached_prefixes: Mutex<HashSet<u64>>,

    /// Slot workers only: the single-entry prefix-cache model (see
    /// [`CacheInvestment`]). `None` for whole-worker handles.
    slot_cache: Mutex<Option<SlotCacheEntry>>,
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
        let model = model.into();
        let runtime = runtime.into();
        let cal = Arc::new(RuntimeCalibration::from_seeds("local", &runtime, &model, 0, defaults));
        Self::build(id, endpoint, model, runtime, cal)
    }

    /// Attach this handle to a (possibly shared) server calibration. All
    /// slot workers of one upstream pass the SAME `Arc`.
    pub fn with_calibration(
        id: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        runtime: impl Into<String>,
        cal: Arc<RuntimeCalibration>,
    ) -> Self {
        Self::build(id, endpoint, model, runtime, cal)
    }

    fn build(
        id: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        runtime: impl Into<String>,
        cal: Arc<RuntimeCalibration>,
    ) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            runtime: runtime.into(),
            max_context: 0,
            slot: None,
            outstanding: AtomicU64::new(0),
            lifecycle: Mutex::new(WorkerLifecycle::new(LifecyclePolicy::default())),
            cal,
            active_estimates: Mutex::new(Vec::new()),
            cached_prefixes: Mutex::new(HashSet::new()),
            slot_cache: Mutex::new(None),
        }
    }

    /// The shared server model this handle prices with.
    pub fn calibration(&self) -> Arc<RuntimeCalibration> {
        self.cal.clone()
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
        self.cal.prefill_base().0
    }

    pub fn prefill_ms_per_tok(&self) -> f64 {
        self.cal.prefill_base().1
    }

    /// Solo decode rate (ms/token). The bucketed view of decode lives in
    /// the shared calibration — this is the unloaded truth for legacy
    /// formulas and admission pricing.
    pub fn decode_ms_per_tok(&self) -> f64 {
        self.cal.decode_rate_solo()
    }

    /// Prefill cost in ms for `input_tokens` estimated tokens of which
    /// `reusable_tokens` are already cached on this worker (solo rates —
    /// contention-aware pricing goes through the calibration directly).
    pub fn prefill_cost_ms(&self, input_tokens: u64, reusable_tokens: u64) -> f64 {
        self.fixed_overhead_ms()
            + (input_tokens.saturating_sub(reusable_tokens)) as f64 * self.prefill_ms_per_tok()
    }

    /// Estimated total service time in ms for a request on this worker.
    /// A known prefix discounts the prefill term (cache hit).
    pub fn service_time_ms(&self, input_tokens: u64, output_tokens: u64, prefix_hit: bool) -> f64 {
        let prefill = self.prefill_ms_per_tok()
            * input_tokens as f64
            * if prefix_hit { 0.3 } else { 1.0 };
        let decode = self.decode_ms_per_tok() * output_tokens as f64;
        self.fixed_overhead_ms() + prefill + decode
    }

    /// The shared CostModel's estimate of this request's service time on
    /// an otherwise idle server — the number admission control budgets
    /// with (identical to the router's starting point, by construction).
    pub fn solo_service_estimate_ms(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.cal.solo_service_ms(input_tokens, output_tokens)
    }

    /// Feed one `worker_first_token_ms` observation (response headers →
    /// first streamed token; queue wait excluded by construction) for the
    /// dispatch `ticket`. The (fixed, ms/token) split refits once observed
    /// prompt sizes vary, and the dispatch's concurrency bucket learns its
    /// contention multiplier.
    pub fn record_prefill(&self, input_tokens: u64, worker_first_token_ms: f64, ticket: &DispatchTicket) {
        self.cal.observe_prefill(input_tokens.max(1), worker_first_token_ms, ticket);
    }

    /// Feed one streaming observation for dispatch `ticket`: ms spent
    /// decoding `output_tokens` under whatever load the server carried.
    pub fn record_decode(&self, output_tokens: u64, streaming_ms: f64, ticket: &DispatchTicket) {
        self.cal.observe_decode(output_tokens.max(1), streaming_ms, ticket);
    }

    // -- active-work tracking (ECT) ------------------------------------------

    /// Register a dispatch with its estimated total AND prefill service
    /// time. Returns the [`DispatchTicket`] to hand back to
    /// [`WorkerHandle::end_work`] exactly once. Reservations made here are
    /// what make concurrent selections under a burst see each other — in
    /// both this worker's queue and the server-wide concurrency buckets.
    pub fn begin_work(&self, est_total_ms: f64, est_prefill_ms: f64) -> DispatchTicket {
        let ticket = self.cal.register_dispatch(est_total_ms, est_prefill_ms);
        if let Ok(mut v) = self.active_estimates.lock() {
            v.push(ActiveEstimate { ticket });
        }
        ticket
    }

    /// Remove a finished dispatch (idempotent: unknown tickets are ignored).
    pub fn end_work(&self, ticket: &DispatchTicket) {
        self.cal.unregister_dispatch(ticket);
        if let Ok(mut v) = self.active_estimates.lock() {
            if let Some(pos) = v.iter().position(|a| a.ticket.id == ticket.id) {
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
                    .map(|a| {
                        (a.ticket.est_total_ms - a.ticket.start.elapsed().as_secs_f64() * 1000.0)
                            .max(0.0)
                    })
                    .sum()
            })
            .unwrap_or(0.0)
    }

    /// True when any active dispatch is (still estimated to be) in its
    /// prefill phase — the prefill-pressure bypass signal.
    pub fn is_prefilling(&self) -> bool {
        self.active_estimates
            .lock()
            .map(|v| {
                v.iter().any(|a| {
                    a.ticket.start.elapsed().as_secs_f64() * 1000.0 < a.ticket.est_prefill_ms
                })
            })
            .unwrap_or(false)
    }

    // -- slot cache model -----------------------------------------------------

    /// Read the slot's cache state for one routing decision. Whole-worker
    /// handles always report Absent — they have no per-slot model.
    pub fn slot_cache_view(&self, fingerprint: u64, prompt_est_tokens: u64) -> SlotCacheView {
        let guard = self.slot_cache.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(e) if e.fingerprint == fingerprint => {
                let reusable = match e.state {
                    CacheInvestment::Absent => 0,
                    CacheInvestment::Populating => e.expected_est_tokens.min(prompt_est_tokens),
                    // RESIDENT: the slot holds the completing request's
                    // whole prompt for this fingerprint — up to that
                    // prompt's real size, translated into this request's
                    // estimated units (a 100-token probe holds 100).
                    CacheInvestment::Resident => e.prompt_real_tokens.min(prompt_est_tokens),
                };
                SlotCacheView {
                    state: e.state,
                    reusable_tokens: reusable.min(prompt_est_tokens),
                    last_used: Some(e.last_used),
                }
            }
            _ => SlotCacheView {
                state: CacheInvestment::Absent,
                reusable_tokens: 0,
                last_used: None,
            },
        }
    }

    /// Record that a request with this fingerprint was dispatched to the
    /// slot. A different fingerprint replaces the entry (single-sequence
    /// slot caches); the same fingerprint refreshes it. No-op for
    /// whole-worker handles.
    pub fn note_slot_dispatch(&self, fingerprint: u64, prompt_est_tokens: u64) {
        if self.slot.is_none() {
            return;
        }
        let mut guard = self.slot_cache.lock().unwrap_or_else(|e| e.into_inner());
        let replace = match guard.as_ref() {
            Some(e) => e.fingerprint != fingerprint,
            None => true,
        };
        if replace {
            *guard = Some(SlotCacheEntry {
                fingerprint,
                state: CacheInvestment::Populating,
                expected_est_tokens: prompt_est_tokens,
                prompt_real_tokens: 0,
                last_reuse_observed_real: 0,
                last_used: Instant::now(),
            });
        } else if let Some(e) = guard.as_mut() {
            e.last_used = Instant::now();
            e.expected_est_tokens = e.expected_est_tokens.max(prompt_est_tokens);
        }
    }

    /// Record a completed request on the slot. Whatever this request
    /// itself reused, the slot now HOLDS its full prompt as reusable
    /// state for the fingerprint — a miss (`reused_observed == 0`) is how
    /// a population looks, not evidence of an empty cache. No-op for
    /// whole-worker handles.
    pub fn note_slot_completion(
        &self,
        fingerprint: u64,
        prompt_real_tokens: u64,
        reused_observed_real_tokens: u64,
    ) {
        if self.slot.is_none() {
            return;
        }
        let mut guard = self.slot_cache.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(guard.as_ref(), Some(e) if e.fingerprint == fingerprint) {
            let e = guard.as_mut().expect("checked above");
            e.state = CacheInvestment::Resident;
            e.prompt_real_tokens = prompt_real_tokens;
            e.last_reuse_observed_real = reused_observed_real_tokens;
            e.last_used = Instant::now();
        } else {
            *guard = Some(SlotCacheEntry {
                fingerprint,
                state: CacheInvestment::Resident,
                expected_est_tokens: 0,
                prompt_real_tokens,
                last_reuse_observed_real: reused_observed_real_tokens,
                last_used: Instant::now(),
            });
        }
    }

    /// Operational snapshot for /status and gauges.
    pub fn slot_cache_snapshot(&self) -> Option<SlotCacheEntry> {
        self.slot_cache.lock().unwrap_or_else(|e| e.into_inner()).clone()
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
    config: RouterConfig,
    /// Serializes selection+reservation so a concurrent burst cannot send
    /// two requests to the same "least loaded" slot before either
    /// registers its work.
    select_lock: Mutex<()>,
}

impl Router {
    pub fn new(workers: Vec<Arc<WorkerHandle>>, policy: RoutingPolicy) -> anyhow::Result<Self> {
        Self::with_config(workers, policy, RouterConfig::default())
    }

    pub fn with_config(
        workers: Vec<Arc<WorkerHandle>>,
        policy: RoutingPolicy,
        config: RouterConfig,
    ) -> anyhow::Result<Self> {
        if workers.is_empty() {
            anyhow::bail!("router needs at least one worker");
        }
        Ok(Self { workers, policy, config, select_lock: Mutex::new(()) })
    }

    pub fn workers(&self) -> &[Arc<WorkerHandle>] {
        &self.workers
    }

    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    pub fn config(&self) -> RouterConfig {
        self.config
    }

    /// Healthy workers only (empty slice means "nobody home").
    pub fn healthy_workers(&self) -> Vec<Arc<WorkerHandle>> {
        self.workers.iter().filter(|w| w.is_healthy()).cloned().collect()
    }

    /// Pick the worker for a request WITHOUT reserving work (compat path
    /// used by tests and the admission estimator's what-ifs). Also records
    /// the prefix on the winner.
    pub fn select(&self, req: RequestShape) -> WorkerOutcome {
        let candidates = self.healthy_workers();
        if candidates.is_empty() {
            return WorkerOutcome::AllUnhealthy;
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
            RoutingPolicy::Native => candidates[0].clone(),
            RoutingPolicy::SlotLeastWork | RoutingPolicy::SlotPrefixOnly
            | RoutingPolicy::SlotCacheAwareEct => {
                // Non-reserving slot selection (compat/tests): same scoring,
                // no reservation, so bursts can race — production must use
                // `select_reserved`.
                let views = self.views(&candidates, &req);
                let chosen = self.pick(&views);
                candidates[chosen].clone()
            }
        };
        winner.remember_prefix(req.prefix_hash);
        WorkerOutcome::Worker(winner)
    }

    /// Pick the worker for a request AND reserve its estimated work on the
    /// winner before returning, atomically. Concurrent handlers dispatching
    /// a burst therefore each see the earlier reservations — eight
    /// simultaneous least-work arrivals land on eight different slots
    /// instead of racing onto one.
    ///
    /// With [`RouterConfig::contention_gate`] on, slot-policy selections
    /// that would push the server past its measured prefill saturation
    /// point return [`SelectOutcome::Held`] instead (nothing reserved).
    pub fn select_reserved(&self, req: RequestShape) -> SelectOutcome {
        self.select_reserved_inner(req, self.config.contention_gate)
    }

    /// Same selection with the contention gate bypassed — the spill path
    /// for an arrival whose hold cap expired.
    pub fn select_reserved_forced(&self, req: RequestShape) -> SelectOutcome {
        self.select_reserved_inner(req, false)
    }

    /// The gate's wake source (Some only when the gate is on): fired
    /// whenever a dispatch leaves the server's registry, so held arrivals
    /// re-price instead of polling.
    pub fn contention_notify(&self) -> Option<Arc<tokio::sync::Notify>> {
        if !self.config.contention_gate {
            return None;
        }
        Some(self.workers.first()?.calibration().contention_notify())
    }

    /// The hold cap for one gate hold (`None` = gate off).
    pub fn contention_gate_hold(&self) -> Option<std::time::Duration> {
        self.config
            .contention_gate
            .then(|| std::time::Duration::from_secs_f64(self.config.gate_max_hold_s.max(0.0)))
    }

    fn select_reserved_inner(&self, req: RequestShape, gate: bool) -> SelectOutcome {
        let _guard = self.select_lock.lock().unwrap_or_else(|e| e.into_inner());
        let candidates = self.healthy_workers();
        if candidates.is_empty() {
            return SelectOutcome::AllUnhealthy;
        }

        // Client pin: an explicit id_slot is honored when the router
        // models slots (external cache warmers teach the model this way).
            if let Some(pin) = req.pinned_slot {
                if let Some(w) = candidates.iter().find(|w| w.slot == Some(pin)) {
                    // Price from the pre-dispatch state, then reserve — the
                    // exported table must show what the DECISION saw, and
                    // the reservation must be exactly what was priced.
                    let view = self.single_view(w, &req);
                    let est_prefill = view.prefill_cost_ms;
                    let est_total = view.prefill_cost_ms + view.decode_cost_ms;
                    let ticket = w.begin_work(est_total, est_prefill);
                w.inc_outstanding();
                w.note_slot_dispatch(req.prefix_hash, req.input_tokens);
                w.remember_prefix(req.prefix_hash);
                return SelectOutcome::Decided(RouteDecision {
                    worker: w.clone(),
                    work_start: ticket,
                    est_total_ms: est_total,
                    est_prefill_ms: est_prefill,
                    policy: self.policy,
                    reason: DecisionReason::ClientPinned,
                    bypass: None,
                    chosen: 0,
                    candidates: vec![view],
                });
            }
            // No modeled slot matches (e.g. native mode): fall through —
            // the request body keeps the client's id_slot untouched.
        }

        let (winner, reason, bypass, views, chosen) = match self.policy {
            RoutingPolicy::LeastOutstandingRequests => {
                let winner = min_by(&candidates, |w| w.outstanding() as f64);
                (winner, DecisionReason::LeastOutstanding, None, Vec::new(), 0)
            }
            RoutingPolicy::LeastEstimatedCompletionTime => {
                let winner = min_by(&candidates, |w| {
                    let prefix_hit = w.has_prefix(req.prefix_hash);
                    w.remaining_work_ms()
                        + w.service_time_ms(req.input_tokens, req.output_tokens, prefix_hit)
                });
                (winner, DecisionReason::LeastEstimatedTime, None, Vec::new(), 0)
            }
            RoutingPolicy::PrefixAffinity => {
                let with_prefix: Vec<_> = candidates
                    .iter()
                    .filter(|w| w.has_prefix(req.prefix_hash))
                    .cloned()
                    .collect();
                let winner = if with_prefix.is_empty() {
                    min_by(&candidates, |w| w.outstanding() as f64)
                } else {
                    min_by(&with_prefix, |w| w.outstanding() as f64)
                };
                (winner, DecisionReason::PrefixAffinity, None, Vec::new(), 0)
            }
            RoutingPolicy::Native => (candidates[0].clone(), DecisionReason::SingleWorker, None, Vec::new(), 0),
            RoutingPolicy::SlotLeastWork | RoutingPolicy::SlotPrefixOnly
            | RoutingPolicy::SlotCacheAwareEct => {
                let views = self.views(&candidates, &req);
                let chosen = self.pick(&views);
                // The contention gate (phase 4C): ask the marginal question
                // for the dispatch that would actually happen. A winner
                // with full cache reuse adds no prefill — joining cache is
                // never an anti-scaling act, so only recomputing dispatches
                // can be held. Client pins never reach this path.
                if gate {
                    if let Some(cal) = candidates.first().map(|w| w.calibration()) {
                        let active = cal.active_prefills();
                        let recompute =
                            req.input_tokens.saturating_sub(views[chosen].reusable_tokens);
                        if recompute > 0 && !cal.prefill_marginal_pays(active) {
                            let gain = if active == 0 {
                                1.0
                            } else {
                                cal.prefill_marginal_gain(active)
                            };
                            return SelectOutcome::Held { active_prefills: active, marginal_gain: gain };
                        }
                    }
                }
                let reason = self.slot_reason(&views, chosen);
                let bypass = self.bypass_record(&views, chosen);
                let winner = candidates[chosen].clone();
                (winner, reason, bypass, views, chosen)
            }
        };

        // Reserve: estimated cost registered before the lock releases, so
        // the next concurrent arrival prices this dispatch into its own
        // decision.
        winner.remember_prefix(req.prefix_hash);
        let (est_total, est_prefill) = match self.policy.is_slot_policy() {
            false => {
                let hit = winner.has_prefix(req.prefix_hash);
                (
                    winner.service_time_ms(req.input_tokens, req.output_tokens, hit),
                    winner.fixed_overhead_ms()
                        + winner.prefill_ms_per_tok()
                            * req.input_tokens as f64
                            * if hit { 0.3 } else { 1.0 },
                )
            }
            true => {
                // Reserve exactly what the chosen view priced (bucketed
                // prefill + bucketed decode) — later arrivals price this
                // dispatch into their remaining-work at the same load.
                let view = &views[chosen];
                (view.prefill_cost_ms + view.decode_cost_ms, view.prefill_cost_ms)
            }
        };
        let work_start = winner.begin_work(est_total, est_prefill);
        winner.inc_outstanding();
        if self.policy.is_slot_policy() {
            winner.note_slot_dispatch(req.prefix_hash, req.input_tokens);
        }

        SelectOutcome::Decided(RouteDecision {
            worker: winner,
            work_start,
            est_total_ms: est_total,
            est_prefill_ms: est_prefill,
            policy: self.policy,
            reason,
            bypass,
            candidates: views,
            chosen,
        })
    }

    /// Candidate table for a slot policy decision: one row per healthy
    /// candidate, priced for THIS request. `reusable_tokens` is the
    /// *effective* reuse (after the minimum-savings floor under
    /// cache-aware ECT); `cache_savings_ms` always reports the *raw*
    /// savings so a floor-refused hit can still be attributed as
    /// `insufficient_benefit` in telemetry.
    ///
    /// Pricing goes through the candidate's shared server calibration at
    /// the concurrency it would JOIN: prefill contends with every active
    /// prefill + itself; decode contends with active decodes AND active
    /// prefills (they finish prefill before this request does). This is
    /// the concurrency-bucketed cost model — the 4B.2 model's seeded
    /// 25 ms/tok decode under-priced loaded decode 10–13× and closed the
    /// load gate too late.
    fn views(&self, candidates: &[Arc<WorkerHandle>], req: &RequestShape) -> Vec<CandidateView> {
        candidates
            .iter()
            .map(|w| {
                let cache = w.slot_cache_view(req.prefix_hash, req.input_tokens);
                let raw_reusable = cache.reusable_tokens;
                let reusable = match self.policy {
                    RoutingPolicy::SlotCacheAwareEct
                        if self.config.min_cache_savings_ms > 0.0
                            && raw_reusable as f64 * w.prefill_ms_per_tok()
                                < self.config.min_cache_savings_ms =>
                    {
                        0
                    }
                    _ => raw_reusable,
                };
                let cal = w.calibration();
                let active_prefills = cal.active_prefills();
                let active_decodes = cal.active_decodes();
                let decode_load = active_decodes + active_prefills + 1;
                let prefill = cal.prefill_ms(req.input_tokens, reusable, active_prefills + 1);
                let decode = cal.decode_rate_for(decode_load) * req.output_tokens as f64;
                let remaining = w.remaining_work_ms();
                // The marginal question for THIS join: does aggregate
                // throughput rise (gain > 1) or does this arrival push a
                // full machine past its measured anti-scaling knee?
                let prefill_gain = if active_prefills == 0 {
                    1.0
                } else {
                    cal.prefill_marginal_gain(active_prefills)
                };
                let decode_gain = if active_decodes + active_prefills == 0 {
                    1.0
                } else {
                    cal.decode_marginal_gain(active_decodes + active_prefills)
                };
                CandidateView {
                    worker_id: w.id.clone(),
                    slot: w.slot,
                    cache_state: cache.state,
                    reusable_tokens: reusable,
                    remaining_work_ms: remaining,
                    prefill_cost_ms: prefill,
                    decode_cost_ms: decode,
                    estimated_ttft_ms: remaining + prefill,
                    estimated_service_ms: prefill + decode,
                    cache_savings_ms: raw_reusable as f64 * w.prefill_ms_per_tok(),
                    prefill_bucket: bucket_label(active_prefills + 1),
                    decode_bucket: bucket_label(decode_load),
                    prefill_marginal_gain: prefill_gain,
                    decode_marginal_gain: decode_gain,
                }
            })
            .collect()
    }

    /// One-row variant for the client-pinned path (no comparison, but the
    /// exported table keeps the same shape).
    fn single_view(&self, w: &Arc<WorkerHandle>, req: &RequestShape) -> CandidateView {
        self.views(std::slice::from_ref(w), req).remove(0)
    }

    /// Winner index under the active slot policy.
    fn pick(&self, views: &[CandidateView]) -> usize {
        let least_work = |v: &CandidateView| {
            // Cache ignored: prefill priced WITHOUT the reuse discount
            // (adding savings back) plus whatever is already ahead.
            v.remaining_work_ms + v.prefill_cost_ms + v.cache_savings_ms + v.decode_cost_ms
        };
        match self.policy {
            RoutingPolicy::SlotLeastWork => min_idx(views, least_work),
            RoutingPolicy::SlotPrefixOnly => {
                let best = views
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.reusable_tokens > 0)
                    .max_by(|(_, a), (_, b)| {
                        a.reusable_tokens
                            .cmp(&b.reusable_tokens)
                            .then(b.estimated_ttft_ms.total_cmp(&a.estimated_ttft_ms))
                    });
                match best {
                    Some((i, _)) => i,
                    None => min_idx(views, least_work),
                }
            }
            // The load gate: a cache hit is worth exactly its savings and
            // only while they exceed the wait — argmin estimated TTFT.
            RoutingPolicy::SlotCacheAwareEct => min_idx(views, |v| v.estimated_ttft_ms),
            _ => 0,
        }
    }

    fn slot_reason(&self, views: &[CandidateView], chosen: usize) -> DecisionReason {
        match self.policy {
            RoutingPolicy::SlotLeastWork => DecisionReason::LeastWorkSpread,
            RoutingPolicy::SlotPrefixOnly => {
                if views[chosen].reusable_tokens > 0 {
                    DecisionReason::PrefixOnlyAffinity
                } else {
                    DecisionReason::NoPrefixAvailable
                }
            }
            RoutingPolicy::SlotCacheAwareEct => {
                let chosen_view = &views[chosen];
                if chosen_view.reusable_tokens > 0 {
                    return DecisionReason::CacheHitCheapest;
                }
                // Refused reuse? Attribute the bypass.
                let Some(best) = views
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.cache_savings_ms > 0.0)
                    .max_by(|(_, a), (_, b)| a.cache_savings_ms.total_cmp(&b.cache_savings_ms))
                else {
                    return DecisionReason::LeastWorkSpread; // genuinely cold
                };
                let (_, best_view) = best;
                if best_view.cache_savings_ms < self.config.min_cache_savings_ms {
                    DecisionReason::InsufficientBenefit
                } else if best_view.cache_state == CacheInvestment::Populating {
                    // The warm slot is mid-population: its head request is
                    // still prefilling the very prefix we would reuse.
                    DecisionReason::PrefillPressureOverrodeAffinity
                } else {
                    DecisionReason::LoadOverrodeAffinity
                }
            }
            _ => DecisionReason::SingleWorker,
        }
    }

    fn bypass_record(&self, views: &[CandidateView], chosen: usize) -> Option<BypassRecord> {
        if self.policy != RoutingPolicy::SlotCacheAwareEct {
            return None;
        }
        let reason = self.slot_reason(views, chosen).bypass_label()?;
        let best = views
            .iter()
            .enumerate()
            .filter(|(i, v)| *i != chosen && v.cache_savings_ms > 0.0)
            .max_by(|(_, a), (_, b)| a.cache_savings_ms.total_cmp(&b.cache_savings_ms))?;
        Some(BypassRecord {
            reason,
            worker_id: best.1.worker_id.clone(),
            slot: best.1.slot,
            forgone_savings_ms: best.1.cache_savings_ms,
            forgone_match_tokens: best.1.reusable_tokens,
        })
    }
}

fn min_idx<T, F: Fn(&T) -> f64>(items: &[T], score: F) -> usize {
    let mut best = 0;
    let mut best_score = f64::INFINITY;
    for (i, item) in items.iter().enumerate() {
        let s = score(item);
        if s < best_score {
            best = i;
            best_score = s;
        }
    }
    best
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

    /// A READY slot worker.
    fn slot(id: &str, n: u32) -> Arc<WorkerHandle> {
        let mut w = WorkerHandle::new(id, format!("http://{id}"), "m", "mock");
        w.slot = Some(n);
        w.record_transport_success();
        w.request_served();
        Arc::new(w)
    }

    fn shape(prefix: u64, in_tok: u64, out_tok: u64) -> RequestShape {
        RequestShape {
            prefix_hash: prefix,
            input_tokens: in_tok,
            output_tokens: out_tok,
            pinned_slot: None,
            service_class: ServiceClass::Standard,
        }
    }

    fn decided(out: SelectOutcome) -> RouteDecision {
        match out {
            SelectOutcome::Decided(d) => d,
            SelectOutcome::Held { active_prefills, marginal_gain } => panic!(
                "expected a decision, got Held (active={active_prefills}, gain={marginal_gain})"
            ),
            SelectOutcome::AllUnhealthy => panic!("expected a decision"),
        }
    }

    #[test]
    fn least_outstanding_picks_idle_worker() {
        let a = worker("a");
        let b = worker("b");
        a.outstanding.store(3, Ordering::Relaxed);
        let r = Router::new(vec![a, b], RoutingPolicy::LeastOutstandingRequests).unwrap();
        match r.select(shape(1, 128, 64)) {
            WorkerOutcome::Worker(w) => assert_eq!(w.id, "b"),
            WorkerOutcome::AllUnhealthy => panic!("expected a worker"),
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
            WorkerOutcome::Worker(w) => assert_eq!(w.id, learned),
            WorkerOutcome::AllUnhealthy => panic!("expected a worker"),
        }
    }

    #[test]
    fn ect_prefers_less_remaining_work_over_fewer_requests() {
        // a: 3 active short dispatches; b: 1 active long dispatch.
        // LOR would pick b (fewer requests); ECT must pick a because b's
        // remaining work dominates.
        let a = worker("a");
        let b = worker("b");
        a.begin_work(100.0, 100.0);
        a.begin_work(100.0, 100.0);
        a.begin_work(100.0, 100.0); // ~300 ms remaining
        b.begin_work(60_000.0, 60_000.0); // ~60 s remaining
        a.outstanding.store(3, Ordering::Relaxed);
        b.outstanding.store(1, Ordering::Relaxed);

        let lor = Router::new(vec![a.clone(), b.clone()], RoutingPolicy::LeastOutstandingRequests)
            .unwrap();
        match lor.select(shape(7, 128, 64)) {
            WorkerOutcome::Worker(w) => assert_eq!(w.id, "b"),
            WorkerOutcome::AllUnhealthy => panic!(),
        }

        let ect = Router::new(
            vec![a.clone(), b.clone()],
            RoutingPolicy::LeastEstimatedCompletionTime,
        )
        .unwrap();
        match ect.select(shape(7, 128, 64)) {
            WorkerOutcome::Worker(w) => assert_eq!(w.id, "a"),
            WorkerOutcome::AllUnhealthy => panic!(),
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
        // the (fixed, rate) split is identifiable. Each observation rides its
        // own dispatch ticket, as the proxy does for real requests.
        for i in 0..200 {
            let toks = 500 + (i % 7) * 350;
            let t = w.begin_work(1.0, f64::MAX);
            w.record_prefill(toks as u64, 500.0 + 10.0 * toks as f64, &t);
            w.end_work(&t);
        }
        assert!((w.prefill_ms_per_tok() - 10.0).abs() < 1.0, "{}", w.prefill_ms_per_tok());
        assert!((w.fixed_overhead_ms() - 500.0).abs() < 400.0, "{}", w.fixed_overhead_ms());
        for _ in 0..200 {
            let t = w.begin_work(1.0, 0.0); // instantly "decoding"
            w.record_decode(100, 5_000.0, &t); // 50 ms/token solo
            w.end_work(&t);
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
            WorkerOutcome::Worker(w) => assert_eq!(w.id, "b"),
            WorkerOutcome::AllUnhealthy => panic!(),
        }
        // Recovery is not one probe either: full ladder, then routable.
        a.record_transport_success();
        a.request_served();
        a.request_served();
        assert_eq!(a.state(), WorkerState::Ready);
        match r.select(shape(2, 8, 8)) {
            WorkerOutcome::Worker(w) => assert_eq!(w.id, "a"),
            WorkerOutcome::AllUnhealthy => panic!(),
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
        assert!(matches!(r.select(shape(1, 8, 8)), WorkerOutcome::AllUnhealthy));
    }

    #[test]
    fn end_work_removes_estimate() {
        let w = worker("w");
        let stored = w.begin_work(1234.0, 0.0);
        assert!((w.remaining_work_ms() - 1234.0).abs() < 5.0);
        // A ticket id that can never match: ids are unique and monotonic,
        // so end_work with an unknown ticket is a no-op (the old
        // Instant-equality matching could collide on same-clock-tick
        // starts — the ticket exists to make that impossible).
        let other = DispatchTicket { id: u64::MAX, ..stored };
        w.end_work(&other);
        assert!(w.remaining_work_ms() > 0.0);
        w.end_work(&stored);
        assert!((w.remaining_work_ms()).abs() < 0.001);
    }

    #[test]
    fn policy_parses() {
        assert_eq!(RoutingPolicy::parse("prefix-affinity"), Some(RoutingPolicy::PrefixAffinity));
        assert_eq!(RoutingPolicy::parse("native"), Some(RoutingPolicy::Native));
        assert_eq!(RoutingPolicy::parse("least-work"), Some(RoutingPolicy::SlotLeastWork));
        assert_eq!(RoutingPolicy::parse("prefix-only"), Some(RoutingPolicy::SlotPrefixOnly));
        assert_eq!(
            RoutingPolicy::parse("cache-aware-ect"),
            Some(RoutingPolicy::SlotCacheAwareEct)
        );
        assert_eq!(
            RoutingPolicy::parse("load-gated-prefix"),
            Some(RoutingPolicy::SlotCacheAwareEct)
        );
        assert_eq!(RoutingPolicy::parse("nope"), None);
    }

    // ---------------------------------------------------------------------
    // Phase 4B.2: slots as schedulable cache domains
    // ---------------------------------------------------------------------

    fn slot_bank(n: usize) -> Vec<Arc<WorkerHandle>> {
        (0..n).map(|i| slot(&format!("s{i}"), i as u32).clone()).collect()
    }

    /// Realistic llama.cpp economics for this machine (estimated units):
    /// 55 s to prefill 8.4K tokens cold, ~6 ms/token, ~25 ms/token decode.
    fn llama_defaults() -> ServiceDefaults {
        ServiceDefaults { fixed_ms: 250.0, prefill_ms_per_tok: 6.2, decode_ms_per_tok: 25.0 }
    }

    fn slots_with(n: usize, policy: RoutingPolicy) -> Router {
        let bank: Vec<_> = slot_bank(n)
            .into_iter()
            .map(|w| {
                Arc::new({
                    let mut h = WorkerHandle::with_defaults(
                        w.id.clone(),
                        &w.endpoint,
                        &w.model,
                        "llama",
                        llama_defaults(),
                    );
                    h.slot = w.slot;
                    h.record_transport_success();
                    h.request_served();
                    h
                })
            })
            .collect();
        Router::new(bank, policy).unwrap()
    }

    /// Slot bank whose handles SHARE one calibration — the topology a real
    /// llama.cpp slot bank has (one server, one model).
    fn slots_shared_cal(n: usize, policy: RoutingPolicy, cal: Arc<RuntimeCalibration>) -> Router {
        let bank: Vec<_> = (0..n)
            .map(|i| {
                let mut h = WorkerHandle::with_calibration(
                    format!("s{i}"),
                    format!("http://s{i}"),
                    "m",
                    "llama",
                    cal.clone(),
                );
                h.slot = Some(i as u32);
                h.record_transport_success();
                h.request_served();
                Arc::new(h)
            })
            .collect();
        Router::new(bank, policy).unwrap()
    }

    #[test]
    fn least_work_reservations_spread_a_burst_across_slots() {
        let r = slots_with(8, RoutingPolicy::SlotLeastWork);
        let mut seen = Vec::new();
        for _ in 0..8 {
            let d = decided(r.select_reserved(shape(7, 8400, 256)));
            seen.push(d.worker.slot.unwrap());
        }
        seen.sort();
        assert_eq!(seen, (0..8).collect::<Vec<_>>(), "each arrival must see prior reservations");
    }

    #[test]
    fn slot_cache_investment_lifecycle() {
        let r = slots_with(2, RoutingPolicy::SlotCacheAwareEct);
        let s0 = r.workers()[0].clone();
        assert_eq!(s0.slot_cache_view(42, 8400).state, CacheInvestment::Absent);

        s0.note_slot_dispatch(42, 8400);
        let v = s0.slot_cache_view(42, 8400);
        assert_eq!(v.state, CacheInvestment::Populating);
        assert_eq!(v.reusable_tokens, 8400, "populating reuse = expected prompt size");

        s0.note_slot_completion(42, 11_400, 0);
        let v = s0.slot_cache_view(42, 8400);
        assert_eq!(v.state, CacheInvestment::Resident);
        // A MISS completion is how a population looks: the request reused
        // nothing, but the slot now HOLDS its whole prompt as reusable
        // state (the 4B.2 study's own telemetry caught the original
        // version of this model conflating the two).
        assert_eq!(v.reusable_tokens, 8400, "resident = full prompt held");
        let snap = s0.slot_cache_snapshot().unwrap();
        assert_eq!(snap.last_reuse_observed_real, 0, "observed reuse kept for telemetry");

        // A different fingerprint replaces the single-sequence entry.
        s0.note_slot_dispatch(43, 8400);
        assert_eq!(s0.slot_cache_view(42, 8400).state, CacheInvestment::Absent);
        assert_eq!(s0.slot_cache_view(43, 8400).state, CacheInvestment::Populating);

        // Whole-worker handles have no slot model.
        let plain = worker("plain");
        plain.note_slot_dispatch(42, 8400);
        assert_eq!(plain.slot_cache_view(42, 8400).state, CacheInvestment::Absent);
    }

    #[test]
    fn prefix_only_piles_onto_the_warm_slot() {
        let r = slots_with(8, RoutingPolicy::SlotPrefixOnly);
        r.workers()[0].note_slot_completion(7, 11_400, 11_400);
        for _ in 0..4 {
            let d = decided(r.select_reserved(shape(7, 8400, 256)));
            assert_eq!(d.reason, DecisionReason::PrefixOnlyAffinity);
            assert_eq!(d.worker.slot, Some(0), "affinity must win over 7 idle slots");
        }
        assert_eq!(r.workers()[0].outstanding(), 4, "the pile-on is the point");
    }

    #[test]
    fn cache_aware_ect_prefers_warm_idle_slot() {
        let r = slots_with(8, RoutingPolicy::SlotCacheAwareEct);
        r.workers()[3].note_slot_completion(7, 11_400, 11_400);
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.worker.slot, Some(3));
        assert_eq!(d.reason, DecisionReason::CacheHitCheapest);
        assert!(d.candidates[d.chosen].cache_savings_ms > 50_000.0);
    }

    #[test]
    fn cache_aware_ect_spreads_cold_burst_during_population() {
        let r = slots_with(8, RoutingPolicy::SlotCacheAwareEct);
        let mut seen = Vec::new();
        for _ in 0..6 {
            let d = decided(r.select_reserved(shape(7, 8400, 256)));
            seen.push(d.worker.slot.unwrap());
            // All arrivals share the fingerprint; the first dispatch makes
            // its slot POPULATING — joining it means waiting out its whole
            // service time, which never beats a cold prefill here.
            assert_ne!(d.reason, DecisionReason::CacheHitCheapest);
        }
        let mut sorted = seen.clone();
        sorted.sort();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>(), "cold burst must spread");
    }

    #[test]
    fn cache_aware_ect_load_overrides_affinity() {
        let r = slots_with(8, RoutingPolicy::SlotCacheAwareEct);
        // Slot 0 resident; every join queues behind its predecessor's
        // ~6.65 s of decode (fixed 250 ms + 256 × 25 ms). The cold
        // alternative costs ~52.3 s of prefill, so affinity wins exactly
        // while the accumulated queue is cheaper — 8 joins, then the gate
        // closes and the 9th arrival recomputes elsewhere.
        r.workers()[0].note_slot_completion(7, 11_400, 11_400);
        for _ in 0..8 {
            let d = decided(r.select_reserved(shape(7, 8400, 256)));
            assert_eq!(d.worker.slot, Some(0), "warm slot keeps winning until the gate closes");
        }
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.worker.slot, Some(1), "queue pressure must override affinity");
        assert_eq!(d.reason, DecisionReason::LoadOverrodeAffinity);
        let bypass = d.bypass.expect("a refused cache hit must be recorded");
        assert_eq!(bypass.reason, "queue_pressure");
        assert_eq!(bypass.slot, Some(0));
        assert!(bypass.forgone_savings_ms > 50_000.0);
    }

    #[test]
    fn loaded_decode_buckets_close_the_load_gate_earlier() {
        // The 4B.2 failure in miniature. Bucket-1 decode is measured at
        // ~100 ms/tok (20 EWMA steps toward a 100 ms/tok observation under
        // 2 streams), so queueing behind a busy warm slot is priced at the
        // LOADED rate and the gate closes at 3 joins — the seeded
        // one-rate model (25 ms/tok everywhere) only closed at 8.
        let cal = Arc::new(RuntimeCalibration::from_seeds(
            "apple-m5", "llama-cpp", "m", 8, llama_defaults(),
        ));
        let peer = cal.register_dispatch(600_000.0, 1.0);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let t = cal.register_dispatch(600_000.0, 1.0);
        for _ in 0..20 {
            cal.observe_decode(64, 6400.0, &t); // 100 ms/tok
        }
        cal.unregister_dispatch(&t);
        cal.unregister_dispatch(&peer);

        let r = slots_shared_cal(8, RoutingPolicy::SlotCacheAwareEct, cal);
        r.workers()[0].note_slot_completion(7, 11_400, 11_400);
        for _ in 0..3 {
            let d = decided(r.select_reserved(shape(7, 8400, 256)));
            assert_eq!(d.worker.slot, Some(0), "warm slot keeps winning while the loaded queue is cheap");
        }
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.worker.slot, Some(1), "loaded decode pricing must close the gate sooner than seeds");
        assert_eq!(d.reason, DecisionReason::LoadOverrodeAffinity);
    }

    #[test]
    fn cache_aware_ect_prefill_pressure_overrides_affinity() {
        let r = slots_with(8, RoutingPolicy::SlotCacheAwareEct);
        // Slot 0 is POPULATING (a same-prefix request in flight): reuse
        // exists, but joining means waiting out a full prefill.
        r.workers()[0].note_slot_dispatch(7, 8400);
        r.workers()[0].begin_work(60_000.0, 55_000.0); // mid-prefill
        r.workers()[0].inc_outstanding();
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.worker.slot, Some(1));
        assert_eq!(d.reason, DecisionReason::PrefillPressureOverrodeAffinity);
        assert_eq!(d.bypass.unwrap().reason, "prefill_pressure");
    }

    #[test]
    fn cache_aware_ect_floor_treats_tiny_reuse_as_absent() {
        let r = Router::with_config(
            slot_bank(4)
                .into_iter()
                .map(|w| {
                    Arc::new({
                        let mut h = WorkerHandle::with_defaults(
                            w.id.clone(), &w.endpoint, &w.model, "llama", llama_defaults(),
                        );
                        h.slot = w.slot;
                        h.record_transport_success();
                        h.request_served();
                        h
                    })
                })
                .collect(),
            RoutingPolicy::SlotCacheAwareEct,
            RouterConfig { min_cache_savings_ms: 10_000.0, ..Default::default() },
        )
        .unwrap();
        // A slot that only ever held a tiny probe prompt (100 real tokens)
        // offers ~620 ms of savings — far under the floor. The warm slot
        // (index 3) must be priced cold; the argmin tie then resolves to
        // index 0, and the floor-refused hit still shows up as an
        // attributed bypass.
        r.workers()[3].note_slot_completion(7, 100, 0);
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.worker.slot, Some(0), "sub-floor reuse must not sticky-route");
        assert_eq!(d.reason, DecisionReason::InsufficientBenefit);
        let bypass = d.bypass.unwrap();
        assert_eq!(bypass.reason, "insufficient_benefit");
        assert_eq!(bypass.slot, Some(3));
    }

    #[test]
    fn client_pin_reserves_the_named_slot_and_teaches_the_model() {
        let r = slots_with(4, RoutingPolicy::SlotCacheAwareEct);
        let shape = RequestShape { pinned_slot: Some(2), ..shape(7, 8400, 256) };
        let d = decided(r.select_reserved(shape));
        assert_eq!(d.worker.slot, Some(2));
        assert_eq!(d.reason, DecisionReason::ClientPinned);
        assert_eq!(d.worker.outstanding(), 1);
        assert_eq!(
            d.worker.slot_cache_view(7, 8400).state,
            CacheInvestment::Populating,
            "external warmers must be visible to the model"
        );
    }

    #[test]
    fn is_prefilling_tracks_the_prefill_phase() {
        let w = worker("w");
        assert!(!w.is_prefilling());
        let t = w.begin_work(10_000.0, 9_000.0);
        assert!(w.is_prefilling());
        w.end_work(&t);
        assert!(!w.is_prefilling());
    }

    // ---------------------------------------------------------------------
    // Phase 4C: the contention gate
    // ---------------------------------------------------------------------

    /// Shared calibration carrying a measured anti-scaling prefill curve
    /// (multipliers 1.0 / 1.43 / 2.86 / 8.33: aggregate throughput peaks
    /// at two concurrent prefills, sags to ~1.05 by bucket 3-4 and 0.60
    /// at 5-8) plus loaded decode rates.
    fn anti_scaling_cal() -> Arc<RuntimeCalibration> {
        let profile = CalibrationProfile {
            schema_version: calibration::SCHEMA_VERSION,
            bucket_labels: ["1".into(), "2".into(), "3-4".into(), "5-8".into()],
            machine: "apple-m5".into(),
            runtime: "llama-cpp".into(),
            model: "m".into(),
            slot_count: 8,
            updated_at_epoch_s: 0,
            prefill: calibration::PrefillCalibration {
                fixed_ms: 250.0,
                ms_per_token: 6.2,
                samples: 40,
                contention_multiplier: [1.0, 1.43, 2.86, 8.33],
                contention_samples: [40, 4, 8, 8],
            },
            decode: calibration::DecodeCalibration {
                ms_per_token: [450.0, 1200.0, 1200.0, 1200.0],
                samples: [10, 4, 6, 8],
            },
        };
        Arc::new(RuntimeCalibration::from_profile(&profile, llama_defaults()))
    }

    fn gated_router(cal: Arc<RuntimeCalibration>) -> Router {
        Router::with_config(
            (0..8)
                .map(|i| {
                    let mut h = WorkerHandle::with_calibration(
                        format!("s{i}"), format!("http://s{i}"), "m", "llama", cal.clone(),
                    );
                    h.slot = Some(i as u32);
                    h.record_transport_success();
                    h.request_served();
                    Arc::new(h)
                })
                .collect(),
            RoutingPolicy::SlotCacheAwareEct,
            RouterConfig { contention_gate: true, gate_max_hold_s: 20.0, ..Default::default() },
        )
        .unwrap()
    }

    /// Release a decision's reservation so a test's registry state stays
    /// exactly what it scripted (decisions reserve on the shared cal).
    fn release(d: &RouteDecision) {
        d.worker.end_work(&d.work_start);
    }

    #[test]
    fn contention_gate_holds_in_the_measured_anti_scaling_region() {
        let cal = anti_scaling_cal();
        let r = gated_router(cal.clone());
        // Two active prefills: joining as the third crosses into bucket
        // 3-4 where the measured aggregate (1.05) is below the current
        // (1.40) — the gate must hold, with nothing reserved.
        let t1 = cal.register_dispatch(600_000.0, f64::MAX);
        let t2 = cal.register_dispatch(600_000.0, f64::MAX);
        match r.select_reserved(shape(7, 8400, 256)) {
            SelectOutcome::Held { active_prefills, marginal_gain } => {
                assert_eq!(active_prefills, 2);
                assert!(marginal_gain < 1.0, "held because gain {marginal_gain} < 1");
            }
            _ => panic!("the gate must hold an anti-scaling join"),
        }
        assert!(r.workers().iter().all(|w| w.outstanding() == 0), "held = no reservation");

        // Full-reuse winner dispatches anyway: joining cache adds no
        // prefill, so the marginal-prefill question does not apply.
        r.workers()[0].note_slot_completion(7, 11_400, 11_400);
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        assert_eq!(d.reason, DecisionReason::CacheHitCheapest);
        release(&d);

        // Spill path: the hold cap expired, dispatch regardless.
        let d = decided(r.select_reserved_forced(shape(7, 8400, 256)));
        assert_eq!(d.candidates.len(), 8);
        release(&d);

        // A prefill finishing drops the load to 1; entering bucket 2 pays
        // (aggregate 1.40 > 1.00) — the gate re-opens.
        cal.unregister_dispatch(&t1);
        cal.unregister_dispatch(&t2);
        let d = decided(r.select_reserved(shape(7, 8400, 256)));
        release(&d);
    }

    #[test]
    fn contention_gate_never_holds_a_flat_or_idle_machine() {
        let cal = Arc::new(RuntimeCalibration::from_seeds(
            "apple-m5", "llama-cpp", "m", 8, llama_defaults(),
        ));
        let r = gated_router(cal.clone());
        // Distinct prefix per probe: no slot may hold reuse for it, or the
        // arrival would be a cache join (which never adds prefill and is
        // never the gate's business).
        let probe = |r: &Router, n: u64| r.select_reserved(shape(n, 8400, 256));
        // Idle: pays by definition. Within-bucket joins: unmeasured, allowed.
        for active in 0..8u64 {
            let tickets: Vec<_> = (0..active)
                .map(|_| cal.register_dispatch(600_000.0, f64::MAX))
                .collect();
            let d = decided(probe(&r, 100 + active));
            release(&d);
            for t in tickets {
                cal.unregister_dispatch(&t);
            }
        }
        // At the envelope (8 active) the gate holds even on a flat curve —
        // the ninth prefill has no measured defense.
        let tickets: Vec<_> = (0..8).map(|_| cal.register_dispatch(600_000.0, f64::MAX)).collect();
        assert!(matches!(probe(&r, 999), SelectOutcome::Held { .. }));
        for t in tickets {
            cal.unregister_dispatch(&t);
        }
    }

    #[test]
    fn client_pins_bypass_the_contention_gate() {
        let cal = anti_scaling_cal();
        let r = gated_router(cal.clone());
        let _t1 = cal.register_dispatch(600_000.0, f64::MAX);
        let _t2 = cal.register_dispatch(600_000.0, f64::MAX);
        // Cache warmers are infrastructure: a pinned dispatch is honored
        // even inside the anti-scaling region.
        let shape = RequestShape { pinned_slot: Some(5), ..shape(7, 8400, 256) };
        let d = decided(r.select_reserved(shape));
        assert_eq!(d.worker.slot, Some(5));
        assert_eq!(d.reason, DecisionReason::ClientPinned);
    }
}
