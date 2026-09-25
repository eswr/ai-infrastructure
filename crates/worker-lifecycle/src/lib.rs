//! worker-lifecycle — the worker health controller shared by the gateway's
//! worker registry and the benchmark harness.
//!
//! Phase 4A established empirically that `process alive ≠ serving traffic`:
//! a wedged mlx_lm.server answered `/health` with 200 while never serving
//! another POST (silent wedge), and vllm-metal answered HTTP 200 with zero
//! generated tokens (silent empty answer). A single up/down boolean
//! collapses those into "healthy". This crate models worker health as a
//! small state machine over four independent dimensions:
//!
//! ```text
//! process     is the OS process alive
//! transport   does the endpoint accept and answer connections
//! functional  does a tiny real generation actually produce tokens
//! semantic    does a completed request carry non-empty output
//! ```
//!
//! States and the transitions this crate emits:
//!
//! ```text
//!          ┌──────────┐  process/transport up   ┌──────────┐
//!          │ STARTING │ ───────────────────────►│ WARMING  │
//!          └──────────┘                         └────┬─────┘
//!                 │  functional pass                 │ recover_probes passes
//!                 ▼                                  ▼
//!          ┌──────────┐  anomaly  ┌──────────┐    ┌────────┐
//!          │  READY   │ ────────► │ DEGRADED │◄───│ READY  │
//!          └──────────┘           └────┬─────┘    └────────┘
//!                ▲   recovery success  │ repeated failures
//!                └─────────────────────┤
//!          ┌───────────┐               ▼
//!          │ RECOVERING│ ◄────── ┌───────────┐
//!          └─────┬─────┘ restart └│ UNHEALTHY │◄── immediate failures
//!                │ live again     └───────────┘
//!                ▼ generation++ (new incarnation)
//!          ┌──────────┐
//!          │ WARMING  │
//!          └──────────┘
//!
//! READY/DEGRADED ──drain──► DRAINING ──outstanding==0──► STOPPED
//! ```
//!
//! Escalation is per failure reason, because the failure classes are not
//! equally trustworthy:
//!
//! | reason                  | class        | effect                        |
//! |-------------------------|--------------|-------------------------------|
//! | `ProcessExited`         | immediate    | UNHEALTHY on first sighting   |
//! | `FunctionalProbeTimeout`| immediate    | UNHEALTHY (the wedge signal)  |
//! | `MemoryPressure`        | immediate    | UNHEALTHY                     |
//! | `ConnectFailed`         | accumulating | 1 → DEGRADED, N → UNHEALTHY   |
//! | `EmptyCompletion`       | accumulating | 1 → DEGRADED, N → UNHEALTHY   |
//! | `StreamStalled`         | accumulating | 1 → DEGRADED, N → UNHEALTHY   |
//! | `FirstTokenTimeout`     | accumulating | 1 → DEGRADED, N → UNHEALTHY   |
//! | `MalformedResponse`     | accumulating | 1 → DEGRADED, N → UNHEALTHY   |
//!
//! Request-path first-token timeouts accumulate rather than escalate
//! immediately on purpose: the watchdog threshold is deployment policy and
//! can misfire on a legitimately slow (8K-prompt) worker, while a functional
//! probe timeout is a *controlled* wedge probe and is trusted immediately.
//! Hysteresis works both directions: `unhealthy_after` accumulating
//! failures to quarantine a worker, `recover_probes` consecutive functional
//! passes to call a restarted worker READY again.
//!
//! DEGRADED still serves traffic (that is how it accumulates evidence and
//! recovers); UNHEALTHY and everything left of WARMING receives none. When
//! a restarted incarnation shows signs of life, the generation epoch
//! increments — `mlx-01/gen-7` wedging and `mlx-01/gen-8` serving are
//! distinct workers in telemetry, not one healthy-looking continuum.
//!
//! The crate is pure logic: no I/O, no clocks beyond `Instant`, no HTTP.
//! Probes and requests produce [`HealthObservation`]s; the gateway and the
//! benchmark harness translate those into the same transitions.

use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// Lifecycle state of one worker. Serialization is via [`WorkerState::as_str`]
/// (metric labels / status JSON), so the strings are the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerState {
    /// Registered, nothing proven yet. Receives no traffic.
    Starting,
    /// Endpoint answers, but no real generation has succeeded yet (or the
    /// recovery probe streak is still short). Receives no traffic.
    Warming,
    /// Functional probe passes; serves traffic.
    Ready,
    /// One strike against a serving worker. Still serves traffic — that is
    /// how it accumulates evidence and how it recovers.
    Degraded,
    /// Quarantined: no new assignments until it proves itself again.
    Unhealthy,
    /// A restart happened (or an UNHEALTHY worker showed signs of life);
    /// waiting for the new incarnation to answer.
    Recovering,
    /// Draining: finish in-flight work, accept nothing new.
    Draining,
    /// No outstanding work after a drain. Terminal.
    Stopped,
}

impl WorkerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Warming => "warming",
            Self::Ready => "ready",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Recovering => "recovering",
            Self::Draining => "draining",
            Self::Stopped => "stopped",
        }
    }

    /// All states, in lifecycle order (for gauge registration).
    pub fn all() -> [WorkerState; 8] {
        [
            Self::Starting,
            Self::Warming,
            Self::Ready,
            Self::Degraded,
            Self::Unhealthy,
            Self::Recovering,
            Self::Draining,
            Self::Stopped,
        ]
    }
}

/// Why a worker's health moved. The set is the lab's empirically observed
/// failure signatures plus the actors a Kubernetes phase will introduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureReason {
    /// OS process is gone.
    ProcessExited,
    /// Connect refused / reset / upstream HTTP refusal to serve.
    ConnectFailed,
    /// Real request never produced a first token inside the watchdog.
    FirstTokenTimeout,
    /// The tiny controlled generation probe timed out — `/health` may still
    /// be 200. This is the silent-wedge signature.
    FunctionalProbeTimeout,
    /// HTTP 200, valid payload, zero generated tokens. The vllm-metal
    /// signature: an HTTP success that is still a failed inference.
    EmptyCompletion,
    /// Answer arrived but could not be parsed as a completion payload.
    MalformedResponse,
    /// Stream started, then stopped producing tokens until the watchdog
    /// tripped (the 193 s mlx-lm mid-stream stall).
    StreamStalled,
    /// Host memory pressure suspected (external actor report).
    MemoryPressure,
    /// A restart was commanded (external actor), not observed.
    Restarting,
}

impl FailureReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProcessExited => "process_exited",
            Self::ConnectFailed => "connect_failed",
            Self::FirstTokenTimeout => "first_token_timeout",
            Self::FunctionalProbeTimeout => "functional_probe_timeout",
            Self::EmptyCompletion => "empty_completion",
            Self::MalformedResponse => "malformed_response",
            Self::StreamStalled => "stream_stalled",
            Self::MemoryPressure => "memory_pressure",
            Self::Restarting => "restarting",
        }
    }

    /// Immediate reasons quarantine a worker on the *first* occurrence.
    /// Everything else must repeat `unhealthy_after` times — one dropped
    /// connection or one empty answer must not flap a serving worker.
    pub fn is_immediate(&self) -> bool {
        matches!(
            self,
            Self::ProcessExited | Self::FunctionalProbeTimeout | Self::MemoryPressure
        )
    }
}

/// Independent health dimension. A worker is never "just down": these four
/// say *how* it is broken (or that we don't know yet).
///
/// Ordering is the dependency ladder: a semantic success implies a
/// functional pass implies a working transport; a higher-dimension success
/// clears the failure streaks of everything it depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dimension {
    Process,
    Transport,
    Functional,
    Semantic,
}

impl Dimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Transport => "transport",
            Self::Functional => "functional",
            Self::Semantic => "semantic",
        }
    }

    fn index(&self) -> usize {
        match self {
            Self::Process => 0,
            Self::Transport => 1,
            Self::Functional => 2,
            Self::Semantic => 3,
        }
    }
}

/// Tristate per dimension: Unknown until first observed. "Unknown" matters —
/// a silent wedge leaves semantic health *unknown*, not "up".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Unknown,
    Up,
    Down,
}

impl Health {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

// ---------------------------------------------------------------------------
// Observations, transitions, condition
// ---------------------------------------------------------------------------

/// One health observation fed into the machine: probe results, request
/// outcomes, process checks.
#[derive(Debug, Clone)]
pub struct HealthObservation {
    pub dimension: Dimension,
    pub ok: bool,
    /// Required when `ok == false`: why.
    pub reason: Option<FailureReason>,
    pub detail: Option<String>,
}

impl HealthObservation {
    pub fn up(dimension: Dimension) -> Self {
        Self {
            dimension,
            ok: true,
            reason: None,
            detail: None,
        }
    }

    pub fn failed(dimension: Dimension, reason: FailureReason) -> Self {
        Self {
            dimension,
            ok: false,
            reason: Some(reason),
            detail: None,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// One state transition, as emitted to metrics and logs.
#[derive(Debug, Clone)]
pub struct Transition {
    pub from: WorkerState,
    pub to: WorkerState,
    pub reason: Option<FailureReason>,
    /// Generation *after* the transition (bumped when a restarted
    /// incarnation starts warming).
    pub generation: u64,
    pub at: Instant,
}

/// Everything an operator (or Kubernetes) needs about one worker, in one
/// snapshot.
#[derive(Debug, Clone)]
pub struct WorkerCondition {
    pub state: WorkerState,
    /// Why we entered this state (None for neutral entries like READY).
    pub reason: Option<FailureReason>,
    pub since: Instant,
    /// Worst consecutive-failure streak across dimensions.
    pub consecutive_failures: u32,
    pub last_success: Option<Instant>,
    /// Incarnation counter; increments when a restarted worker starts
    /// warming. Telemetry keyed by `(worker, generation)` never blends a
    /// wedged incarnation with its healthy successor.
    pub generation: u64,
    pub dimensions: [Health; 4],
}

impl WorkerCondition {
    pub fn dimension(&self, d: Dimension) -> Health {
        self.dimensions[d.index()]
    }
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LifecyclePolicy {
    /// Consecutive accumulating-class failures that escalate to UNHEALTHY
    /// (the 1 → DEGRADED → N → UNHEALTHY ladder).
    pub unhealthy_after: u32,
    /// Consecutive functional passes a *restarted* worker needs in WARMING
    /// before READY. The asymmetry vs `unhealthy_after` is deliberate
    /// hysteresis: recovery must be proven, not assumed.
    pub recover_probes: u32,
    /// First warmup after registration needs a single pass (a booting worker
    /// has no prior failures to distrust).
    /// (Not configurable: fixed at 1.)
    pub history_len: usize,
}

impl Default for LifecyclePolicy {
    fn default() -> Self {
        Self {
            unhealthy_after: 3,
            recover_probes: 2,
            history_len: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct WorkerLifecycle {
    policy: LifecyclePolicy,
    state: WorkerState,
    reason: Option<FailureReason>,
    since: Instant,
    generation: u64,
    /// Failure streak per dimension. A success clears its own dimension and
    /// every dimension it implies (semantic ⇒ functional ⇒ transport), so a
    /// connect blip heals on the next connect but repeated empty completions
    /// still accumulate across interleaved healthy connects.
    fail_counts: [u32; 4],
    /// Consecutive functional passes while WARMING (the recovery streak).
    warmup_passes: u32,
    /// Passes required to leave WARMING: 1 on first boot,
    /// `recover_probes` after a restart.
    warmup_goal: u32,
    /// The dimension that got us DEGRADED — recovery must match.
    degraded_dimension: Dimension,
    last_success: Option<Instant>,
    dimensions: [Health; 4],
    history: VecDeque<Transition>,
    total_transitions: u64,
}

impl WorkerLifecycle {
    pub fn new(policy: LifecyclePolicy) -> Self {
        Self {
            warmup_goal: 1,
            degraded_dimension: Dimension::Transport,
            policy,
            state: WorkerState::Starting,
            reason: None,
            since: Instant::now(),
            generation: 1,
            fail_counts: [0; 4],
            warmup_passes: 0,
            last_success: None,
            dimensions: [Health::Unknown; 4],
            history: VecDeque::new(),
            total_transitions: 0,
        }
    }

    // -- inputs -------------------------------------------------------------

    /// Feed one observation; returns transitions it caused (0–2: recovery
    /// can hop RECOVERING → WARMING → READY in one step when a single pass
    /// suffices).
    pub fn observe(&mut self, obs: &HealthObservation) -> Vec<Transition> {
        if obs.ok {
            self.on_success(obs.dimension)
        } else {
            let reason = obs.reason.unwrap_or(FailureReason::MalformedResponse);
            self.on_failure(obs.dimension, reason)
        }
    }

    /// A real request completed with ≥1 generated token — the strongest
    /// functional + semantic evidence there is.
    pub fn request_served(&mut self) -> Vec<Transition> {
        self.on_success(Dimension::Semantic)
    }

    /// A restart was commanded by an external actor (the gateway cannot
    /// restart native processes; the harness or a future Kubernetes
    /// controller can). Only meaningful from UNHEALTHY (reactive) or
    /// DEGRADED (preventive).
    pub fn begin_restart(&mut self) -> Vec<Transition> {
        if !matches!(self.state, WorkerState::Unhealthy | WorkerState::Degraded) {
            return vec![];
        }
        self.transition(WorkerState::Recovering, Some(FailureReason::Restarting))
    }

    /// Stop assigning new work. Allowed from every live state; orthogonal to
    /// health on purpose (a healthy worker may be drained for replacement).
    pub fn drain(&mut self) -> Vec<Transition> {
        if matches!(self.state, WorkerState::Draining | WorkerState::Stopped) {
            return vec![];
        }
        self.transition(WorkerState::Draining, None)
    }

    /// The gateway calls this when outstanding work hits zero. Only
    /// completes a drain.
    pub fn notify_drained(&mut self) -> Vec<Transition> {
        if self.state != WorkerState::Draining {
            return vec![];
        }
        self.transition(WorkerState::Stopped, None)
    }

    // -- reads --------------------------------------------------------------

    pub fn state(&self) -> WorkerState {
        self.state
    }

    pub fn reason(&self) -> Option<FailureReason> {
        self.reason
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn consecutive_failures(&self) -> u32 {
        *self.fail_counts.iter().max().unwrap_or(&0)
    }

    pub fn last_success(&self) -> Option<Instant> {
        self.last_success
    }

    /// READY and DEGRADED receive traffic; nothing else does.
    pub fn serves_traffic(&self) -> bool {
        matches!(self.state, WorkerState::Ready | WorkerState::Degraded)
    }

    pub fn is_terminal(&self) -> bool {
        self.state == WorkerState::Stopped
    }

    pub fn condition(&self) -> WorkerCondition {
        WorkerCondition {
            state: self.state,
            reason: self.reason,
            since: self.since,
            consecutive_failures: self.consecutive_failures(),
            last_success: self.last_success,
            generation: self.generation,
            dimensions: self.dimensions,
        }
    }

    pub fn history(&self) -> &VecDeque<Transition> {
        &self.history
    }

    pub fn total_transitions(&self) -> u64 {
        self.total_transitions
    }

    // -- internals ----------------------------------------------------------

    fn on_success(&mut self, dim: Dimension) -> Vec<Transition> {
        self.last_success = Some(Instant::now());
        self.dimensions[dim.index()] = Health::Up;
        match dim {
            Dimension::Process => self.fail_counts[0] = 0,
            Dimension::Transport => {
                self.fail_counts[1] = 0;
            }
            Dimension::Functional => {
                self.fail_counts[1] = 0;
                self.fail_counts[2] = 0;
                self.warmup_passes += 1;
            }
            Dimension::Semantic => {
                self.fail_counts[1] = 0;
                self.fail_counts[2] = 0;
                self.fail_counts[3] = 0;
                self.warmup_passes += 1;
            }
        }

        let mut out = vec![];
        match self.state {
            WorkerState::Starting => match dim {
                Dimension::Process => {}
                Dimension::Transport => out.extend(self.transition(WorkerState::Warming, None)),
                // Real traffic (or a probe) succeeded straight away: no
                // reason to make it pass through WARMING.
                Dimension::Functional | Dimension::Semantic => {
                    out.extend(self.transition(WorkerState::Ready, None))
                }
            },
            WorkerState::Warming => {
                if dim >= Dimension::Functional && self.warmup_passes >= self.warmup_goal {
                    out.extend(self.transition(WorkerState::Ready, None));
                }
            }
            WorkerState::Ready => {}
            WorkerState::Degraded => {
                let required = match self.degraded_dimension {
                    Dimension::Transport => Dimension::Transport,
                    // A probe (functional-grade) can clear a semantic
                    // suspicion; a mere connect cannot.
                    Dimension::Functional | Dimension::Semantic => Dimension::Functional,
                    Dimension::Process => Dimension::Functional,
                };
                if dim >= required {
                    out.extend(self.transition(WorkerState::Ready, None));
                }
            }
            // Signs of life: somebody restarted the worker out-of-band.
            // Quarantine lifts only through the full recovery ladder.
            WorkerState::Unhealthy => {
                if dim >= Dimension::Transport {
                    out.extend(self.transition(WorkerState::Recovering, None));
                }
            }
            WorkerState::Recovering => {
                if dim >= Dimension::Transport {
                    // The new incarnation is answering: epoch boundary.
                    self.generation += 1;
                    self.warmup_goal = self.policy.recover_probes;
                    self.warmup_passes = if dim >= Dimension::Functional { 1 } else { 0 };
                    out.extend(self.transition(WorkerState::Warming, None));
                    if dim >= Dimension::Functional && self.warmup_passes >= self.warmup_goal {
                        out.extend(self.transition(WorkerState::Ready, None));
                    }
                }
            }
            WorkerState::Draining | WorkerState::Stopped => {}
        }
        out
    }

    fn on_failure(&mut self, dim: Dimension, reason: FailureReason) -> Vec<Transition> {
        self.fail_counts[dim.index()] = self.fail_counts[dim.index()].saturating_add(1);
        self.dimensions[dim.index()] = Health::Down;
        self.warmup_passes = 0;

        // Draining/Stopped workers stay put; health is still recorded so a
        // drain of a dying worker still lands on truthful state.
        if matches!(self.state, WorkerState::Draining | WorkerState::Stopped) {
            return vec![];
        }

        if reason.is_immediate() {
            return self.transition(WorkerState::Unhealthy, Some(reason));
        }

        if self.fail_counts[dim.index()] >= self.policy.unhealthy_after {
            return self.transition(WorkerState::Unhealthy, Some(reason));
        }

        match self.state {
            // Non-serving states must never route through DEGRADED (which
            // serves traffic): they simply keep accumulating until the
            // unhealthy threshold.
            WorkerState::Ready => {
                self.degraded_dimension = dim;
                self.transition(WorkerState::Degraded, Some(reason))
            }
            WorkerState::Degraded => {
                if dim > self.degraded_dimension {
                    self.degraded_dimension = dim;
                }
                self.reason = Some(reason);
                vec![]
            }
            _ => vec![],
        }
    }

    fn transition(&mut self, to: WorkerState, reason: Option<FailureReason>) -> Vec<Transition> {
        let t = Transition {
            from: self.state,
            to,
            reason,
            generation: self.generation,
            at: Instant::now(),
        };
        self.state = to;
        self.reason = reason;
        self.since = t.at;
        if to == WorkerState::Ready {
            self.fail_counts = [0; 4];
            self.warmup_passes = 0;
            self.warmup_goal = 1;
        }
        self.total_transitions += 1;
        self.history.push_back(t.clone());
        while self.history.len() > self.policy.history_len {
            self.history.pop_front();
        }
        vec![t]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> WorkerLifecycle {
        WorkerLifecycle::new(LifecyclePolicy::default())
    }

    /// Drive STARTING → WARMING → READY the ordinary way.
    fn warm_up(lc: &mut WorkerLifecycle) {
        let ts = lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(ts.len(), 1, "STARTING → WARMING emits a transition");
        assert_eq!(ts[0].to, WorkerState::Warming);
        let ts = lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].to, WorkerState::Ready);
    }

    fn state_of(lc: &WorkerLifecycle) -> WorkerState {
        lc.state()
    }

    #[test]
    fn happy_path_starting_warming_ready() {
        let mut lc = machine();
        assert_eq!(state_of(&lc), WorkerState::Starting);
        assert!(!lc.serves_traffic());

        let ts = lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(lc.state(), WorkerState::Warming);
        assert_eq!(ts.len(), 1);
        assert!(!lc.serves_traffic(), "warming workers get no traffic");

        let ts = lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
        assert_eq!(ts[0].from, WorkerState::Warming);
        assert_eq!(ts[0].to, WorkerState::Ready);
        assert!(lc.serves_traffic());
        // A functional pass straight from STARTING skips warming: real
        // generation already proved everything.
        let mut lc2 = machine();
        lc2.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc2.state(), WorkerState::Ready);
    }

    #[test]
    fn accumulating_failure_ladder_ready_degraded_unhealthy() {
        let mut lc = machine();
        warm_up(&mut lc);

        let ts = lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);
        assert_eq!(ts[0].reason, Some(FailureReason::EmptyCompletion));
        assert!(lc.serves_traffic(), "degraded still serves");

        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);

        let ts = lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Unhealthy);
        assert_eq!(ts[0].from, WorkerState::Degraded);
        assert!(!lc.serves_traffic());
        assert_eq!(lc.condition().reason, Some(FailureReason::EmptyCompletion));
    }

    #[test]
    fn immediate_reasons_quarantine_on_first_sighting() {
        for (dim, reason) in [
            (Dimension::Functional, FailureReason::FunctionalProbeTimeout),
            (Dimension::Process, FailureReason::ProcessExited),
            (Dimension::Process, FailureReason::MemoryPressure),
        ] {
            let mut lc = machine();
            warm_up(&mut lc);
            let ts = lc.observe(&HealthObservation::failed(dim, reason));
            assert_eq!(lc.state(), WorkerState::Unhealthy, "{reason:?}");
            assert_eq!(ts.len(), 1);
        }
    }

    #[test]
    fn first_token_timeout_accumulates_like_other_request_failures() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FirstTokenTimeout,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);
    }

    #[test]
    fn degraded_recovers_on_matching_success() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);

        // A probe pass (functional grade) clears a semantic suspicion.
        let ts = lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
        assert_eq!(ts[0].from, WorkerState::Degraded);
        assert_eq!(ts[0].to, WorkerState::Ready);
        assert_eq!(lc.reason(), None);

        // Transport-caused degradation needs at least transport evidence to
        // heal; functional evidence also heals it.
        lc.observe(&HealthObservation::failed(
            Dimension::Transport,
            FailureReason::ConnectFailed,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);
        lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
    }

    #[test]
    fn transport_blip_does_not_accumulate_across_functional_failures() {
        let mut lc = machine();
        warm_up(&mut lc);
        // One connect blip, then a good request (semantic success resets the
        // transport streak).
        lc.observe(&HealthObservation::failed(
            Dimension::Transport,
            FailureReason::ConnectFailed,
        ));
        assert_eq!(lc.state(), WorkerState::Degraded);
        lc.request_served();
        assert_eq!(lc.state(), WorkerState::Ready);
        assert_eq!(lc.consecutive_failures(), 0);
    }

    #[test]
    fn the_silent_wedge_signature_end_to_end() {
        // /health 200 + probe POST hangs: process up, transport up,
        // functional down → immediate quarantine despite all that green.
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::up(Dimension::Transport)); // /health lies
        let ts = lc.observe(&HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FunctionalProbeTimeout,
        ));
        assert_eq!(lc.state(), WorkerState::Unhealthy);
        assert_eq!(ts[0].reason, Some(FailureReason::FunctionalProbeTimeout));
        let cond = lc.condition();
        assert_eq!(cond.dimension(Dimension::Transport), Health::Up);
        assert_eq!(cond.dimension(Dimension::Functional), Health::Down);
        assert_eq!(cond.dimension(Dimension::Semantic), Health::Unknown);
    }

    #[test]
    fn recovery_requires_two_probes_and_bumps_generation() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FunctionalProbeTimeout,
        ));
        assert_eq!(lc.state(), WorkerState::Unhealthy);
        assert_eq!(lc.generation(), 1);

        // An external actor restarts the process; first transport success is
        // only "signs of life".
        lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(lc.state(), WorkerState::Recovering);
        assert!(!lc.serves_traffic());

        // First probe pass: new incarnation is WARMING, epoch increments.
        let ts = lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Warming);
        assert_eq!(lc.generation(), 2);
        assert_eq!(ts.len(), 1);
        assert!(!lc.serves_traffic());

        // Second pass: ready.
        let ts = lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
        assert_eq!(ts[0].from, WorkerState::Warming);
        assert_eq!(ts[0].generation, 2);
    }

    #[test]
    fn commanded_restart_walks_the_same_ladder() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Functional,
            FailureReason::FunctionalProbeTimeout,
        ));
        let ts = lc.begin_restart();
        assert_eq!(lc.state(), WorkerState::Recovering);
        assert_eq!(ts[0].reason, Some(FailureReason::Restarting));
        assert_eq!(
            lc.generation(),
            1,
            "generation bumps when the new incarnation warms, not when the old dies"
        );

        lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(lc.state(), WorkerState::Warming);
        assert_eq!(lc.generation(), 2);

        // One pass is not enough with recover_probes = 2.
        lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Warming);
        lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
    }

    #[test]
    fn warming_workers_are_not_degraded_into_traffic() {
        // A warming worker with accumulating failures must NOT route through
        // DEGRADED (which serves traffic) — it stays non-serving until it
        // either passes or quarantines.
        let mut lc = machine();
        lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(lc.state(), WorkerState::Warming);
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Warming);
        assert!(!lc.serves_traffic());
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        assert_eq!(lc.state(), WorkerState::Unhealthy);
    }

    #[test]
    fn drain_is_orthogonal_to_health_and_terminal_at_zero_work() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        // Drain a *degraded* worker: allowed, health orthogonal.
        let ts = lc.drain();
        assert_eq!(lc.state(), WorkerState::Draining);
        assert_eq!(ts[0].from, WorkerState::Degraded);
        assert!(!lc.serves_traffic());

        // Health observations still record but never move a draining worker.
        assert!(lc
            .observe(&HealthObservation::failed(
                Dimension::Semantic,
                FailureReason::EmptyCompletion
            ))
            .is_empty());
        assert_eq!(lc.state(), WorkerState::Draining);
        assert!(lc
            .observe(&HealthObservation::up(Dimension::Functional))
            .is_empty());
        assert_eq!(lc.state(), WorkerState::Draining);

        let ts = lc.notify_drained();
        assert_eq!(lc.state(), WorkerState::Stopped);
        assert_eq!(ts[0].to, WorkerState::Stopped);
        assert!(lc.is_terminal());

        // notify_drained outside a drain is a no-op; double drain is a no-op.
        let mut lc2 = machine();
        warm_up(&mut lc2);
        assert!(lc2.notify_drained().is_empty());
        assert_eq!(lc2.drain().len(), 1);
        assert_eq!(lc2.state(), WorkerState::Draining);
        assert!(lc2.drain().is_empty());
    }

    #[test]
    fn repeat_failure_inside_degraded_upgrades_required_recovery() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Transport,
            FailureReason::ConnectFailed,
        ));
        assert_eq!(lc.degraded_dimension, Dimension::Transport);
        // A worse signal arrives while degraded: transport success alone can
        // no longer heal this worker.
        lc.observe(&HealthObservation::failed(
            Dimension::Semantic,
            FailureReason::EmptyCompletion,
        ));
        lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(
            lc.state(),
            WorkerState::Degraded,
            "transport success can't clear semantic suspicion"
        );
        lc.request_served();
        assert_eq!(lc.state(), WorkerState::Ready);
    }

    #[test]
    fn unhealthy_worker_revives_out_of_band_through_the_ladder() {
        // The gateway cannot restart native processes; when the harness (or
        // k8s) does it anyway, transport success must start recovery.
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Process,
            FailureReason::ProcessExited,
        ));
        assert_eq!(lc.state(), WorkerState::Unhealthy);
        lc.observe(&HealthObservation::up(Dimension::Transport));
        assert_eq!(lc.state(), WorkerState::Recovering);
        lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Warming);
        lc.observe(&HealthObservation::up(Dimension::Functional));
        assert_eq!(lc.state(), WorkerState::Ready);
        assert_eq!(lc.generation(), 2);
    }

    #[test]
    fn condition_and_history_are_consistent() {
        let mut lc = machine();
        warm_up(&mut lc);
        lc.observe(&HealthObservation::failed(
            Dimension::Functional,
            FailureReason::StreamStalled,
        ));
        lc.request_served();

        let cond = lc.condition();
        assert_eq!(cond.state, WorkerState::Ready);
        assert_eq!(cond.consecutive_failures, 0);
        assert!(cond.last_success.is_some());
        assert_eq!(cond.generation, 1);
        assert_eq!(cond.dimension(Dimension::Semantic), Health::Up);

        assert_eq!(lc.history().len(), lc.total_transitions() as usize);
        assert_eq!(lc.history()[0].from, WorkerState::Starting);
        assert_eq!(lc.history()[0].to, WorkerState::Warming);
    }

    #[test]
    fn history_is_bounded() {
        let mut lc = WorkerLifecycle::new(LifecyclePolicy {
            history_len: 5,
            ..Default::default()
        });
        warm_up(&mut lc);
        for i in 0..20 {
            lc.observe(&HealthObservation::failed(
                Dimension::Transport,
                FailureReason::ConnectFailed,
            ));
            if i % 3 == 2 {
                // eventually unhealthy; recover via begin_restart ladder
                lc.begin_restart();
                lc.observe(&HealthObservation::up(Dimension::Transport));
                lc.observe(&HealthObservation::up(Dimension::Functional));
                lc.observe(&HealthObservation::up(Dimension::Functional));
            }
        }
        assert_eq!(lc.history().len(), 5);
        assert!(lc.total_transitions() > 5);
    }

    #[test]
    fn reason_strings_are_stable_metric_labels() {
        assert_eq!(WorkerState::Ready.as_str(), "ready");
        assert_eq!(FailureReason::EmptyCompletion.as_str(), "empty_completion");
        assert_eq!(
            FailureReason::FunctionalProbeTimeout.as_str(),
            "functional_probe_timeout"
        );
        assert_eq!(Dimension::Semantic.as_str(), "semantic");
    }
}
