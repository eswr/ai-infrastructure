//! Prometheus metric surface for the gateway (`/metrics`).
//!
//! Everything an on-call needs to answer "what's slow?" and "why?" without
//! reading logs: request outcomes, the TTFT *decomposition* (gateway /
//! admission wait / worker connect / worker first token), ITL/E2E, workload
//! accounting in tokens and estimated seconds, admission state, scheduler
//! decisions, and the adaptive worker service model itself.

use inference_lab_scheduler::WorkerHandle;
use inference_lab_worker_lifecycle::WorkerState;
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry, TextEncoder,
};

/// Wide latency buckets: first token on a big prompt can legitimately take
/// minutes (45 s at 8K on this machine), ITL lives in milliseconds.
fn latency_buckets() -> Vec<f64> {
    vec![
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        600.0,
    ]
}

fn itl_buckets() -> Vec<f64> {
    vec![
        0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
    ]
}

/// A stream that stalls can stall for a long time; this is the "one user
/// stared at a frozen spinner for a minute" scale, not the ITL scale.
fn stall_buckets() -> Vec<f64> {
    vec![
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ]
}

fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> anyhow::Result<IntGaugeVec> {
    let g = IntGaugeVec::new(Opts::new(name, help), labels)?;
    Ok(g)
}

macro_rules! reg {
    ($registry:expr, $metric:expr) => {
        $registry.register(Box::new($metric.clone()))?
    };
}

pub struct Metrics {
    pub registry: Registry,

    // -- request outcomes ---------------------------------------------------
    /// Terminal outcome of every request: ok | client_cancelled |
    /// upstream_error | upstream_http_error | first_token_timeout |
    /// no_worker | rejected | queued_cancelled.
    pub requests_total: IntCounterVec,
    /// Gauges currently in a handler (admitted or queued).
    pub requests_in_flight: IntGauge,
    pub input_tokens_total: IntCounter,
    pub output_tokens_total: IntCounter,

    // -- TTFT decomposition (seconds) ----------------------------------------
    pub ttft: Histogram,
    pub gateway_processing: Histogram,
    pub admission_wait: Histogram,
    pub worker_connect: Histogram,
    /// Headers → first token. NOT "prefill": it also contains the runtime's
    /// own queueing, tokenization and scheduling. Renamed deliberately.
    pub worker_first_token: Histogram,
    pub itl: Histogram,
    pub e2e: Histogram,

    // -- stream fairness (per-request aggregates) -----------------------------
    // Pooled ITL/E2E hide a starved minority: one stream with 10 s gaps
    // among nine healthy ones barely moves the pooled percentiles. These
    // are per-request aggregates, so starvation stays visible.
    /// Per-request *mean* inter-token latency across the decode phase.
    pub stream_inter_token: Histogram,
    /// Per-request worst gap between consecutive tokens.
    pub stream_max_stall: Histogram,
    /// Per-request stream wall time (response headers → stream end).
    pub stream_completion: Histogram,

    // -- queue / admission ----------------------------------------------------
    pub requests_active: IntGauge,
    pub requests_queued: IntGauge,
    pub queued_input_tokens: IntGauge,
    pub queued_output_tokens: IntGauge,
    pub estimated_queue_seconds: prometheus::Gauge,
    /// 0 normal | 1 degraded | 2 overloaded.
    pub load_state: IntGauge,
    pub admission_rejections_total: IntCounterVec,
    pub stream_cancelled_total: IntCounterVec,
    /// Estimated compute that never ran because a request was cancelled
    /// (queued or mid-stream), in seconds.
    pub saved_compute_seconds: prometheus::Counter,
    pub cancel_cleanup_seconds: Histogram,

    // -- scheduler / workers ---------------------------------------------------
    pub scheduler_decisions_total: IntCounterVec,
    pub worker_errors_total: IntCounterVec,
    pub workers_healthy: IntGaugeVec,
    pub worker_outstanding: IntGaugeVec,
    pub worker_prefill_ms_per_tok: IntGaugeVec,
    pub worker_decode_ms_per_tok: IntGaugeVec,
    pub worker_fixed_overhead_ms: IntGaugeVec,
    pub worker_remaining_work_seconds: IntGaugeVec,

    // -- worker lifecycle (phase 4D) --------------------------------------------
    /// 1 on the worker's current state, 0 on the rest: the state-timeline
    /// panel's data source. `process alive ≠ serving traffic` is a *state*,
    /// not a boolean.
    pub worker_state: IntGaugeVec,
    /// Every lifecycle edge, labeled by from/to/reason — the audit trail
    /// that says HOW a worker got where it is.
    pub worker_state_transitions_total: IntCounterVec,
    /// Wall time of the tiny real-generation probe.
    pub worker_functional_probe_seconds: HistogramVec,
    /// Probe failures by reason (functional_probe_timeout = silent wedge).
    pub worker_functional_probe_failures_total: IntCounterVec,
    /// Completed restarts (generation epochs begun).
    pub worker_restarts_total: IntCounterVec,
    pub worker_consecutive_failures: IntGaugeVec,
    /// Incarnation epoch; changes on every completed restart.
    pub worker_generation: IntGaugeVec,
    /// HTTP-200 responses that carried zero generated tokens — the
    /// phase-4A proof that a clean HTTP answer can still be a failed
    /// inference.
    pub empty_completions_total: IntCounterVec,
}

impl Metrics {
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            Opts::new(
                "inference_requests_total",
                "Chat completion requests by terminal outcome",
            ),
            &["outcome"],
        )?;
        // Pre-register the outcome labels so they appear at scrape time
        // before the first occurrence of each.
        for outcome in [
            "ok",
            "client_cancelled",
            "queued_cancelled",
            "upstream_error",
            "upstream_http_error",
            "first_token_timeout",
            "stream_stalled",
            "no_worker",
            "rejected",
        ] {
            requests_total.with_label_values(&[outcome]);
        }

        let requests_in_flight = IntGauge::new(
            "inference_requests_in_flight",
            "Requests inside a gateway handler",
        )?;
        let input_tokens_total = IntCounter::new(
            "inference_input_tokens_total",
            "Estimated prompt tokens admitted (calibrated estimator)",
        )?;
        let output_tokens_total = IntCounter::new(
            "inference_output_tokens_total",
            "Generated tokens observed server-side at the gateway (streamed chunks)",
        )?;

        let h = |name: &str, help: &str, buckets: Vec<f64>| -> anyhow::Result<Histogram> {
            Ok(Histogram::with_opts(
                HistogramOpts::new(name, help).buckets(buckets),
            )?)
        };
        let ttft = h(
            "inference_ttft_seconds",
            "Time to first token, gateway entry → first streamed token",
            latency_buckets(),
        )?;
        let gateway_processing = h(
            "inference_gateway_processing_seconds",
            "Gateway-side processing before dispatch (estimate + admission decision)",
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )?;
        let admission_wait = h(
            "inference_admission_wait_seconds",
            "Time spent waiting in the admission queue",
            latency_buckets(),
        )?;
        let worker_connect = h(
            "inference_worker_connect_seconds",
            "Upstream request send → response headers",
            vec![
                0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        )?;
        let worker_first_token = h(
            "inference_worker_first_token_seconds",
            "Response headers → first token (runtime queue + tokenize + prefill; NOT pure prefill)",
            latency_buckets(),
        )?;
        let itl = h(
            "inference_itl_seconds",
            "Inter-token latency observed at the gateway",
            itl_buckets(),
        )?;
        let e2e = h(
            "inference_e2e_seconds",
            "End-to-end request time, gateway entry → stream end",
            latency_buckets(),
        )?;
        let stream_inter_token = h(
            "inference_stream_inter_token_seconds",
            "Per-stream mean inter-token latency (decode phase average, not pooled)",
            itl_buckets(),
        )?;
        let stream_max_stall = h(
            "inference_stream_max_stall_seconds",
            "Per-stream longest gap between consecutive tokens (stream starvation signal)",
            stall_buckets(),
        )?;
        let stream_completion = h(
            "inference_stream_completion_seconds",
            "Per-request stream duration (response headers → stream end)",
            latency_buckets(),
        )?;

        let requests_active = IntGauge::new(
            "inference_requests_active",
            "Requests currently dispatched to a worker",
        )?;
        let requests_queued = IntGauge::new(
            "inference_requests_queued",
            "Requests waiting for admission",
        )?;
        let queued_input_tokens = IntGauge::new(
            "inference_queued_input_tokens",
            "Estimated input tokens waiting in the admission queue",
        )?;
        let queued_output_tokens = IntGauge::new(
            "inference_queued_output_tokens",
            "Requested output tokens waiting in the admission queue",
        )?;
        let estimated_queue_seconds = prometheus::Gauge::new(
            "inference_estimated_queue_seconds",
            "Estimated drain time of the admission queue for a new arrival",
        )?;
        let load_state = IntGauge::new(
            "inference_load_state",
            "Gateway load state: 0=normal 1=degraded 2=overloaded",
        )?;
        let admission_rejections_total = IntCounterVec::new(
            Opts::new(
                "inference_admission_rejections_total",
                "Requests refused at admission",
            ),
            &["reason"],
        )?;
        let stream_cancelled_total = IntCounterVec::new(
            Opts::new(
                "inference_stream_cancelled_total",
                "Client cancellations by phase",
            ),
            &["phase"],
        )?;
        stream_cancelled_total.with_label_values(&["queued"]);
        stream_cancelled_total.with_label_values(&["active"]);
        let saved_compute_seconds = prometheus::Counter::new(
            "inference_saved_estimated_compute_seconds",
            "Estimated compute avoided by cancelling requests (queued or mid-stream)",
        )?;
        let cancel_cleanup_seconds = h(
            "inference_cancel_cleanup_seconds",
            "Time to release capacity after a cancellation",
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )?;

        let scheduler_decisions_total = IntCounterVec::new(
            Opts::new(
                "scheduler_decisions_total",
                "Worker selections by policy and worker",
            ),
            &["policy", "worker"],
        )?;
        let worker_errors_total = IntCounterVec::new(
            Opts::new("worker_errors_total", "Worker failures by kind"),
            &["worker", "reason"],
        )?;
        let workers_healthy = gauge_vec(
            "gateway_workers_healthy",
            "Worker health flag (1/0)",
            &["worker"],
        )?;
        let worker_outstanding = gauge_vec(
            "gateway_worker_outstanding",
            "Outstanding requests per worker",
            &["worker"],
        )?;
        let worker_prefill_ms_per_tok = gauge_vec(
            "gateway_worker_prefill_ms_per_token",
            "Adaptive prefill cost estimate (EWMA regression intercept+rate model)",
            &["worker"],
        )?;
        let worker_decode_ms_per_tok = gauge_vec(
            "gateway_worker_decode_ms_per_token",
            "Adaptive decode cost estimate (EWMA)",
            &["worker"],
        )?;
        let worker_fixed_overhead_ms = gauge_vec(
            "gateway_worker_fixed_overhead_ms",
            "Adaptive fixed per-request overhead estimate",
            &["worker"],
        )?;
        let worker_remaining_work_seconds = gauge_vec(
            "gateway_worker_remaining_work_seconds",
            "Estimated remaining work of active dispatches",
            &["worker"],
        )?;

        let worker_state = gauge_vec(
            "inference_worker_state",
            "Worker lifecycle state, 1 on the current state 0 elsewhere",
            &["worker", "runtime", "state"],
        )?;
        let worker_state_transitions_total = IntCounterVec::new(
            Opts::new(
                "inference_worker_state_transitions_total",
                "Lifecycle transitions by from/to/reason",
            ),
            &["from", "to", "reason"],
        )?;
        let worker_functional_probe_seconds = HistogramVec::new(
            HistogramOpts::new(
                "inference_worker_functional_probe_seconds",
                "Wall time of the tiny real-generation readiness probe",
            )
            .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
            &["worker"],
        )?;
        let worker_functional_probe_failures_total = IntCounterVec::new(
            Opts::new(
                "inference_worker_functional_probe_failures_total",
                "Functional-probe failures by reason",
            ),
            &["worker", "reason"],
        )?;
        let worker_restarts_total = IntCounterVec::new(
            Opts::new(
                "inference_worker_restarts_total",
                "Completed restarts (new generation epochs begun)",
            ),
            &["worker", "reason"],
        )?;
        let worker_consecutive_failures = gauge_vec(
            "inference_worker_consecutive_failures",
            "Worst consecutive-failure streak across health dimensions",
            &["worker"],
        )?;
        let worker_generation = gauge_vec(
            "inference_worker_generation",
            "Worker incarnation epoch; increments on every completed restart",
            &["worker"],
        )?;
        let empty_completions_total = IntCounterVec::new(
            Opts::new(
                "inference_empty_completions_total",
                "HTTP-200 completions that carried zero generated tokens",
            ),
            &["worker"],
        )?;

        reg!(registry, requests_total);
        reg!(registry, requests_in_flight);
        reg!(registry, input_tokens_total);
        reg!(registry, output_tokens_total);
        reg!(registry, ttft);
        reg!(registry, gateway_processing);
        reg!(registry, admission_wait);
        reg!(registry, worker_connect);
        reg!(registry, worker_first_token);
        reg!(registry, itl);
        reg!(registry, e2e);
        reg!(registry, stream_inter_token);
        reg!(registry, stream_max_stall);
        reg!(registry, stream_completion);
        reg!(registry, requests_active);
        reg!(registry, requests_queued);
        reg!(registry, queued_input_tokens);
        reg!(registry, queued_output_tokens);
        reg!(registry, estimated_queue_seconds);
        reg!(registry, load_state);
        reg!(registry, admission_rejections_total);
        reg!(registry, stream_cancelled_total);
        reg!(registry, saved_compute_seconds);
        reg!(registry, cancel_cleanup_seconds);
        reg!(registry, scheduler_decisions_total);
        reg!(registry, worker_errors_total);
        reg!(registry, workers_healthy);
        reg!(registry, worker_outstanding);
        reg!(registry, worker_prefill_ms_per_tok);
        reg!(registry, worker_decode_ms_per_tok);
        reg!(registry, worker_fixed_overhead_ms);
        reg!(registry, worker_remaining_work_seconds);
        reg!(registry, worker_state);
        reg!(registry, worker_state_transitions_total);
        reg!(registry, worker_functional_probe_seconds);
        reg!(registry, worker_functional_probe_failures_total);
        reg!(registry, worker_restarts_total);
        reg!(registry, worker_consecutive_failures);
        reg!(registry, worker_generation);
        reg!(registry, empty_completions_total);

        Ok(Self {
            registry,
            requests_total,
            requests_in_flight,
            input_tokens_total,
            output_tokens_total,
            ttft,
            gateway_processing,
            admission_wait,
            worker_connect,
            worker_first_token,
            itl,
            e2e,
            stream_inter_token,
            stream_max_stall,
            stream_completion,
            requests_active,
            requests_queued,
            queued_input_tokens,
            queued_output_tokens,
            estimated_queue_seconds,
            load_state,
            admission_rejections_total,
            stream_cancelled_total,
            saved_compute_seconds,
            cancel_cleanup_seconds,
            scheduler_decisions_total,
            worker_errors_total,
            workers_healthy,
            worker_outstanding,
            worker_prefill_ms_per_tok,
            worker_decode_ms_per_tok,
            worker_fixed_overhead_ms,
            worker_remaining_work_seconds,
            worker_state,
            worker_state_transitions_total,
            worker_functional_probe_seconds,
            worker_functional_probe_failures_total,
            worker_restarts_total,
            worker_consecutive_failures,
            worker_generation,
            empty_completions_total,
        })
    }

    /// Render the registry in Prometheus text format.
    pub fn encode(&self) -> anyhow::Result<Vec<u8>> {
        let encoder = TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&self.registry.gather(), &mut buf)?;
        Ok(buf)
    }

    /// Feed lifecycle transitions into the metric surface: the edge audit
    /// counter, restart accounting on epoch boundaries, and a warn log on
    /// the edges an on-call cares about.
    pub fn record_transitions(&self, worker: &WorkerHandle, transitions: &[inference_lab_worker_lifecycle::Transition]) {
        for t in transitions {
            self.worker_state_transitions_total
                .with_label_values(&[
                    t.from.as_str(),
                    t.to.as_str(),
                    t.reason.map(|r| r.as_str()).unwrap_or("none"),
                ])
                .inc();
            if t.to == WorkerState::Unhealthy {
                tracing::warn!(
                    worker = %worker.id,
                    reason = ?t.reason,
                    "worker quarantined: UNHEALTHY, no new traffic"
                );
            }
            // RECOVERING → WARMING means the restarted incarnation answered:
            // the restart is complete, a new generation epoch begins.
            if t.from == WorkerState::Recovering && t.to == WorkerState::Warming {
                let reason = worker
                    .condition()
                    .reason
                    .map(|r| r.as_str())
                    .unwrap_or("unknown");
                self.worker_restarts_total
                    .with_label_values(&[worker.id.as_str(), reason])
                    .inc();
                self.worker_generation
                    .with_label_values(&[worker.id.as_str()])
                    .set(worker.generation() as i64);
                tracing::info!(
                    worker = %worker.id,
                    generation = worker.generation(),
                    "new worker generation warming up"
                );
            }
        }
    }
}
