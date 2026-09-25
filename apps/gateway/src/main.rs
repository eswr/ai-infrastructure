//! gateway — OpenAI-compatible inference edge in front of the lab's workers.
//!
//! Request path (phase 4D):
//!
//! ```text
//! request → estimate tokens → admission control (budget by estimated work;
//!           queue with token/seconds accounting, or 429 + Retry-After when
//!           overloaded) → scheduler (LOR / adaptive ECT / prefix affinity,
//!           lifecycle-aware, re-route on connect failure) → worker →
//!           instrumented stream (TTFT decomposition, ITL, token counts,
//!           first-token + mid-stream stall watchdogs, OTel spans) → client
//! ```
//!
//! Worker lifecycle: every worker walks STARTING → WARMING → READY before
//! it receives traffic, driven by a `/health` transport poll AND a tiny
//! real-generation functional probe (`/health` 200 alone is never READY —
//! the phase-4A mlx-lm silent wedge answered /health for hours). Failures
//! escalate by reason class (immediate quarantine for probe timeouts and
//! process death; 3-strike accumulation for connect/empty/stall), recovery
//! requires repeated probe passes, and each restart begins a new generation
//! epoch. Admin surface: `POST /admin/workers/{id}/drain`,
//! `POST /admin/workers/{id}/probe`.
//!
//! Every request leaves: a distributed trace (OTLP → collector → Tempo),
//! Prometheus metrics (`/metrics`), and log lines correlated by trace id
//! (OTLP → Loki). Client disconnects cancel upstream generation and release
//! admission capacity immediately — including requests still in the queue
//! (via HTTP/2 stream reset; on HTTP/1.1 a queued-request disconnect is
//! only observable once the response is written).
//!
//! Config (env):
//! * `GATEWAY_ADDR`       listen address            (default 0.0.0.0:8080)
//! * `WORKERS` — `name=url[,name=url...]`, optional `#model` suffix
//!   (default mlx=http://127.0.0.1:8082)
//! * `ROUTE_POLICY`       lor | ect | prefix-affinity (default lor)
//! * `ADMIT_MAX_ACTIVE_INPUT_TOKENS`  prefill budget, 0 = off (default 0)
//! * `ADMIT_MAX_ACTIVE_OUTPUT_TOKENS` decode budget, 0 = off (default 0)
//! * `ADMIT_MAX_WORK_MS`  per-request estimated-work cap, 0 = off
//! * `ADMIT_DEGRADED_QUEUE_S` / `ADMIT_OVERLOADED_QUEUE_S` (5 / 30)
//! * `FIRST_TOKEN_TIMEOUT_S` abort streams with no first token (0 = off)
//! * `STREAM_STALL_TIMEOUT_S` abort streams that stop producing tokens
//!   mid-generation (0 = off; the 193 s mlx-lm stall detector)
//! * `HEALTH_INTERVAL_S`  worker /health transport-poll interval (default 5)
//! * `FUNCTIONAL_PROBE_INTERVAL_S`        probe cadence for non-READY
//!   workers (default 5) — the restart-to-READY latency
//! * `FUNCTIONAL_PROBE_INTERVAL_READY_S`  steady-state probe cadence for
//!   READY workers (default 30, 0 = off — the silent-wedge detector; set 0
//!   for measured benchmark runs)
//! * `FUNCTIONAL_PROBE_TIMEOUT_S` (10) / `FUNCTIONAL_PROBE_MAX_TOKENS` (2)
//! * `PROFILE_FIXED_MS` / `PROFILE_PREFILL_MS_PER_TOK` /
//!   `PROFILE_DECODE_MS_PER_TOK`  service-model seeds (phase-2 defaults:
//!   750 / 5.35 / 76)
//! * `OTEL_EXPORTER_OTLP_ENDPOINT`  enables OTLP export (e.g. http://127.0.0.1:4318)

mod admission;
mod lifecycle;
mod metrics;
mod proxy;
mod telemetry;

use admission::{
    AdmissionController, AdmissionDecision, AdmissionPolicy, RequestSize, TokenEstimator,
};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use inference_lab_protocol::ChatCompletionRequest;
use inference_lab_scheduler::{
    RequestShape, Router as WorkerRouter, SelectOutcome, ServiceDefaults, WorkerHandle,
};
use inference_lab_worker_lifecycle::{Dimension, FailureReason, WorkerState};
use metrics::Metrics;
use opentelemetry::trace::{SpanKind, TraceContextExt};
use opentelemetry::KeyValue;
use proxy::{InFlightGuard, InstrumentedStream, RequestTimeline, StreamCtx, TraceSpans};
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct AppState {
    router: Arc<WorkerRouter>,
    client: reqwest::Client,
    metrics: Arc<Metrics>,
    admission: Arc<AdmissionController>,
    estimator: Arc<TokenEstimator>,
    first_token_timeout: Option<Duration>,
    stream_stall_timeout: Option<Duration>,
    lifecycle_cfg: lifecycle::LifecycleConfig,
}

fn error_json(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({"error": {"message": msg, "type": "gateway_error"}});
    (status, Json(body)).into_response()
}

/// Hash the leading prompt content — the part a prefix cache would key on.
fn prefix_hash(req: &ChatCompletionRequest) -> u64 {
    let mut h = DefaultHasher::new();
    for m in req.messages.iter().take(2) {
        let len = m.content.len().min(2048);
        h.write(&m.content.as_bytes()[..len]);
    }
    h.finish()
}

/// Best-case (fastest healthy worker) estimated service time, used by
/// admission for work accounting.
fn est_work_ms(st: &AppState, est_in: u64, out_tok: u64) -> f64 {
    let healthy = st.router.healthy_workers();
    let candidates: &[Arc<WorkerHandle>] = if healthy.is_empty() {
        st.router.workers()
    } else {
        &healthy
    };
    candidates
        .iter()
        .map(|w| w.service_time_ms(est_in, out_tok, false))
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Chat completions: the full path
// ---------------------------------------------------------------------------

/// Ends queued-request bookkeeping when the handler future is dropped
/// (client disconnect) — the entry leaves the queue and the estimated work
/// it held is reported as saved.
struct QueuedGuard {
    ctrl: Arc<AdmissionController>,
    metrics: Arc<Metrics>,
    entry: Arc<admission::Queued>,
}

impl Drop for QueuedGuard {
    fn drop(&mut self) {
        let saved = self.ctrl.cancel_queued(&self.entry);
        if saved > 0.0 {
            self.metrics.stream_cancelled_total.with_label_values(&["queued"]).inc();
            self.metrics.requests_total.with_label_values(&["queued_cancelled"]).inc();
            self.metrics.saved_compute_seconds.inc_by(saved / 1000.0);
        }
    }
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    st.metrics.requests_in_flight.inc();
    let in_flight = InFlightGuard(st.metrics.clone());

    let stream_mode = req.stream;
    let model = req.model.clone();
    let est_in = st.estimator.estimate(&req);
    let out_tok = req.max_tokens as u64;
    let hash = prefix_hash(&req);

    let root = telemetry::root_span(
        "POST /v1/chat/completions",
        SpanKind::Server,
        vec![
            KeyValue::new("gen_ai.operation.name", "chat"),
            KeyValue::new("gen_ai.request.model", model.clone()),
            KeyValue::new("request.input_tokens_estimated", est_in as i64),
            KeyValue::new("request.output_tokens_requested", out_tok as i64),
            KeyValue::new("scheduler.policy", st.router.policy().as_str()),
            KeyValue::new("cache.prefix_hash", format!("{hash:016x}")),
        ],
    );
    telemetry::instant_span(
        &root,
        "tokens.estimate",
        vec![
            KeyValue::new("request.input_tokens_estimated", est_in as i64),
            KeyValue::new("inference.token_calib_ratio", st.estimator.ratio()),
        ],
    );

    // ---- admission ---------------------------------------------------------
    let size = RequestSize {
        input_tokens: est_in,
        output_tokens: out_tok,
        est_work_ms: est_work_ms(&st, est_in, out_tok),
    };

    let admission_span =
        telemetry::child_span(&root, "admission", SpanKind::Internal, vec![]);
    let decision = st.admission.decide(size);
    let snap = st.admission.snapshot();
    root.span().set_attribute(KeyValue::new("scheduler.queue_depth", snap.queued_requests as i64));
    root.span().set_attribute(KeyValue::new("scheduler.active_requests", snap.active_requests as i64));
    root.span().set_attribute(KeyValue::new("inference.load_state", snap.load_state.as_str()));
    admission_span.span().end();

    let (permit, mut timeline) = match decision {
        AdmissionDecision::Reject { reason, retry_after_s } => {
            st.metrics.admission_rejections_total.with_label_values(&[reason]).inc();
            st.metrics.requests_total.with_label_values(&["rejected"]).inc();
            root.span().set_attribute(KeyValue::new("inference.admission.state", "rejected"));
            root.span().set_attribute(KeyValue::new("inference.admission.reject_reason", reason));
            root.span().end();
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, HeaderValue::from(retry_after_s))],
                Json(serde_json::json!({
                    "error": {"message": format!("gateway overloaded ({reason})"),
                              "type": "gateway_admission",
                              "retry_after_s": retry_after_s}
                })),
            )
                .into_response();
        }
        AdmissionDecision::Admit(p) => {
            root.span().set_attribute(KeyValue::new("inference.admission.state", "immediate"));
            (p, RequestTimeline::new(est_in, out_tok))
        }
        AdmissionDecision::Queue(entry) => {
            root.span().set_attribute(KeyValue::new("inference.admission.state", "queued"));
            let mut tl = RequestTimeline::new(est_in, out_tok);
            tl.queued_at = Some(Instant::now());
            let wait_span = telemetry::child_span(
                &root,
                "queue.wait",
                SpanKind::Internal,
                vec![KeyValue::new(
                    "inference.queue.depth_at_enqueue",
                    AdmissionController::queue_depth_at_enqueue(&entry) as i64,
                )],
            );
            let _guard = QueuedGuard {
                ctrl: st.admission.clone(),
                metrics: st.metrics.clone(),
                entry: entry.clone(),
            };
            match st.admission.wait_for_grant(&entry).await {
                Some(p) => {
                    tl.granted_at = Some(Instant::now());
                    wait_span.span().set_attribute(KeyValue::new(
                        "inference.admission_wait_s",
                        st.admission.queue_wait(&entry).as_secs_f64(),
                    ));
                    wait_span.span().end();
                    (p, tl)
                }
                None => {
                    // Defensive: the only canceller is the guard above, which
                    // has already run its bookkeeping on drop.
                    wait_span.span().end();
                    root.span().set_attribute(KeyValue::new("inference.outcome", "queued_cancelled"));
                    root.span().end();
                    return (
                        StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                        Json(serde_json::json!({
                            "error": {"message": "client disconnected while queued",
                                      "type": "gateway_admission"}
                        })),
                    )
                        .into_response();
                }
            }
        }
    };

    // ---- scheduling --------------------------------------------------------
    let shape =
        RequestShape { prefix_hash: hash, input_tokens: est_in, output_tokens: out_tok };
    let sel_span = telemetry::child_span(&root, "scheduler.select", SpanKind::Internal, vec![]);
    let first_choice = match st.router.select(shape) {
        SelectOutcome::Worker(w) => w,
        SelectOutcome::AllUnhealthy => {
            sel_span.span().end();
            st.metrics.requests_total.with_label_values(&["no_worker"]).inc();
            root.span().set_attribute(KeyValue::new("inference.outcome", "no_worker"));
            root.span().end();
            return error_json(StatusCode::SERVICE_UNAVAILABLE, "no healthy workers");
        }
    };
    sel_span.span().end();

    // ---- dispatch (one re-route attempt on connect failure) -----------------
    let mut worker = first_choice;

    let (resp, est_total, work_start, worker_span) = loop {
        st.metrics
            .scheduler_decisions_total
            .with_label_values(&[st.router.policy().as_str(), &worker.id])
            .inc();
        worker.inc_outstanding();
        timeline.dispatched_at = Some(Instant::now());

        let est_total = worker.service_time_ms(est_in, out_tok, worker.has_prefix(hash));
        let work_start = worker.begin_work(est_total);

        let url = format!("{}/v1/chat/completions", worker.endpoint.trim_end_matches('/'));
        let mut out_headers = HeaderMap::new();
        telemetry::inject_traceparent(&root, &mut out_headers);

        let worker_span = telemetry::child_span(
            &root,
            "worker.request",
            SpanKind::Client,
            vec![
                KeyValue::new("http.request.method", "POST"),
                KeyValue::new("server.address", worker.endpoint.clone()),
                KeyValue::new("http.url", url.clone()),
                KeyValue::new("inference.worker.id", worker.id.clone()),
                KeyValue::new("inference.runtime", worker.runtime.clone()),
                KeyValue::new("inference.estimated_work_ms", est_total),
            ],
        );
        root.span().set_attribute(KeyValue::new("inference.worker.id", worker.id.clone()));
        root.span().set_attribute(KeyValue::new("inference.runtime", worker.runtime.clone()));
        let connect_span = telemetry::child_span(&root, "worker.connect", SpanKind::Internal, vec![]);

        match st.client.post(&url).headers(out_headers).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                connect_span.span().set_attribute(KeyValue::new(
                    "inference.connect_ms",
                    work_start.elapsed().as_secs_f64() * 1000.0,
                ));
                connect_span.span().end();
                break (resp, est_total, work_start, worker_span);
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                st.metrics.worker_errors_total.with_label_values(&[&worker.id, "http"]).inc();
                // The worker answered at transport level but refused
                // service — accumulating-class failure (3 strikes
                // quarantine; one blip degrades without flapping).
                let transitions =
                    worker.record_failure(Dimension::Transport, FailureReason::ConnectFailed);
                st.metrics.record_transitions(&worker, &transitions);
                worker.end_work(work_start, est_total);
                let drained = worker.dec_outstanding();
                st.metrics.record_transitions(&worker, &drained);
                worker_span.span().set_attribute(KeyValue::new(
                    "inference.outcome",
                    "upstream_http_error",
                ));
                worker_span.span().set_attribute(KeyValue::new(
                    "http.response.status_code",
                    status.as_u16() as i64,
                ));
                worker_span.span().end();
                connect_span.span().end();
                st.metrics.requests_total.with_label_values(&["upstream_http_error"]).inc();
                root.span().set_attribute(KeyValue::new("inference.outcome", "upstream_http_error"));
                root.span().end();
                return error_json(
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    &format!(
                        "upstream {}: {}",
                        worker.id,
                        body.chars().take(300).collect::<String>()
                    ),
                );
            }
            Err(e) => {
                st.metrics.worker_errors_total.with_label_values(&[&worker.id, "connect"]).inc();
                let transitions =
                    worker.record_failure(Dimension::Transport, FailureReason::ConnectFailed);
                st.metrics.record_transitions(&worker, &transitions);
                worker.end_work(work_start, est_total);
                let drained = worker.dec_outstanding();
                st.metrics.record_transitions(&worker, &drained);
                worker_span.span().set_attribute(KeyValue::new("inference.outcome", "upstream_error"));
                worker_span.span().set_attribute(KeyValue::new("inference.error", format!("{e}")));
                worker_span.span().end();
                connect_span.span().end();
                // Re-route once if another healthy worker exists.
                if let SelectOutcome::Worker(w2) = st.router.select(shape) {
                    if w2.id != worker.id {
                        tracing::warn!(from = %worker.id, to = %w2.id, "connect failure, re-routing");
                        worker = w2;
                        continue;
                    }
                }
                st.metrics.requests_total.with_label_values(&["upstream_error"]).inc();
                root.span().set_attribute(KeyValue::new("inference.outcome", "upstream_error"));
                root.span().end();
                return error_json(
                    StatusCode::BAD_GATEWAY,
                    &format!("upstream {}: {e}", worker.id),
                );
            }
        }
    };
    timeline.headers_at = Some(Instant::now());

    let first_token_span = telemetry::child_span(
        &root,
        "worker.waiting_for_first_token",
        SpanKind::Internal,
        vec![KeyValue::new(
            "inference.note",
            "runtime queue + tokenization + prefill; NOT pure prefill",
        )],
    );

    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or(HeaderValue::from_static(if stream_mode {
            "text/event-stream"
        } else {
            "application/json"
        }));

    // A runtime that answers JSON to a streaming request has declined to
    // stream (vllm-metal 0.1.0 does). Treat the body as non-streaming so
    // usage parses, and forward the honest content-type to the client.
    let upstream_streams = content_type
        .to_str()
        .map(|c| c.starts_with("text/event-stream"))
        .unwrap_or(!stream_mode);
    let stream_mode = stream_mode && upstream_streams;

    let ctx = StreamCtx {
        metrics: st.metrics.clone(),
        worker: worker.clone(),
        estimator: st.estimator.clone(),
        _permit: permit,
        _in_flight: in_flight,
        timeline,
        spans: TraceSpans {
            root: Some(root),
            worker_request: Some(worker_span),
            first_token: Some(first_token_span),
            streaming: None,
        },
        first_token_deadline: st.first_token_timeout.map(|t| Instant::now() + t),
        stream_stall_timeout: st.stream_stall_timeout,
        stream_mode,
        work_estimate: (work_start, est_total),
        est_total_ms: est_total,
    };

    let stream = InstrumentedStream::new(resp.bytes_stream(), ctx);
    match Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from_stream(stream))
    {
        Ok(r) => r,
        Err(e) => error_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("build response: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Supporting endpoints & tasks
// ---------------------------------------------------------------------------

async fn list_models(State(st): State<Arc<AppState>>) -> Response {
    let seen: std::collections::BTreeSet<String> =
        st.router.workers().iter().map(|w| w.model.clone()).collect();
    let data: Vec<serde_json::Value> = seen
        .into_iter()
        .map(|m| serde_json::json!({"id": m, "object": "model"}))
        .collect();
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

async fn healthz() -> Response {
    Json(serde_json::json!({"status": "ok"})).into_response()
}

/// Rich operational snapshot for tests and humans.
async fn status(State(st): State<Arc<AppState>>) -> Response {
    let snap = st.admission.snapshot();
    let workers: Vec<serde_json::Value> = st
        .router
        .workers()
        .iter()
        .map(|w| {
            let cond = w.condition();
            serde_json::json!({
                "id": w.id,
                "endpoint": w.endpoint,
                "runtime": w.runtime,
                "model": w.model,
                // "healthy" = receives traffic (READY or DEGRADED).
                "healthy": w.is_healthy(),
                "state": cond.state.as_str(),
                "state_reason": cond.reason.map(|r| r.as_str()),
                "state_since_s": cond.since.elapsed().as_secs_f64(),
                "generation": cond.generation,
                "consecutive_failures": cond.consecutive_failures,
                "transitions_total": w.total_transitions(),
                "dimensions": {
                    "process": cond.dimension(inference_lab_worker_lifecycle::Dimension::Process).as_str(),
                    "transport": cond.dimension(inference_lab_worker_lifecycle::Dimension::Transport).as_str(),
                    "functional": cond.dimension(inference_lab_worker_lifecycle::Dimension::Functional).as_str(),
                    "semantic": cond.dimension(inference_lab_worker_lifecycle::Dimension::Semantic).as_str(),
                },
                "last_success_age_s": cond
                    .last_success
                    .map(|t| t.elapsed().as_secs_f64()),
                "outstanding": w.outstanding(),
                "prefill_ms_per_token": w.prefill_ms_per_tok(),
                "fixed_overhead_ms": w.fixed_overhead_ms(),
                "decode_ms_per_token": w.decode_ms_per_tok(),
                "remaining_work_s": w.remaining_work_ms() / 1000.0,
            })
        })
        .collect();
    Json(serde_json::json!({
        "admission": {
            "enabled": snap.enabled,
            "active_requests": snap.active_requests,
            "active_input_tokens": snap.active_input_tokens,
            "active_output_tokens": snap.active_output_tokens,
            "queued_requests": snap.queued_requests,
            "queued_input_tokens": snap.queued_input_tokens,
            "queued_output_tokens": snap.queued_output_tokens,
            "estimated_queue_seconds": snap.estimated_queue_seconds,
            "load_state": snap.load_state.as_str(),
        },
        "policy": st.router.policy().as_str(),
        "workers": workers,
    }))
    .into_response()
}

/// Stop assigning new work to a worker; in-flight requests finish. The
/// worker's lifecycle moves READY/DEGRADED → DRAINING, and once the last
/// outstanding request completes, DRAINING → STOPPED.
async fn drain_worker(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(w) = st.router.workers().iter().find(|w| w.id == id) else {
        return error_json(StatusCode::NOT_FOUND, &format!("unknown worker {id}"));
    };
    let transitions = w.drain();
    st.metrics.record_transitions(w, &transitions);
    let cond = w.condition();
    Json(serde_json::json!({
        "worker": w.id,
        "state": cond.state.as_str(),
        "generation": cond.generation,
        "outstanding": w.outstanding(),
    }))
    .into_response()
}

/// Run one functional probe immediately (the readiness contract: a tiny
/// real generation). Useful after config changes and in tests.
async fn probe_worker(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(w) = st.router.workers().iter().find(|w| w.id == id) else {
        return error_json(StatusCode::NOT_FOUND, &format!("unknown worker {id}"));
    };
    let (obs, dur) = lifecycle::run_functional_probe(&st.client, w, &st.lifecycle_cfg).await;
    st.metrics
        .worker_functional_probe_seconds
        .with_label_values(&[w.id.as_str()])
        .observe(dur.as_secs_f64());
    let transitions = w.observe(&obs);
    st.metrics.record_transitions(w, &transitions);
    let cond = w.condition();
    Json(serde_json::json!({
        "worker": w.id,
        "probe": {
            "ok": obs.ok,
            "dimension": obs.dimension.as_str(),
            "reason": obs.reason.map(|r| r.as_str()),
            "detail": obs.detail,
            "elapsed_s": dur.as_secs_f64(),
        },
        "state": cond.state.as_str(),
        "generation": cond.generation,
    }))
    .into_response()
}

async fn metrics_endpoint(State(st): State<Arc<AppState>>) -> Response {
    match st.metrics.encode() {
        Ok(buf) => (
            [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"))],
            buf,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn root() -> Response {
    Json(serde_json::json!({
        "service": "inference-lab gateway",
        "endpoints": [
            "/v1/chat/completions", "/v1/models", "/healthz", "/status", "/metrics",
            "/admin/workers/{id}/drain", "/admin/workers/{id}/probe"
        ],
    }))
    .into_response()
}

/// Mirror admission + worker state into gauges every 500 ms so Prometheus
/// (5 s scrape) always sees fresh queue/work state.
async fn gauge_sampler(st: Arc<AppState>) {
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        let s = st.admission.snapshot();
        let m = &st.metrics;
        m.requests_active.set(s.active_requests as i64);
        m.requests_queued.set(s.queued_requests as i64);
        m.queued_input_tokens.set(s.queued_input_tokens as i64);
        m.queued_output_tokens.set(s.queued_output_tokens as i64);
        m.estimated_queue_seconds.set(s.estimated_queue_seconds);
        m.load_state.set(s.load_state as i64);
        for w in st.router.workers() {
            let id = w.id.as_str();
            let cond = w.condition();
            m.workers_healthy.with_label_values(&[id]).set(w.is_healthy() as i64);
            m.worker_outstanding.with_label_values(&[id]).set(w.outstanding() as i64);
            m.worker_prefill_ms_per_tok
                .with_label_values(&[id])
                .set(w.prefill_ms_per_tok() as i64);
            m.worker_decode_ms_per_tok
                .with_label_values(&[id])
                .set(w.decode_ms_per_tok() as i64);
            m.worker_fixed_overhead_ms
                .with_label_values(&[id])
                .set(w.fixed_overhead_ms() as i64);
            m.worker_remaining_work_seconds
                .with_label_values(&[id])
                .set((w.remaining_work_ms() / 1000.0) as i64);
            // Lifecycle: one-hot state vector per worker + failure/epoch
            // gauges. The state gauge is the Grafana timeline's source.
            for s in WorkerState::all() {
                m.worker_state
                    .with_label_values(&[id, &w.runtime, s.as_str()])
                    .set((cond.state == s) as i64);
            }
            m.worker_consecutive_failures
                .with_label_values(&[id])
                .set(cond.consecutive_failures as i64);
            m.worker_generation.with_label_values(&[id]).set(cond.generation as i64);
        }
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn parse_workers(spec: &str) -> Vec<(String, String, String)> {
    // spec: "name=url[,name=url...]" with optional "#model" suffix:
    // "mlx=http://127.0.0.1:8082#mlx-community/Qwen3.5-9B-4bit"
    spec.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|pair| {
            let (name, rest) = pair.split_once('=').unwrap_or(("worker", pair));
            let (url, model) = match rest.split_once('#') {
                Some((u, m)) => (u, m.to_string()),
                None => (rest, "default".to_string()),
            };
            (name.trim().to_string(), url.trim().to_string(), model)
        })
        .collect()
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let telemetry_guard = telemetry::init(env!("CARGO_PKG_VERSION"))?;

    let addr = std::env::var("GATEWAY_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let workers_spec =
        std::env::var("WORKERS").unwrap_or_else(|_| "mlx=http://127.0.0.1:8082".into());
    let policy = inference_lab_scheduler::RoutingPolicy::parse(
        &std::env::var("ROUTE_POLICY").unwrap_or_else(|_| "lor".into()),
    )
    .unwrap_or_default();

    let service_defaults = ServiceDefaults {
        fixed_ms: env_f64("PROFILE_FIXED_MS", 750.0),
        prefill_ms_per_tok: env_f64("PROFILE_PREFILL_MS_PER_TOK", 5.35),
        decode_ms_per_tok: env_f64("PROFILE_DECODE_MS_PER_TOK", 76.0),
    };

    let handles = parse_workers(&workers_spec)
        .into_iter()
        .map(|(name, url, model)| {
            let runtime = name.split('-').next().unwrap_or("unknown").to_string();
            Arc::new(WorkerHandle::with_defaults(name, url, model, runtime, service_defaults))
        })
        .collect::<Vec<_>>();

    for w in &handles {
        tracing::info!(worker = %w.id, endpoint = %w.endpoint, model = %w.model, "registered worker");
    }
    let admission_policy = AdmissionPolicy::from_env();
    tracing::info!(
        ?policy,
        admission_enabled = admission_policy.enabled(),
        max_active_input_tokens = admission_policy.max_active_input_tokens,
        max_active_output_tokens = admission_policy.max_active_output_tokens,
        max_estimated_work_ms = admission_policy.max_estimated_work_ms,
        "routing policy + admission"
    );

    let router = Arc::new(WorkerRouter::new(handles, policy)?);
    let admission = AdmissionController::new(admission_policy);
    let first_token_timeout = env_f64("FIRST_TOKEN_TIMEOUT_S", 0.0);
    let stream_stall_timeout = env_f64("STREAM_STALL_TIMEOUT_S", 0.0);
    let lifecycle_cfg = lifecycle::LifecycleConfig::from_env();
    let state = Arc::new(AppState {
        router: router.clone(),
        client: reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()?,
        metrics: Arc::new(Metrics::new()?),
        admission: admission.clone(),
        estimator: Arc::new(TokenEstimator::new()),
        first_token_timeout: (first_token_timeout > 0.0)
            .then(|| Duration::from_secs_f64(first_token_timeout)),
        stream_stall_timeout: (stream_stall_timeout > 0.0)
            .then(|| Duration::from_secs_f64(stream_stall_timeout)),
        lifecycle_cfg: lifecycle_cfg.clone(),
    });

    let health_interval = Duration::from_secs_f64(env_f64("HEALTH_INTERVAL_S", 5.0).max(0.5));
    tokio::spawn(gauge_sampler(state.clone()));
    tokio::spawn(lifecycle::transport_poller(state.clone(), health_interval));
    tokio::spawn(lifecycle::functional_prober(state.clone(), lifecycle_cfg));

    let app = Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/status", get(status))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/admin/workers/{id}/drain", post(drain_worker))
        .route("/admin/workers/{id}/probe", post(probe_worker))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("gateway listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;

    telemetry_guard.shutdown();
    Ok(())
}
