//! Instrumented streaming proxy.
//!
//! Forwards upstream bytes to the client **unchanged** while feeding every
//! byte through the SSE parser to timestamp tokens at the gateway: first
//! token, inter-token gaps, token counts, and the final usage chunk. From
//! those observations it (a) decomposes TTFT into its components,
//! (b) updates the worker's adaptive service model, (c) fills the
//! Prometheus histograms and (d) closes out the request's OTel spans.
//!
//! Lifecycle: the stream ends either naturally (upstream `None`), with an
//! upstream error, with a first-token timeout, or by being dropped — which
//! is exactly what happens when the client disconnects mid-stream (hyper
//! drops the body, our `Drop` runs, the upstream reqwest stream drops too,
//! closing the worker connection). All four paths converge on
//! [`StreamCtx::finish`], which is idempotent and owns capacity release via
//! the admission permit's `Drop`.

use crate::admission::{AdmissionPermit, TokenEstimator};
use crate::metrics::Metrics;
use futures_util::{Stream, StreamExt};
use inference_lab_protocol::{chunk_text, ChatCompletionChunk, Usage};
use inference_lab_scheduler::{DispatchTicket, WorkerHandle};
use inference_lab_streaming::SseParser;
use inference_lab_worker_lifecycle::{Dimension, FailureReason};
use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
use std::future::Future as _;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

/// Cap on JSON body buffering (non-streaming) for usage extraction.
const MAX_JSON_BUFFER: usize = 256 * 1024;

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// Gateway-side timestamps for one request, from handler entry to stream
/// end. All the TTFT components the lab reports derive from these.
#[derive(Debug)]
pub struct RequestTimeline {
    pub t0: Instant,
    pub est_input_tokens: u64,
    pub requested_output_tokens: u64,
    pub queued_at: Option<Instant>,
    pub granted_at: Option<Instant>,
    pub dispatched_at: Option<Instant>,
    pub headers_at: Option<Instant>,
    pub first_token_at: Option<Instant>,
    pub ended_at: Option<Instant>,
    /// Tokens counted at the gateway (streamed content/reasoning chunks).
    pub output_tokens: u64,
    pub prompt_tokens_reported: Option<u64>,
    pub completion_tokens_reported: Option<u64>,
    /// Server-reported reusable-prefix tokens (llama.cpp
    /// `timings.cache_n` on the final chunk / JSON body) — the realized
    /// counterpart of the router's match estimate.
    pub reused_tokens_reported: Option<u64>,
}

impl RequestTimeline {
    pub fn new(est_input_tokens: u64, requested_output_tokens: u64) -> Self {
        Self {
            t0: Instant::now(),
            est_input_tokens,
            requested_output_tokens,
            queued_at: None,
            granted_at: None,
            dispatched_at: None,
            headers_at: None,
            first_token_at: None,
            ended_at: None,
            output_tokens: 0,
            prompt_tokens_reported: None,
            completion_tokens_reported: None,
            reused_tokens_reported: None,
        }
    }

    /// Time in the admission queue (0 when admitted immediately).
    pub fn admission_wait(&self) -> Duration {
        match (self.queued_at, self.granted_at) {
            (Some(q), Some(g)) => g.duration_since(q),
            _ => Duration::ZERO,
        }
    }

    /// Gateway-side processing before dispatch: estimate + admission
    /// decision + scheduling (everything except the queue wait).
    pub fn gateway_processing(&self) -> Option<Duration> {
        self.dispatched_at
            .map(|d| d.duration_since(self.t0) - self.admission_wait())
    }

    /// Upstream request send → response headers.
    pub fn worker_connect(&self) -> Option<Duration> {
        match (self.dispatched_at, self.headers_at) {
            (Some(d), Some(h)) => Some(h.duration_since(d)),
            _ => None,
        }
    }

    /// Response headers → first token. Deliberately NOT called "prefill":
    /// it also contains the runtime's queueing, tokenization and scheduling.
    pub fn worker_first_token(&self) -> Option<Duration> {
        match (self.headers_at, self.first_token_at) {
            (Some(h), Some(f)) => Some(f.duration_since(h)),
            _ => None,
        }
    }

    pub fn ttft(&self) -> Option<Duration> {
        self.first_token_at.map(|f| f.duration_since(self.t0))
    }

    pub fn decode_phase(&self) -> Option<Duration> {
        match (self.first_token_at, self.ended_at) {
            (Some(f), Some(e)) => Some(e.duration_since(f)),
            _ => None,
        }
    }

    pub fn e2e(&self) -> Option<Duration> {
        self.ended_at.map(|e| e.duration_since(self.t0))
    }
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

/// Spans owned by the stream (created during dispatch, closed at finish).
/// The root span is included so the trace stays open until the last byte.
pub struct TraceSpans {
    pub root: Option<crate::telemetry::SpanHandle>,
    pub worker_request: Option<crate::telemetry::SpanHandle>,
    pub first_token: Option<crate::telemetry::SpanHandle>,
    pub streaming: Option<crate::telemetry::SpanHandle>,
}

// ---------------------------------------------------------------------------
// Terminal outcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Stream ended normally.
    Completed,
    /// Body dropped before upstream finished (client disconnect).
    ClientCancelled,
    /// Upstream body errored mid-stream.
    UpstreamError,
    /// FIRST_TOKEN_TIMEOUT_S elapsed without a token.
    FirstTokenTimeout,
    /// Stream produced tokens, then went silent past
    /// STREAM_STALL_TIMEOUT_S (the 193 s mlx-lm stall class).
    StreamStalled,
}

impl Outcome {
    fn metric_label(&self) -> &'static str {
        match self {
            Self::Completed => "ok",
            Self::ClientCancelled => "client_cancelled",
            Self::UpstreamError => "upstream_error",
            Self::FirstTokenTimeout => "first_token_timeout",
            Self::StreamStalled => "stream_stalled",
        }
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Everything the stream needs when it finishes. Constructed during
/// dispatch in the handler; the handler keeps none of it.
pub struct StreamCtx {
    pub metrics: Arc<Metrics>,
    pub worker: Arc<WorkerHandle>,
    pub estimator: Arc<TokenEstimator>,
    /// Holds active admission capacity until the last byte is out.
    pub _permit: AdmissionPermit,
    /// Decrements `inference_requests_in_flight` when dropped.
    pub _in_flight: InFlightGuard,
    pub timeline: RequestTimeline,
    pub spans: TraceSpans,
    /// The prompt-prefix fingerprint the router keyed its cache model on.
    pub prefix_hash: u64,
    /// The chosen slot's estimated TTFT at decision time (ms); the other
    /// half of the prediction-error measurement in `finish`.
    pub estimated_ttft_ms: Option<f64>,
    pub first_token_deadline: Option<Instant>,
    /// Mid-stream stall watchdog: abort when no token arrives for this long
    /// after generation started (0/off = None).
    pub stream_stall_timeout: Option<Duration>,
    pub stream_mode: bool,
    /// The dispatch reservation registered with `worker.begin_work` —
    /// passed back to `end_work` and used to attribute service-model
    /// observations to the right dispatch.
    pub work_estimate: DispatchTicket,
    /// Total estimated service time for saved-compute accounting.
    pub est_total_ms: f64,
}

pub struct InFlightGuard(pub Arc<Metrics>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.requests_in_flight.dec();
    }
}

pub struct InstrumentedStream<S> {
    inner: S,
    ctx: StreamCtx,
    parser: SseParser,
    /// Non-streaming body accumulation for usage extraction (bounded).
    json_buf: Vec<u8>,
    last_token_at: Option<Instant>,
    /// Sum and max of this stream's inter-token gaps: the per-stream
    /// fairness view (mean ITL + worst stall) pooled ITL would wash out.
    gap_sum_s: f64,
    gap_max_s: f64,
    finished: bool,
    /// Arms the first-token watchdog: a real timer future so a *silent*
    /// upstream still wakes this stream's poll at the deadline (polling is
    /// otherwise driven purely by upstream data).
    watchdog: Option<Pin<Box<tokio::time::Sleep>>>,
    /// Same trick, mid-stream: re-armed on every token, fires when the
    /// upstream stops producing. The mlx-lm 193 s mid-stream stall is the
    /// motivating signature.
    stall_watchdog: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl<S> InstrumentedStream<S> {
    pub fn new(inner: S, ctx: StreamCtx) -> Self {
        let watchdog = ctx
            .first_token_deadline
            .map(|d| Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(d))));
        Self {
            inner,
            ctx,
            parser: SseParser::new(),
            json_buf: Vec::new(),
            last_token_at: None,
            gap_sum_s: 0.0,
            gap_max_s: 0.0,
            finished: false,
            watchdog,
            stall_watchdog: None,
        }
    }

    /// Observe a raw upstream chunk: parse (streaming) or buffer
    /// (non-streaming), timestamp tokens, never modify the bytes.
    fn observe_chunk(&mut self, bytes: &[u8]) {
        if !self.ctx.stream_mode {
            if self.ctx.timeline.first_token_at.is_none() {
                // Non-streaming: the first body byte is our best gateway-side
                // "first token" proxy (the whole body arrives ~at once).
                self.on_first_token();
            }
            if self.json_buf.len() < MAX_JSON_BUFFER {
                self.json_buf.extend_from_slice(bytes);
            }
            return;
        }

        for msg in self.parser.feed(bytes) {
            if msg.data == "[DONE]" || msg.data.is_empty() {
                continue;
            }
            // llama.cpp puts `timings` (incl. cache_n = tokens served from
            // the reusable-prefix cache) on the final stream chunk — the
            // realized-reuse signal the slot cache model learns from.
            if msg.data.contains("\"timings\"") {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg.data) {
                    if let Some(cn) = v.pointer("/timings/cache_n").and_then(|x| x.as_u64()) {
                        self.ctx.timeline.reused_tokens_reported = Some(cn);
                    }
                }
            }
            let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(&msg.data) else {
                continue; // keepalives / non-chat events
            };
            if chunk.choices.is_empty() {
                if let Some(usage) = &chunk.usage {
                    self.ctx.timeline.prompt_tokens_reported = Some(usage.prompt_tokens);
                    self.ctx.timeline.completion_tokens_reported = Some(usage.completion_tokens);
                    if usage.prompt_tokens > 0 {
                        self.ctx
                            .estimator
                            .calibrate(self.ctx.timeline.est_input_tokens, usage.prompt_tokens);
                    }
                }
                continue;
            }
            if chunk_text(&chunk).is_some() {
                self.on_token();
            }
        }
    }

    fn on_first_token(&mut self) {
        let now = Instant::now();
        self.ctx.timeline.first_token_at = Some(now);
        self.watchdog = None; // token arrived: disarm
        self.arm_stall_watchdog(now);
                              // Close the waiting-for-first-token span, open the streaming span.
        if let Some(cx) = self.ctx.spans.first_token.take() {
            cx.span().end();
        }
        if let Some(root) = self.ctx.spans.root.as_ref() {
            root.span().add_event(
                "first_token",
                vec![KeyValue::new(
                    "stream.first_token_ms",
                    self.ctx
                        .timeline
                        .ttft()
                        .map(|d| d.as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                )],
            );
            self.ctx.spans.streaming = Some(crate::telemetry::child_span(
                root,
                "worker.streaming",
                opentelemetry::trace::SpanKind::Internal,
                vec![],
            ));
        }
    }

    fn on_token(&mut self) {
        let now = Instant::now();
        if self.ctx.timeline.first_token_at.is_none() {
            self.on_first_token();
        } else if let Some(last) = self.last_token_at {
            let gap = now.duration_since(last).as_secs_f64();
            self.ctx.metrics.itl.observe(gap);
            self.gap_sum_s += gap;
            self.gap_max_s = self.gap_max_s.max(gap);
        }
        self.last_token_at = Some(now);
        self.arm_stall_watchdog(now);
        self.ctx.timeline.output_tokens += 1;
    }

    /// (Re-)arm the mid-stream stall deadline from the given instant.
    fn arm_stall_watchdog(&mut self, from: Instant) {
        if let Some(timeout) = self.ctx.stream_stall_timeout {
            self.stall_watchdog =
                Some(Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(
                    from + timeout,
                ))));
        }
    }

    /// Terminal bookkeeping — exactly once per request.
    fn finish(&mut self, outcome: Outcome) {
        if self.finished {
            return;
        }
        self.finished = true;
        let cleanup_start = Instant::now();
        let tl = &mut self.ctx.timeline;
        tl.ended_at = Some(Instant::now());

        let m = &self.ctx.metrics;
        let w = &self.ctx.worker;

        m.requests_total
            .with_label_values(&[outcome.metric_label()])
            .inc();
        if tl.dispatched_at.is_some() {
            m.input_tokens_total.inc_by(tl.est_input_tokens);
            m.output_tokens_total.inc_by(tl.output_tokens);
        }

        // Histograms. The decomposition is only complete when a token was
        // seen; partial requests still contribute what they have.
        if let Some(gw) = tl.gateway_processing() {
            m.gateway_processing.observe(gw.as_secs_f64());
        }
        if tl.queued_at.is_some() {
            m.admission_wait.observe(tl.admission_wait().as_secs_f64());
        }
        if let Some(c) = tl.worker_connect() {
            m.worker_connect.observe(c.as_secs_f64());
        }
        if let Some(ft) = tl.worker_first_token() {
            m.worker_first_token.observe(ft.as_secs_f64());
        }
        if let Some(t) = tl.ttft() {
            m.ttft.observe(t.as_secs_f64());
        }
        if let Some(e) = tl.e2e() {
            m.e2e.observe(e.as_secs_f64());
        }
        let toks = tl.output_tokens;
        if toks >= 2 {
            m.stream_inter_token
                .observe(self.gap_sum_s / (toks - 1) as f64);
            m.stream_max_stall.observe(self.gap_max_s);
        }
        if let (Some(h), Some(e)) = (tl.headers_at, tl.ended_at) {
            m.stream_completion
                .observe(e.duration_since(h).as_secs_f64());
        }

        // Routing model bookkeeping (phase 4B.2). Prediction error only
        // exists where the model made an estimate and the request ran to
        // a real first token; observed reuse and slot-cache learning only
        // exist where the runtime reported its own telemetry.
        if let (Some(actual), Some(estimated), true) = (
            tl.ttft(),
            self.ctx.estimated_ttft_ms,
            matches!(outcome, Outcome::Completed),
        ) {
            let error_s = actual.as_secs_f64() - estimated / 1000.0;
            m.scheduler_prediction_error_seconds.observe(error_s);
            m.scheduler_prediction_abs_error_seconds.observe(error_s.abs());
        }
        if let Some(reused) = tl.reused_tokens_reported {
            m.scheduler_observed_reused_tokens_total.inc_by(reused);
        }
        if w.slot.is_some() {
            if let Some(prompt_real) = tl.prompt_tokens_reported {
                if matches!(outcome, Outcome::Completed) {
                    w.note_slot_completion(
                        self.ctx.prefix_hash,
                        prompt_real,
                        tl.reused_tokens_reported.unwrap_or(0).min(prompt_real),
                    );
                }
            }
        }

        // Adaptive service model: observations attribute to this dispatch's
        // ticket, so the shared server calibration can bucket them by the
        // concurrency it actually ran under.
        if let Some(wft) = tl.worker_first_token() {
            w.record_prefill(tl.est_input_tokens, wft.as_secs_f64() * 1000.0, &self.ctx.work_estimate);
        }
        if matches!(outcome, Outcome::Completed) {
            if let (Some(dec), 2..) = (tl.decode_phase(), tl.output_tokens) {
                w.record_decode(tl.output_tokens, dec.as_secs_f64() * 1000.0, &self.ctx.work_estimate);
            }
        }

        // Worker lifecycle: failures that implicate the worker feed its
        // state machine, classified by reason (escalation class differs —
        // see inference-lab-worker-lifecycle). Client-side cancels are
        // neutral: they say nothing about worker health.
        match outcome {
            Outcome::Completed => {
                if tl.output_tokens > 0 {
                    let transitions = w.request_served();
                    m.record_transitions(w, &transitions);
                } else {
                    // The phase-4A signature: HTTP-clean, finish_reason
                    // valid, zero generated tokens. A failed inference that
                    // looks like a success must never count as one.
                    m.empty_completions_total.with_label_values(&[w.id.as_str()]).inc();
                    let transitions = w.record_failure(
                        Dimension::Semantic,
                        FailureReason::EmptyCompletion,
                    );
                    m.record_transitions(w, &transitions);
                    tracing::warn!(
                        worker = %w.id,
                        "empty completion: HTTP 200 with zero generated tokens"
                    );
                }
            }
            Outcome::UpstreamError | Outcome::FirstTokenTimeout | Outcome::StreamStalled => {
                let (dimension, reason, metric_kind) = match outcome {
                    Outcome::UpstreamError => (
                        Dimension::Transport,
                        FailureReason::ConnectFailed,
                        "stream",
                    ),
                    Outcome::FirstTokenTimeout => (
                        Dimension::Functional,
                        FailureReason::FirstTokenTimeout,
                        "first_token_timeout",
                    ),
                    _ => (
                        Dimension::Functional,
                        FailureReason::StreamStalled,
                        "stream_stalled",
                    ),
                };
                m.worker_errors_total
                    .with_label_values(&[w.id.as_str(), metric_kind])
                    .inc();
                let transitions = w.record_failure(dimension, reason);
                m.record_transitions(w, &transitions);
            }
            Outcome::ClientCancelled => {
                m.stream_cancelled_total
                    .with_label_values(&["active"])
                    .inc();
                // Saved compute: if we never got a token, the whole estimate;
                // otherwise the undelivered portion of decode.
                let saved_ms = if tl.first_token_at.is_none() {
                    self.ctx.est_total_ms
                } else {
                    let remaining_tokens =
                        tl.requested_output_tokens.saturating_sub(tl.output_tokens);
                    w.decode_ms_per_tok() * remaining_tokens as f64
                };
                m.saved_compute_seconds.inc_by(saved_ms / 1000.0);
            }
        }

        // Worker bookkeeping. A draining worker's last completion ends the
        // drain (DRAINING → STOPPED); those transitions hit metrics too.
        w.end_work(&self.ctx.work_estimate);
        let drained = w.dec_outstanding();
        m.record_transitions(w, &drained);

        // Spans: final attributes, then close in creation order.
        if let Some(root) = self.ctx.spans.root.as_ref() {
            let rs = root.span();
            rs.set_attribute(KeyValue::new(
                "stream.output_tokens",
                tl.output_tokens as i64,
            ));
            rs.set_attribute(KeyValue::new(
                "stream.first_token_ms",
                tl.ttft().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(-1.0),
            ));
            if toks >= 2 {
                rs.set_attribute(KeyValue::new(
                    "stream.mean_itl_ms",
                    self.gap_sum_s / (toks - 1) as f64 * 1000.0,
                ));
                rs.set_attribute(KeyValue::new(
                    "stream.max_stall_ms",
                    self.gap_max_s * 1000.0,
                ));
            }
            rs.set_attribute(KeyValue::new("inference.outcome", outcome.metric_label()));
            rs.set_attribute(KeyValue::new(
                "http.client_cancelled",
                matches!(outcome, Outcome::ClientCancelled),
            ));
            rs.set_attribute(KeyValue::new(
                "inference.input_tokens_estimated",
                tl.est_input_tokens as i64,
            ));
            if let Some(pt) = tl.prompt_tokens_reported {
                rs.set_attribute(KeyValue::new("gen_ai.usage.input_tokens", pt as i64));
            }
            if let Some(ct) = tl.completion_tokens_reported {
                rs.set_attribute(KeyValue::new("gen_ai.usage.output_tokens", ct as i64));
            }
            rs.end();
        }
        if let Some(cx) = self.ctx.spans.streaming.take() {
            cx.span().set_attribute(KeyValue::new(
                "stream.output_tokens",
                tl.output_tokens as i64,
            ));
            cx.span().end();
        }
        if let Some(cx) = self.ctx.spans.first_token.take() {
            cx.span().end();
        }
        if let Some(cx) = self.ctx.spans.worker_request.take() {
            let s = cx.span();
            s.set_attribute(KeyValue::new(
                "stream.output_tokens",
                tl.output_tokens as i64,
            ));
            s.set_attribute(KeyValue::new("inference.outcome", outcome.metric_label()));
            s.end();
        }

        m.cancel_cleanup_seconds
            .observe(cleanup_start.elapsed().as_secs_f64());

        tracing::info!(
            worker = %w.id,
            outcome = outcome.metric_label(),
            output_tokens = tl.output_tokens,
            ttft_ms = ?tl.ttft().map(|d| d.as_millis()),
            "request finished"
        );
    }
}

impl<S> Stream for InstrumentedStream<S>
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
{
    type Item = reqwest::Result<bytes::Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        // First-token watchdog: the timer registers our waker, so even a
        // fully silent upstream gets aborted at the deadline.
        if let Some(sleep) = self.watchdog.as_mut() {
            if matches!(sleep.as_mut().poll(cx), Poll::Ready(())) {
                self.watchdog = None;
                if !self.finished && self.ctx.timeline.first_token_at.is_none() {
                    self.finish(Outcome::FirstTokenTimeout);
                    return Poll::Ready(None);
                }
            }
        }

        // Mid-stream stall watchdog: re-armed on every token, so firing
        // means the upstream went silent for the full stall timeout after
        // generation had already started.
        if let Some(sleep) = self.stall_watchdog.as_mut() {
            if matches!(sleep.as_mut().poll(cx), Poll::Ready(())) {
                self.stall_watchdog = None;
                if !self.finished && self.ctx.timeline.first_token_at.is_some() {
                    tracing::warn!(
                        worker = %self.ctx.worker.id,
                        stall_s = self.gap_max_s.max(
                            self.last_token_at.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
                        ),
                        "mid-stream stall: no token past STREAM_STALL_TIMEOUT_S"
                    );
                    self.finish(Outcome::StreamStalled);
                    return Poll::Ready(None);
                }
            }
        }

        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                self.observe_chunk(&bytes);
                Poll::Ready(Some(Ok(bytes))) // forward unchanged
            }
            Poll::Ready(Some(Err(e))) => {
                tracing::warn!(worker = %self.ctx.worker.id, error = %e, "upstream stream error");
                self.finish(Outcome::UpstreamError);
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                // Natural end. For non-streaming requests extract usage now.
                if !self.ctx.stream_mode && self.json_buf.len() <= MAX_JSON_BUFFER {
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&self.json_buf) {
                        if let Some(usage) = v.get("usage") {
                            let u: Usage =
                                serde_json::from_value(usage.clone()).unwrap_or_default();
                            self.ctx.timeline.prompt_tokens_reported = Some(u.prompt_tokens);
                            self.ctx.timeline.completion_tokens_reported =
                                Some(u.completion_tokens);
                            if u.prompt_tokens > 0 {
                                self.ctx
                                    .estimator
                                    .calibrate(self.ctx.timeline.est_input_tokens, u.prompt_tokens);
                            }
                            if u.completion_tokens > self.ctx.timeline.output_tokens {
                                self.ctx.timeline.output_tokens = u.completion_tokens;
                            }
                        }
                        // llama.cpp JSON bodies carry the same reuse
                        // telemetry under `timings.cache_n`.
                        if let Some(cn) =
                            v.pointer("/timings/cache_n").and_then(|x| x.as_u64())
                        {
                            self.ctx.timeline.reused_tokens_reported = Some(cn);
                        }
                    }
                }
                self.finish(Outcome::Completed);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for InstrumentedStream<S> {
    fn drop(&mut self) {
        // Body dropped before natural end = client went away (or gateway
        // shutdown). Releasing here both frees admission capacity and drops
        // the upstream reqwest stream, aborting generation at the worker.
        if !self.finished {
            self.finish(Outcome::ClientCancelled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{
        AdmissionController, AdmissionDecision, AdmissionPolicy, RequestSize, TokenEstimator,
    };
    use bytes::Bytes;
    use inference_lab_scheduler::WorkerHandle;

    fn sse(tok: &str) -> String {
        format!(
            concat!(
                "data: {{\"id\":\"t\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",",
                "\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{tok}\"}},\"finish_reason\":null}}]}}\n\n"
            ),
            tok = tok
        )
    }

    fn test_ctx() -> (Arc<Metrics>, StreamCtx) {
        let metrics = Arc::new(Metrics::new().unwrap());
        let worker = Arc::new(WorkerHandle::new("t", "http://127.0.0.1:1", "m", "test"));
        let ctrl = AdmissionController::new(AdmissionPolicy::default());
        let permit = match ctrl.decide(RequestSize {
            input_tokens: 10,
            output_tokens: 10,
            est_work_ms: 100.0,
        }) {
            AdmissionDecision::Admit(p) => p,
            _ => panic!("disabled policy must admit immediately"),
        };
        let mut timeline = RequestTimeline::new(10, 10);
        // The handler sets this on dispatch; the stream-fairness completion
        // histogram needs it to have a start to measure from.
        timeline.headers_at = Some(Instant::now());
        let work_estimate = worker.begin_work(100.0, 100.0);
        let ctx = StreamCtx {
            metrics: metrics.clone(),
            worker,
            estimator: Arc::new(TokenEstimator::new()),
            _permit: permit,
            _in_flight: InFlightGuard(metrics.clone()),
            timeline,
            spans: TraceSpans {
                root: None,
                worker_request: None,
                first_token: None,
                streaming: None,
            },
            prefix_hash: 0,
            estimated_ttft_ms: None,
            first_token_deadline: None,
            stream_stall_timeout: None,
            stream_mode: true,
            work_estimate,
            est_total_ms: 100.0,
        };
        (metrics, ctx)
    }

    /// Upstream that streams `n` tokens, sleeping `gap` between tokens 2..n:
    /// a real (std-clock) inter-token gap, small enough for tests, large
    /// enough to land in a meaningful stall bucket.
    fn gappy_stream(
        tok: &str,
        n: usize,
        gap: Duration,
    ) -> Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>> {
        let tok = tok.to_string();
        Box::pin(futures_util::stream::unfold(0usize, move |i| {
            let tok = tok.clone();
            async move {
                if i >= n {
                    return None;
                }
                if i > 0 {
                    tokio::time::sleep(gap).await;
                }
                Some((Ok(Bytes::from(sse(&tok))), i + 1))
            }
        }))
    }

    fn hist_count(fams: &[prometheus::proto::MetricFamily], name: &str) -> u64 {
        fams.iter()
            .find(|f| f.name() == name)
            .map(|f| f.get_metric()[0].get_histogram().get_sample_count())
            .unwrap_or(0)
    }

    fn counter_for(
        fams: &[prometheus::proto::MetricFamily],
        name: &str,
        label_name: &str,
        label_value: &str,
    ) -> f64 {
        fams.iter()
            .find(|f| f.name() == name)
            .and_then(|f| {
                f.get_metric().iter().find(|m| {
                    m.get_label()
                        .iter()
                        .any(|l| l.name() == label_name && l.value() == label_value)
                })
            })
            .map(|m| m.get_counter().value())
            .unwrap_or(0.0)
    }

    /// One token, then eternal silence: the mid-stream stall signature.
    fn one_token_then_silence(
        tok: &str,
    ) -> Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>> {
        let tok = tok.to_string();
        Box::pin(
            futures_util::stream::iter(vec![Ok(Bytes::from(sse(&tok)))])
                .chain(futures_util::stream::pending()),
        )
    }

    #[tokio::test]
    async fn mid_stream_stall_watchdog_trips_and_frees_capacity() {
        let (metrics, mut ctx) = test_ctx();
        ctx.stream_stall_timeout = Some(Duration::from_millis(50));
        let mut st = InstrumentedStream::new(one_token_then_silence("x"), ctx);
        let start = Instant::now();
        while st.next().await.is_some() {}
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "stall watchdog must abort the silent stream, took {start:?}"
        );
        let fams = metrics.registry.gather();
        assert_eq!(
            counter_for(&fams, "inference_requests_total", "outcome", "stream_stalled"),
            1.0,
            "aborted stream must land on the stream_stalled outcome"
        );
    }

    #[tokio::test]
    async fn stream_fairness_metrics_track_each_stream() {
        let (metrics, ctx) = test_ctx();
        let mut st = InstrumentedStream::new(gappy_stream("x", 4, Duration::from_millis(80)), ctx);
        while st.next().await.is_some() {}

        let fams = metrics.registry.gather();
        // 4 tokens = 3 gaps: one observation per stream-level histogram.
        assert_eq!(hist_count(&fams, "inference_stream_inter_token_seconds"), 1);
        assert_eq!(hist_count(&fams, "inference_stream_max_stall_seconds"), 1);
        assert_eq!(hist_count(&fams, "inference_stream_completion_seconds"), 1);
        // ~80 ms gaps must land in a sub-second stall bucket, not +Inf —
        // the buckets are the operational signal ("was anyone stalled ≥10 s?").
        let fam = fams
            .iter()
            .find(|f| f.name() == "inference_stream_max_stall_seconds")
            .unwrap();
        let first_populated = fam.get_metric()[0]
            .get_histogram()
            .get_bucket()
            .iter()
            .find(|b| b.cumulative_count() > 0)
            .map(|b| b.upper_bound())
            .unwrap_or(f64::INFINITY);
        assert!(
            first_populated <= 0.5,
            "80ms stall landed in bucket {first_populated}"
        );
    }

    #[tokio::test]
    async fn stream_fairness_metrics_skip_single_token_streams() {
        let (metrics, ctx) = test_ctx();
        let mut st =
            InstrumentedStream::new(gappy_stream("only", 1, Duration::from_millis(1)), ctx);
        while st.next().await.is_some() {}

        let fams = metrics.registry.gather();
        // One token: no inter-token gap exists, so the gap-derived
        // histograms must stay empty rather than observe a fake zero.
        assert_eq!(hist_count(&fams, "inference_stream_inter_token_seconds"), 0);
        assert_eq!(hist_count(&fams, "inference_stream_max_stall_seconds"), 0);
        assert_eq!(hist_count(&fams, "inference_stream_completion_seconds"), 1);
    }
}
