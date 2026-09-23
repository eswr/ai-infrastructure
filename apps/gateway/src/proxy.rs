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
use inference_lab_scheduler::WorkerHandle;
use inference_lab_streaming::SseParser;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
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
        self.dispatched_at.map(|d| d.duration_since(self.t0) - self.admission_wait())
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
}

impl Outcome {
    fn metric_label(&self) -> &'static str {
        match self {
            Self::Completed => "ok",
            Self::ClientCancelled => "client_cancelled",
            Self::UpstreamError => "upstream_error",
            Self::FirstTokenTimeout => "first_token_timeout",
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
    pub first_token_deadline: Option<Instant>,
    pub stream_mode: bool,
    /// (start, est_ms) pair registered with `worker.begin_work`.
    pub work_estimate: (Instant, f64),
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
    finished: bool,
}

impl<S> InstrumentedStream<S> {
    pub fn new(inner: S, ctx: StreamCtx) -> Self {
        Self {
            inner,
            ctx,
            parser: SseParser::new(),
            json_buf: Vec::new(),
            last_token_at: None,
            finished: false,
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
            let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(&msg.data) else {
                continue; // keepalives / non-chat events
            };
            if chunk.choices.is_empty() {
                if let Some(usage) = &chunk.usage {
                    self.ctx.timeline.prompt_tokens_reported = Some(usage.prompt_tokens);
                    self.ctx.timeline.completion_tokens_reported = Some(usage.completion_tokens);
                    if usage.prompt_tokens > 0 {
                        self.ctx.estimator.calibrate(
                            self.ctx.timeline.est_input_tokens,
                            usage.prompt_tokens,
                        );
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
        // Close the waiting-for-first-token span, open the streaming span.
        if let Some(cx) = self.ctx.spans.first_token.take() {
            cx.span().end();
        }
        if let Some(root) = self.ctx.spans.root.as_ref() {
            root.span().add_event(
                "first_token",
                vec![KeyValue::new(
                    "stream.first_token_ms",
                    self.ctx.timeline.ttft().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0),
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
            self.ctx.metrics.itl.observe(now.duration_since(last).as_secs_f64());
        }
        self.last_token_at = Some(now);
        self.ctx.timeline.output_tokens += 1;
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

        m.requests_total.with_label_values(&[outcome.metric_label()]).inc();
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

        // Adaptive service model.
        if let Some(wft) = tl.worker_first_token() {
            w.record_prefill(tl.est_input_tokens, wft.as_secs_f64() * 1000.0);
        }
        if matches!(outcome, Outcome::Completed) {
            if let (Some(dec), 2..) = (tl.decode_phase(), tl.output_tokens) {
                w.record_decode(tl.output_tokens, dec.as_secs_f64() * 1000.0);
            }
        }

        // Worker health: failures that implicate the worker, not the client.
        match outcome {
            Outcome::Completed => w.record_success(),
            Outcome::UpstreamError | Outcome::FirstTokenTimeout => {
                let reason = if matches!(outcome, Outcome::UpstreamError) {
                    "stream"
                } else {
                    "first_token_timeout"
                };
                m.worker_errors_total.with_label_values(&[w.id.as_str(), reason]).inc();
                if w.record_dispatch_failure() {
                    tracing::warn!(worker = %w.id, "worker marked unhealthy after stream failure");
                }
            }
            Outcome::ClientCancelled => {
                m.stream_cancelled_total.with_label_values(&["active"]).inc();
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

        // Worker bookkeeping.
        w.end_work(self.ctx.work_estimate.0, self.ctx.work_estimate.1);
        w.dec_outstanding();

        // Spans: final attributes, then close in creation order.
        if let Some(root) = self.ctx.spans.root.as_ref() {
            let rs = root.span();
            rs.set_attribute(KeyValue::new("stream.output_tokens", tl.output_tokens as i64));
            rs.set_attribute(KeyValue::new(
                "stream.first_token_ms",
                tl.ttft().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(-1.0),
            ));
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
            cx.span().set_attribute(KeyValue::new("stream.output_tokens", tl.output_tokens as i64));
            cx.span().end();
        }
        if let Some(cx) = self.ctx.spans.first_token.take() {
            cx.span().end();
        }
        if let Some(cx) = self.ctx.spans.worker_request.take() {
            let s = cx.span();
            s.set_attribute(KeyValue::new("stream.output_tokens", tl.output_tokens as i64));
            s.set_attribute(KeyValue::new("inference.outcome", outcome.metric_label()));
            s.end();
        }

        m.cancel_cleanup_seconds.observe(cleanup_start.elapsed().as_secs_f64());

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
        // First-token watchdog: checked on every wakeup so a hung worker
        // cannot hold a slot forever.
        if !self.finished {
            if let (Some(deadline), None) = (self.ctx.first_token_deadline, self.ctx.timeline.first_token_at) {
                if Instant::now() >= deadline {
                    self.finish(Outcome::FirstTokenTimeout);
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
                            let u: Usage = serde_json::from_value(usage.clone()).unwrap_or_default();
                            self.ctx.timeline.prompt_tokens_reported = Some(u.prompt_tokens);
                            self.ctx.timeline.completion_tokens_reported =
                                Some(u.completion_tokens);
                            if u.prompt_tokens > 0 {
                                self.ctx.estimator.calibrate(
                                    self.ctx.timeline.est_input_tokens,
                                    u.prompt_tokens,
                                );
                            }
                            if u.completion_tokens > self.ctx.timeline.output_tokens {
                                self.ctx.timeline.output_tokens = u.completion_tokens;
                            }
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
