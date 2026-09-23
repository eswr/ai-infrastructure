//! gateway — OpenAI-compatible edge in front of the lab's inference workers.
//!
//! Responsibilities:
//! * terminate client SSE, proxy to a worker chosen by the routing policy
//! * keep per-worker outstanding-request counts accurate even when a client
//!   disconnects mid-stream (dropping the body stream drops the upstream
//!   reqwest stream → connection aborted → worker stops generating)
//! * expose Prometheus metrics on /metrics
//!
//! Config (env):
//! * `GATEWAY_ADDR`       listen address           (default 0.0.0.0:8080)
//! * `WORKERS`            `name=url[,name=url...]` (default mlx=http://127.0.0.1:8082)
//! * `ROUTE_POLICY`       lor | ect | prefix-affinity (default lor)

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{Stream, StreamExt};
use inference_lab_protocol::ChatCompletionRequest;
use inference_lab_scheduler::{RoutingPolicy, Router as WorkerRouter, WorkerHandle};
use prometheus::{Encoder, IntCounter, IntCounterVec, IntGauge, Registry, TextEncoder};
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

#[derive(Clone)]
struct AppState {
    router: Arc<WorkerRouter>,
    client: reqwest::Client,
    metrics: Arc<Metrics>,
}

struct Metrics {
    registry: Registry,
    requests_total: IntCounter,
    upstream_failures: IntCounterVec,
    active_requests: IntGauge,
}

impl Metrics {
    fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();
        let requests_total =
            IntCounter::new("gateway_requests_total", "Chat completion requests received")?;
        let upstream_failures = IntCounterVec::new(
            prometheus::Opts::new(
                "gateway_upstream_failures_total",
                "Upstream request failures",
            ),
            &["worker", "reason"],
        )?;
        let active_requests =
            IntGauge::new("gateway_active_requests", "Requests currently streaming/proxying")?;
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(upstream_failures.clone()))?;
        registry.register(Box::new(active_requests.clone()))?;
        Ok(Self { registry, requests_total, upstream_failures, active_requests })
    }
}

/// Stream wrapper that runs a cleanup closure when the body is dropped —
/// whether that's normal completion, an upstream error, or a client
/// disconnect. This is what keeps outstanding counts honest and what turns
/// "browser closed the tab" into "abort the upstream request".
struct OnDropStream<S, F: FnOnce()> {
    inner: S,
    cleanup: Option<F>,
}

impl<S, F> Stream for OnDropStream<S, F>
where
    S: Stream + Unpin,
    F: FnOnce() + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }
}

impl<S, F: FnOnce()> Drop for OnDropStream<S, F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

fn error_json(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({
        "error": {"message": msg, "type": "gateway_error"}
    });
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

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    st.metrics.requests_total.inc();
    st.metrics.active_requests.inc();

    let hash = prefix_hash(&req);
    let worker = st.router.select(hash, req.max_tokens);
    worker.outstanding.fetch_add(1, Ordering::Relaxed);
    tracing::info!(worker = %worker.id, model = %req.model, stream = req.stream, "routing request");

    let worker_for_err = worker.clone();
    let metrics = st.metrics.clone();
    let cleanup = move || {
        worker_for_err.outstanding.fetch_sub(1, Ordering::Relaxed);
        metrics.active_requests.dec();
    };

    let url = format!("{}/v1/chat/completions", worker.endpoint.trim_end_matches('/'));
    let upstream = match st.client.post(&url).json(&req).send().await {
        Ok(r) => r,
        Err(e) => {
            st.metrics
                .upstream_failures
                .with_label_values(&[&worker.id, "connect"])
                .inc();
            cleanup();
            return error_json(
                StatusCode::BAD_GATEWAY,
                &format!("upstream {}: {e}", worker.id),
            );
        }
    };

    let status = upstream.status();
    if !status.is_success() {
        let body = upstream.text().await.unwrap_or_default();
        st.metrics
            .upstream_failures
            .with_label_values(&[&worker.id, "http"])
            .inc();
        cleanup();
        return error_json(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            &format!("upstream {}: {}", worker.id, body.chars().take(300).collect::<String>()),
        );
    }

    let mut upstream_headers = HeaderMap::new();
    if let Some(ct) = upstream.headers().get(header::CONTENT_TYPE) {
        upstream_headers.insert(header::CONTENT_TYPE, ct.clone());
    }
    // SSE passthrough: bytes flow through unchanged; when the client hangs up,
    // hyper drops this body → OnDropStream::drop fires → upstream stream is
    // dropped too, closing the worker connection.
    let stream = OnDropStream {
        inner: upstream.bytes_stream(),
        cleanup: Some(cleanup),
    };

    match Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK))
        .header(header::CONTENT_TYPE, upstream_headers.get(header::CONTENT_TYPE).cloned().unwrap_or(HeaderValue::from_static("application/json")))
        .body(Body::from_stream(stream))
    {
        Ok(resp) => resp,
        Err(e) => error_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("build response: {e}")),
    }
}

async fn list_models(State(st): State<Arc<AppState>>) -> Response {
    let mut seen = std::collections::BTreeSet::new();
    for w in st.router.workers() {
        seen.insert(w.model.clone());
    }
    let data: Vec<serde_json::Value> = seen
        .into_iter()
        .map(|m| serde_json::json!({"id": m, "object": "model"}))
        .collect();
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

async fn healthz() -> Response {
    Json(serde_json::json!({"status": "ok"})).into_response()
}

async fn metrics_endpoint(State(st): State<Arc<AppState>>) -> Response {
    let encoder = TextEncoder::new();
    let mut buf = Vec::new();
    if encoder.encode(&st.metrics.registry.gather(), &mut buf).is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"))],
        buf,
    )
        .into_response()
}

async fn root() -> Response {
    Json(serde_json::json!({
        "service": "inference-lab gateway",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/healthz", "/metrics"],
    }))
    .into_response()
}

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gateway=info".into()),
        )
        .init();

    let addr = std::env::var("GATEWAY_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let workers_spec = std::env::var("WORKERS").unwrap_or_else(|_| "mlx=http://127.0.0.1:8082".into());
    let policy = RoutingPolicy::parse(
        &std::env::var("ROUTE_POLICY").unwrap_or_else(|_| "lor".into()),
    )
    .unwrap_or_default();

    let handles = parse_workers(&workers_spec)
        .into_iter()
        .map(|(name, url, model)| {
            let runtime = name.split('-').next().unwrap_or("unknown").to_string();
            Arc::new(WorkerHandle::new(name, url, model, runtime))
        })
        .collect::<Vec<_>>();

    for w in &handles {
        tracing::info!(worker = %w.id, endpoint = %w.endpoint, model = %w.model, "registered worker");
    }
    tracing::info!(?policy, "routing policy");

    let router = Arc::new(WorkerRouter::new(handles, policy)?);

    let state = Arc::new(AppState {
        router,
        client: reqwest::Client::builder().build()?,
        metrics: Arc::new(Metrics::new()?),
    });

    let app = Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("gateway listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
