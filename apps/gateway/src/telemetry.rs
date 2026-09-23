//! OpenTelemetry wiring for the gateway.
//!
//! Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (the standard OTel env
//! knob — `make gateway` points it at the local collector's OTLP/HTTP port).
//! Traces go to the collector → Tempo, and `tracing` log events ride along as
//! OTel logs → Loki, so a trace and its log lines share a trace_id.
//!
//! When OTLP is not configured everything here degrades to no-ops: spans
//! still get created through the global no-op tracer, so request handling
//! code has a single unconditional code path.

use opentelemetry::global::set_tracer_provider;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::SpanKind;
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;

/// Build the resource describing this gateway process in every signal.
fn resource(service_version: &str) -> Resource {
    Resource::builder()
        .with_service_name("gateway")
        .with_attributes([
            KeyValue::new("service.version", service_version.to_string()),
            KeyValue::new("deployment.environment", "inference-lab"),
        ])
        .build()
}

/// Trace provider exporting OTLP/HTTP (binary protobuf) to the collector.
/// The endpoint comes from `OTEL_EXPORTER_OTLP_ENDPOINT` (OTel default env),
/// e.g. `http://127.0.0.1:4318`.
fn tracer_provider(service_version: &str) -> anyhow::Result<SdkTracerProvider> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .map_err(|e| anyhow::anyhow!("building OTLP span exporter: {e}"))?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource(service_version))
        .build())
}

/// Logger provider for the `tracing` → OTel logs bridge (→ collector → Loki).
fn logger_provider(
    service_version: &str,
) -> anyhow::Result<opentelemetry_sdk::logs::SdkLoggerProvider> {
    let exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .build()
        .map_err(|e| anyhow::anyhow!("building OTLP log exporter: {e}"))?;
    Ok(opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource(service_version))
        .build())
}

/// True when `OTEL_EXPORTER_OTLP_ENDPOINT` names a collector.
pub fn otlp_configured() -> bool {
    std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok()
}

/// Install tracing + OTel. Returns a guard whose drop flushes and shuts the
/// providers down (batch exporters hold spans in memory otherwise).
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    logger_provider: Option<opentelemetry_sdk::logs::SdkLoggerProvider>,
}

impl TelemetryGuard {
    pub fn shutdown(self) {
        if let Some(p) = self.tracer_provider {
            if let Err(e) = p.shutdown() {
                eprintln!("tracer provider shutdown: {e:?}");
            }
        }
        if let Some(p) = self.logger_provider {
            if let Err(e) = p.shutdown() {
                eprintln!("logger provider shutdown: {e:?}");
            }
        }
    }
}

/// Set up `tracing_subscriber`: fmt to stdout plus, when OTLP is configured,
/// spans/logs exported through the OTel pipeline.
pub fn init(service_version: &str) -> anyhow::Result<TelemetryGuard> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "gateway=info".into());
    let fmt_layer = tracing_subscriber::fmt::layer();

    if !otlp_configured() {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
        return Ok(TelemetryGuard { tracer_provider: None, logger_provider: None });
    }

    let tracer_provider = tracer_provider(service_version)?;
    set_tracer_provider(tracer_provider.clone());
    let logger_provider = logger_provider(service_version)?;
    let otel_logs =
        opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&logger_provider);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(otel_logs)
        .init();

    Ok(TelemetryGuard {
        tracer_provider: Some(tracer_provider),
        logger_provider: Some(logger_provider),
    })
}

/// The gateway's tracer (named after the component, per OTel convention).
pub fn tracer() -> opentelemetry::global::BoxedTracer {
    opentelemetry::global::tracer("gateway")
}

// ---------------------------------------------------------------------------
// Small helpers so span creation stays terse at the call sites
// ---------------------------------------------------------------------------

/// A span handle. In OTel 0.33 spans are not cloneable; the ownable handle
/// is the `Context` carrying the span, which *is* cloneable and exposes
/// `&self` mutators through [`opentelemetry::trace::TraceContextExt::span`].
pub type SpanHandle = opentelemetry::Context;

/// Start the request's root (SERVER) span, parented to the ambient context.
pub fn root_span(name: &str, kind: SpanKind, attrs: Vec<KeyValue>) -> SpanHandle {
    use opentelemetry::trace::{TraceContextExt, Tracer};
    let span = tracer()
        .span_builder(name.to_string())
        .with_attributes(attrs)
        .with_kind(kind)
        .start(&tracer());
    opentelemetry::Context::current_with_span(span)
}

/// Start a child span. Explicit context parenting (rather than ambient) is
/// what makes the waterfall render correctly in Tempo — spans created
/// without it become disconnected roots.
pub fn child_span(
    parent: &SpanHandle,
    name: &str,
    kind: SpanKind,
    attrs: Vec<KeyValue>,
) -> SpanHandle {
    use opentelemetry::trace::{TraceContextExt, Tracer};
    let span = tracer()
        .span_builder(name.to_string())
        .with_attributes(attrs)
        .with_kind(kind)
        .start_with_context(&tracer(), parent);
    parent.with_span(span)
}

/// Record an INTERNAL span that starts and ends immediately (micro-phases
/// like token estimation, where the span exists to anchor timings and
/// attributes in the trace view).
pub fn instant_span(parent: &SpanHandle, name: &str, attrs: Vec<KeyValue>) {
    use opentelemetry::trace::TraceContextExt;
    let cx = child_span(parent, name, SpanKind::Internal, attrs);
    cx.span().end();
}

/// Inject the request's trace context into outbound headers as W3C
/// `traceparent`, so workers (or a future instrumented runtime) join the
/// same trace.
pub fn inject_traceparent(cx: &SpanHandle, headers: &mut reqwest::header::HeaderMap) {
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut ReqHeaderInjector(headers));
    });
}

struct ReqHeaderInjector<'a>(&'a mut reqwest::header::HeaderMap);

impl Injector for ReqHeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::try_from(key),
            reqwest::header::HeaderValue::try_from(value),
        ) {
            self.0.insert(name, val);
        }
    }
}
