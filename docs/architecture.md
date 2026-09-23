# Architecture

```text
                           CLIENTS (loadgen, curl, apps)
                              │
                 OpenAI-compatible API
                              │
              ┌───────────────▼────────────────┐
              │        Gateway (Rust)          │
              │           Axum / Tokio         │
              │                                │
              │  estimate tokens (calibrated)  │
              │        ↓                       │
              │  admission control             │  budgets in tokens + estimated ms,
              │  queue (work-accounted FIFO)   │  429 + Retry-After when overloaded
              │        ↓                       │
              │  scheduler                     │  LOR · adaptive ECT · prefix-affinity
              │  (health-aware, re-route ×1)   │  service model learned online
              └───────────────┬────────────────┘
                              │ trace context (W3C traceparent)
          ┌───────────────────┼──────────────────┐
          ▼                   ▼                  ▼
   vLLM-Metal Worker     MLX-LM Worker     llama.cpp Worker   mock-gpu (testing)
   real vLLM scheduler   raw MLX path      GGUF / Metal       failure injection
   paged KV, prefix cat. │                  │
          │              └──────────────────┴──────────┐
          ▼                                          ▼
                    Apple M5 GPU · unified memory (native macOS)

                              │ telemetry (OTLP + Prometheus)
              ┌───────────────┼──────────────┐
              ▼               ▼              ▼
        OTel Collector   Prometheus       Loki
              │               │              │
              └───────┬───────┴──────────────┘
                      ▼
                   Grafana        (+ Tempo for distributed traces)

                k3d / Kubernetes (phase 4: gateway/router/mock-worker control plane;
                Metal workers stay native, exposed via host endpoints)
```

## Components

| Component | Language/stack | Role |
| --- | --- | --- |
| `apps/gateway` | Rust, Axum | OpenAI-compatible edge: token estimation → admission control → scheduling → instrumented streaming proxy; OTel traces/logs; Prometheus metrics |
| `apps/loadgen` | Rust, Tokio/reqwest | Open-loop arrivals + closed-loop concurrency cap; client-side TTFT/ITL/TPOT/E2E; JSON+CSV artifacts; matrix runner |
| `crates/scheduler` | Rust | worker registry + online service-time model (EWMA prefill regression, decode EWMA); routing policies; health tracking |
| `crates/streaming` | Rust | incremental SSE parser (spec-correct across chunk boundaries) |
| `crates/metrics` | Rust | request records → percentiles/throughput aggregation |
| `crates/workload` | Rust | scenario TOMLs, deterministic prompt synthesis, arrival processes |
| `workers/*` | Python / C++ | native macOS model servers (MLX-LM, vLLM-Metal, llama.cpp) + dependency-free mock |
| `benchmarks/failures` | Python | automated failure-injection suite (`make failcheck`) |
| `deploy/compose` | Docker | OTel Collector, Prometheus, Tempo, Loki, Grafana (provisioned dashboard) |

## The gateway request path

```text
request → estimate input tokens (words×3/4, EWMA-calibrated from usage chunks)
        → admission.decide(work = fixed + prefill·in + decode·out)
             fits + queue empty  → dispatch
             doesn't fit         → FIFO queue (tokens + estimated seconds accounted)
             est wait ≥ overload → 429 + Retry-After
        → scheduler.select (ECT: remaining worker work + service time)
        → worker (traceparent injected; one re-route on connect failure)
        → instrumented stream: first token, ITL, token counts, usage chunk
        → finish(outcome): histograms, service-model updates, spans closed
```

### TTFT decomposition

The gateway can only see what crosses it, so it splits TTFT into exactly
the components it *knows*, naming them for what they are:

| component | measures |
| --- | --- |
| `gateway_processing` | estimation + admission decision + scheduling |
| `admission_wait` | time in the admission queue |
| `worker_connect` | upstream send → response headers |
| `worker_first_token` | headers → first token: **runtime queue + tokenization + prefill + scheduler overhead** — deliberately *not* called "prefill" |

Distinguishing the last one into true prefill/queue/tokenization needs
runtime-side instrumentation; until then the honest name is the honest
label.

### Admission control

Budgets are denominated in the units the service model speaks, not
request counts (`ADMIT_MAX_ACTIVE_INPUT_TOKENS`,
`ADMIT_MAX_ACTIVE_OUTPUT_TOKENS`, `ADMIT_MAX_WORK_MS`; 0 disables).
Queueing is FIFO with bounded skip-ahead: a small request may pass a head
that cannot fit, so a giant prompt never causes head-of-line death.
Overload is explicit — `NORMAL → DEGRADED → OVERLOADED` on the estimated
queue wait, with `429 + Retry-After` past the threshold instead of silent
latency growth. Cancellation releases capacity in both phases (queued
entries via handler-future drop; active streams via upstream connection
abort) and credits `inference_saved_estimated_compute_seconds`.

All of this is off by default: with no budgets set the gateway behaves
like the phase-2 system, which is what makes A/B saturation experiments
possible.

### Adaptive service model

Each worker learns, online:

```text
prefill_ms ≈ fixed + prefill_ms_per_tok × input_tokens   (EWMA least squares)
decode_ms  ≈ decode_ms_per_tok × output_tokens           (EWMA)
```

Seeds come from the phase-2 study (750 ms + 5.35 ms/token prefill, 76
ms/token decode for Qwen3.5-9B-4bit on this M5). The live parameters are
exported as `gateway_worker_*` gauges and feed both ECT scoring and
admission work estimates. The scheduler therefore gets smarter about
*this machine* as it serves traffic — no hard-coded constants survive
contact with a new runtime or quantization.

## Design rules

1. **Model processes stay native on macOS.** Metal/unified-memory access is
   not equivalent inside Linux containers on Apple Silicon. Infrastructure
   (metrics, tracing, dashboards) is containerized — that's the control
   plane; the workers are the "accelerator nodes."
2. **Everything speaks the OpenAI chat API.** Any worker is swappable behind
   the same loadgen/gateway; comparisons are apples-to-apples.
3. **Measurement happens at the client.** TTFT/ITL are what the user
   experiences; server-internal counters (when available) are secondary.
   The gateway's own decomposition is labelled by what it can actually see.
4. **Bound what you admit, don't just queue and pray.** Phase 2 measured
   the failure mode (8K×c8: 0/16 first tokens in 900 s); admission control
   turns that measurement into a policy.
5. **One repo, small slices.** Each phase (benchmarks → routing →
   tracing/admission → k8s) produces a working, demonstrable system.

## Failure-path contract (gateway)

```text
client disconnect (active stream)
       ↓ gateway body stream dropped (hyper)
       ↓ InstrumentedStream::drop → finish(ClientCancelled)
       ↓   capacity released, saved compute credited, spans closed
       ↓ upstream reqwest stream dropped → connection aborted
       ↓ worker sees cancelled request → generation stops

client disconnect (queued request, HTTP/2 reset)
       ↓ handler future dropped → QueuedGuard
       ↓ admission.cancel_queued → budget freed, waiter never dispatched
       (HTTP/1.1: observable only at response-write time)

worker silent (no first token)
       ↓ watchdog timer (tokio::time::Sleep fused into the stream poll)
       ↓ FirstTokenTimeout → capacity released, worker failure counted
```

No tokens are generated for nobody, and no slot is held forever.
