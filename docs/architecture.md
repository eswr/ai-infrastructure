# Architecture

```text
                           CLIENTS (loadgen, curl, apps)
                              │
                 OpenAI-compatible API
                              │
                    ┌─────────▼─────────┐
                    │  Gateway (Rust)   │   routing policies:
                    │  Axum / Tokio     │   least-outstanding · ECT · prefix-affinity
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼──────────────────┐
          ▼                   ▼                  ▼
   vLLM-Metal Worker     MLX-LM Worker     llama.cpp Worker   mock-gpu (testing)
   real vLLM scheduler   raw MLX path      GGUF / Metal       failure injection
   paged KV, prefix cat. │                  │
          │              └──────────────────┴──────────┐
          ▼                                          ▼
                    Apple M5 GPU · unified memory (native macOS)

                              │ telemetry
              ┌───────────────┼──────────────┐
              ▼               ▼              ▼
        OTel Collector   Prometheus       Loki
              │               │              │
              └───────┬───────┴──────────────┘
                      ▼
                   Grafana        (+ Tempo for distributed traces)

                k3d / Kubernetes (phase 2: gateway/router/mock-worker control plane;
                Metal workers stay native, exposed via host endpoints)
```

## Components

| Component | Language/stack | Role |
| --- | --- | --- |
| `apps/gateway` | Rust, Axum | OpenAI-compatible edge; SSE passthrough; cancellation propagates to workers (client disconnect → upstream abort); Prometheus metrics |
| `apps/loadgen` | Rust, Tokio/reqwest | Open-loop arrivals + closed-loop concurrency cap; client-side TTFT/ITL/TPOT/E2E; JSON+CSV artifacts |
| `crates/scheduler` | Rust | worker registry; routing policies incl. prefix-affinity (KV-cache-aware routing, simplified) |
| `crates/streaming` | Rust | incremental SSE parser (spec-correct across chunk boundaries) |
| `crates/metrics` | Rust | request records → percentiles/throughput aggregation |
| `crates/workload` | Rust | scenario TOMLs, deterministic prompt synthesis, arrival processes |
| `workers/*` | Python / C++ | native macOS model servers (MLX-LM, vLLM-Metal, llama.cpp) + dependency-free mock |
| `deploy/compose` | Docker | OTel Collector, Prometheus, Tempo, Loki, Grafana |

## Design rules

1. **Model processes stay native on macOS.** Metal/unified-memory access is
   not equivalent inside Linux containers on Apple Silicon. Infrastructure
   (metrics, tracing, dashboards) is containerized — that's the control
   plane; the workers are the "accelerator nodes."
2. **Everything speaks the OpenAI chat API.** Any worker is swappable behind
   the same loadgen/gateway; comparisons are apples-to-apples.
3. **Measurement happens at the client.** TTFT/ITL are what the user
   experiences; server-internal counters (when available) are secondary.
4. **One repo, small slices.** Each phase (benchmarks → routing → tracing →
   k8s → failure injection) produces a working, demonstrable system.

## Failure-path contract (gateway)

```text
browser disconnect
       ↓ gateway body stream dropped (hyper)
       ↓ OnDropStream::drop → outstanding--, active--
       ↓ upstream reqwest stream dropped → connection aborted
       ↓ worker sees cancelled request → generation stops
```

No tokens are generated for nobody.
