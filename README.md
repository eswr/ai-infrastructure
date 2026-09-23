# inference-lab

> A production-style LLM inference platform built and benchmarked on Apple
> Silicon: pluggable inference runtimes, OpenAI-compatible streaming, a
> KV-aware Rust gateway/router, a Rust load generator measuring TTFT / ITL /
> TPOT under controlled workloads, failure injection, and a full local
> observability stack.

Models and workers run **natively on macOS** (Metal GPU, unified memory);
infrastructure services run in Docker. That split mirrors a real datacenter's
control-plane vs accelerator-worker distinction — see
[docs/production-gpu-migration.md](docs/production-gpu-migration.md) for what
maps to what.

## Quickstart

```bash
brew install llama.cpp                     # once
cargo build --workspace && cargo test      # Rust gateway + loadgen

make up                                    # Prometheus/Grafana/Tempo/Loki/OTel
#   → Grafana http://localhost:3000  Prometheus http://localhost:9090

make smoke                                 # terminal 2: MLX-LM 0.8B on :8082
make bench S=smoke                         # terminal 1: loadgen vs the worker

# full studies: matrix of runs (warmup + cooldowns + aggregate artifacts)
make worker-mlx                            # terminal 2: 9B-4bit on :8082
make gateway                               # terminal 3
make matrix MATRIX=mlx-prompt-concurrency  # terminal 1: 16 measured runs
make analyze STUDY=<results dir printed>   # tables + SVG charts
```

Swap workers freely — they all speak the OpenAI API on known ports:

| Worker | Port | Start |
| --- | --- | --- |
| MLX-LM | 8082 | `make worker-mlx` (9B-4bit default, `make smoke` for 0.8B) |
| llama.cpp | 8081 | `make worker-llama` |
| vLLM-Metal | 8083 | `make worker-vllm` (first: `cd workers/vllm-metal && uv sync`) |
| mock-gpu | 8090 | `make worker-mock` (failure injection) |
| gateway | 8080 | `WORKERS=mlx=http://127.0.0.1:8082,llama=http://127.0.0.1:8081 make gateway` |

## Layout

```text
apps/          gateway (Rust/Axum router+proxy) · loadgen (Rust load generator)
crates/        protocol · scheduler (routing policies) · streaming (SSE) ·
               metrics (TTFT/ITL/TPOT aggregation) · workload (scenarios)
workers/       mlx-lm · llama-cpp · vllm-metal · mock-gpu
benchmarks/    scenarios (TOML) · matrices (studies) · results (gitignored) · reports
deploy/        compose (observability stack) · k8s/helm (phase 2)
observability/ otel · prometheus · tempo · loki · grafana provisioning
experiments/   batching · prefix-cache · kv-cache · quantization · speculative · streaming
docs/          architecture · benchmark-methodology · failure-testing · production-gpu-migration · results (study writeups)
```

## Roadmap

- [x] Repo + Rust workspace (gateway, loadgen, protocol/scheduler/streaming/metrics/workload)
- [x] Observability stack (Prometheus, Grafana, Tempo, Loki, OTel Collector)
- [x] MLX-LM + llama.cpp + mock workers runnable
- [x] Experiment runner: matrix manifests + analyzer (benchmarks/matrices/)
- [x] Concurrency × prompt-length study on MLX-LM (docs/results/)
- [ ] vLLM-Metal benchmarks (needs `uv sync` in workers/vllm-metal)
- [ ] Cross-runtime comparison (MLX-LM vs llama.cpp vs vLLM-Metal)
- [ ] Gateway OTel tracing → Tempo trace per request
- [ ] k3d control-plane simulation, fake accelerator node labels
- [ ] Failure-injection suite (docs/failure-testing.md)
