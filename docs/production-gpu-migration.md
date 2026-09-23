# Production GPU migration

How this local implementation maps to a real datacenter — what transfers,
what doesn't, and what to learn next when GPU access exists.

| Local (M5, 24 GB unified) | Production GPU cluster |
| --- | --- |
| vLLM-Metal | vLLM (CUDA) |
| Metal | CUDA |
| unified memory | HBM (per-device) + host RAM |
| local process worker | GPU pod (GPU Operator managed) |
| mock-gpu workers | real GPU nodes |
| gateway router | llm-d / NVIDIA Dynamo (KV-aware routing) |
| llama.cpp (GGUF/Metal) | TensorRT-LLM / SGLang |
| k3d | EKS / GKE / bare-metal K8s |
| Prometheus + host metrics | Prometheus + DCGM exporter |
| MLX/Metal profiling (Instruments, mlx.core.synchronize timing) | Nsight Systems |
| local network (host.docker.internal) | NVLink / InfiniBand / RoCE |
| single node | disaggregated prefill/decode, multi-node KV transfer |

## Concepts that transfer directly

- continuous batching effects on TTFT vs throughput (same scheduler physics)
- prefix caching economics (same hit/miss math)
- client-side latency methodology (identical)
- routing policies (least-outstanding, ECT, prefix-affinity — same inputs,
  same tradeoffs)
- streaming + cancellation correctness (identical HTTP semantics)
- observability architecture (OTel/Prometheus/Grafana unchanged)

## Concepts that don't transfer

- Metal kernel-level tuning ≠ CUDA kernel tuning (different kernels, same
  *methodology*)
- unified-memory KV sizing ≠ HBM + host-offload tiering
- single-node queueing ≠ distributed KV-aware scheduling at cluster scale
- host.docker.internal ≠ RDMA transport (latency and bandwidth class differ
  by orders of magnitude)

## Reference systems to read

1. [vllm-metal](https://github.com/vllm-project/vllm-metal) — the runtime this
   lab benchmarks on Apple Silicon
2. [vLLM](https://github.com/vllm-project/vllm) — scheduler, PagedAttention,
   benchmark suite (`benchmarks/`)
3. [llama.cpp](https://github.com/ggml-org/llama.cpp) — low-level GGUF/Metal path
4. [MLX-LM](https://github.com/ml-explore/mlx-lm) — Apple-native model execution
5. [vLLM Production Stack](https://github.com/vllm-project/production-stack) —
   K8s/Helm deployment patterns
6. [llm-d](https://github.com/llm-d/llm-d) — distributed inference on K8s
   (prefix-aware routing, disaggregated serving)
7. [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — datacenter-scale
   orchestration (Rust core)
8. [OpenTelemetry Demo](https://github.com/open-telemetry/opentelemetry-demo) —
   instrumentation patterns
