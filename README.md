# inference-lab

**A production inference control-plane lab.** A Rust gateway (admission
control, calibrated scheduling, cache-aware routing, functional worker
readiness) proven under real Apple-Silicon inference, then extended into
Kubernetes operations, a simulated datacenter control plane, and a
source-level comparison against llm-d, Dynamo, vLLM, GPU Operator,
dcgm-exporter and Slurm.

**Results first:**

* discovered MLX long-context saturation and silent-wedge behavior under
  concurrent load
* built token-budget admission control that turned the same saturation
  workload from **0/16 delivered → 16/16 delivered**
* measured cache-locality economics (warm TTFT ×24.7) and the cold-burst
  trap, then built load-gated cache routing with learned cost calibration
  (routing prediction MAE **−37%**)
* protected mixed interactive/batch workloads that otherwise wedged
  llama-server for hours (interactive TTFT p50 118 s → 47.7 s)
* built Kubernetes worker discovery, functional readiness, graceful
  draining, queue-signal autoscaling, and a k3d→native-Metal bridge
* simulated heterogeneous H100/L40S/A10 placement (TP groups, VRAM
  accounting, KV bounds) and GPU failure drills (XID/thermal/OOM) with
  differentiated control-plane responses
* mapped the whole design against production systems at source —
  [architecture-mapping.md](docs/architecture-mapping.md)

Models and workers run **natively on macOS** (Metal GPU, unified memory);
infrastructure services run in Docker. That split mirrors a real datacenter's
control-plane vs accelerator-worker distinction — see
[docs/production-gpu-migration.md](docs/production-gpu-migration.md) for what
maps to what.

## Quickstart

```bash
brew install llama.cpp                     # once (optional; mock worker needs nothing)
make setup                                 # toolchain checks + workspace build
cargo test                                 # Rust gateway + loadgen

make up                                    # Prometheus/Grafana/Tempo/Loki/OTel
#   → Grafana http://localhost:3000  Prometheus http://localhost:9090

make smoke                                 # terminal 2: MLX-LM 0.8B on :8082
make bench S=smoke                         # terminal 1: loadgen vs the worker

# full studies: matrix of runs (warmup + cooldowns + aggregate artifacts)
make worker-mlx                            # terminal 2: 9B-4bit on :8082
make gateway                               # terminal 3
make matrix MATRIX=mlx-prompt-concurrency  # terminal 1: 16 measured runs
make analyze STUDY=<results dir printed>   # tables + SVG charts

# the centerpiece experiment: the 8K×c8 saturation collapse, unbounded vs
# admission-controlled (~55 min, real GPU; docs/results/admission-saturation.md).
# Artifacts land in benchmarks/results/<study>/ with a provenance manifest.
make experiment-admission-saturation

# automated failure-injection suite (mock worker, no GPU needed)
make failcheck

# the six worker-lifecycle acceptance scenarios: silent wedge, connection
# failure, empty completion, mid-stream stall, restart/recovery, drain
make lifecycle-check

# phase 5: the control plane under kubernetes (k3d) — discovery, readiness,
# draining, queue-signal autoscaling, native-Metal bridge; minutes, mock workers
make k8s-up        # once: build images + create cluster + deploy
make k8s-check     # six acceptance scenarios (~3 min)
make k8s-demo      # the six-beat milestone demo (1.8 min)

# phase 6: datacenter control-plane SIMULATION — heterogeneous GPU fleet,
# placement/memory estimation, mock-DCGM telemetry (simulation="true"),
# node lifecycle + failure exercises; plus a Slurm batch-side lab (fake GRES)
make fleet-check   # six fleet scenarios (<1 min)
make slurm-up && make slurm-demo   # sinfo/sbatch --gres/squeue/drain
```

Swap workers freely — they all speak the OpenAI API on known ports:

| Worker | Port | Start |
| --- | --- | --- |
| MLX-LM | 8082 | `make worker-mlx` (9B-4bit default, `make smoke` for 0.8B) |
| llama.cpp | 8081 | `make worker-llama` |
| vLLM-Metal | 8083 | `make worker-vllm` (first: `cd workers/vllm-metal && uv sync`) |
| mock-gpu | 8090 | `make worker-mock` (failure injection) |
| gateway | 8080 | `WORKERS=mlx=http://127.0.0.1:8082,llama=http://127.0.0.1:8081 make gateway` |

## The gateway, in one screen

```text
request → estimate tokens → admission control → scheduler → worker → stream
```

- **Estimation**: words×3/4, calibrated online from server usage chunks.
- **Admission control** budgets *estimated work* (prefill tokens, decode
  tokens, milliseconds — from the adaptive service model), queues what
  doesn't fit in a work-accounted FIFO, and answers `429 + Retry-After`
  once the estimated wait crosses the overload threshold. Off by default;
  `ADMIT_MAX_ACTIVE_INPUT_TOKENS=…` turns it on.
- **Scheduling**: least-outstanding, prefix-affinity, or ECT — score each
  worker by *remaining active work + this request's service time*, where the
  service-time model (fixed + ms/token prefill + ms/token decode) is learned
  online from the worker's actual behavior. Unhealthy workers get no traffic.
- **Observability**: every request emits a Tempo trace
  (`admission → queue.wait → scheduler.select → worker.request →
  waiting_for_first_token → streaming`), log lines correlated by trace id in
  Loki, and Prometheus metrics including the TTFT decomposition
  (gateway / admission wait / worker connect / worker first-token).

See [docs/architecture.md](docs/architecture.md) for the full picture and
[docs/results/](docs/results/) for measured studies.

## Interview demo kit

Minute-scale, deterministic demo flows and the eight-number results page:

- [docs/interview-demo.md](docs/interview-demo.md) — demo modes, talk
  tracks, `make demo-metal | demo-wedge | demo-admission | demo-k8s |
  demo-placement | demo-fleet-failure | demo-slurm` (each < 3 min, no
  `demo-all` by design), and frozen fallback transcripts
- [docs/results/interview-scorecard.md](docs/results/interview-scorecard.md)
  — the results slide: eight verified numbers with their frozen artifacts
- [docs/interview-walkthrough.md](docs/interview-walkthrough.md) — the
  10-minute three-incident narrative

## Layout

```text
apps/          gateway (Rust/Axum: admission+scheduler+telemetry proxy) · loadgen
crates/        protocol · scheduler (service model + policies) · streaming (SSE) ·
               metrics (TTFT/ITL/TPOT aggregation) · workload (scenarios)
workers/       mlx-lm · llama-cpp · vllm-metal · mock-gpu
benchmarks/    scenarios · matrices · failures (automated injection suite) ·
               results (gitignored) · reports
deploy/        compose (observability stack) · k8s/helm (phase 4)
observability/ otel · prometheus · tempo · loki · grafana provisioning
experiments/   admission (saturation A/B) · batching · prefix-cache · kv-cache ·
               quantization · speculative · streaming
docs/          architecture · benchmark-methodology · failure-testing ·
               production-gpu-migration · results (study writeups)
```

## Roadmap

- [x] Repo + Rust workspace (gateway, loadgen, protocol/scheduler/streaming/metrics/workload)
- [x] Observability stack (Prometheus, Grafana, Tempo, Loki, OTel Collector)
- [x] MLX-LM + llama.cpp + mock workers runnable
- [x] Experiment runner: matrix manifests + analyzer (benchmarks/matrices/)
- [x] Concurrency × prompt-length study on MLX-LM (docs/results/) — found the
      saturation knee and the 8K×c8 collapse
- [x] Gateway OTel tracing → Tempo trace per request (with streaming lifecycle)
- [x] Prometheus metrics: TTFT decomposition, workload accounting, service model
- [x] Adaptive ECT scheduling + worker health
- [x] Admission control: work-budgeted queueing, overload 429, cancellation
- [x] Failure-injection suite (`make failcheck`)
- [x] Admission-vs-unbounded saturation study (docs/results/admission-saturation.md):
      0/16 unbounded vs 16/16 admission-controlled — tag `phase-3-admission-control`
- [x] Experiment provenance manifests (benchmarks/manifest.py): git/machine/mlx/model
      quant + exact args, written next to every experiment's artifacts
- [x] Worker lifecycle controller (crates/worker-lifecycle): four health dimensions,
      functional-probe READY gate, per-reason escalation, recovery hysteresis,
      generation epochs, drain — `make lifecycle-check` (tag: phase-4D)
- [x] Prefix-cache efficiency study (docs/results/prefix-cache-efficiency.md):
      warm reuse erases prefill (TTFT ×24.7 at c1), savings/token ×270 from c1→c8,
      and the cold-population trap: same-prefix bursts serialize on LCP slot
      affinity (253 s vs 19 s all-miss) — tag `phase-4b-prefix-cache`
- [x] Cross-runtime comparison at matched model size — tag `phase-4-runtime-comparison`
- [x] Load-gated prefix-aware routing + persistent calibrated cost model
      (tags `phase-4b2-prefix-routing`, `phase-4b3-…`)
- [x] Serving interactions: contention curves, marginal-cost gate, service
      classes (tag `phase-4c-serving-interactions`)
- [x] k3d control plane: kubernetes worker discovery, lifecycle-driven
      readiness, graceful draining, queue-signal autoscaling, native-Metal
      bridge — `make k8s-check` (tag `phase-5-k3d-control-plane`), demo:
      `make k8s-demo` (docs/results/phase5-demo/)
- [x] Datacenter control-plane simulation (phase 6): heterogeneous GPU fleet
      (H100×2/L40S/A10 labels), memory-estimator placement gate
      (reject-before-dispatch, TP groups, KV bounds), mock-DCGM exporter
      (simulation="true"), node lifecycle (provision/ready/cordon/drain/
      maintenance/failed/recovering), failure exercises (XID, thermal, OOM)
      — `make fleet-check`; Slurm batch-side lab with FAKE GRES GPUs
      (`make slurm-up` / `slurm-demo`) — tag `phase-6-datacenter-control-plane`
- [x] Phase 7 — architecture/source study against production systems
      (llm-d, NVIDIA Dynamo, vLLM, GPU Operator, dcgm-exporter, Slurm):
      one execution-path trace each, mapped back to inference-lab in
      [docs/architecture-mapping.md](docs/architecture-mapping.md) with
      final synthesis table; two tiny adoptions implemented (exploration
      probability on inferred cache affinity; scaler startup guard),
      two interface contracts defined (runtime scheduler-state, node
      conditions) — tag `phase-7-production-architecture-mapping`
- [ ] Phase 8 — consumable accelerator placement accounting (the one
      candidate kept from the Phase-7 review): per-GPU reservation ledger
      (reserved VRAM, allocation lifecycle, atomic TP reservations) —
      turns the PlacementGate from filter + estimator into filter +
      estimator + accounting. Everything else stays frozen.

**Portfolio entry points:**
[docs/portfolio-architecture.md](docs/portfolio-architecture.md) (the
one-page narrative and evidence tiers) ·
[docs/interview-walkthrough.md](docs/interview-walkthrough.md) (three
incidents, 10 minutes) ·
[docs/architecture-mapping.md](docs/architecture-mapping.md) (production
comparison)

## Scope: measured vs simulated

Everything under [x] above was **measured or exercised locally** on this
Mac (M5, 24 GB unified memory) and in its k3d cluster. **NVIDIA-specific
measurements — CUDA kernel performance, HBM bandwidth/behavior, NVLink/NCCL
scaling, and real DCGM telemetry — are outside the locally validated
scope.** Phase 6's GPU fleet, its DCGM-shaped metrics
(`simulation="true"`), and its Slurm GRES resources are explicitly labeled
simulations of the *control plane*: the scheduling, placement, lifecycle,
and failure-response logic is real code with real tests; the accelerators
behind it are not. A real dcgm-exporter or GPU node drops in later without
changing a dashboard, a metric name, or a line of scheduler logic.
