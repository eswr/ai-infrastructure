# inference-lab — the architecture, in one page

A production-style LLM inference control plane, built from first principles
on Apple Silicon, then compared against the production systems that solve
the same problems. One request walks through all of it:

```text
request
  → admission control          does this request enter the system now,
                               or queue / get refused?
  → calibrated scheduler       when and where should it run?
  → cache-aware routing        is there a replica whose KV state makes
                               it the cheapest choice — and is that
                               worth the queueing cost?
  → worker lifecycle           is this worker actually able to serve?
                               (functional readiness, not /health)
  → Kubernetes discovery       which workers exist? what can they do?
  → heterogeneous placement    can this model fit this accelerator?
  → node lifecycle             is this node cordoned, draining, failed?
  → telemetry                  is anyone throttling, erroring, overheating?
  → autoscaling                should the fleet grow or shrink?
```

## How each piece is built

* **Gateway** (Rust, Axum): OpenAI-compatible edge. Token estimation,
  estimated-work admission (interactive/standard/batch priority), LOR/ECT/
  cache-aware-ECT routing over slot-and-capability-aware worker handles,
  instrumented streaming (TTFT decomposition, ITL, stall watchdogs),
  OTel traces + Prometheus on every decision.
* **Worker lifecycle** (Rust crate): four health dimensions
  (process/transport/functional/semantic), immediate vs accumulating
  escalation, recovery hysteresis, generation epochs — a worker that
  answers `/health` but cannot generate never comes back as "the same"
  worker. Node lifecycle above it: PROVISIONING → READY → CORDONED →
  DRAINING → MAINTENANCE → FAILED → RECOVERING, driving per-worker
  placement gates.
* **Scheduler calibration**: the cost model learns from its own misses —
  per-concurrency-bucket decode rates, prefill contention curves,
  persisted and reloaded; prediction error is a labeled metric family.
* **Placement gate**: model-registry memory estimates (weights + KV
  context + overhead × safety margin), tensor-parallel groups, residency
  tracking, refusal-before-dispatch.
* **Fleet controller + mock DCGM**: simulated H100/L40S/A10 nodes,
  DCGM-shaped telemetry (`simulation="true"`), differentiated failure
  responses (XID → fail node; thermal → cordon, keep serving; OOM →
  quarantine worker).
* **Slurm lab**: the batch side — partitions, GRES, drain/resume — on
  fake accelerators.

Everything runs on one MacBook: Metal workers native, the control plane in
k3d, observability in Docker, mock workers as the failure-injection fleet.

## Evidence tiers — what was measured, exercised, simulated, validated

This is the most important distinction in the project. Four tiers,
explicitly:

| tier | meaning | examples |
| --- | --- | --- |
| **MEASURED** | real inference behavior, timed and recorded | Apple Silicon MLX/llama.cpp/vLLM-Metal benchmark studies; saturation collapse (0/16 unbounded vs 16/16 admission-controlled); cache reuse economics (TTFT ×24.7 warm); calibration MAE −37% |
| **EXERCISED** | real control-plane code driven through real Kubernetes, with deterministic assertions | worker/node lifecycle transitions, readiness gating, draining, autoscaling scale-up/down, failure drills (`k8s-check`, `fleet-check`, `failcheck`, `lifecycle-check` — all green, minutes each) |
| **SIMULATED** | the *shape* of a GPU fleet, honestly labeled | H100/L40S/A10 nodes, DCGM-shaped metrics (`simulation="true"`), fake Slurm GRES GPUs |
| **SOURCE-VALIDATED** | design compared against production code at HEAD | llm-d, Dynamo, vLLM, GPU Operator, dcgm-exporter, Slurm — six execution-path traces in [architecture-mapping.md](architecture-mapping.md) |

NVIDIA-specific numbers — CUDA kernel performance, HBM bandwidth, NVLink/
NCCL scaling, real DCGM telemetry — are **out of scope and out of the
claims**. The simulated tiers exercise control-plane logic; they measure
nothing about real accelerators.

## What the production comparison showed

* **Matched**: filter→score→pick routing; placement-before-dispatch;
  queue-signal (never CPU) autoscaling; decision observability as a
  first-class output; cache locality treated as a trade, not a rule.
* **Diverged**: cache truth (we infer residency; llm-d consumes
  authoritative runtime KV events); cost basis (affinity-score blending
  vs our time-denominated estimates); control-plane decomposition (we
  fuse lifecycle response into the request-path owner; Dynamo and the
  GPU Operator split it).
* **Justified simplifications**: single-process fleet controller; one
  explicit node state machine; compile-time policies. Each documented in
  the mapping doc with the production alternative named.
* **Adopted from production, kept tiny**: exploration probability on
  inferred cache affinity (llm-d); scaling-in-progress guard (Dynamo
  planner); runtime scheduler-state contract (vLLM) and Kubernetes-shaped
  node conditions (GPU Operator) — both implemented as honest interface
  stubs.

## Where to look next

* [`../docs/architecture-mapping.md`](architecture-mapping.md) — the full
  production comparison, per-system traces, final synthesis table.
* [`../docs/phase7-reading-plan.md`](phase7-reading-plan.md) — the study
  method, if you want to re-run the comparison against repo HEADs.
* [`../docs/results/`](../results/) — the measured studies with artifacts
  and provenance manifests.
* Milestones: `phase-3-admission-control`, `phase-4-runtime-comparison`,
  `phase-5-k3d-control-plane`, `phase-6-datacenter-control-plane`,
  `phase-7-production-architecture-mapping`.
