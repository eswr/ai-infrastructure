# k3d operations (phase 5)

The control plane under Kubernetes. The gateway and the mock workers run as
pods; the real Metal worker stays native on macOS — never force Metal into
Linux containers.

```text
                k3d (cluster "inference-lab")
                 │
          ┌──────▼───────┐   readiness of every worker pod =
          │   Gateway    |   gateway lifecycle state (functional
          │ scheduler /  |   generation probe; /health==200 is
          │ admission /  |   never a readiness signal)
          │ discovery    │
          └──────┬───────┘
                 │ endpoint IPs from pods+endpoints watch
        ┌────────┼────────┐
        │        │        │     mock workers: labels ARE the
     mock-1   mock-2   mock-3    registration (runtime/model);
        │                 │     scaler adjusts replicas
        │            host.docker.internal (hostAlias, render.sh)
        │                 │
     host.docker.internal   native macOS
                          │
                    llama.cpp / MLX (Metal)
                          │
                         M5
```

## Quick start

```bash
make k8s-up            # build images, create cluster, deploy, wait ready
make k8s-check         # six acceptance scenarios, ~3 min, mock workers only
make k8s-bridge-check  # native Metal worker + gateway in k3d, 4 concurrent reqs
make k8s-status        # kubectl get pods -n inference-lab
make k8s-down          # delete the cluster
```

The gateway is reachable on the host at `http://localhost:8080` (k3d
nodePort 30080) — the observability compose stack's existing Prometheus job
(`host.docker.internal:8080`) scrapes the IN-CLUSTER gateway unchanged.

## The four mechanisms

1. **Worker discovery** (`WORKER_DISCOVERY=kubernetes`,
   `apps/gateway/src/discovery.rs`). The gateway polls pods + endpoints
   carrying `inference-lab.io/worker=true`. The pod LABELS are the worker
   registration (`inference-lab.io/runtime`, `inference-lab.io/model`);
   there is no hard-coded worker list anywhere. Poll, not watch-stream: at
   lab scale a 1 s list is faster than the probe cadence that gates
   traffic, with no resourceVersion machinery. `EXTERNAL_WORKERS`
   (`name=url#model`) bridges out-of-cluster workers; discovery never
   removes those it did not register.

2. **Readiness mapping.** The lifecycle controller stays authoritative:
   `READY | DEGRADED` → routable; `STARTING | WARMING | UNHEALTHY |
   RECOVERING | DRAINING | STOPPED` → readiness false. Worker pods'
   readinessProbe hits their own `/readyz`, which is the conjunction of a
   real mini-generation AND the gateway's verdict
   (`/admin/workers/{id}/readiness`, fail-closed). Endpoints are OBSERVED,
   not obeyed: a NotReady pod stays registered so the gateway can keep
   probing it by IP — that is what lets it walk UNHEALTHY → RECOVERING →
   WARMING → READY (generation epoch bump) and re-enter service.

3. **Graceful draining.** `preStop` calls the worker's `/drain`: refuse new
   work, finish active streams, exit at active == 0
   (`terminationGracePeriodSeconds` bounds it). Gateway-side,
   `POST /admin/workers/{id}/drain` does the same for registry entries;
   DRAINING → STOPPED only when outstanding hits 0.

4. **Autoscaling** (`deploy/k8s/scaler.py`, a Deployment — deliberately no
   HPA). Scales `deploy/mock-workers` on the gateway's own queue signals:
   `inference_estimated_queue_seconds` (threshold), `queued` work,
   interactive SLO violations (`inference_interactive_slo_violations_total`,
   rate) scale up; an empty queue for a cooldown scales down. Never CPU.
   Admission must be ON (the deployment sets `ADMIT_MAX_ACTIVE_OUTPUT_TOKENS`)
   because the queue signal only exists when admission tracks queued work.

## Acceptance suite — `make k8s-check`

| scenario | proves |
|---|---|
| worker discovery | registry = kubernetes pods; labels → capabilities |
| readiness false on functional-probe failure | wedge → UNHEALTHY → pod NotReady while /health stays 200; traffic moves |
| worker recovers and re-enters service | unwedge → RECOVERING → generation++ → READY → pod Ready |
| drain removes worker from new traffic | in-flight finishes, zero new work, DRAINING → STOPPED |
| rolling replacement preserves active stream | pod deleted mid-stream; stream completes, 0 client errors |
| autoscaler adds mock capacity under queue pressure | queue signal → replicas up; idle cooldown → back down |

## Phase 6: datacenter control-plane simulation

On top of the phase-5 stack, the cluster now simulates a heterogeneous GPU
fleet — explicitly labeled as simulation everywhere (every DCGM-shaped
series carries `simulation="true"`; the README scope note separates
measured from simulated):

* **6A heterogeneous fleet** (`50-fleet.yaml`): three nodes — 2×H100
  (tp-group `tg-h100-01`), 1×L40S, 1×A10 — registered purely through pod
  labels (`inference-lab.io/node|accelerator|vram-gb|tp-group|gpu-uuid`).
  The native M5 joins via `EXTERNAL_WORKERS;accelerator=apple-m5;vram_gb=24`.
* **6C placement gate** (`crates/scheduler/src/placement.rs`): requests
  carry `placement` requirements (`model_memory_gb` overrides, `context_length`,
  `tensor_parallel_size`, `accelerator`). The gateway estimates
  `weights + KV(context) + overhead` with a safety margin and refuses
  impossible placements BEFORE dispatch (HTTP 422 +
  `inference_placement_rejections_total`): 70B-fp16 fits nowhere
  single-GPU, fits TP-2 on the H100 pair; 70B-q4 never lands on the A10;
  a 400K-context request blows every VRAM budget on KV alone. Residency is
  tracked: an already-loaded model skips the weights term. Capability-less
  workers (the generic phase-5 pool) refuse placement requests — you cannot
  claim a fit you cannot verify.
* **6B mock DCGM** (`deploy/k8s/images/mock-dcgm/`, scraped by the
  observability Prometheus as job `dcgm-sim`, dashboard
  `gpu-fleet-simulation.json`): DCGM_FI_* metric names and label shapes,
  inventory derived from the gateway /status, admin injections
  (`/inject/xid|thermal|oom|clear`).
* **6D node lifecycle** (`crates/worker-lifecycle/src/node.rs`,
  gateway `fleet.rs`): PROVISIONING → READY → CORDONED → DRAINING →
  MAINTENANCE → RECOVERING, FAILED — above the worker machines. Admin:
  `POST /admin/nodes/{id}/{action}`; states drive per-worker PLACEMENT
  GATES (READY open, CORDONED resident-only, rest closed).
* **6E failure exercises** (`make fleet-check`): XID 48 → node FAILED →
  drain → replacement → validation; thermal → cordon + DEGRADED with
  traffic continuing; OOM → immediate worker quarantine → probe recovery.

## Gotchas (all found live)

* `$(POD_NAME)` does NOT expand in probe `httpGet.path` — readiness must be
  pod-local, hence the worker-side `/readyz` consulting the gateway.
* kubelet probe `host:` fields resolve via the NODE resolver, not CoreDNS.
* The scaler must ADOPT externally-set replica counts (drift window) and
  never write back its stale internal count, or it fights `kubectl scale`.
* mlx-lm ≥ 0.31 resolves repo-id model names against the hub PER REQUEST
  (404/crash offline) — `workers/mlx-lm/run.sh` serves from the local
  snapshot path; its `--model` default is addressed as `default_model`.
* `estimated_queue_seconds` is queued-work-based: no admission budgets, no
  signal. `ADMIT_MAX_ACTIVE_OUTPUT_TOKENS` gives the mock pool a real one.
* DCGM FB_USED is MiB; capabilities are GB — normalize before comparing
  (a raw MiB-vs-GB compare quarantined the whole fleet on the first scrape).
* k3d port mappings are fixed at cluster creation — adding a nodePort means
  recreating the cluster (or kubectl port-forward).
* Deployment `spec.selector` must match template label KEYS exactly
  (`inference-lab.io/node`, not `node`).
