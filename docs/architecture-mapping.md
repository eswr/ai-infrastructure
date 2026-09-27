# Architecture mapping — inference-lab vs production systems

> Phase 7 artifact. Claim of intellectual honesty this document makes:
> inference-lab's concepts were built **from first principles, on one Mac**,
> then compared against production implementations. Design-level mappings
> below are asserted from architecture knowledge; **file-level evidence is
> confirmed during the Phase 7 reading plan**
> ([phase7-reading-plan.md](phase7-reading-plan.md)) against repo HEADs
> (paths verified 2026-09-28). Paths drift — the *concepts* don't.

## The mapping table

| inference-lab concept | where it lives here | real-system analogue | where it lives there |
| --- | --- | --- | --- |
| `WorkerLifecycle` (4-dimension health, probe-gated READY, escalation, recovery hysteresis, generation epochs, drain) | `crates/worker-lifecycle` | readiness/health controllers: llm-d EPP scheduler filters; vLLM's per-worker engine health; K8s readiness itself | llm-d `llm-d-router` `pkg/epp/scheduling/` filters; K8s `pkg/apis/core/v1` node/cond conditions |
| `NodeLifecycle` (PROVISIONING/READY/CORDONED/DRAINING/MAINTENANCE/FAILED/RECOVERING) | `crates/worker-lifecycle/src/node.rs` | GPU node maintenance/failure state machines | GPU Operator `controllers/state_manager.go` component states; Slurm node states (`DRAIN/RESUME`, `sinfo`) |
| `PlacementGate` (memory estimator, accelerator match, TP groups, KV bounds, reject-before-dispatch) | `crates/scheduler/src/placement.rs` | accelerator/resource placement | Slurm `select/cons_tres` `job_test.c` (GRES-aware node selection); llm-d EPP filters; K8s scheduler predicates/filters |
| Cache-aware ECT routing (prefix fingerprint, cache-domain investment model, marginal-cost contention gate) | `crates/scheduler/src/lib.rs`, `calibration.rs` | prefix/KV-aware routing | llm-d prefix scorer + KV block index; Dynamo KV router (`kv_router.rs`, radix-tree indexer) |
| Admission controller (estimated-work budgeting, class priority, overload contract) | `apps/gateway/src/admission.rs` | overload/backpressure policy | vLLM v1 `scheduler.py` waiting-queue + `Abort`/preemption semantics; K8s Gateway API inference-extension queue-depth hints |
| Queue-signal autoscaler (est. queue seconds, SLO violation rate — never CPU) | `deploy/k8s/scaler.py` | inference-aware autoscaling | Dynamo planner (`components/src/dynamo/planner/core/load_scaling.py`); K8s HPA/KEDA with custom metrics |
| Mock DCGM (`simulation="true"`, DCGM_FI_* shapes, XID/thermal/OOM injections) | `deploy/k8s/images/mock-dcgm/` | dcgm-exporter | NVIDIA/dcgm-exporter `internal/pkg/collector/`, `etc/default-counters.csv` |
| k3d discovery (pods+endpoints → capability-labeled registry) | `apps/gateway/src/discovery.rs` | Kubernetes endpoint/service discovery | K8s EndpointSlice + controller-manager; llm-d inference pools; GPU Operator node labeling (GFD) |
| Fleet controller (telemetry → node transitions: XID→FAILED, thermal→cordon, OOM→quarantine) | `apps/gateway/src/fleet.rs` | GPU node failure controllers | GPU Operator state manager; DCGM XID semantics (`dcgm-exporter` `xid_collector.go`) |
| Slurm lab (fake GRES `gpu:a100:2`, drain/resume, partitions) | `deploy/slurm/` | GPU GRES scheduling | SchedMD/slurm `src/plugins/select/cons_tres/job_test.c`, `src/interfaces/gres.c` |
| Mock worker (failure injection: wedge, empty completion, stall) | `workers/mock-gpu/` | chaos/fault injection | (no direct analogue — this is the lab's own accelerator stand-in) |

## Trace 1 — one request through llm-d

```text
client → Gateway API (Envoy) → EPP (inference-scheduler)
  → filter plugins → scorer plugins (prefix-aware, load-aware)
  → picker (max-score) → chosen inference-model pod
  → vLLM pod serves; KV events → ZMQ → KV block index
```

**Production files** (llm-d org): `llm-d-router`
`cmd/epp/main.go` → `pkg/epp/scheduling/scheduler.go` (filter→scorer→picker
pipeline) → `pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity/plugin.go`
and `scheduling/scorer/prefix/plugin.go`; KV index in `llm-d-kv-cache`
`pkg/kvcache/kvblock/index.go`, event ingestion `pkg/kvevents/zmq_subscriber.go`.

**Maps to inference-lab**: the EPP filter/scorer/picker pipeline is the
same shape as `Router::select_reserved` (filter by placement/health → price
by ECT/cache → pick). Our prefix fingerprint ↔ their block-hash KV index;
our per-slot cache investment model ↔ their KV event-driven index (they
consume *runtime-reported* KV events from vLLM — we simulate residency).
Their plugin trait boundaries are the main lesson: our policies are
compile-time enum arms; theirs are runtime plugins.

### Session record 1 — llm-d (read 2026-09-28, repos at HEAD)

Files actually read: `pkg/epp/scheduling/scheduler.go`,
`scheduler_profile.go`, `weighted_scorer.go`,
`framework/plugins/scheduling/scorer/prefix/plugin.go`,
`filter/prefixcacheaffinity/plugin.go`, `scorer/loadaware/load_aware.go`,
`scorer/nohitlru/no_hit_lru.go`, `requestcontrol/dataproducer/
{preciseprefixcache,approximateprefix,prefixhash,inflightload,predictedlatency}/`,
`llm-d-kv-cache/pkg/kvcache/{indexer.go,kvblock/index.go,kvblock_scorer.go}`.

**Verified end-to-end path.** `cmd/epp/main.go` → Gateway-API extension
handler → `Scheduler.Schedule` (scheduler.go:73): a *profile handler* picks
profiles; each `SchedulerProfile` runs filters → scorers → picker;
`WeightedScorer` (weighted_scorer.go:17) pairs each scorer with a weight;
`ProcessResults` merges profile results. Prefix truth flows the other way:
data *producers* (`requestcontrol/dataproducer/preciseprefixcache` —
tokenize → hash KV-block keys → index lookup → per-endpoint
`PrefixCacheMatchInfo`) publish into the datalayer, and scorers *consume*
it (`Consumes()` declares `PrefixCacheMatchInfo`).

**Step-by-step vs inference-lab:**

| llm-d step | inference-lab analogue | difference |
| --- | --- | --- |
| `Scheduler.Schedule` profile pipeline (scheduler.go:73) | `Router::select_reserved_inner` | same idea, different implementation — profiles are runtime-configured plugin sets; ours are compile-time policy arms |
| filters (health, utilization, prefix affinity) | `healthy_workers()` + `PlacementGate` | same idea, different implementation — their filters are also plugins; our placement gate is richer on memory/TP, theirs has none at this layer |
| prefix scorer (prefix/plugin.go:120): `matchRatio = matchBlocks/totalBlocks`, optional quadratic absolute-length term | CacheAwareECT's `(prompt − reusable) × prefill_rate` priced into ECT | same goal, different math — they score 0-1 affinity for a weighted blend; we convert reuse into *time saved* inside an estimate. Ours needs the cost model theirs doesn't |
| affinity FILTER (prefixcacheaffinity/plugin.go): threshold 0.80 → "sticky" set; exploration probability skips the gate; no-sticky → keep all; TTFT load gate BREAKS stickiness when `bestStickyTTFT − bestNonStickyTTFT > MaxTTFTPenaltyMs` | our bypass machinery: `scheduler_cache_affinity_bypassed_total{queue_pressure,prefill_pressure,…}` | production feature we lack (partially): an explicit sticky-set gate with exploration + a *measured-TTFT* override. We bypass on *predicted* cost; they override on *observed* TTFT |
| load-aware scorer (loadaware/load_aware.go:96): queue depth → score in [0, 0.5] | least-outstanding + remaining-work ECT terms | same idea, different implementation — theirs is a bounded heuristic score blended by weight; ours is a work estimate folded into completion time |
| picker (`maxscore`, `weightedrandom`) | argmin in policy arms | intentionally simpler on our side |
| KV state: `llm-d-kv-cache` block index (`Lookup(keys, pods)`, in-memory HMA / Redis backends) fed by **vLLM KV events over ZMQ** (`pkg/kvevents/zmq_subscriber.go`) | per-slot cache investment model (ABSENT→POPULATING→RESIDENT) learned from our own dispatches + `cache_n` telemetry | production feature we lack: authoritative runtime eviction events. Our residency is inferred and can be wrong |
| **two producers for the same truth**: `preciseprefixcache` (tokenize+hash exact blocks) vs `approximateprefix` (cheap approximation) | one prefix fingerprint (`DefaultHasher` on the first 4 KB of messages) | production feature we lack: an explicit accuracy/cost dial on cache knowledge |

**The five questions, with evidence:**

1. *How prefix/KV locality enters scoring* — as a normalised affinity
   score (matchRatio, default; optional quadratic absolute-length term,
   `matchLengthWeight`), produced by a data producer, consumed by a scorer,
   then weighted against non-affinity scorers by profile-configured
   weights. Locality is one *vote*, priced in ratio space — not, as in our
   ECT, a time-saved term inside a completion-time estimate.
2. *Where load overrides locality* — in TWO places: the affinity filter's
   TTFT gate (if the sticky set's best *observed* TTFT is worse than the
   non-sticky best by more than `MaxTTFTPenaltyMs`, the gate is broken and
   all endpoints compete — `outcomeLoadOverride`), and the weighted blend
   itself, where load scores pull a cached pod down.
3. *Approximate vs authoritative state* — runtime KV events are the
   authoritative layer (evictions are *reported by vLLM*, not guessed);
   the router keeps an approximate index with two accuracy tiers
   (precise tokenized-hash vs approximate) and treats match info as
   advisory. Nothing in the router pretends to know a pod's cache.
4. *How decisions are observable* — every filter/scorer/picker emits an
   outcome record + OTel span attributes (`outcomeSticky`, `outcomeLoadOverride`,
   `outcomeNoMatch`, `outcomeExploration`), per-extension-point latency
   histograms, and the KV index has instrumented/traced wrappers. Directly
   comparable to our `scheduler_decisions_total`, bypass counters, and
   per-candidate decision tables — this is the closest match in the whole
   phase.
5. *Stale cache knowledge* — three mitigations: exploration probability
   (occasionally ignore affinity to re-discover), no-sticky-keep-all (a
   wrong threshold never empties the pool), and the `nohitlru` scorer (cold
   requests — no hit anywhere — get scored by replica LRU, deliberately
   spreading cache-cold traffic onto least-recently-used pods). Our
   equivalent is the miss-completion demotion fix (52d29d2) and ECT's
   natural spread of colds — weaker, and we learned the cold-serialization
   trap the hard way in 4B.

**Adoption candidate (exactly one): exploration probability on prefix
affinity.** `CacheAwareECT` never re-considers a bypassed affinity decision
at random; adding an `EXPLORATION_PROBABILITY` knob (default ~0.02) to the
slot-cache affinity path would let a wrongly-trusted residency entry heal
itself by occasionally dispatching against it — the exact failure mode of
our inferred (not authoritative) residency. Accepted: ~20 lines in one
policy arm + a bypass counter label, directly improves the control-plane
story (self-healing cache knowledge), no new subsystem. **Rejected**:
converting policies to runtime plugins (framework rewrite, no story gain)
and a latency-predictor data producer (duplicates our calibrated cost
model at 10× scope).

## Trace 2 — one request through NVIDIA Dynamo

```text
client → frontend (FastAPI) → preprocessor → KV router (Rust)
  → worker selection (KV-cache match + load) → vLLM/worker handler
  (separate prefill/decode workers when disaggregated)
  → planner watches load → scales workers
```

**Production files** (ai-dynamo/dynamo): `components/src/dynamo/frontend/main.py`
→ `lib/llm/src/preprocessor.rs` → `lib/llm/src/kv_router.rs` +
`lib/kv-router/src/indexer/kv_indexer.rs` (constrained radix tree of KV
blocks) → `lib/router-plugins/builtin/src/default/{scorer,picker}.rs` →
`components/src/dynamo/vllm/handlers.py`; planner:
`components/src/dynamo/planner/core/load_scaling.py`.

**Maps to inference-lab**: their KV router's radix tree of block hashes is
the production version of our prefix-fingerprint → cache-domain model; their
planner is our queue-signal scaler with a performance model
(`perf_model/`) — the same "learn the service curve, scale on the queue"
idea we do with the calibrated cost model + `estimated_queue_seconds`.

## Trace 3 — one request through vLLM

```text
OpenAI api_server → AsyncLLM → EngineCore → scheduler.schedule()
  → KV cache manager (PagedAttention block allocation, preemption)
  → gpu_model_runner (continuous batching step, TP collectives)
```

**Production files** (vllm-project/vllm): `vllm/entrypoints/openai/api_server.py`
→ `vllm/v1/engine/async_llm.py` → `vllm/v1/engine/core.py` →
`vllm/v1/core/sched/scheduler.py` (`schedule()`: admission, waiting→running
promotion, preemption) → `vllm/v1/core/kv_cache_manager.py` +
`block_pool.py` → `vllm/v1/worker/gpu_model_runner.py`; TP:
`vllm/distributed/parallel_state.py`.

**Maps to inference-lab**: vLLM's scheduler is *intra-worker* batching
(token-budgeted continuous batching, preemption) — the layer below our
gateway. Our admission controller decides *whether* a request enters the
system; vLLM's scheduler decides *when it runs* and *what gets evicted*.
Our contention gate (anti-scaling prefill holds) has no direct vLLM
analogue at this layer because PagedAttention + chunked prefill dissolve
the prefill/decode interference we measured on Metal — that difference is
exactly why our interactive SLO needs chunked prefill on real GPUs.

## Trace 4 — a GPU node through the GPU Operator

```text
ClusterPolicy applied → clusterpolicy_controller → state manager
  → per-component states: driver → device-plugin → DCGM host monitoring
    → dcgm-exporter → GPU-Feature-Discovery labels
  → node gets resource slots (nvidia.com/gpu) + labels → schedulable
```

**Production files** (NVIDIA/gpu-operator): `cmd/gpu-operator/main.go` →
`controllers/clusterpolicy_controller.go` → `controllers/state_manager.go`
→ `internal/state/{manager,driver,dcgm_exporter,dcgm}.go`; CRD:
`api/nvidia/v1/clusterpolicy_types.go`.

**Maps to inference-lab**: our `NodeLifecycle` + discovery labels are a
one-process sketch of what the operator does with CRD-driven component
states. Their "state manager applies components in order and reports one
node-level state" is precisely our PROVISIONING → READY validation ladder.

## Trace 5 — one scrape through dcgm-exporter

```text
Prometheus scrape → HTTP server → collector loop → DCGM (nvml/DCGM API)
  → counters filtered by default-counters.csv → DCGM_FI_* series
  (gpu/UUID/device/modelName/Hostname labels) → XID collector maps
  error codes to DCGM_FI_DEV_XID_ERRORS
```

**Production files** (NVIDIA/dcgm-exporter): `cmd/dcgm-exporter/main.go` →
`internal/pkg/server/` → `internal/pkg/collector/gpu_collector.go` +
`xid_collector.go`; counters from `etc/default-counters.csv` via
`internal/pkg/counters/counter_config.go`.

**Maps to inference-lab**: our mock exporter copies the metric-name and
label contract (`gpu`, `UUID`, `device`, `modelName`, `Hostname`) and the
XID-as-latest-code semantics, with `simulation="true"` on everything. The
unit lesson we hit live (DCGM reports framebuffer in **MiB**) is a real
dcgm-exporter contract.

## Trace 6 — one job through Slurm

```text
sbatch --gres=gpu:1 → slurmctld job manager → priority ordering
  → select/cons_tres job_test.c: evaluate candidate nodes against
    GRES/memory/CPU requests → allocate → slurmd launches
  → node drain/resume (scontrol) → accounting
```

**Production files** (SchedMD/slurm, `master`): `src/slurmctld/` job manager
→ `src/plugins/select/cons_tres/job_test.c` (+ `gres_select_filter.c`) →
GRES bookkeeping `src/interfaces/gres.c` → priority
`src/plugins/priority/multifactor/priority_multifactor.c`.

**Maps to inference-lab**: `job_test.c` is the production version of our
placement gate — node-by-node "can this request's resource vector fit?"
with GRES filtering. Our drain/resume lab exercises map 1:1 onto
CORDONED/DRAINING.

## Where inference-lab matched, diverged, and which simplifications were justified

*Updated after each Phase-7 session. Current entries: llm-d (session 1).*

**Matched (same concept, independently derived):**

* Filter → score → pick as the routing pipeline's three phases; and — the
  strongest match — *decision observability as a first-class output*
  (their per-outcome filter records + spans vs our bypass counters +
  per-candidate decision tables).
* Cache-aware routing priced as a *trade*, not a rule: they weight an
  affinity score against load scores; we fold cache savings into a
  completion-time estimate and refuse reuse below a floor. Both systems
  treat locality as advisory, never mandatory.
* Load-awareness as queue state, and autoscaling on queue signals rather
  than CPU (Dynamo's planner and our scaler agree on this from first
  principles).
* Cold-cache handling as a first-class case: their `nohitlru` scorer and
  our ECT cold-spreading both exist because of the same failure — warm
  affinity serializing cold traffic onto the wrong replicas.

**Diverged (and why):**

* *Source of cache truth.* llm-d consumes authoritative KV events from the
  runtime (vLLM reports block eviction over ZMQ); we infer residency from
  our own dispatches. Ours is approximate by construction — defensible
  only because our mock worker *is* the runtime and can be wrong in
  realistic ways (that's the wedge/empty-completion family).
* *Units of routing currency.* They score in normalized affinity space and
  blend with weights; we price in milliseconds and pick the argmin. Their
  weights are operational knobs; our estimates are calibrated against
  measured TTFT (4B.3) with prediction-error telemetry — more machinery,
  but it answers "how wrong was the routing decision" numerically.
* *Scope of the gate.* Their affinity filter is a single threshold gate
  with escape hatches; our placement gate is a resource-feasibility check
  (memory/TP/accelerator). Different problems sharing a pipeline position —
  llm-d assumes capacity exists (K8s scheduled the pods); we simulate the
  capacity decision too.

**Justified simplifications:**

* Compile-time policy arms instead of runtime plugin registries — one
  developer, one process, zero need for third-party policies. Revisit only
  if policies multiply.
* Simulated residency instead of a KV-event consumer — a real event
  consumer needs a real runtime emitting events; on Metal, mlx-lm/llama.cpp
  offer partial telemetry (`cache_n`), which we already consume.
* No flow-control framework: llm-d's flowcontrol tree (fairness, ordering,
  saturation detectors, usage limits) is what our single admission
  controller + service classes collapse into. Their taxonomy confirms our
  one-controller design covers the same decisions at lab scale.

## Cross-cutting differences (the part worth writing home about)

1. **Plugin boundaries.** llm-d and Dynamo expose scheduler policy as
   plugins/filters/scorers/pickers; inference-lab compiles policies as enum
   arms. First-principles version is simpler; production needs the seam.
2. **Runtime-reported KV truth.** We *simulate* residency and prefix
   caches; llm-d/Dynamo ingest KV events from the runtime (ZMQ/HTTP).
   Simulating first made the event schemas legible later.
3. **In-worker batching is the missing layer.** vLLM's continuous batching,
   PagedAttention and preemption live *below* our gateway abstraction —
   on Metal, mlx-lm had none of it, which is precisely why our admission +
   contention gate existed. On CUDA that layer does the heavy lifting.
4. **CRD-driven component orchestration.** GPU Operator orders driver →
   toolkit → device-plugin → monitoring with a single node-level state —
   the production answer to the same problem our NodeLifecycle solves with
   a state machine + labels.
