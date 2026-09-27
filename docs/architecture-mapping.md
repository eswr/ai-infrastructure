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

### Session record 2 — NVIDIA Dynamo, control-plane decomposition (read 2026-09-28, repos at HEAD)

Files actually read: `components/src/dynamo/frontend/main.py`,
`components/src/dynamo/planner/__main__.py`, `planner/core/load_scaling.py`
(1129 lines: `LoadScalingMixin`, `_advance_load_{single,disagg,agg}`,
prefill/decode decisions, scaling budgets),
`planner/core/state_machine.py` (`PlannerScalingState`: `begin_tick`,
`observe_worker_counts`, `observe_fpm`, `advance_load`),
`planner/core/types.py`, `planner/connectors/` (observe side: kubernetes,
mdc, global_planner, virtual), `planner/environment/runtime.py` (actuate
side: kubernetes client), `lib/runtime/src/discovery/` (etcd leases, kube,
mock stores), plus the previously-read `lib/llm/src/kv_router.rs`.

**The boundary, precisely.** Dynamo separates THREE control concerns that
inference-lab fuses into one gateway process:

1. *Request routing* — data plane, per request, milliseconds:
   frontend (HTTP) → Rust KV router → worker. State: the KV block index
   (worker-reported, event-fed) + scoring gauges. Never scales anything.
2. *Planning* — control plane, per tick, fleet-wide: a SEPARATE component
   (`planner/`) that observes (connectors: metrics/kubernetes), decides
   (core/state_machine + load_scaling: per-component prefill/decode
   decisions, scaling budgets, `_pending_startup` accounting, SLA vs
   "easy" modes, throughput lower bounds), and actuates
   (`environment/runtime.py` → Kubernetes deployments/scale).
3. *Liveness* — etcd LEASES in `lib/runtime/src/discovery/`: workers
   register with a lease; expiry is automatic. There is no DRAINING/
   MAINTENANCE vocabulary here — liveness is delegated to leases +
   Kubernetes; node maintenance is someone else's problem entirely.

**The six questions:**

1. *Routing vs planning boundary* — exactly where our scaler.py sits
   relative to our gateway: the router never scales, the planner never
   picks workers. Dynamo draws the line as a PROCESS boundary; we draw it
   as a process boundary too (scaler.py), but our gateway also folds in
   the lifecycle-response (fleet controller) that Dynamo leaves to leases
   + K8s.
2. *Router state vs planner state* — fully disjoint. Router: KV block
   index + load gauges, per-request-relevant. Planner: worker counts,
   scaling budgets, pending-startup, perf regressions, decision reasons
   (`_diag_load_reason`). Our equivalent: router holds cache/load state ✓,
   scaler holds its own desired/adopted replicas ✓ — but our fleet
   controller's node gates are read BY the router while being WRITTEN by
   the controller, a shared-state seam Dynamo avoids by not having node
   lifecycle in the request path at all.
3. *Worker capacity & model residency* — residency belongs to the ROUTER
   (KV block index); capacity belongs to the PLANNER (perf model fed by
   `connectors/` runtime config: `max_num_batched_tokens`, `max_kv_tokens`,
   `context_length` — see `connectors/global_planner.py`). inference-lab
   puts both in `PlacementGate`: vram_gb + residency in one check. Same
   inputs, different ownership.
4. *Event-driven vs reconciled* — request routing and worker liveness are
   event-driven (KV events, etcd leases); planning is tick-reconciled
   (`begin_tick` → observe → decide → actuate) with a state machine that
   explicitly BLOCKS new scale-ups while a previous adjustment is still
   starting ("scaling_in_progress, observing only",
   load_scaling.py:63-65) and caps scale-downs with throughput bounds.
5. *Local vs cluster-wide* — router decisions are local to a frontend
   instance (index + gauges; cross-instance sync only for KV events);
   planner decisions are cluster-wide by construction (Prometheus fleet
   view, K8s actuation). Nothing in the request path needs the cluster
   view.
6. *Mapping our three pieces* — `PlacementGate` ≈ router-side filter
   (plus capacity feasibility that Dynamo gets from K8s scheduling for
   free); `NodeLifecycle` has NO Dynamo analogue (delegated to leases +
   K8s + the operator layer — the operator session will cover it); our
   `scaler.py` ≈ the planner, but with ~5% of its machinery (no state
   machine, no budgets, no startup accounting).

**Is inference-lab too centralized?** Partly, and the interesting part is
precise: the routing/planning split EXISTS in inference-lab (gateway vs
scaler.py are separate processes, like frontend+router vs planner). What
we centralize that Dynamo does not is *node/worker lifecycle inside the
request-path owner* — our gateway watches pods, writes placement gates,
drains workers, and answers readiness. Dynamo delegates all of it. At
laptop scale the centralization is justified (one process, one state
truth, no leases to expire wrongly); the seam to preserve is that the
router only ever READS the planner/controller's published gates.

**Matched / diverged / justified updates:**

* Matched: placement before dispatch; explicit worker/resource state;
  runtime health feeding routing (their FPM observations and worker
  counts ↔ our gauge sampler); queue/load as the scaling signal.
* Diverged: lifecycle-in-request-path (ours) vs lifecycle-by-delegation
  (theirs: etcd leases for liveness, K8s for scale, operator for nodes).
* Justified simplification: single-process fleet controller — the drift-
  adoption + counters give observability without a second control loop.

**Adoption candidate (exactly one): the planner's scaling-in-progress
guard + startup accounting.** Our scaler counts `replicas` it asked for,
not replicas that are READY — under sustained pressure it can stack
scale-ups while previous pods are still starting, and scale-down decisions
can race a startup. Adopt `desired_pending` accounting: while
`ready_replicas < desired`, observe only (log the hold) — mirroring
`_load_scaling_blocked`/`_pending_startup`. Accepted: ~15 lines in
scaler.py, prevents scale-up thrash, no new subsystem. **Rejected**:
etcd-lease liveness (k8s pod presence already is the lease in our world)
and splitting the fleet controller out of the gateway (two processes for
one laptop's control plane is the complexity Dynamo earns with real
fleets).

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

### Session record 3 — vLLM v1 in-worker scheduler (read 2026-09-28, sparse checkout at HEAD)

Files actually read: `vllm/v1/core/sched/scheduler.py` (schedule() loop,
allocation-failure preemption, `_preempt_request`, waiting promotion),
`vllm/config/scheduler.py`, structure of `vllm/v1/core/kv_cache_manager.py`
(956 lines) and `block_pool.py` (903 lines).

**The verified step-by-step.** Each `schedule()` iteration (scheduler.py:557):
a per-iteration **token budget** (`max_num_scheduled_tokens` /
`max_num_batched_tokens`); RUNNING requests are served first, each getting
`num_new_tokens` capped by budget, `long_prefill_token_threshold` (chunked
prefill), and encoder budgets; then WAITING requests are promoted while
budget and `max_num_active_reqs` allow. For every request,
`kv_cache_manager.allocate_slots(...)` reserves KV blocks **before** the
step is scheduled; on failure the scheduler preempts the lowest-priority
running request and retries. The step's output is a `SchedulerOutput` the
model runner executes in one batched forward pass.

**The seven questions:**

1. *Which sequences enter the next iteration* — running first (index
   order), each clipped to the remaining token budget; then waiting
   promotions; `num_new_tokens == 0` → `continue` (scheduler.py:724),
   deliberately breaking FCFS so one unrunnable head cannot idle the batch.
2. *Prefill vs decode* — the comment at scheduler.py:560 is the answer:
   "There's no decoding phase nor prefill phase in the scheduler. Each
   request just has `num_computed_tokens` and `num_tokens_with_spec`."
   A "prefill" is simply a request still catching up. inference-lab's
   `active_prefills`/`active_decodes` split is a dispatch-time *estimate*
   of the same distinction; vLLM's version is per-token bookkeeping, not a
   phase label.
3. *Continuous batching in code* — the iteration repeats on a fixed token
   budget and re-forms the batch every step: a running request continues
   with whatever budget remains, a waiting request can join mid-stream,
   and a chunked prefill resumes where it stopped. There is no batch
   boundary to wait for. That is the whole trick.
4. *KV block pressure changes scheduling* — through `allocate_slots`
   returning None and through the chunk caps: KV pressure converts
   directly into smaller `num_new_tokens` (chunking), skipped promotions,
   and preemption.
5. *When a request cannot get enough KV blocks* — the retry loop
   (scheduler.py:750-800) preempts the lowest-priority running request
   (`_preempt_request`: free blocks, `num_computed_tokens = 0`, status
   PREEMPTED, back to waiting) and retries allocation; if the request
   itself is the victim it waits. v1 preemption is **recompute**, not
   swap — prefix caching softens the re-prefill.
6. *Long prefills chunked or preempted — the Phase 4C answer* — YES, and
   explicitly for our failure mode: `long_prefill_token_threshold` "exists
   to stop a long prefill from starving other requests of the token
   budget" (scheduler.py:577-585), with an adaptive variant flooring the
   cap at `max_num_batched_tokens / num_eligible_reqs`. An 8K prefill that
   held the Metal runtime for 40-90 s (our ITL p95 0.3→15.1 s stall) would
   have been sliced into budget-sized chunks with interactive decodes
   interleaved between chunks. llama.cpp and mlx-lm expose no equivalent
   control surface — which is exactly why the gateway needed a contention
   gate instead.
7. *Authoritative state the gateway only estimates* — per-request
   `num_computed_tokens`/`num_tokens_with_spec`, live block tables,
   queue depths, preemption counts, cache hit/eviction events. Everything
   our admission budget, ECT terms, and contention gate *predict* is, one
   layer down, *measured* here every step.

**Matched / diverged / justified updates:**

* Matched: explicit prefill/decode distinction (we model it; they
  dissolve it into per-token catch-up — the distinction survives as
  bookkeeping); KV pressure treated as scheduling state; fairness vs
  throughput as an explicit scheduler concern (their budget/floor knobs ↔
  our contention gate + service classes).
* Diverged: the dispatch boundary. inference-lab estimates accelerator
  contention BEFORE dispatch; vLLM schedules with authoritative per-step
  state AFTER dispatch. Our prefill/decode classification is an estimate
  from a calibration model — vLLM's is per-token truth.
* Justified simplification: the gateway-level contention model was
  necessary because llama.cpp/MLX did not expose a rich internal
  scheduler control surface — vLLM's `long_prefill_token_threshold` is
  the parameterization of the problem we solved one layer up, under a
  runtime that offered no knob at all.

**Adoption candidate (exactly one): a runtime scheduler-state interface
contract — telemetry, not scheduling logic.** A production runtime should
expose, per worker: `running` and `waiting` request counts, KV utilization
(fraction of blocks in use), preemption count, and a prefill-chunk
indicator — machine-readable at `/scheduler_state` (or the equivalent
Prometheus series: vLLM already exports `num_requests_running`,
`num_requests_waiting`, `gpu_cache_usage_perc`, `num_preemptions`). The
gateway's cost model would then consume *runtime-reported* queue state as
a calibration target instead of predicting it from seeds. Implemented
here as the contract's reference half: the mock worker now serves
`GET /scheduler_state` (running/waiting/kv — honest about what a mock
cannot know); the gateway consumption side is deliberately deferred —
wiring per-step runtime truth into ECT duplicates the runtime's own
scheduler, and the contract's value is that it documents exactly what a
real vLLM deployment hands the control plane for free. **Rejected**:
adopting any vLLM scheduling algorithm locally (token-budget iteration,
preemption) — our runtimes give no control surface to drive, and
re-implementing a scheduler that never executes is scope theater.

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

### Session record 4 — GPU Operator, node-as-state-machine (read 2026-09-28, sparse checkout at HEAD)

Files actually read: `controllers/state_manager.go` (1062 lines: node
labels, operand registration, SyncState aggregation),
`controllers/clusterpolicy_controller.go` (Reconcile, `updateCRState`),
`internal/state/manager.go`, `internal/state/state.go` (`SyncState`:
ready/notReady/ignore/reset/error), `internal/state/results.go`
(one NotReady ⇒ aggregate NotReady), `internal/state/dcgm_exporter.go`,
`driver.go` (shared `configurableState` skeleton), plus greps for
unschedulable/taint/condition across controllers/ and internal/.

**The verified path from "GPU node exists" to "GPU consumable":**

1. A node appears with an NVIDIA GPU; the operator's node-labeling marks
   eligibility with its OWN label vocabulary
   (`nvidia.com/gpu.present`, `nvidia.com/gpu.deploy.<operand>=true`,
   `nvidia.com/mig.capable` — state_manager.go:41-82). These are
   *scheduling-for-operands* labels: which DaemonSets should run here.
2. `clusterpolicy_controller.Reconcile` → `state_manager.SyncState` runs
   one `State` per operand (driver, container-toolkit, device-plugin,
   DCGM host monitoring, dcgm-exporter, GFD — mostly a shared
   `configurableState` skeleton rendering per-node DaemonSets). Each
   returns `SyncState`; `Results` aggregates (results.go: one NotReady ⇒
   whole-NodePolicy NotReady). ClusterPolicy `status.state` carries the
   aggregate.
3. **`nvidia.com/gpu` becomes a schedulable resource NOT via the
   operator**: the device-plugin DaemonSet registers with the KUBELET,
   and the kubelet's device manager advertises the extended resource.
   The operator only ensures the plugin runs.
4. Capability labels (`nvidia.com/gpu.product`, counts, driver version)
   are written by GPU-Feature-Discovery — a separate operand DaemonSet —
   again not the operator's controllers.

**The seven questions:**

1. *Node preparation vs device advertisement* — split three ways: the
   operator prepares the node (drivers/toolkit DaemonSets + operand
   labels); the DEVICE PLUGIN advertises devices (via kubelet
   registration → extended resources); GFD advertises *descriptive*
   labels. Preparation, advertisement, and description are three owners.
2. *Which operands a node needs* — decided by node labels + ClusterPolicy
   spec, encoded back as `nvidia.com/gpu.deploy.*` labels, then each
   state's control functions render/patch the operand DaemonSet with node
   affinity onto exactly those labels. Desired state ping-pongs through
   the Kubernetes API, never through an internal list.
3. *Where `nvidia.com/gpu` becomes schedulable* — in the KUBELET, via
   device-plugin registration. The operator's job ends at "the plugin is
   running and healthy on this node."
4. *State in Kubernetes objects vs internal enums* — almost everything is
   K8s-shaped: operand DaemonSets (existence = deployment intent),
   node labels (eligibility + capability), ClusterPolicy
   `status.state` (aggregate ready/notReady). The ONLY internal enum is
   the per-operand `SyncState` — and it is short-lived reconciliation
   bookkeeping, not a lifecycle vocabulary.
5. *Unhealthy GPUs — who reacts* — **the operator does NOT decide
   "GPU unhealthy → stop scheduling."** No unschedulable/taint/condition
   logic exists in its controllers (verified by grep; the only
   NoSchedule reference is a comment). The chain is: device plugin
   unhealthy ⇒ KUBELET withdraws/marks the `nvidia.com/gpu` resources ⇒
   the scheduler stops placing GPU pods there. DCGM exporter = pure
   observability; *consumers* (or a human) turn XID/thermal signals into
   taints/cordons. This is the split the session was designed to find:
   **inference-lab's fleet controller combines responsibilities that the
   production stack deliberately separates** — telemetry (exporter),
   advertisement (device plugin + kubelet), and enforcement (scheduler).
6. *Continuous reconcile vs configure-once* — everything operand-shaped
   is continuously reconciled (DaemonSets re-rendered on node/CR churn;
   per-node state re-synced); nothing is "configured once" — even node
   labels are re-derived. This is reconciliation-over-declaration vs our
   imperative sync loop; same destination, different control philosophy.
7. *Where DCGM fits* — two separate operands: `dcgm` (host monitoring
   DaemonSet, DCGM REST API for the exporter/operator diagnostics) and
   `dcgm-exporter` (Prometheus surface). Pure observability both: the
   operator deploys them, consumes nothing from them for scheduling
   decisions.

**Matched / diverged / justified updates:**

* Matched: node eligibility influences placement (their deploy-labels +
  GFD capability labels ↔ our capability labels + placement gates);
  health/state must affect schedulability; observability and lifecycle
  are separate concerns (their dcgm-exporter deploys but decides nothing
  ↔ our exporter observes while fleet.rs decides).
* Diverged: ONE explicit node state machine (ours) vs state distributed
  across Kubernetes objects, DaemonSets, labels and continuous
  reconciliation (theirs). Our NodeLifecycle declares transitions; the
  operator never declares — it converges desired state and lets K8s
  objects BE the state.
* Justified simplification: the explicit machine makes failure exercises
  deterministic and legible locally (fleet-check's XID → FAILED →
  replace → validate is one line to read); in the operator's world the
  same behavior emerges from reconciliation of half a dozen objects and
  is only legible through their logs.

**Adoption candidate (exactly one): a Kubernetes-shaped condition/label
contract on our node statuses** — not another controller. inference-lab's
node states are an internal enum visible only through /status; giving
each simulated node a `conditions` array (`{type, status, reason,
message}`, K8s condition semantics) — Ready/Draining/Failed with reasons
— makes the fleet legible to any Kubernetes-shaped consumer (dashboards,
alerting, a future real operator) without changing the state machine.
**Rejected**: moving state into actual Kubernetes labels/taints (needs a
controller watching our synthetic nodes — a controller to run a
controller) and adding device-plugin-style advertisement modeling
(duplicates what K8s already does in production).

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

### Session record 5 — dcgm-exporter, telemetry semantics only (read 2026-09-28, sparse checkout at HEAD)

Files actually read: `etc/default-counters.csv` (126 lines),
`internal/pkg/counters/counter_config.go` (258),
`internal/pkg/collector/gpu_collector.go` (1049, label/label-skip paths),
`internal/pkg/collector/xid_collector.go` (84), plus greps across
`internal/pkg/` for XID aggregates and event emission.

**The verified path:** `cmd/dcgm-exporter/main.go` → `internal/pkg/server/`
scrape → collectors built from a counter catalog CSV → per-GPU DCGM field
reads → label/value conversion (`gpu_collector.go:778-794`: `UUID`,
`Hostname`, `Labels` attached per metric; `counter.IsLabel()` entries
become labels instead of series) → Prometheus exposition.

**Answers:**

1. *Default counters* — `etc/default-counters.csv`: 26 enabled, more
   commented; plain `DCGM_FI_…, type, help` rows. Custom sets = a
   different CSV via config; the exporter computes additional
   EXP-prefixed aggregates itself (`DCGM_EXP_XID_ERRORS_COUNT` windowed
   gauge, `DCGM_EXP_XID_ERRORS_TOTAL` counter — via
   `xid_total_collector.go`, disabled by default).
2. *Units* (all in the help strings — the catalog IS the unit
   documentation): FB_USED/FREE **MiB**; POWER_USAGE **W**; GPU_TEMP **C**;
   GPU_UTIL **%**; XID_ERRORS dimensionless "value of the last XID error
   encountered" (a GAUGE of the latest code); PCIe in the default set is
   `DCGM_FI_PROF_PCIE_{TX,RX}_BYTES` in **bytes per second** — not the
   older `DCGM_FI_DEV_PCIE_*_THROUGHPUT` MB/s fields our mock invented.
3. *Unsupported/unavailable fields* — failed field reads are skipped
   (`continue` in the collect loops), not exported as sentinels; the
   metric's existence is availability.
4. *GPU vs workload identity* — GPU identity: `gpu` (index), `UUID`,
   `device`, `modelName`, `Hostname`. Workload identity: a kubeclient
   pod-mapper adds pod/namespace/container labels for GPUs bound to pods,
   and config exposes GPU bind/unbind event watching
   (`appconfig/types.go:202`). Our mock's `worker` label plays the pod
   role.
5. *Severity interpretation* — none in the metric path. The exporter
   publishes the raw latest-XID-code gauge plus computed windowed
   totals; classifying 48/74/79 as "critical" is a CONSUMER's job. Our
   mock also passes the code through uninterpreted — the critical-set
   lives in `fleet.rs` (gateway), which is the correct side.
6. *XID events: metrics, logs, or events?* — metrics (the gauge) and
   computed metrics (EXP counters); no Kubernetes event emission in the
   verified tree. Anything event-shaped happens outside the exporter.
7. *What the mock got right / invented* — right: XID-as-latest-code
   gauge, FB in MiB, gauge/counter typing, label shapes (UUID/device/
   modelName/Hostname), catalog-style help strings. Invented (now
   explicit): `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` in "MB/s" (real default
   catalog uses PROF_*_BYTES in bytes/s — normalized in this commit);
   an actuator half (fault injection) that real dcgm-exporter cannot
   have — it changes no GPU state; and model-aware FB baselines, which
   real framebuffer readings get from actual allocations.

**Measurement / interpretation / policy / action — now explicit:**

| layer | real NVIDIA stack | inference-lab |
| --- | --- | --- |
| measurement | dcgm-exporter (raw DCGM fields) | mock exporter (simulated fields) |
| interpretation | consumers (none built-in) | fleet.rs (XID critical set, thresholds) |
| policy | operator/cluster admins (taints, conditions) | fleet.rs decisions (cordon/fail/quarantine) |
| action | kubelet + scheduler (withdraw resources, stop placing) | fleet.rs + discovery (drain, remove, re-validate) |

**Adoption candidate (exactly one): normalize the mock's metric metadata
against the real catalog** — PCIe fields corrected to bytes/sec semantics,
unit help strings matched to `default-counters.csv`, and a catalog-
provenance comment. Implemented in this commit. **Rejected**: adding
DCGM_EXP-style windowed aggregates and severity classification to the
mock — both belong to consumers, and adopting them would re-fuse the
measurement/interpretation split session 4 established.

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

### Session record 6 — Slurm, the allocation layer (read 2026-09-28, sparse checkout at HEAD)

Files actually read (structure + core paths): `job_test.c` (4093 lines —
header contract, `_get_job_node_req`, `_select_nodes` entry),
`gres_select_filter.c` (1944 lines — consumable GRES selection:
`nonalloc_gres`, per-socket counts, `max_node_gres`),
`src/interfaces/gres.c` (11633 lines — GRES bookkeeping),
`priority/multifactor/priority_multifactor.c`.

**The verified path:** `sbatch` job record → `job_test.c` "Determine if
job can be allocated resources" → `_get_job_node_req` (exclusive vs shared
vs overcommit, from partition `OverSubscribe`/`max_share`) →
`_select_nodes` → topology-aware `eval_nodes` with GRES filtering
(`gres_select_filter.c`: the job's GRES vector vs each node's
NON-ALLOCATED GRES counts) → allocation decrements consumable state →
job/node state updates; priority ordering comes in from
`priority/multifactor` before selection.

**The central comparison answered:** inference-lab's PlacementGate is
closer to a **filter with an estimator**; Slurm is a filter **plus a
consumable-resource accounting system**. The differences:

* Slurm tracks per-node *remaining* resources (allocated vs free) as
  authoritative state that every scheduling decision decrements and every
  job release restores; overcommit is an explicit policy knob
  (`NODE_CR_RESERVED | ONE_ROW | AVAILABLE`). inference-lab's gate reads
  current VRAM capability + one loaded model, but never *reserves* placed
  capacity against future placements — our only reservation is ECT's
  time-based `begin_work` estimate, which is about time, not memory.
* Slurm filters and accounts against TOPOLOGY (sockets, GRES binding);
  our gate is topology-free by design (one simulated GPU per worker).
* Priority ordering sits BEFORE feasibility in Slurm's pipeline (queue
  order → selection), where ours applies admission priority at the queue
  and feasibility at selection — the same two concerns in the opposite
  layering.

**Adoption candidate (exactly one): a per-GPU placement ledger — memory
reservations decremented on dispatch and restored on release.** With
one-model-per-GPU residency it is mostly redundant today (which is why it
is a candidate and not an implementation), but it is the missing piece
that separates "is this request placeable" from "is this fleet
schedulable", and it becomes necessary the moment a simulated GPU hosts
two models. Accepted as a Phase-8 candidate, not implemented now: the
current single-resident-model semantics make the ledger dead weight.
**Rejected**: porting `job_test.c`'s topology/weight machinery — nothing
in a one-GPU-per-worker fleet for it to weight.

## Final synthesis — Phase 7

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

| Area | Production system | inference-lab | Verdict |
| --- | --- | --- | --- |
| Cache/routing | llm-d (EPP filter→scorer→picker, KV block index) | CacheAwareECT + prefix fingerprint + exploration | matched with a different cost basis: affinity vote (theirs) vs time-saved estimate (ours) |
| Planning | Dynamo (planner: observe→decide→actuate) | scaler + fleet controller | mostly matched, more centralized; startup-guard adopted |
| In-worker scheduling | vLLM (per-token budget, preemption-recompute) | pre-dispatch estimates + contention gate | intentionally different layer; their `long_prefill_token_threshold` is the knob our 4C stall needed |
| Node/device lifecycle | GPU Operator (desired-state reconciliation) | NodeLifecycle + fleet gates | simplified but legible; responsibilities production splits are fused here — now documented |
| GPU telemetry | dcgm-exporter (counter catalog, raw exposition) | mock DCGM (simulated fields + actuator) | matched on names/units/labels after normalization; measurement/interpretation/policy/action now explicitly separated |
| Resource allocation | Slurm (cons_tres: filter + consumable accounting) | PlacementGate | filter + estimator, not accounting; per-GPU ledger is the named Phase-8 candidate |

The closing answer to *what matched, what diverged, and why the
simplifications were reasonable* is the table above plus the per-session
records and the matched/diverged/justified section that precedes it.

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
