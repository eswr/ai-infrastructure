# Phase 7 — reading plan: production systems, one trace at a time

> Goal: map what inference-lab built to how serious systems do it —
> **reading and mapping, not experiments**. Every repo path below was
> verified against the default branch at HEAD on 2026-09-28; paths drift,
> so re-verify the tree (`git ls-files` / GitHub tree view) before diving.
>
> Method per system (the same trace template every time):
>
> ```text
> request / event → entrypoint → scheduler / controller →
> state transition → worker / node action → metrics / recovery
> ```
>
> Output per system: annotated answers in this file's "record here"
> boxes, a diff to `docs/architecture-mapping.md`, and at least one
> **adoption candidate** (a concrete thing inference-lab could steal).
> Timebox: one focused session per system (~2 h). Read one execution path
> end-to-end rather than exploring broadly.

---

## 1. llm-d — prefix-aware routing as scheduler plugins (session 1)

```bash
git clone https://github.com/llm-d/llm-d-router          # EPP scheduler
git clone https://github.com/llm-d/llm-d-kv-cache        # KV block index
git clone https://github.com/llm-d/llm-d                 # architecture docs
```

Read, in order:

1. `llm-d/docs/architecture/README.md` — the umbrella: inference pools,
   Gateway API + EPP, P/D disaggregation vocabulary.
2. `llm-d-router/cmd/epp/main.go` — entrypoint; note the plugin
   registration list.
3. `llm-d-router/pkg/epp/scheduling/scheduler.go` +
   `scheduler_profile.go` — the filter → scorer → picker pipeline.
4. `pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity/plugin.go`
   and `scheduling/scorer/prefix/plugin.go` — how prefix affinity is scored.
5. `llm-d-kv-cache/pkg/kvcache/kvblock/index.go` and
   `pkg/kvcache/kv_block_scorer.go` — the block-hash index that backs the
   prefix scores; then `pkg/kvevents/zmq_subscriber.go` — how vLLM's KV
   events flow in.

**Trace to write down**: request → EPP HTTP endpoint → scheduler profile →
filter → scorers → picker → pod. Compare each stage against
`Router::select_reserved` (healthy filter → policy scoring → reserve).

**Questions to answer in the mapping doc**:
- Their filter/scorer/picker seams vs our enum-arm policies — what would a
  `RoutingPolicy` trait object need?
- Their KV block index consumes *runtime events*; we infer residency from
  dispatches. What would an `inference-lab` KV-event endpoint on the mock
  worker look like?
- How do they express "SLO-aware" placement (if at all) vs our service
  classes?

## 2. NVIDIA Dynamo — KV router + planner (session 2)

```bash
git clone https://github.com/ai-dynamo/dynamo
```

Read, in order:

1. `components/src/dynamo/frontend/main.py` → `vllm_processor.py` — the
   HTTP entrypoint and how a request becomes a Dynamo `Request`.
2. `lib/llm/src/preprocessor.rs` — where routing decisions begin.
3. `lib/llm/src/kv_router.rs` and `lib/kv-router/src/indexer/kv_indexer.rs`
   (+ `indexer/concurrent_radix_tree_compressed/`) — the KV-block radix
   tree; compare with our `prefix_hash` → cache-domain model.
4. `lib/router-plugins/builtin/src/default/scorer.rs` + `picker.rs` — the
   pluggable policy seam.
5. `components/src/dynamo/planner/core/load_scaling.py` + `core/perf_model/`
   — their autoscaler: signals, decision rule, perf model. Compare with
   `deploy/k8s/scaler.py` + our calibrated cost model.

**Trace to write down**: request → frontend → preprocessor → KV router
scorer/picker → worker handler (`components/src/dynamo/vllm/handlers.py`);
and separately: metrics → planner → scale decision.

**Questions**:
- Their planner's scaling signals vs our `estimated_queue_seconds` +
  SLO-violation rate — same or different? What do they know that we don't?
- Prefill/decode disaggregation: how does `prefill_router/` decide who
  prefills? (Our contention gate was the Metal-era stand-in for this.)
- What does their Rust/Python split teach about our gateway being Rust?

## 3. vLLM — the layer below the gateway (session 3)

```bash
git clone https://github.com/vllm-project/vllm
```

Read, in order (v1 engine):

1. `vllm/entrypoints/openai/api_server.py` → `vllm/v1/engine/async_llm.py`
   → `vllm/v1/engine/core.py` — how an API request becomes engine input.
2. `vllm/v1/core/sched/scheduler.py` — `schedule()`: token budget,
   waiting→running promotion, preemption (`request_queue.py`,
   `async_scheduler.py` alongside).
3. `vllm/v1/core/kv_cache_manager.py` + `block_pool.py` — PagedAttention's
   block allocation: the production answer to KV placement (our
   `context_length` KV estimate is the placement-time shadow of this).
4. `vllm/v1/worker/gpu_model_runner.py` — one batching step; where
   continuous batching actually happens.
5. `vllm/distributed/parallel_state.py` — TP process groups; compare with
   our TP-group placement simulation.
6. Config seams: `vllm/config/cache.py`, `vllm/config/scheduler.py`.

**Questions**:
- What does vLLM's preemption/eviction do to our admission controller's
  assumptions (we budget estimated work; they can revoke running work)?
- Where exactly does chunked prefill live, and why does it dissolve the
  prefill/decode anti-scaling our contention gate models?
- What would `--tensor-parallel-size` change at the *scheduler* level that
  our TP-group placement simulates coarsely?

## 4. NVIDIA GPU Operator — node-as-state-machine (session 4)

```bash
git clone https://github.com/NVIDIA/gpu-operator
```

Read, in order:

1. `api/nvidia/v1/clusterpolicy_types.go` — the CRD: what an operator
   declares about a whole GPU fleet.
2. `controllers/clusterpolicy_controller.go` → `controllers/state_manager.go`
   — how one reconcile drives many component states.
3. `internal/state/manager.go`, then skim `driver.go`,
   `dcgm_exporter.go`, `dcgm.go` — ordering, readiness gating, and how a
   component failure marks the node.
4. Find where GPU-Feature-Discovery labels land on the node object — the
   production version of our capability labels.

**Questions**:
- Their component ordering (driver → toolkit → device-plugin → monitoring)
  vs our PROVISIONING → READY ladder — what are the intermediate states
  and who reports them?
- How does the operator express "node failed, stop scheduling" vs our
  `NodeLifecycle::FAILED` + drain?
- What is `nvidia.com/gpu` extended-resource accounting doing that our
  `vram_gb` placement gate models coarsely?

## 5. DCGM exporter — real telemetry semantics (session 5, half day)

```bash
git clone https://github.com/NVIDIA/dcgm-exporter
```

Read, in order:

1. `cmd/dcgm-exporter/main.go` → `internal/pkg/server/` — scrape serving.
2. `internal/pkg/collector/gpu_collector.go` — how DCGM_FI_* series and
   their labels are produced (compare with our mock's label set:
   `gpu`, `UUID`, `device`, `modelName`, `Hostname`, + our
   `simulation="true"`).
3. `etc/default-counters.csv` + `internal/pkg/counters/counter_config.go`
   — the metric catalog: which DCGM_FI fields ship by default.
4. `internal/pkg/collector/xid_collector.go` + `xid_errors.go` — XID
   semantics: latest-code reporting, severity classification. Compare with
   our `CRITICAL_XIDS = [48, 74, 79, 95]`.

**Questions**:
- Which units does each DCGM_FI_DEV_* field use (we were bitten: FB_USED
  is MiB)? Make a one-table unit reference in the mapping doc.
- How do real XID severities map to our critical/non-critical split?
- What does the real exporter know about *containers/pods* (device
  ownership labels) that our fleet simulates with worker labels?

## 6. Slurm — placement inside cons_tres (session 6, half day)

```bash
git clone https://github.com/SchedMD/slurm
```

Read, in order:

1. `src/plugins/select/cons_tres/job_test.c` — the placement evaluation:
   how a job's TRES vector (CPU/mem/GRES) is tested against nodes. This is
   the production `PlacementGate`.
2. `src/plugins/select/cons_tres/gres_select_filter.c` — GRES-aware node
   filtering; `src/interfaces/gres.c` — GRES bookkeeping (our fake
   `gpu:a100:2` with `File=` device mapping lives here in production).
3. `src/plugins/priority/multifactor/priority_multifactor.c` (+ `basic/`)
   — queue priority; compare with admission's class priority.
4. Skim `src/slurmctld/` node state handling for DRAIN/RESUME semantics.

**Questions**:
- How does `job_test.c` order/break ties between feasible nodes (best-fit?
  first-fit? weight-driven)? Compare with `lor`/ECT.
- What exactly does a drained node refuse vs finish (compare our DRAINING)?
- What does accounting (TRES tracking) assume about *reported* resources —
  the same "capability report" our placement gate requires?

---

## Standing rules for Phase 7

* **Timebox**: one session per system; if a file dive exceeds ~30 min,
  zoom back out to the trace level.
* **Verify paths at HEAD** before reading; the tree above is a snapshot.
* **Every session ends in the mapping doc**: table additions, one adoption
  candidate, one "we were right / we were wrong" note.
* **No new simulation features** during Phase 7 except tiny shims that
  prove a mapping (e.g., a KV-event endpoint on the mock worker) — and
  only if it stays under an hour.
* Keep the README scope note intact: measured ≠ simulated; nothing in
  this phase upgrades a simulation into a measurement.
