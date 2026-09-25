# Phase 4B.2 — Load-Gated Prefix-Aware Routing

**The question phase 4B left open.** Phase 4B measured prefix-cache
*economics* and found the trap: llama.cpp's native LCP slot selection
makes cache locality **absolute** — a cold same-prefix burst all queued
for the busy LCP-matching slot (`f_sim=1.000`), serializing eight 8K
prefills into **253 s TTFT p50 vs 19 s for the all-miss baseline**. Caching
wasn't the problem; *cache locality without load-aware routing* was.

This phase makes the gateway the scheduler. One `llama-server -np 8`
is modeled as **eight schedulable cache domains** (`WORKER_SLOTS=8`,
one `WorkerHandle` per slot, `id_slot` injected per request) and four
routing policies are compared against exactly the pathological workloads:

| Policy (`ROUTE_POLICY`) | Study name | Decision |
| --- | --- | --- |
| `native` | `native` | no `id_slot` — llama.cpp keeps its own LCP choice (the 4B baseline) |
| `least-work` | `least_work` | lowest estimated work, caches ignored (token-aware LB baseline) |
| `prefix-only` | `prefix_only` | largest reusable prefix wins, load be damned (explicit hotspot) |
| `cache-aware-ect` | `cache_aware_ect` | **the feature**: argmin over the routing equation |

## The routing equation

For every slot, the estimated TTFT an arrival would experience:

```text
recompute_tokens(slot) = prompt_tokens − reusable_prefix_tokens(slot)
prefill_cost(slot)     = fixed + recompute_tokens(slot) × prefill_ms_per_tok
estimated_ttft(slot)   = slot_remaining_work + prefill_cost(slot)
choose                 = argmin estimated_ttft(slot)
```

A cache hit is worth exactly `reusable_tokens × prefill_ms_per_tok` and is
taken **only while** that saving exceeds the wait for the warm slot — the
load gate llama.cpp's LCP selection lacks. Cache knowledge is per-slot and
single-entry (llama.cpp slot caches hold one evolving sequence), with an
explicit investment state:

```text
ABSENT      nothing reusable known
POPULATING  a request carrying the prefix is in flight there — joining
            means waiting out its whole service time (the cold-start
            question: wait for population, or prefill independently?)
RESIDENT    reuse observed from the runtime's own cache_n telemetry
```

The model learns from the gateway's own dispatches **and** from
client-pinned `id_slot` warmers, so external population teaches it
directly (the approximate → precise knowledge progression llm-d describes).

## The 4 × 4 matrix

Four cache topologies × four policies, at c8 / 8 requests ×
(8192 shared + 128 unique tokens) / out 256 / burst / streaming:

| Topology | Warm slots | Purpose |
| --- | --- | --- |
| `all-cold` | 0 | cold-start fan-out (does the router spread or pile on?) |
| `one-warm` | 0 | locality vs parallelism (the phase-4B pathology, instrumented) |
| `four-warm` | 0–3 | partial replication |
| `all-warm` | 0–7 | ideal cache state |

Warmups are `loadgen --pin-slot i` one-shots **through the gateway** with
the same seed as the measured run (byte-identical shared prefix; Rust
StdRng synthesis is not reproducible from Python).

Plus the two-prefix A/B experiment: A and B warm on slots 0/1, then
4+4 interleaved requests arrive — a miniature multi-tenant prefix-aware
cluster. Each prefix should ride its warm capacity; a saturated warm slot
must spill rather than starve.

## What is measured

* Standard client-side: TTFT p50/p95, ITL p50/p99, output tok/s, max
  stream stall, completion spread (loadgen summaries).
* Realized reuse: `timings.cache_n` per request (never inferred from
  latency).
* **Deliberate bypasses**: `scheduler_cache_affinity_bypassed_total`
  by reason (`queue_pressure` | `prefill_pressure` |
  `insufficient_benefit` | `worker_degraded`) against
  `scheduler_affinity_opportunities_total` — "locality was available for
  X% of requests; the router ignored it on purpose for Y%".
* **Prediction error**: per request, `actual_TTFT − estimated_TTFT`
  observed into `scheduler_prediction_{,abs_}error_seconds` (bias / MAE /
  p50 / p95) — the router is a cost model, so the model is tested.
* **Where requests ran**: 1 Hz llama.cpp `/slots` occupancy trace per leg
  (`native` should show the hotspot; the gate should not).
* Every decision exports its full candidate table to the
  `scheduler.select` span and a `routing decision` log line: estimated
  TTFT / queue / savings / match tokens per slot + `decision_reason`
  (`cache_hit_cheapest`, `load_overrode_affinity`,
  `prefill_pressure_overrode_affinity`, `insufficient_benefit`, ...).

## Run

```bash
make experiment-prefix-router            # full 16-leg matrix + two-prefix (~1.5-2 h)
python3 benchmarks/prefix_router.py --smoke   # pipeline validation (~10 min)
python3 benchmarks/router_econ.py benchmarks/results/prefix-router-<ts>
```

Artifacts: `benchmarks/results/prefix-router-<ts>/` (per-leg
`matrix.json`, `gateway_metrics.{txt,json}`, `routing_decisions.log`,
`slots_occupancy.csv`, `slots_snapshot.json`, study `lifecycle.json`,
`memory.csv`), analyzed into `analysis/router-economics.md`.

Report: `docs/results/prefix-router.md`.
