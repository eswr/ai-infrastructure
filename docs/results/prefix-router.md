# Load-gated prefix routing — reusing cache when it pays, spreading when it doesn't

*Phase 4B.2. Raw artifacts:
`benchmarks/results/prefix-router-20260926-003834/` (16 legs × loadgen
records, gateway `/metrics` scrapes, routing-decision log extracts, 1 Hz
llama.cpp `/slots` occupancy traces, provenance in `lifecycle.json`) and
`benchmarks/results/prefix-router-twoprefix-v2-20260926-053655/` (the A/B
experiment); curated evidence in
[`prefix-router-20260926-003834/`](prefix-router-20260926-003834/);
economics tables in
[`prefix-router-20260926-003834/router-economics.md`](prefix-router-20260926-003834/router-economics.md).*

## Question

> **When the gateway owns slot assignment, which routing policy reuses
> cache when reuse pays — and refuses it when load makes reuse the slower
> choice?**

Phase 4B found the trap this question comes from: llama.cpp's native LCP
slot selection makes cache locality *absolute*, so a cold same-prefix c8
burst serialized onto the busy hot slot — **253 s TTFT p50 vs 19 s for
the all-miss baseline**. Caching was never the problem; *cache locality
without load-aware routing* was. 4B.2 makes the gateway the scheduler and
prices the tradeoff explicitly.

## Method

One `llama-server` (Qwen3.5-9B Q4_K_M, ctx 98304, `-np 8`) modeled by the
gateway as **eight schedulable cache domains** (`WORKER_SLOTS=8`, one
`WorkerHandle` per slot, `id_slot` + `cache_prompt` injected per
dispatch). Four policies × four cache topologies:

| Policy | Decision |
| --- | --- |
| `native` | no `id_slot` — llama.cpp keeps its own LCP choice (the 4B baseline) |
| `least-work` | lowest estimated work, caches ignored (token-aware LB baseline) |
| `prefix-only` | largest reusable prefix wins, load be damned (the explicit hotspot) |
| `cache-aware-ect` | **the feature**: per slot, `estimated_ttft = remaining_work + fixed + (prompt − reusable) × prefill_ms_per_tok`, pick the argmin |

Workload per leg: 8 requests × (8192 shared + 128 unique nominal tokens),
out 256, burst, c8, streaming, via the gateway. Topologies set how many
slots hold the shared prefix before the measured run: **all-cold** (0),
**one-warm** (slot 0), **four-warm** (0–3), **all-warm** (0–7). Warmups
are `loadgen --pin-slot i` one-shots *through the gateway* with the
measured run's seed (byte-identical prefix), so the router's cache model
learns the topology from real completions. Every leg: cache reset (process
restart) → functional gate → warmup → measured run → post-run probe →
evidence capture.

The router's knowledge is an explicit per-slot, single-entry
**cache-investment state** — `ABSENT → POPULATING (a request carrying the
prefix is in flight; joining means waiting it out) → RESIDENT (the slot
holds the prefix)`. Reuse below `SCHEDULER_MIN_CACHE_SAVINGS_MS` is
priced as absent. Selection and work reservation are atomic, so a burst
prices earlier arrivals in. Every decision exports its full candidate
table (estimated TTFT / queue / savings / match tokens per slot), a
decision reason, and — when available reuse was refused — a bypass record
counted in `scheduler_cache_affinity_bypassed_total{reason}`. The model's
prediction is tested against reality per request
(`scheduler_prediction_{,abs_}error_seconds`). This is the same
prefix-affinity-with-load-gate tension production routers (llm-d, vLLM
production stack) solve across machines — reproduced here on one 24 GB
Mac with slots as logical workers.

## Results — the 4 × 4 matrix

TTFT p50 is the headline; OK is delivered/8. All values from the loadgen
client; reuse is the runtime's own `cache_n` telemetry.

| Topology | Policy | OK | TTFT p50 | ITL p50 | Out tok/s | Reused tok/req |
|---|---|---:|---:|---:|---:|---:|
| all-cold | native | 8/8 | 237.9s | 468ms | 3.4 | 0 |
| all-cold | least_work | 8/8 | 388.8s | 468ms | 2.7 | 0 |
| all-cold | prefix_only | 8/8 | 281.6s | **93ms** | 4.3 | 9377 |
| all-cold | cache_aware_ect | 8/8 | 335.5s | 430ms | 2.6 | 0 |
| one-warm | native | 8/8 | 285.3s | 553ms | 3.0 | 1455 |
| one-warm | least_work | 8/8 | 299.5s | 560ms | 2.6 | 2783 |
| one-warm | prefix_only | 8/8 | **231.6s** | **101ms** | **3.9** | 10766 |
| one-warm | cache_aware_ect | 8/8 | 410.4s | 110ms | 2.5 | 10252 |
| four-warm | **native** | 8/8 | **42.6s** | 560ms | 3.5 | 7021 |
| four-warm | least_work | 8/8 | 236.4s | 310ms | 3.6 | 11196 |
| four-warm | prefix_only | 7/8 | 222.2s | 344ms | 3.0 | 11132 |
| four-warm | cache_aware_ect | 8/8 | 219.6s | 284ms | 4.0 | 11196 |
| all-warm | native | 8/8 | 124.7s | 557ms | 6.3 | 9804 |
| all-warm | least_work | 8/8 | 252.8s | 335ms | 2.9 | 11195 |
| all-warm | prefix_only | **5/8** | 72.0s | 477ms | 1.4 | 11131 |
| all-warm | **cache_aware_ect** | 8/8 | **42.4s** | 458ms | **6.4** | 11195 |

Full tables (TTFT p95, ITL p99, stalls, completion spread, per-slot
occupancy, per-leg bypass/prediction accounting):
[`router-economics.md`](prefix-router-20260926-003834/router-economics.md).

## What the experiment found

**1. The load gate is real, legible, and it fires exactly where 4B
needed it.** In the all-cold burst — the population transient that gave
4B its 253 s — cache-aware ECT priced the POPULATING slot honestly and
**refused it seven times** (`prefill_pressure_overrode_affinity`,
bypass reason `prefill_pressure`), spreading instead; the decisions and
their numbers are in the
[decision-log excerpt](prefix-router-20260926-003834/routing-decision-excerpt.txt).
"Locality was available, and the router ignored it on purpose" is now a
measured quantity, not a story: opportunities and attributed bypasses per
leg land in `scheduler_affinity_opportunities_total` and
`scheduler_cache_affinity_bypassed_total`.

**2. The unexpected systems finding: prefill on Metal anti-scales across
slots, so "spread the load" is a trap in the all-cold regime.** With eight
11.4K-token prefills to do, deliberate spreading (least_work, one per
slot, **388.8 s**) was *worse* than native's accidental pile-on
(237.9 s). Slot-parallel prefill on this GPU collapses aggregate
throughput — the 8-way all-warm warmups crawled at ~110–140 tok/s
aggregate vs ~200+ tok/s for one prefill alone, and a two-way warm took
~130–196 s per prompt vs ~55 s solo. Prefill is a compute wall here, not
a parallelizable queue; decode is the opposite (it batches beautifully).
The ITL data says the same thing from the client side: pile-on legs
streamed at 93–110 ms ITL p50 while spread legs stalled at 430–560 ms —
every foreign-slot prefill walls the shared batch. So in the cold regime
*no* policy is good, and the router's refusal to join population costs
nothing: the independent prefill it chooses is equally slow.

**3. When warm capacity exists, affinity dominates — and the all-warm
cell is the phase's clean win.** cache-aware ECT delivered the best cell
in the study: **42.4 s p50, 8/8, 6.4 tok/s** — ×6 faster than least_work
(252.8 s) in the same topology and ×2.9 faster than native, with full
reuse (11,195 tok/req) and the best worst-stall in the matrix (11.9 s).
four-warm showed the same physics from the other side: llama.cpp's own
LCP across four warm slots (native) hit **42.6 s** — the
spread-across-warm-capacity ideal — while the gateway policies that
over-concentrated onto fewer warm slots paid ~220 s. And prefix_only
shows what unconditional affinity costs at saturation: it piled all eight
onto slot 0 and **lost 3 of 8 requests** in all-warm (transport errors
under full contention). Reuse valuable, hotspot fatal — in one table.

**4. The binding constraint is cost-model calibration, and the study
measures its own model.** The prediction-error table is the phase's most
consequential artifact: MAE 170–360 s with bias ≈ −MAE — the model
*under*-predicts TTFT by 3–10×. The mechanism is identified: the seeded
decode rate (25 ms/est-token, from unloaded single-stream estimates)
under-prices *loaded* decode by ~10–13× (a 256-token decode behind a
contended-slot chain costs 60–110 s, not 6.4 s), so every queued join
looked ~7 s instead of ~95 s and the gate closed too late — in one-warm,
cache-aware ECT joined all 8 requests onto slot 0 (8 × 7.3 s estimated <
52.3 s cold; reality ~95 s per chain step → 410.4 s, the worst warm leg).
The gateway *does* refit prefill cost online from completions (visible in
the two-prefix leg, where the estimated cache savings tripled mid-leg as
the rate absorbed real timing), but a burst is decided entirely on seeds
— completions arrive too late to help it. Production directions this
points at: persist calibrated parameters across restarts, pre-calibrate
with a cheap probe wave, or make the decode rate load-aware. The policy
shape was right more often than its arithmetic.

**5. The experiment caught a bug in its own model — the strongest
possible argument for decision telemetry.** Two-prefix v1 exposed a
learning flaw: the *populating* request completes as a miss
(`cache_n = 0`), and the original model recorded "slot holds 0 reusable
tokens" — conflating *what this request reused* with *what the slot now
holds*. One warm slot became invisible 7 seconds after being populated
while its twin (still POPULATING) looked like a full match. Fixed in this
phase: a completion now marks the slot as *holding* its full prompt
(`RESIDENT`), and the request's observed reuse is kept as telemetry only,
with a regression test stating the distinction. The pinned-request
decision table now also exports pre-dispatch state, and the two-prefix
warmup warms A and B with their own seeds.

**6. The two-prefix A/B experiment — the miniature multi-tenant cluster
works.** With prefix A resident on slot 0 and prefix B on slot 1 and
4+4 interleaved requests arriving (v2, fixed model, 8/8 delivered,
healthy):

```text
B requests: 4/4 → slot1 (cache_hit_cheapest), full reuse each (~11.1K tokens)
A requests: 2/4 → slot0 (cache_hit_cheapest), full reuse
            1   → slot0, partial reuse (4.1K — mid-contention match)
            1   → cold slot, spilled (2 attributed prefill_pressure bypasses)
```

Each prefix rode its warm capacity; when a warm slot saturated, the
router **spilled to cold capacity instead of starving** — the property
that keeps a viral context from taking its tenant down. This is
prefix-aware routing with a load gate doing exactly what llm-d describes,
on one machine.

**7. Operational lesson: functional probes need load-awareness.** With
the default 10 s probe timeout, a slot busy with an 8K prefill cannot
answer, and `FunctionalProbeTimeout` is an *immediate*-quarantine class —
one 808 s eight-way populate flapped a slot through 14
quarantine/recovery generations. Slot legs now run with
`FUNCTIONAL_PROBE_TIMEOUT_S=120`; the production answer is probe only
idle slots (or queue probes behind work) so readiness gating never
punishes a busy-but-healthy worker.

## Honest accounting

* Single runs per cell, sequential over ~4.5 h on one machine: thermal
  drift is real (the worst warm leg ran right after prefix_only's
  440 s sustained load). Within a topology, all legs share the protocol,
  so policy comparisons stand; absolute cross-topology times carry that
  drift.
* Four requests across the 16 legs (all in `prefix_only` pile-on legs)
  failed on upstream transport errors under full contention — reported,
  not scrubbed; they are themselves evidence of what unconditional
  affinity does at saturation.
* The occupancy traces sample 1 Hz and the `/slots` endpoint of
  llama.cpp 0.4.1 exposes only `is_processing` — enough to show *where*
  requests ran (native all-cold spread; prefix_only pinned one slot at
  100 %), not cache state.
* `cache_n` counts cache positions and can exceed the API prompt size;
  economics here use per-request observed reuse as reported, with the
  gateway capping at the reported prompt.

## Framing the 4B result correctly

Phase 4B is not "prefix caching makes cold traffic slower". It is:
**unbounded cache affinity caused a cache-locality hotspot**. Warm reuse
itself was the best thing measured in 4B (×24.7 TTFT at c1) and is again
here (the 42 s cells). The failure mode is specifically
*locality-without-load-awareness during population* — and 4B.2 is the
fix shape: price reuse, count refusals, attribute reasons.

## What this sets up

* **Phase 4C — batching × cache × admission:** the routing equation now
  prices per-slot TTFT; the open question is how continuous batching
  changes the denominators (does joining an efficient batch elsewhere
  beat waiting for a warm slot?). The decision table + prediction-error
  machinery built here is the measurement apparatus for it.
* **Model calibration persistence** (finding 4) is the highest-leverage
  router upgrade before any of that: the gate is only as good as its
  per-token costs.
* **Load-aware probing** (finding 7) belongs with the 4C lifecycle work.
