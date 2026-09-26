# Phase 4B.3 — persistent calibration: validation results

*Study `calibration-router-20260926-221552` (8 legs + calibration wave,
all healthy, 8/8 every leg, 0 restarts; llama.cpp `-np 8`, Qwen3.5-9B
Q4_K_M, 8×[8192 shared + 128 unique] est tokens, out=256, c8 burst, via
gateway). Design + schema: [docs/calibration.md](../calibration.md).
Raw study: `benchmarks/results/calibration-router-20260926-221552/`
(gitignored); analysis JSON alongside it under `analysis/`.*

## The question

4B.2 measured its own cost model and found **TTFT prediction MAE
170–360 s with bias ≈ +MAE**: every burst was priced on startup seeds
(25 ms/token decode) while loaded decode really cost 10–13× that, so
the cache-aware load gate opened too late, and every gateway restart
threw the online model away. 4B.3 made the model (a) **persist** across
restarts, keyed by machine × model × runtime × slot count, and (b)
**stop using one decode rate for all load levels** — rates are EWMA'd
per server-wide concurrency bucket (`1 / 2 / 3-4 / 5-8`), with a
measured prefill contention multiplier per bucket.

Acceptance criterion (set before the run): *materially reduce scheduler
TTFT prediction MAE versus the frozen 4B.2 model without degrading the
already-correct routing cases.*

## Verdict

**Accepted.** Every cell improved, nothing degraded:

Prediction MAE by cell (seconds, same metric definition as 4B.2):

| Topology | Policy | 4B.2 MAE | 4B.3 MAE | Δ | 4B.2 bias | 4B.3 bias |
|---|---|---:|---:|---:|---:|---:|
| all-cold | cache_aware_ect | 283.3 | 173.4 | −39% | +283 | +159 |
| one-warm | cache_aware_ect | 337.4 | 101.4 | −70% | +337 | +17 |
| four-warm | cache_aware_ect | 217.3 | 195.0 | −10% | +215 | +35 |
| all-warm | cache_aware_ect | 210.3 | 155.7 | −26% | +210 | +153 |
| all-cold | least_work | 327.0 | 155.3 | −53% | +327 | −117 |
| one-warm | least_work | 220.8 | 67.5 | −69% | +221 | −41 |
| four-warm | least_work | 222.3 | 129.0 | −42% | +222 | +12 |
| all-warm | least_work | 358.1 | 144.1 | −60% | +356 | +144 |
| **pooled** | cache_aware_ect | **250.6** | **158.5** | **−37%** | +250 | — |
| **pooled** | least_work | **288.9** | **126.7** | **−56%** | +289 | — |

The qualitative change matters more than the percentages: 4B.2's bias
was ≈ **+MAE everywhere** — a one-sided model that always under-predicted,
so the load gate *always* opened too late. 4B.3's bias signs are **mixed**
(one-warm cae +16.7 s, all-cold least_work −116.9 s): the systematic
under-pricing is gone; what remains is queue-order variance, not a
one-sided model error.

Routing quality did not degrade — it improved across the board, because
a policy with honest numbers routes better even when the policy itself
is unchanged:

| Cell | 4B.2 TTFT p50 | 4B.3 TTFT p50 | 4B.2 p95 | 4B.3 p95 |
|---|---:|---:|---:|---:|
| all-warm cache_aware_ect (the 4B.2 best cell) | 42.4 s | **25.9 s** | 182.5 s | **26.6 s** |
| one-warm cache_aware_ect (4B.2's worst warm leg) | 410.4 s | **126.9 s** | 696.8 s | 249.7 s |
| four-warm cache_aware_ect | 219.6 s | 75.8 s | 422.0 s | 133.8 s |
| all-cold cache_aware_ect | 335.5 s | 316.9 s | 553.0 s | 488.0 s |
| all-warm least_work | 252.8 s | **26.0 s** | 568.4 s | **26.9 s** |

The one-warm worst case (410 s — over-concentration on the warm slot
under-priced into a pile-on) collapsed to 127 s: the loaded-decode
buckets now close the gate while the queue is still worth joining. And
the all-warm least_work pile-on (253 s) became a full 8-slot spread at
26 s — with honest queue pricing, even the cache-blind baseline policy
naturally spreads onto warm slots and every request gets full reuse
(11195 tok/req observed).

## Where the remaining error lives

The 4B.3 regime split (`scheduler_*_prediction_*_{seconds}{prompt,conc,cache}`
— new this phase) answers "where is the model still wrong?":

- **Cache hits are now accurately priced** — the decision the gate
  actually makes: all-warm cae hit rows MAE **24.9 s** (rel err **0.12**),
  four-warm hits 25.8 s (0.27), one-warm hit 23.3 s (0.20). In 4B.2
  these same decisions carried the full ~300 s under-prediction.
- **The residual is cold recompute under saturation** (cache-miss ×
  conc 5–8: MAE ~225–287 s, rel err ≳ 1). This is queue-order
  stochasticity — which requests spill where, and how the burst's
  compound service interleaves — not a fixable one-rate bias. It is the
  measured-contention / mixed-workload question 4C was carved out for.

## What the machine taught the model

Final persisted profile (`calibration_profile_final.json` in the study
dir; learned across the wave + 8 legs, 101 prefill / 109 decode
observations):

```text
prefill: fixed ≈ 0 ms, rate 11.2 ms/est-tok (solo regression)
contention multipliers: [0.50, 0.80, 0.50, 1.00]  (buckets 1/2/3-4/5-8)
decode ms/tok:         [450, 1151, 1184, 1039]    (samples 20/15/28/46)
```

- **Loaded decode is 2.3–2.6× solo on this workload** (450 → ~1100
  ms/tok), and solo decode itself measured **450 ms/tok — 18× the 25
  ms/tok seed 4B.2 priced with** for 256-token generations. The 4B.2
  seed was not just missing contention; it was wrong about the solo
  rate for this model's generation behavior (Qwen3.5 thinking tokens
  included). Only measured, persisted data could reveal that.
- The prefill contention multiplier learned toward the **0.5 clamp
  floor** at bucket 1: the (rate × multiplier) product self-corrects
  toward the true ~47 s solo 8K prefill, but the decomposition between
  "rate" and "multiplier" is degenerate — the multiplier is not yet a
  trustworthy standalone contention number. Notably, **no bucket
  learned a >1 prefill penalty** in this study: Metal prefill
  anti-scaling across slots (4B.2's 150 vs 200+ tok/s) shows up in
  observed TTFT error rather than in the multiplier. Measured
  multi-slot prefill contention is explicitly 4C scope.
- Decode buckets are non-monotonic (bucket 3-4 ≈ bucket 5-8 ≈ 1.0–1.2
  K ms/tok): buckets sample different queue-order mixes. Accept noise
  for now; the fallback chain (exact → higher → lower → seed) prices
  conservatively.

## Cost and mechanics

- The calibration wave (2 rounds × {solo 1K, solo 8K, 8×8K cold burst},
  distinct seed per run) costs ~30 min of GPU and buys every later leg
  — and every future gateway restart — a measured starting point.
- Persistence worked as designed: each leg's gateway loaded the
  previous legs' profile from `calibration/apple-m5/
  qwen3.5-9b-q4_k_m.gguf/llama-8slots.json` (logged at boot), and the
  periodic save (2 s cadence) captured intra-leg learning.
- New operational surface used here: `inference_active_prefills` /
  `inference_active_decodes` gauges, per-bucket calibration gauges, and
  the regime-split prediction-error families — all scraped per leg into
  `gateway_metrics.json` and turned into the tables above by
  `router_econ.py`.

## Threats to validity

- The 4B.2 baseline numbers are frozen from a different day, not an
  interleaved A/B. Mitigation: `make experiment-calibration-ab` runs the
  same 8 legs with `CALIBRATION_DISABLED=1` (seed-only) on demand; the
  metric definition (histogram family, buckets, scrape point) is
  unchanged between the two runs.
- One study, one model, one runtime — llama.cpp only, per the 4B
  finding that it is the only honest cache runtime here. Bucket
  semantics are runtime-agnostic; CUDA is the later test.
- The wave's out=64 decode observations slightly over-weight
  post-prefill stalls; legs' own 256-token observations dominate by the
  second leg (decode samples 20/15/28/46 at study end).

## Next

Phase 4C (cache × batching × admission) is now unblocked: its
marginal-cost scoreboard (`waiting + recompute + contention + decode
load − cache savings`) has measured contention inputs instead of
constants, and its stability verdicts can read the lifecycle
instrumentation that 4B.3's metrics extend.
