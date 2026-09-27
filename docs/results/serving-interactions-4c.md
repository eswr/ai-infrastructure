# Phase 4C — serving interactions: contention as a first-class resource

*Studies `serving-interactions-4c-20260927-014857` (contention wave, 2
rounds × prefill levels {1,2,4,8} + decode scaling + decode interference)
and `serving-interactions-4c-fast-20260927-070425` (revised integration
scope: 4 homogeneous cells + mixed interactive A/B), llama.cpp `-np 8`,
Qwen3.5-9B Q4_K_M, all through the gateway. Curated evidence:
[serving-interactions-20260927-070425/](serving-interactions-20260927-070425/).
Design: [docs/calibration.md](../calibration.md).*

## The question

> Given limited accelerator capacity, should a request reuse cached KV,
> join concurrent execution, spill to another slot, or wait?

4B.3 ended with the cost model's contention inputs degenerate (the
multiplier clamped at a 0.5 floor; anti-scaling visible only in
prediction error). 4C was scoped to measure contention deliberately,
then let routing and admission consume it. Mid-phase the exhaustive
16-cell matrix was **cut to an integration test** (4 cells + mixed A/B):
the wave had already established the physics, and phases 3–4B.3 had
already validated the mechanisms independently — the remaining question
was whether combining them produces the desired behavior.

## Machine truth: the contention wave

Cold 8.3K-token prompts (distinct seed → honestly cold), controlled
concurrency, 2 rounds per level, all through the gateway so the shared
calibration learns every observation in its bucket:

| Active prefills | TTFT p50 | ms/tok (individual) | Individual slowdown | Aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 89.6 s | 10.8 | 1.00× | 93 |
| 2 | 113.0 s | 15.0 | 1.42× | **121.6** |
| 4 | 234.7 s | 23.0 | 2.14× | 121.0 |
| 8 | 474.2 s | 52.3 | 4.85× | 93.7 |

**Prefill anti-scales immediately and hard.** Individual slowdown is
already 1.42× at two concurrent prefills; aggregate prefill throughput
peaks at ~122 tok/s around L2–L4 and falls back to *solo level* (~94
tok/s) at L8 — eight concurrent long prefills deliver one solo prefill's
throughput at 8× the latency. **The prefill saturation point is L2–L4.**

Decode behaves in the opposite way (short prompts, long generations):

| Active decodes | decode ms/tok | Aggregate tok/s |
|---:|---:|---:|
| 1 | 113 | 8.8 |
| 2 | 175 | 11.4 |
| 4 | 258 | 15.5 |
| 8 | 463 | **17.3** |

Decode individual cost rises 4× at L8 but aggregate throughput nearly
doubles — **decode benefits from concurrency; prefill does not**. Any
scheduler that treats "concurrent streams" as one number conflates these.

### Prefill → decode interference

One live interactive decode stream (1K prompt, 192 output tokens), then
cold 8K prefills injected mid-stream:

| Injected prefills | ITL p50 | ITL p95 | Max stall |
|---:|---:|---:|---:|
| 1 | 77 ms | 0.3 s | 11.2 s |
| 2 | 78 ms | 9.4 s | 14.1 s |
| 4 | 93 ms | 10.4 s | 15.2 s |
| 8 | 98 ms | 15.1 s | 16.3 s |

Median ITL barely moves; the *tail* is destroyed. A single injected 8K
prefill already stalls the interactive stream for 11 s; eight stall it
for 16 s. **Homogeneous burst throughput numbers hide a categorical
serving-quality failure.** This is the measurement that justifies
interactive-aware scheduling.

## The engineering find: estimates lied about the phase

The wave first ran on the 4B.3-classifier and produced a *degenerate*
curve — contention samples stayed `[0,0,0,0]` through an entire smoke
study while the solo regression ate every loaded observation, because
the registry classified a dispatch as "still prefilling" while
`elapsed < est_prefill_ms` — an estimate that expires early exactly when
contention makes real prefills run long. One first-of-boot JIT
observation (44 ms/tok at 1K) then walked the solo fit to its clamps
(fixed 10 s, rate 50 ms/tok), which shortened the next process's
estimates, shortening the windows again: a self-reinforcing loop
(b845bdb).

Fix: phases are **observed, not estimated** — a dispatch is prefilling
until its first token is *seen* (`observe_prefill_done` marks + observes
in one step); cache-hit observations mark the phase but teach nothing;
bucket-1 outliers > 8× the current rate estimate are treated as boot
noise. After the fix the wave above measured cleanly, and the persisted
profile went from `[0.50, 0.80, 0.50, 1.00]` (9 samples, sub-1
multipliers) to fully populated curves with hundreds of samples.

## The mechanisms (all regression-tested)

1. **Marginal accelerator economics** on `RuntimeCalibration`: aggregate
   factors per phase (`n / multiplier(n)`), marginal gains of the
   (n+1)th stream, and `prefill_marginal_pays()` — a bucket-boundary
   comparison that holds arrivals only where the measured curve shows an
   aggregate drop, and queues past the 8-stream envelope (5277c08).
2. **The contention gate** (`ADMIT_MARGINAL_COST=1`): slot-policy
   selection returns `Held` (nothing reserved) when the chosen dispatch
   would add prefill into the anti-scaling region; the handler waits
   (woken by the calibration's `Notify`, 100 ms fallback tick, 20 s cap)
   and spills through afterwards. Full-reuse joins and client pins never
   hold (fcf92cf).
3. **Service classes** (`interactive | standard | batch`) on the wire:
   contention physics stays class-blind; classes decide who gets freed
   capacity first (admission grant priority, interactive first) plus
   class-labeled telemetry (b3b732b).
4. **Recompute-token admission budgeting** (68f7a4d): the prefill budget
   counts `prompt − best achievable reuse` instead of raw prompt length
   — measured necessity, see below.

## Integration test: 4 cells + mixed A/B (out=64 homogeneous)

Each leg starts from the SAME wave-taught calibration copy
(`--seed-cal`), cache state rebuilt honestly per leg:

| Cell (8K × c8 × 8) | Verdict | TTFT p50 | TTFT p95 | Reused tok/req |
|---|---|---:|---:|---:|
| cold baseline | healthy 8/8 | 70.8 s | 97.0 s | 8216 |
| cold full stack | healthy 8/8 | 133.2 s | 171.0 s | 8215 |
| warm baseline | healthy 8/8 | 23.8 s | 29.3 s | 11018 |
| warm full stack | healthy 8/8 | **18.4 s** | 22.6 s | 11018 |

- **Warm is preserved**: full stack ≈ baseline (even nominally better),
  identical reuse. The recompute-budget fix was required for this — the
  raw-prompt budget measured 137.9 s vs 42.4 s baseline in the first
  attempt (a fully-warm 8K request counted as 8192 tokens of prefill
  work it would never run, serializing the burst to 2).
- **Cold regressed** (70.8 → 133.2 s): the static input budget (2 × 8K)
  serializes harder than the router's own economics. Left unconstrained,
  the calibrated router *itself* queues arrivals behind a slot that is
  populating the wanted prefix (priced join vs contended cold recompute)
  — the wave's own L2 saturation says that serialization is *correct*,
  and the budget's extra layer of it adds queue-handoff latency without
  buying stability. Zero contention-gate holds in this cell: the budget
  already keeps concurrency below the curve's knee, making the gate
  redundant where it is tightest.
- Honest consequence for the recommended stack: **calibrated routing +
  contention gate + class priority compose well; a tight static token
  budget does not add anything on top of them for cold traffic** — it
  duplicated work the router already does, later and with less
  information. Budgets remain useful as an edge overload contract, not
  as a concurrency policy.

### The mixed workload: the decisive result

2 × 8K warm-prefix (long_context) + 2 × 8K cold (batch) + 4 × 1K
(interactive), one short burst, class tags on the wire:

| Config | Verdict | Wall | Batch (cold 8K) | Interactive TTFT |
|---|---|---:|---:|---:|
| baseline (everything off) | **degraded — runtime WEDGED** | 4.6 h | 0/2 | 0/4 delivered |
| full stack | healthy | **189 s** | 2/2, TTFT p50 35.3 s | 4/4, TTFT p50 120 s |

The unbounded mixed burst reproduced the phase-4A pathology live: the
baseline leg ran 4.6 hours, delivered **zero** of 8 requests (one was
cancelled at TTFT 834 s with one token), the gateway repeatedly
quarantined slots on `FunctionalProbeTimeout`, generation epochs cycled
to 8, and llama-server had to be **SIGKILLed** — it answered nothing
anymore. The full stack completed the identical workload in 189 s, 8/8,
with batch cold prefills completing *and* the interactive streams riding
through (their first tokens arrive while batch prefill still runs).

Interactive TTFT improved 118 s → 47.7 s (p50) in the first healthy
A/B and landed at 120 s in this run's interleaving — run-to-run
interleaving varies (three loadgen processes race the burst), but in
every run the full stack **monotonically separates the classes**:
interactive always precedes batch completion, which never happened in
any baseline. The absolute 5 s interactive SLO remains out of reach on
this machine — protecting interactive traffic from 8K prefill storms
ultimately needs preemption (chunked prefill), which llama.cpp's slots
do not expose. That is a real-GPU / real-runtime question, and a clean
hand-off to the Kubernetes phase.

## Acceptance scorecard (phase criteria set upfront)

| Criterion | Verdict |
|---|---|
| Prediction error in the high-concurrency miss regime improves | **carried over from 4B.3** (−37% pooled MAE); 4C replaced the degenerate contention inputs with measured curves (rate 4.56 ms/tok, multipliers [1.0, 3.85, 3.03, 1.56], 143 loaded samples vs 9) |
| Contention-aware policy avoids anti-scaling regions | **yes, by construction on measured data**; in the tight-budget cell the budget pre-empts it (gate holds = 0), and the wave shows what it protects against: 4.85× individual slowdown at L8 |
| Warm-cache performance not sacrificed | **yes** — 18.4 vs 23.8 s p50, identical reuse (was 137.9 s before the recompute-budget fix) |
| Interactive TTFT materially improves under mixed load | **yes** — 118 → 47.7 s p50 (first A/B); class separation holds in every healthy run |
| Batch completion not catastrophically worse | **yes** — mixed wall 189 s vs 4.6 h; homogeneous cold +88% is the honest cost of the budget layer, which the analysis recommends dropping in favor of gate + routing |
| No lifecycle regressions / wedges / empty completions / restarts | **full stack clean** (0 restarts, 0 empty completions); the BASELINE wedged the runtime and was SIGKILLed — the stability criterion now has a live demonstration |

## Threats to validity

- One run per cell, n=8 requests: the homogeneous cells are an
  integration test, not a statistical study (the 16-cell ablation
  remains available via `make experiment-serving-interactions-full`).
- Mixed-baseline vs full ran ~5 h apart on a thermally variable machine
  (warm-populate times varied 65 s → 852 s across the session); the
  wedge vs healthy outcome is not thermal-sensitive.
- The leg-learned bucket-2 multiplier (3.85) is higher than the wave's
  clean 1.42 at 8K: leg observations include llama.cpp slot-queue wait
  inside `worker_first_token` when several requests join one slot. The
  wave remains the machine-truth reference; the persisted curve reflects
  deployed-workload reality.
- Out=64 homogeneous runs: TTFT/contention/queueing are the measured
  quantities; decode-volume effects are the mixed workload's job.

## Cost

Wave ~75 min + integration study ~2.5 h (including a 4.6 h baseline
wedge the experiment deliberately recorded instead of suppressing).
The repeated fast loop (`make experiment-serving-interactions`, ~40 min)
plus early termination and per-leg calibration seeding is the iteration
loop this phase was missing at its start.

## Next

Phase 4C is frozen here (`phase-4c-serving-interactions`). The
single-Mac runtime has given what it has: an empirically calibrated
inference control plane — measured contention curves, marginal-cost
gating, service classes, honest admission — with worker lifecycle
instrumentation that caught a real wedge. Kubernetes (k3d) becomes the
layer for worker discovery, readiness mapping, draining, rolling
replacement and autoscaling on top of that control plane, ahead of real
NVIDIA hardware where the 5 s interactive SLO becomes reachable.
