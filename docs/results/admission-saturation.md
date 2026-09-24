# Admission control vs unbounded admission — the 8K × c8 saturation A/B

The phase-2 study ([mlx-concurrency-study.md](mlx-concurrency-study.md)) ended
on a collapse: at 8192-token prompts and client concurrency 8, **0 of 16
requests produced a single token within 900 s**. This experiment is the
payoff of phase 3 — the same pathological workload run twice through the
gateway, once behaving exactly like phase 2 (admission off) and once with
admission control on, to show that a measured service-time model converted
into an admission policy turns a collapsed system into a stable bounded one.

**Result: the admission-controlled run completed all 16 requests with
healthy inter-token latency — in less wall time than the unbounded run
wasted on producing nothing.**

| | Run A — unbounded | Run B — admission-controlled |
| --- | --- | --- |
| requests ok | **0 / 16** | **16 / 16** |
| tokens produced | 0 | 4066 |
| wall time | 1800 s (two timeout waves) | **786 s** |
| TTFT p50 | — (no first token ever) | 365 s (finite, queue position + prefill) |
| ITL p50 | — | **63.7 ms** |
| E2E p50 | 900 s → client timeout | 382 s, every stream completed |

- **Worker**: MLX-LM `mlx_lm.server` 0.31.3, `mlx-community/Qwen3.5-9B-4bit`
- **Path**: loadgen → gateway (127.0.0.1:8080) → worker (:8082)
- **Workload** (both runs): prompt 8192 tokens × 16 requests, c8, out 256,
  burst arrivals, streaming, unique prompts (prefix-cache-miss by design)
- **Run B gateway env**: `ADMIT_MAX_ACTIVE_INPUT_TOKENS=8192`,
  `ADMIT_OVERLOADED_QUEUE_S=1200`, `ADMIT_DEGRADED_QUEUE_S=300`
  (queue-don't-reject: the worst-case estimated wait ≈ 15 × 64 s stays
  under the gate, so everything queues and nothing gets 429'd)
- **Run A gateway env**: no `ADMIT_*` (phase-2 behavior)
- **Machine**: MacBook Air M5, 24 GB unified memory, on power; other
  containers idle. Reproduce: `bash experiments/admission/run.sh` (~55 min).
  Raw artifacts: `benchmarks/results/admission-saturation-20260924-214429/`
  (gitignored): per-run JSON/CSV, both gateway logs, worker log, consoles
  (curated copies of the consoles in `run-A-console.txt` / `run-B-console.txt`
  next to this file).

## Run A — unbounded: total collapse, reproduced

16 requests, 0 tokens, wall 1800 s: the first wave of 8 held the worker
for 900 s producing nothing (each client timed out), then the second wave
did the same. `mlx_lm.server` has no continuous batching, so 8 concurrent
8K prefills time-slice each other into nothing — TTFT never arrives. This
is exactly the phase-2 result, now reproduced under the phase-3 gateway
with admission disabled (the A/B is apples-to-apples).

**A worse variant exists.** In the first execution attempt of this study,
the same workload did not merely starve the worker — the Metal runtime
ran out of memory mid-prefill:

```text
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

and the worker was left **wedged**: it answered `/health` but never served
another POST (the failed generation held the server's generation lock).
Unbounded admission doesn't just destroy latency — it can crash the
runtime into a state that passes health checks while serving nothing.
That failure mode is the strongest argument for admitting work by
estimated cost: the harness now restarts the worker between runs and
probes an actual POST before trusting `/health`.

## Run B — admission-controlled: the queue is visible, the worker never collapses

The gateway's 8192-token input budget admits exactly one 8K prefill at a
time; everyone else waits in the admission queue, accounted in tokens and
estimated seconds. The per-request staircase shows the controller working:

| request | TTFT (s) | E2E (s) | tokens |
| --- | ---: | ---: | ---: |
| #4 (first dispatched) | 20.8 | 34.0 | 254 |
| #6 | 56.7 | 69.9 | 254 |
| #7 | 94.2 | 107.9 | 254 |
| #2 | 136.3 | 152.4 | 254 |
| #1 | 184.5 | 200.5 | 254 |
| #0 | 233.2 | 250.2 | 254 |
| #3 | 290.0 | 309.1 | 254 |
| #5 | 355.9 | 373.0 | 254 |
| #8–15 (2nd client wave) | 374–417 | 391–434 | 254–256 |

Readings:

- **Dispatch spacing ≈ 38–47 s** = one full 8K prefill, served at full
  speed instead of eight-way time-sliced. Request #4's 20.8 s TTFT is the
  uncontended prefill + runtime overhead — the `worker_first_token`
  component; everything the other requests wait is *queue*, and the queue
  is measured.
- **TTFT decomposes honestly**: TTFT(k) ≈ k × service_time + prefill. The
  phase-2 mystery ("8K × c8 TTFT = ∞") becomes an accounting identity:
  `admission_wait` (visible, bounded, your queue position) +
  `worker_first_token` (~20 s here, still named honestly — it contains
  runtime queueing + tokenization + prefill).
- **Decode stays healthy**: ITL p50 63.7 ms / p95 84 ms — each stream
  decodes at full speed because prefills no longer interleave.
- **The trade**: run B's TTFTs are terrible by design (queue position ×
  45 s), but finite and *ordered*; run A's were infinite. Batch makespan
  786 s vs 1800 s of pure waste.

## What the live gauges caught along the way

The first run-B attempt deadlocked, and the observability found it: with
the input budget set to exactly the workload's nominal token count,
estimator jitter (`words × 3/4` → 8199 tokens vs budget 8192) made every
request unfittable — active 0, queued 8, worker idle, nobody moving. A
budget that can never fit is a deadlock, so admission now treats the
budget as soft when the machine is idle (bounded overshoot by one
max-size request) and force-grants the queue head if a release would
otherwise grant nothing. (`fix(admission): soft budget when idle`.)

This is the phase-3 thesis working end to end: measure → see the
anomaly in gauges → fix the policy → regression-test it.

## Takeaways

1. **Admission control converts collapse into bounded latency.** Same
   workload, same worker: 0/16 → 16/16, and the controlled run finishes
   the whole batch in less wall time than the unbounded run spent
   accomplishing nothing.
2. **Overload should be explicit.** Queues with token/seconds accounting
   make wait time predictable (and 429-able) instead of smeared into
   infinite TTFT.
3. **Unbounded admission is worse than slow — it can wedge the runtime**
   (GPU OOM → generation lock held → health-passing, work-refusing
   server). Bound what reaches the model.
4. **The queue is observable.** Every wait in run B has a name, a metric,
   and a trace span; nothing in run A did.
