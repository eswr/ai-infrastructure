# Admission control vs unbounded admission — the 8K × c8 saturation A/B

The phase-2 study ([docs/results/mlx-concurrency-study.md](../../docs/results/mlx-concurrency-study.md))
ended on a collapse: at 8192-token prompts with client concurrency 8,
**0 of 16 requests produced a single token within 900 s**. mlx_lm.server
has no continuous batching, so 8 concurrent prefills time-slice each
other into nothing.

This experiment is the payoff of phase 3: take the *same pathological
workload* and run it twice through the gateway — once behaving exactly
like phase 2, once with admission control enabled — to show that a
measured service-time model, converted into an admission policy, converts
a collapsed system into a stable bounded one.

## Design

Both runs: MLX-LM `mlx-community/Qwen3.5-9B-4bit` behind the gateway,
prompt 8192 tokens, out 256, burst arrivals, 16 requests, c8, streaming,
unique prompts (prefix-cache-miss by design). `bash run.sh` orchestrates:

| | run A `unbounded-8k-c8` | run B `admission-controlled-8k-c8` |
| --- | --- | --- |
| admission | off (no budgets) | `ADMIT_MAX_ACTIVE_INPUT_TOKENS=8192` |
| overload gate | n/a | `ADMIT_OVERLOADED_QUEUE_S=1200` (queue-don't-reject: worst-case estimated wait ≈ 15 × 64 s ≈ 960 s) |
| client timeout | 900 s (phase-2 conditions) | 2400 s (queue waits are the point) |
| expected | 0/16 first tokens, 2 × 900 s of waste | 1 active prefill at a time, ~16 × 65 s ≈ 17 min, finite TTFT for everyone |

The 8192-token input budget means one 8K prefill runs at a time — the
admission controller serializes exactly the phase that time-slicing
destroyed, while the queue keeps order and books estimated seconds.

## What to look for in the results

- **first tokens**: A → 0; B → 16/16 with finite TTFT.
- **TTFT decomposition** (Prometheus `inference_*` histograms, Tempo
  traces): B's TTFT should split into `admission_wait` (queue position ×
  ~65 s) + `worker_first_token` (~45 s prefill + overhead) — the queue is
  visible instead of being smeared into a 900 s mystery.
- **stability**: B's worker never sees more than one 8K prefill;
  `gateway_worker_remaining_work_seconds` drains smoothly to zero.
- **the trade**: B's total batch time (~17 min) is comparable to A's
  wasted 30 min — but every request completes, and ITL stays healthy.

Artifacts land in `benchmarks/results/admission-saturation-<ts>/`:
per-run JSON/CSV, gateway logs for both runs, and the console summaries
(`run-A-console.txt`, `run-B-console.txt`). Write-up:
[docs/results/admission-saturation.md](../../docs/results/admission-saturation.md).

Run it:

```bash
bash experiments/admission/run.sh          # ~55 min incl. model load
KEEP_MLX=1 bash experiments/admission/run.sh   # leave the worker running
```
