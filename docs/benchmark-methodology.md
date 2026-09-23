# Benchmark methodology

## Metrics (all client-side, per request)

| Metric | Definition |
| --- | --- |
| **TTFT** | time from request start to first generated token |
| **ITL** | inter-token latency: gaps between consecutive tokens |
| **TPOT** | mean time per output token over decode: `(e2e − ttft) / (tokens − 1)` |
| **E2E** | request start to response end |
| **request throughput** | completed requests / wall time |
| **output tok/s** | generated tokens / wall time (aggregate) |
| **prompt tok/s** | prompt tokens / wall time |

Report **p50 / p95 / p99**, never just means. Wall time spans first request
release to last completion.

Token counting uses the standard client-side approximation: **one streamed
content chunk = one token**. When a server honors `include_usage`, its
reported counts are captured too (`records[].prompt_tokens` /
`completion_tokens` in the result JSON).

## Arrival model

Open-loop release at the scenario's offsets (burst / constant / Poisson),
combined with a closed-loop concurrency cap (semaphore). This reproduces
both queueing behaviour (Poisson) and saturation sweeps (burst + cap).

## Experiment matrix

| Variable | Values |
| --- | --- |
| Input length | 128 / 512 / 2K / 8K / 16K |
| Output length | 32 / 128 / 512 / 1K |
| Concurrency | 1 / 2 / 4 / 8 / 16 |
| Runtime | mlx-lm / vllm-metal / llama.cpp |
| Prefix cache | off / on (`shared_prefix_tokens`) |
| Warm state | cold (first run after load) / warm (repeat) |
| Streaming | on / off |
| Speculative decoding | off / on (worker flag) |
| Arrival | burst / constant / Poisson |
| Model size | 0.8B (smoke) / 9B-4bit (main) |

Hold everything constant except the variable under test. Model sizes chosen
for 24 GB unified memory: 9B-4bit leaves headroom for KV cache + OS.

## Run hygiene

1. **Warmup**: one unmeasured pass before recording (model load, JIT, page-in).
2. **Seeds**: scenarios are seeded; reruns are reproducible (same prompts,
   same arrivals). Change only via `seed =`.
3. **Repeats**: ≥3 runs per configuration; report medians of the aggregates,
   note variance.
4. **Machine state**: plugged into power, no other GPU users; note `powermetrics`
   style context in the report when in doubt.
5. **Record everything**: model + revision, runtime version (`pip show` /
   `llama-server --version`), scenario file, date, temperature of the room if
   it matters. A benchmark you can't rerun is a rumor.
6. Results land in `benchmarks/results/` (gitignored); curated writeups +
   copies of the JSON go in `benchmarks/reports/<date>-<topic>.md`.

## Pitfalls

- TTFT vs prompt length conflates tokenization + prefill; keep the tokenizer
  fixed (same model family) when scanning.
- Streaming chunk coalescing (Nagle/proxy buffering) can hide ITL — loadgen
  measures what arrives over the socket, which is the honest number for the
  client, but document hops (e.g. through gateway) in the report.
- First-run numbers are always garbage (weights loading from disk). Discard
  cold runs or report them separately as "cold start" experiments.
