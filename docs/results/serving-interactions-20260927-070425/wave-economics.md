# Serving interactions — serving-interactions-4c-20260927-014857

Phase 4C study on llama-cpp / Qwen3.5-9B-Q4_K_M.gguf, created 20260927-014857. Question: phase-4C: measure accelerator contention as a first-class resource — prefill slowdown vs aggregate throughput per concurrent level, decode scaling, and prefill->decode interference — before routing/admission consume the curve.

## Contention wave — the machine, measured

### Prefill contention (cold identical prompts, out=1)

| Active prefills | Runs | TTFT p50 | ms/token (individual) | Aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 2 | 89.6s | 10.77 | 93.1 |
| 2 | 2 | 113.0s | 15.02 | 121.6 |
| 4 | 2 | 234.7s | 23.02 | 121.0 |
| 8 | 2 | 474.2s | 52.32 | 93.7 |

**Prefill saturation point: L=4** (first level whose aggregate tok/s stops improving on the previous).

### Decode scaling (short prompts, long decode)

| Active decodes | Runs | decode ms/token (mean) | decode ms/token p50 |
|---:|---:|---:|---:|
| 1 | 2 | 113.4 | 113.4 |
| 2 | 2 | 174.7 | 174.5 |
| 4 | 2 | 258.0 | 258.0 |
| 8 | 2 | 462.9 | 486.4 |

### Prefill → decode interference (one live interactive stream)

| Injected prefills | ITL p50 | ITL p95 | ITL p99 | Max stall | Interactive TTFT | Injected TTFT p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.9ms | 299.9ms | 10803.2ms | 11.19s | 4.54s | 62.82s |
| 2 | 77.5ms | 9391.0ms | 11912.1ms | 14.07s | 4.51s | 83.97s |
| 4 | 93.3ms | 10353.5ms | 12651.9ms | 15.21s | 27.90s | 207.56s |
| 8 | 97.6ms | 15142.9ms | 16148.6ms | 16.30s | 41.60s | 426.79s |

## The learned curve (persisted calibration profile)

```text
prefill solo: fixed 0 ms, 42.01 ms/est-tok (33 samples)
bucket                 1       2     3-4     5-8
prefill mult        1.00    4.21    3.33    2.09
  samples              0      26      42      67
decode ms/tok        339     993    1052     909
  samples             33      25      44      80
```

Client-side aggregate tok/s by level (wave): `{"1": 93.1, "2": 121.6, "4": 121.0, "8": 93.7}`
