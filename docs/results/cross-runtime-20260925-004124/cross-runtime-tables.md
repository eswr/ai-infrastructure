# Cross-runtime comparison — cross-runtime-apple-silicon

- created: `20260925-024135` | legs: mlx-lm, llama-cpp, vllm-metal

## Per-run results

| runtime | prompt | c | ok | delivered | ttft p50 | itl p50 | stall max | spread s | ×sf | tok/s |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx-lm | 128 | 1 | 8/8 | 8 | 1.08s | 50 | 1.45s | 12.5 | 1.93 | 12.9 |
| mlx-lm | 128 | 4 | 8/8 | 8 | 3.14s | 110 | 1.19s | 1.9 | 1.06 | 28.1 |
| mlx-lm | 128 | 8 | 6/8 | 6 | 5.13s | 137 | 811 | 0.1 | 1.00 | 34.6 |
| mlx-lm | 1024 | 1 | 8/8 | 8 | 4.07s | 53 | 899 | 3.5 | 1.20 | 12.8 |
| mlx-lm | 1024 | 4 | 8/8 | 8 | 32.0s | 149 | 1.89s | 2.2 | 1.03 | 12.7 |
| mlx-lm | 1024 | 8 | 6/8 | 6 | 50.8s | 212 | 2.00s | 0.2 | 1.00 | 12.9 |
| mlx-lm | 8192 | 1 | 8/8 | 8 | 44.3s | 68 | 2.00s | 16.4 | 1.27 | 3.7 |
| mlx-lm | 8192 | 4 | 8/8 | 8 | 271.6s | 166 | 193.0s | 140.7 | 1.43 | 2.6 |
| mlx-lm | 8192 | 8 | 6/8 | 0 | n/a | n/a | n/a | 4.0 | 1.00 | 0.0 |
| llama-cpp | 128 | 1 | 8/8 | 8 | 1.14s | 77 | 1.75s | 17.1 | 2.14 | 10.2 |
| llama-cpp | 128 | 4 | 8/8 | 8 | 601 | 169 | 1.01s | 4.0 | 1.08 | 20.1 |
| llama-cpp | 128 | 8 | 8/8 | 8 | 1.37s | 313 | 1.16s | 0.0 | 1.00 | 21.8 |
| llama-cpp | 1024 | 1 | 8/8 | 8 | 6.58s | 88 | 1.45s | 10.2 | 1.40 | 8.0 |
| llama-cpp | 1024 | 4 | 8/8 | 8 | 1.55s | 290 | 2.08s | 17.1 | 1.20 | 10.9 |
| llama-cpp | 1024 | 8 | 8/8 | 8 | 1.35s | 349 | 2.73s | 0.0 | 1.00 | 18.2 |
| llama-cpp | 8192 | 1 | 8/8 | 8 | 51.9s | 90 | 1.54s | 10.3 | 1.14 | 3.2 |
| llama-cpp | 8192 | 4 | 8/8 | 8 | 9.97s | 298 | 2.38s | 25.8 | 1.27 | 9.4 |
| llama-cpp | 8192 | 8 | 8/8 | 8 | 19.0s | 476 | 2.55s | 0.0 | 1.00 | 13.1 |
| vllm-metal | 128 | 1 | 8/8 | 8 | 17.7s | n/a | n/a | 17.2 | 9.30 | 8.7 |
| vllm-metal | 128 | 4 | 8/8 | 8 | 63.6s | n/a | n/a | 58.9 | 4.10 | 6.7 |
| vllm-metal | 128 | 8 | 8/8 | 7 | 57.8s | n/a | n/a | 84.4 | 6.25 | 9.5 |
| vllm-metal | 1024 | 1 | 8/8 | 8 | 13.6s | n/a | n/a | 14.6 | 3.54 | 6.8 |
| vllm-metal | 1024 | 4 | 8/8 | 7 | 56.4s | n/a | n/a | 55.2 | 4.03 | 8.6 |
| vllm-metal | 1024 | 8 | 8/8 | 8 | 66.8s | n/a | n/a | 109.7 | 6.68 | 5.9 |
| vllm-metal | 8192 | 1 | 8/8 | 7 | 47.7s | n/a | n/a | 18.5 | 1.56 | 2.4 |
| vllm-metal | 8192 | 4 | 8/8 | 8 | 158.8s | n/a | n/a | 133.0 | 4.13 | 2.6 |
| vllm-metal | 8192 | 8 | 8/8 | 8 | 168.7s | n/a | n/a | 287.5 | 10.93 | 2.8 |

## Prefill scaling (c=1 fit: TTFT ≈ fixed + rate × prompt)

| runtime | fixed (s) | rate (ms/tok) | points |
| --- | ---: | ---: | --- |
| mlx-lm | -0.51 | 5.46 | 3 |
| llama-cpp | 0.23 | 6.30 | 3 |
| vllm-metal | 13.54 | 4.12 | 3 |

## Concurrency scaling (c8 vs c1 per prompt)

| runtime | prompt | tok/s ×c8/c1 | TTFT ×c8/c1 | stall max c8 |
| mlx-lm | 128 | 2.67 | 4.73 | 811 |
| mlx-lm | 1024 | 1.01 | 12.48 | 2.00s |
| mlx-lm | 8192 | 0.00 | n/a | n/a |
| llama-cpp | 128 | 2.14 | 1.20 | 1.16s |
| llama-cpp | 1024 | 2.27 | 0.20 | 2.73s |
| llama-cpp | 8192 | 4.08 | 0.37 | 2.55s |
| vllm-metal | 128 | 1.09 | 3.26 | n/a |
| vllm-metal | 1024 | 0.87 | 4.90 | n/a |
| vllm-metal | 8192 | 1.16 | 3.53 | n/a |

## Lifecycle, recovery and memory (from runtime_harness.py)

| runtime | startup gen | post-benchmark verdict | /health said | restart needed | peak tree RSS | peak footprint | peak sys used |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx-lm | ok | silent_wedge | health: 200 models: 200 | True | 5372 MB | 18534 MB | 21418 MB |
| llama-cpp | ok | healthy | health: 200 models: 200 (mode=sse) | False | 10325 MB | 15258 MB | 16371 MB |
| vllm-metal | ok | healthy | health: 200 models: 200 (mode=json) | False | 3565 MB | 10 MB | 15481 MB |
