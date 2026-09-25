# Prefix-cache economics — prefix-cache-reuse-20260925-222628

## Central table

| c | Cache | TTFT p50 | TTFT p95 | Out tok/s | Max stall | Spread | Reused tok/req |
| -:|---|---|---|---|---|---| ---: |
| 1 | off (frozen 4A) | 51.9s | 54.2s | 3.2 | 1539 | 10.3s | 0 |
| 1 | cold | 3028 | 22.6s | 8.2 | 1029 | 27.1s | 9740 |
| 1 | warm | 2103 | 6102 | 7.4 | 3978 | 7962 | 11577 |
| 4 | off (frozen 4A) | 9971 | 15.7s | 9.4 | 2381 | 25.8s | 0 |
| 4 | cold | 43.6s | 240.1s | 5.2 | 13.5s | 269.5s | 5564 |
| 4 | warm | 6397 | 10.2s | 9.5 | 2634 | 17.5s | 11641 |
| 8 | off (frozen 4A) | 19.0s | 19.0s | 13.1 | 2549 | 2 | 0 |
| 8 | cold | 253.0s | 385.9s | 4.2 | 12.3s | 11.0s | 0 |
| 8 | warm | 5054 | 5288 | 20.8 | 1125 | 66 | 11578 |

## What the cache bought

| c | TTFT saved (cold−warm) | Warm reused tok/req | ms saved per cached token | Effective prefill tok/req (share of server prompt) |
| -:|---|---|---|---|
| 1 | 925 | 8384 | 0.11 | 0 (0.0%) |
| 4 | 37.2s | 8384 | 4.43 | 0 (0.0%) |
| 8 | 247.9s | 8384 | 29.57 | 0 (0.0%) |

## Memory footprint by checkpoint (vmmap, MB)

| phase | samples | peak footprint | peak RSS |
|---|---:|---:|---:|
| loading | 6 | 3584 | 8975 |
| reset | 30 | 3686 | 9121 |
| cold-c1 | 213 | 4198 | 9291 |
| post_probe | 27 | 10445 | 10945 |
| idle | 3 | n/a | 9226 |
| warm-c1 | 521 | 10138 | 10807 |
| cold-c4 | 296 | 4813 | 9404 |
| warm-c4 | 527 | 10138 | 9793 |
| cold-c8 | 393 | 5734 | 9539 |
| warm-c8 | 406 | 9421 | 9941 |

Reused tokens come from the runtime's own telemetry (llama.cpp timings.cache_n / usage cached_tokens); absence of telemetry is reported as n/a, never inferred from TTFT.
