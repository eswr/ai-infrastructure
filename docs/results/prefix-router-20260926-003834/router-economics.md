# Prefix-router economics — prefix-router-20260926-003834

## Policy x topology matrix (c8, 8x [8192 shared + 128 unique], out=256)

| Topology | Policy | OK | TTFT p50 | TTFT p95 | ITL p50/p99 | Out tok/s | Max stall | Spread | Reused tok/req |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-cold | native | 8/8 | 237.9s | 429.0s | 468/12.7s | 3.4 | 15.0s | 15.1s | 0 |
| all-cold | least_work | 8/8 | 388.8s | 603.4s | 468/15.3s | 2.7 | 17.3s | 14.5s | 0 |
| all-cold | prefix_only | 8/8 | 281.6s | 426.2s | 93/1847 | 4.3 | 3374 | 324.7s | 9377 |
| all-cold | cache_aware_ect | 8/8 | 335.5s | 553.0s | 430/15.3s | 2.6 | 17.2s | 123.5s | 0 |
| one-warm | native | 8/8 | 285.3s | 496.6s | 553/14.2s | 3.0 | 15.1s | 22.1s | 1455 |
| one-warm | least_work | 8/8 | 299.5s | 519.2s | 560/18.5s | 2.6 | 24.4s | 117.1s | 2783 |
| one-warm | prefix_only | 8/8 | 231.6s | 440.1s | 101/2653 | 3.9 | 8028 | 417.7s | 10766 |
| one-warm | cache_aware_ect | 8/8 | 410.4s | 696.8s | 110/5311 | 2.5 | 19.0s | 685.9s | 10252 |
| four-warm | native | 8/8 | 42.6s | 355.0s | 560/24.1s | 3.5 | 27.2s | 18.1s | 7021 |
| four-warm | least_work | 8/8 | 236.4s | 407.8s | 310/11.7s | 3.6 | 34.7s | 188.1s | 11196 |
| four-warm | prefix_only | 7/8 | 222.2s | 419.5s | 344/8323 | 3.0 | 19.2s | 385.9s | 11132 |
| four-warm | cache_aware_ect | 8/8 | 219.6s | 422.0s | 284/5962 | 4.0 | 14.2s | 310.0s | 11196 |
| all-warm | native | 8/8 | 124.7s | 128.7s | 557/4927 | 6.3 | 17.4s | 6950 | 9804 |
| all-warm | least_work | 8/8 | 252.8s | 568.4s | 335/9793 | 2.9 | 13.9s | 469.3s | 11195 |
| all-warm | prefix_only | 5/8 | 72.0s | 73.0s | 477/8876 | 1.4 | 11.5s | 6639 | 11131 |
| all-warm | cache_aware_ect | 8/8 | 42.4s | 182.5s | 458/6040 | 6.4 | 11.9s | 89.8s | 11195 |

## Locality vs load: was reuse available, and who refused it?

| Topology | Policy | Affinity opportunities | Bypassed (by reason) | Observed reused tokens (total) |
|---|---|---:|---|---:|
| all-cold | native | 0 | 0 | 0 |
| all-cold | least_work | 7 | 0 | 0 |
| all-cold | prefix_only | 7 | 0 | 75019 |
| all-cold | cache_aware_ect | 7 | prefill_pressure: 7 | 0 |
| one-warm | native | 0 | 0 | 11643 |
| one-warm | least_work | 9 | 0 | 22262 |
| one-warm | prefix_only | 8 | 0 | 86130 |
| one-warm | cache_aware_ect | 9 | 0 | 82017 |
| four-warm | native | 0 | 0 | 56167 |
| four-warm | least_work | 12 | 0 | 89565 |
| four-warm | prefix_only | 12 | 0 | 77927 |
| four-warm | cache_aware_ect | 12 | 0 | 89565 |
| all-warm | native | 0 | 0 | 78429 |
| all-warm | least_work | 16 | 0 | 89560 |
| all-warm | prefix_only | 16 | 0 | 55655 |
| all-warm | cache_aware_ect | 16 | 0 | 89561 |

## Was the cost model honest? (estimated vs actual TTFT)

| Topology | Policy | Est TTFT mean | MAE | |err| p50 | |err| p95 | Bias (signed mean) | n |
|---|---|---:|---:|---:|---:|---:|---:|
| all-cold | native | n/a | n/a | n/a | n/a | n/a | 0 |
| all-cold | least_work | 52.0 | 327.0 | 300.0 | 570.0 | 327.0 | 8 |
| all-cold | prefix_only | 75.2 | 208.2 | 210.0 | 540.0 | 208.2 | 8 |
| all-cold | cache_aware_ect | 52.0 | 283.3 | 300.0 | 570.0 | 283.3 | 8 |
| one-warm | native | n/a | n/a | n/a | n/a | n/a | 0 |
| one-warm | least_work | 46.2 | 220.8 | 210.0 | 555.0 | 220.8 | 9 |
| one-warm | prefix_only | 72.7 | 173.8 | 142.5 | 465.0 | 164.5 | 9 |
| one-warm | cache_aware_ect | 28.6 | 337.4 | 337.5 | 600.0 | 337.4 | 9 |
| four-warm | native | n/a | n/a | n/a | n/a | n/a | 0 |
| four-warm | least_work | 54.3 | 222.3 | 300.0 | 570.0 | 222.3 | 12 |
| four-warm | prefix_only | 28.9 | 202.3 | 187.5 | 545.0 | 202.3 | 11 |
| four-warm | cache_aware_ect | 30.0 | 217.3 | 210.0 | 555.0 | 215.1 | 12 |
| all-warm | native | n/a | n/a | n/a | n/a | n/a | 0 |
| all-warm | least_work | 35.8 | 358.1 | 270.0 | 600.0 | 356.3 | 16 |
| all-warm | prefix_only | 29.7 | 308.2 | 255.0 | 600.0 | 308.2 | 13 |
| all-warm | cache_aware_ect | 26.6 | 210.3 | 165.0 | 552.0 | 210.3 | 16 |

## Who actually ran? (slot busy share, 1 Hz /slots trace)

| Topology | Policy | busy% per slot |
|---|---|---|
| all-cold | native | s0: 97%, s1: 98%, s2: 98%, s3: 99%, s4: 99%, s5: 100%, s6: 100%, s7: 100% |
| all-cold | least_work | s0: 99%, s1: 98%, s2: 98%, s3: 98%, s4: 98%, s5: 99%, s6: 99%, s7: 99% |
| all-cold | prefix_only | s0: 100% |
| all-cold | cache_aware_ect | s0: 81%, s1: 97%, s2: 98%, s3: 99%, s4: 99%, s5: 100%, s6: 100%, s7: 100% |
| one-warm | native | s0: 97%, s1: 98%, s2: 98%, s3: 99%, s4: 99%, s5: 99%, s6: 100%, s7: 100% |
| one-warm | least_work | s0: 87%, s1: 99%, s2: 99%, s3: 99%, s4: 100%, s5: 99%, s6: 100%, s7: 100% |
| one-warm | prefix_only | s0: 9%, s1: 99% |
| one-warm | cache_aware_ect | s0: 100%, s1: 2% |
| four-warm | native | s0: 98%, s1: 98%, s2: 98%, s3: 98%, s4: 99%, s5: 98%, s6: 99%, s7: 100% |
| four-warm | least_work | s0: 22%, s1: 0%, s2: 9%, s4: 99%, s5: 99%, s6: 98%, s7: 100% |
| four-warm | prefix_only | s0: 86%, s1: 7%, s2: 82%, s3: 100% |
| four-warm | cache_aware_ect | s0: 13%, s1: 70%, s2: 70%, s3: 100% |
| all-warm | native | s0: 99%, s1: 98%, s2: 99%, s3: 99%, s4: 100%, s5: 99%, s6: 100%, s7: 100% |
| all-warm | least_work | s0: 99%, s1: 6%, s2: 3%, s6: 74%, s7: 99% |
| all-warm | prefix_only | s0: 33%, s1: 1%, s2: 0%, s3: 29%, s4: 29%, s5: 29%, s6: 29%, s7: 100% |
| all-warm | cache_aware_ect | s0: 80%, s1: 11%, s2: 75%, s3: 76%, s4: 75%, s5: 76%, s6: 76%, s7: 98% |

## Two-prefix A/B interleaved (mini multi-tenant cluster)

| Variant | OK | TTFTs (s, by request id) | Reused tok/req | Mean reused |
|---|---:|---|---:|---:|
| a (seed 42) | 4/4 | [136.8, 408.5, 273.7, 229.4] | [4114, 11131, 4114, 0] | 4840 |
| b (seed 43) | 4/4 | [477.2, 409.0, 136.8, 538.9] | [11586, 11075, 11074, 11074] | 11202 |

Warm slots: 0←A, 1←B (pinned, own seeds). Affinity = each prefix riding its warm slot; spill = a saturated warm slot's traffic rerouted cold (attributed bypasses) instead of queueing indefinitely.
