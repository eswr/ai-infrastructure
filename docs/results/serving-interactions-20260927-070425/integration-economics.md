# Serving interactions — serving-interactions-4c-fast-20260927-070425

Phase 4C study on llama-cpp / Qwen3.5-9B-Q4_K_M.gguf, created 20260927-070425. Question: phase-4C: measure accelerator contention as a first-class resource — prefill slowdown vs aggregate throughput per concurrent level, decode scaling, and prefill->decode interference — before routing/admission consume the curve.

## Contention wave — the machine, measured

## The serving-interaction matrix (cache x admission x contention gate)

| Cell | Verdict | Term | OK | TTFT p50 | TTFT p95 | ITL p50 | Out tok/s | Reused tok/req | Gate holds | Spills |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long-cold-admit0-gate0 | healthy | - | 8/8 | 70.8s | 97.0s | 0.1s | 4.8 | 8216 | 0 | 0 |
| long-cold-admit1-gate1 | healthy | - | 8/8 | 133.2s | 171.0s | 0.1s | 2.8 | 8215 | 0 | 0 |
| long-warm-admit0-gate0 | healthy | - | 8/8 | 23.8s | 29.3s | 0.4s | 9.2 | 11018 | 0 | 0 |
| long-warm-admit1-gate1 | healthy | - | 8/8 | 18.4s | 22.6s | 0.4s | 10.2 | 11018 | 0 | 0 |

## Mixed interactive workload

Interactive SLO: TTFT <= 5 s. 2x8K warm-prefix (long_context/standard) + 2x8K cold (batch) + 4x1K (interactive) over one short burst.

| Config | Class | OK | TTFT p50 | TTFT p95 | E2E p50 | SLO violations | Out tok/s | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | long-context | 0/2 | n/a | n/a | n/a | 0/2 (0.0%) | n/a |
| baseline | batch | 0/2 | n/a | n/a | n/a | 0/2 (0.0%) | n/a |
| baseline | interactive | 0/4 | n/a | n/a | n/a | 0/4 (0.0%) | n/a |

| full | long-context | 2/2 | 4.8s | 10.5s | 72.3s | 1/2 (50.0%) | 7.1 |
| full | batch | 2/2 | 35.3s | 149.2s | 78.5s | 2/2 (100.0%) | 2.7 |
| full | interactive | 4/4 | 120.0s | 120.0s | 189.0s | 4/4 (100.0%) | 5.4 |

### Baseline vs full stack

- **baseline** (admission=False, gate=False): interactive TTFT p50 n/a, p95 n/a, SLO violations 0.0%, wall 16460.6s
- **full** (admission=True, gate=True): interactive TTFT p50 120.0s, p95 120.0s, SLO violations 100.0%, wall 189.4s
