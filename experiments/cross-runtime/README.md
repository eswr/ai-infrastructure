# cross-runtime: MLX-LM vs llama.cpp vs vLLM-Metal on one Apple M5

Phase 4A. Phase 2 proved a performance pathology and phase 3 explained and
controlled it — on **one** runtime. This study asks the prerequisite question
for everything after (prefix caching, continuous batching, k8s):

> How much of that behavior is MLX-LM-specific, and how much is fundamental
> to the machine and workload?

## Matched, not identical

Same model **family** (Qwen3.5-9B), not bit-identical weights everywhere:

| leg | runtime | weights | server |
| --- | --- | --- | --- |
| `mlx-lm` | mlx_lm.server 0.31.3 | `mlx-community/Qwen3.5-9B-4bit` (MLX affine 4-bit, group 64) | `python3 -m mlx_lm.server` |
| `llama-cpp` | llama-server (brew) | `unsloth/Qwen3.5-9B-GGUF` **Q4_K_M** — closest GGUF analogue | `llama-server -c 73728 -np 8 --metrics` |
| `vllm-metal` | vllm-metal 0.1.0 | **same MLX 4-bit weights as mlx-lm** | `vllm-metal --model ...` (defaults — no tunable flags exposed) |

Quantization differences are recorded in `runtime_info.json` per leg and
discussed in the report's *Model equivalence* section — MLX 4-bit (affine,
group 64) and GGUF Q4_K_M are close in bits/weight but not equivalent;
llama.cpp additionally compiles its own Metal kernels and (by default)
quantizes KV to f16. llama.cpp context is sized `-c 73728 -np 8` (9216 per
slot) so the 8K×c8 workload fits without truncation. mlx-lm and vllm-metal
run at their defaults, as in phases 2–3.

Port note: llama.cpp serves on **:8085** — :8081 is squatted by an unrelated
Docker proxy on this machine.

## Fixed conditions

```
prompt        {128, 1024, 8192}   (per-request unique, shared_prefix = 0)
concurrency   {1, 4, 8}           (closed-loop cap, burst arrival)
requests      8 per run           → 9 runs × 3 runtimes = 27 measured runs
output_tokens 256, stream = true
temperature   1.0 (loadgen default, identical everywhere; seeded prompts)
gateway       :8080, ROUTE_POLICY=lor, admission OFF, FIRST_TOKEN_TIMEOUT_S=900
seed          42
```

## Lifecycle (why the harness exists)

Phase 3 showed /health can lie while a wedged server holds its generation
lock. Every leg therefore walks, via `benchmarks/runtime_harness.py`:

```
start → wait_live → wait_functional (real SSE generation)
      → benchmark (the 9-run matrix behind the gateway)
      → functional_probe immediately after the worst workload
      → verdict: healthy | silent_wedge | loud_failure | dead
      → restart if degraded → post_restart probe → stop
```

with 1 Hz memory sampling (process-tree RSS + vmmap physical footprint +
system used/free + kernel pressure level) for the whole walk.

## Run

```bash
make experiment-cross-runtime              # all three legs (~2–3 h, real GPU)
make experiment-cross-runtime RT=vllm-metal   # one leg
python3 benchmarks/analyze.py --compare benchmarks/results/cross-runtime-<ts>/cross-runtime.json
```

Artifacts per leg in `benchmarks/results/cross-runtime-<ts>/<leg>/`:
`runtime.log`, `gateway.log`, `memory.csv`, `lifecycle.json`,
`runtime_info.json`, `manifest.json`, and the loadgen matrix dir
(`<study>-<ts>/matrix.json` + per-run JSON/CSV). `../cross-runtime.json`
merges all legs for the analyzer.

## Not in this phase

Prefix caching (4B) and continuous-batching-focused runs (4C) come after
this baseline: enabling either would confound "runtime behavior" with
"cache/scheduler implementation". Admission stays OFF so each runtime's
*own* queueing behavior is what saturates.
