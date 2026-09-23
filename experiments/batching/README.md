# batching

How does continuous batching change TTFT/TPOT as concurrency rises? Sweep concurrency 1/2/4/8/16 with fixed input/output; compare mlx-lm vs vllm-metal batching behaviour. Run: scenario concurrency-sweep.toml with --concurrency overrides.

Hypothesis, method, kept results and writeup live here as the experiment lands.
See docs/benchmark-methodology.md for measurement rules.
