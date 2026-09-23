# Experiment matrices

A matrix manifest describes a whole study: shared `[defaults]`, optional
`[[warmup]]` runs (executed first, recorded but flagged `warmup = true`,
abort the matrix on failure) and `[[runs]]` executed sequentially with
`cooldown_s` between them. Each run sets only the Scenario fields it
varies; everything else falls through run → defaults → baseline.
Run names must be unique — they name the per-run artifacts.

```bash
make matrix MATRIX=mlx-prompt-concurrency     # run a study
make analyze STUDY=mlx-prompt-concurrency-<ts>  # tables + charts
```

Results land in `benchmarks/results/<study>-<timestamp>/`:
one `<run>.json`/`<run>.csv` per run (same format as single-run
artifacts) plus `matrix.json` aggregating parameters + summaries, which
`benchmarks/analyze.py` consumes.

Fields every run may set (all optional): `name`, `endpoint`, `model`,
`concurrency`, `num_requests`, `input_tokens`, `output_tokens`, `stream`,
`shared_prefix_tokens`, `arrival` (`burst` | `constant:<s>` |
`poisson:<s>`), `timeout_s`, `seed` — see `benchmarks/scenarios/README.md`
for semantics.

| Matrix | Question it answers |
| --- | --- |
| `mlx-prompt-concurrency.toml` | how do TTFT/ITL/throughput move with prompt length and concurrency on mlx-lm? |
