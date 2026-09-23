# Benchmark scenarios

Each TOML is one loadgen run: `make bench S=<name>` runs `<name>.toml`.
Every field can be overridden on the CLI
(`cargo run --release -p loadgen -- --scenario ... --concurrency 16`).

| Scenario | What it varies |
| --- | --- |
| `smoke.toml` | tiny end-to-end sanity run (0.8B model, seconds) |
| `ttft-scan.toml` | prompt length pressure on time-to-first-token |
| `concurrency-sweep.toml` | batching/queueing behaviour under load |
| `prefix-shared.toml` | shared-prefix requests (prefix-cache hit/miss) |
| `stream-vs-nostream.toml` | streaming overhead check |

Keep results out of git — they land in `benchmarks/results/` (ignored).
Document every kept run in `benchmarks/reports/` with the exact scenario,
model, runtime version and machine state (see docs/benchmark-methodology.md).
