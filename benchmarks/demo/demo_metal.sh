#!/usr/bin/env bash
# Interview demo 1: real inference measurement on the M5 (docs/interview-demo.md).
# Self-contained: starts the native MLX worker, waits for live, runs the
# loadgen smoke scenario, stops the worker. Deterministic, ~1 min.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PORT=8082
SCENARIO=benchmarks/scenarios/smoke.toml   # 4 requests, 128 in / 32 out, stream

if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "error: something already serves :${PORT} — stop it first (lsof -i :${PORT})" >&2
  exit 1
fi

cargo build --release -p loadgen 1>&2

WORKER_PID=""
cleanup() {
  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill "${WORKER_PID}" 2>/dev/null || true
    wait "${WORKER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

bash workers/mlx-lm/run.sh > /tmp/demo-metal-worker.log 2>&1 &
WORKER_PID=$!

echo "waiting for the worker to load the model..." >&2
for _ in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "error: worker never became healthy — log: /tmp/demo-metal-worker.log" >&2
  exit 1
fi

cat <<'HEADER'
Model:      mlx-community/Qwen3.5-9B-4bit  (real weights, ~5.5 GB)
Runtime:    mlx-lm.server
Hardware:   Apple M5, 24 GB unified memory (Metal GPU)
Streaming:  yes (SSE)
Requests:   4, burst, 128-token prompts, 32 output tokens
Measured:   client-side per-stream (loadgen crate)
HEADER
echo

./target/release/loadgen --scenario "${SCENARIO}"
