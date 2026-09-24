#!/usr/bin/env bash
# The admission-control centerpiece experiment:
#   the phase-2 pathological workload (8K prompt × c8 × 16 requests,
#   which collapsed to 0/16 first tokens in 900 s) run twice —
#   A: admission control OFF (phase-2 behavior)
#   B: admission control ON (one 8K prefill at a time, token-accounted queue)
#
# Artifacts: benchmarks/results/admission-saturation-<ts>/, one
# <name>.json/.csv per run (loadgen single-run artifacts, renamed).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="mlx-community/Qwen3.5-9B-4bit"
GW_PID=""
MLX_PID=""
cleanup() {
  [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null || true
  if [ "${KEEP_MLX:-0}" != "1" ] && [ -n "$MLX_PID" ]; then
    kill "$MLX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- worker -----------------------------------------------------------------
# Each run must start from a healthy worker: run A (unbounded) can wedge the
# worker with GPU OOM (see the study write-up), so the worker is restarted
# between runs. The post-restart POST probe guards against a wedged server
# that still answers /health.
MLX_LOG="$PWD/worker-mlx.log"
start_worker() {
  pkill -f "mlx_lm.server" 2>/dev/null || true
  sleep 2
  if [ "${KEEP_MLX:-0}" = "1" ] && curl -sf -m 2 http://127.0.0.1:8082/v1/models > /dev/null 2>&1; then
    echo "using already-running mlx-lm worker on :8082"
    return 0
  fi
  echo "starting mlx-lm worker on :8082 ($MODEL)…"
  bash workers/mlx-lm/run.sh >> "$MLX_LOG" 2>&1 &
  MLX_PID=$!
  for i in $(seq 1 120); do
    curl -sf -m 2 http://127.0.0.1:8082/v1/models > /dev/null 2>&1 && break
    [ "$i" = 120 ] && { echo "worker never became ready"; exit 1; }
    sleep 2
  done
}
post_ok() {
  curl -sf -m 60 http://127.0.0.1:8082/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"worker readiness probe\"}],\"stream\":true,\"max_tokens\":4}" \
    | grep -q 'data:'
}
start_worker
post_ok || { echo "worker accepts connections but never serves a POST (wedged?)"; exit 1; }
echo "worker ready and serving"

TS=$(date +%Y%m%d-%H%M%S)
OUT="benchmarks/results/admission-saturation-$TS"
# worker log lives inside the artifacts dir once it exists
mkdir -p "$OUT"
MLX_LOG="$OUT/worker-mlx.log"
mkdir -p "$OUT"

start_gateway() {  # $1 = extra env assignments
  env GATEWAY_ADDR=0.0.0.0:8080 \
      WORKERS="mlx=http://127.0.0.1:8082#$MODEL" \
      ROUTE_POLICY=lor \
      OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
      $1 \
      ./target/release/gateway > "$OUT/gateway-$2.log" 2>&1 &
  GW_PID=$!
  for i in $(seq 1 50); do
    curl -sf -m 2 http://127.0.0.1:8080/healthz > /dev/null 2>&1 && return 0
    sleep 0.2
  done
  echo "gateway ($2) never became ready"; exit 1
}

# --- run A: unbounded ---------------------------------------------------------
echo "=== run A: admission OFF (phase-2 conditions) ==="
start_gateway "" unbounded
./target/release/loadgen --scenario experiments/admission/unbounded-8k-c8.toml \
  --out-dir "$OUT" 2>&1 | tee "$OUT/run-A-console.txt"
mv "$OUT/unbounded-8k-c8"-*.json "$OUT/" 2>/dev/null || true
kill "$GW_PID"; wait "$GW_PID" 2>/dev/null || true; GW_PID=""

echo "restarting worker for run B (run A may have degraded it)…"
start_worker
post_ok || { echo "worker wedged after run A — this is itself a finding; see study doc"; exit 1; }
echo "cooldown 15 s…"; sleep 15

# --- run B: admission-controlled ----------------------------------------------
echo "=== run B: admission ON (ADMIT_MAX_ACTIVE_INPUT_TOKENS=8192, overload gate 1200 s) ==="
start_gateway "ADMIT_MAX_ACTIVE_INPUT_TOKENS=8192 ADMIT_OVERLOADED_QUEUE_S=1200 ADMIT_DEGRADED_QUEUE_S=300" admission
./target/release/loadgen --scenario experiments/admission/admission-controlled-8k-c8.toml \
  --out-dir "$OUT" 2>&1 | tee "$OUT/run-B-console.txt"
mv "$OUT/admission-controlled-8k-c8"-*.json "$OUT/" 2>/dev/null || true

kill "$GW_PID"; wait "$GW_PID" 2>/dev/null || true; GW_PID=""
echo "artifacts in $OUT"
