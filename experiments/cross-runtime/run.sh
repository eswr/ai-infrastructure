#!/usr/bin/env bash
# The matched cross-runtime study (phase 4A):
#   MLX-LM vs llama.cpp vs vLLM-Metal, same Qwen3.5-9B model family,
#   same 3x3 prompt x concurrency grid, same gateway + loadgen,
#   admission OFF, streaming, burst.
#
# Each leg is driven by benchmarks/runtime_harness.py: the runtime walks
# start -> wait_live -> wait_functional -> benchmark -> functional_probe
# -> [restart if degraded] -> stop, with 1 Hz memory sampling throughout
# and a recovery probe immediately after the worst workload (8K x c8).
#
# Usage:    bash experiments/cross-runtime/run.sh [runtime ...]
#           (default: all three, in that order)
# Artifacts: benchmarks/results/cross-runtime-<ts>/<runtime>/
#            + cross-runtime.json merge for the analyzer.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -gt 0 ]; then
  RUNTIMES=("$@")
else
  RUNTIMES=(mlx-lm llama-cpp vllm-metal)
fi

TS=$(date +%Y%m%d-%H%M%S)
OUT="benchmarks/results/cross-runtime-$TS"
mkdir -p "$OUT"
echo "cross-runtime study → $OUT  (runtimes: ${RUNTIMES[*]})"

for rt in "${RUNTIMES[@]}"; do
  echo
  echo "================= leg: $rt ================="
  # The harness returns 1 if the runtime never started; the study should
  # keep going with the remaining legs but record the failure loudly.
  if ! python3 benchmarks/runtime_harness.py run \
        --runtime "$rt" \
        --matrix "experiments/cross-runtime/$rt.toml" \
        --out "$OUT" 2>&1 | tee "$OUT/$rt-console.txt"; then
    echo "!!! leg $rt FAILED (see $OUT/$rt-console.txt)"
  fi
  # Provenance manifest per leg: git/OS/chip/python/rust + this runtime's
  # versions, weights and flags (fragment written by the harness).
  python3 benchmarks/manifest.py write --out "$OUT/$rt" \
      --experiment "cross-runtime-$rt" \
      --fragment "$OUT/$rt/runtime_info.json" || true
  python3 benchmarks/manifest.py finalize --out "$OUT/$rt" || true
done

echo
echo "================= merge ================="
python3 benchmarks/runtime_harness.py merge --out "$OUT"
echo "artifacts in $OUT"
echo "next: python3 benchmarks/analyze.py --compare $OUT/cross-runtime.json"
