#!/usr/bin/env bash
# MLX-LM worker — native macOS, Apple GPU via MLX.
# Uses the mlx-lm installed in the current Python environment.
# Env overrides: MODEL, PORT (default 9B 4-bit ≈ 5.5 GB)
set -euo pipefail

PORT="${PORT:-8082}"
MODEL="${MODEL:-mlx-community/Qwen3.5-9B-4bit}"

# mlx-lm >= 0.31 hits the Hugging Face API when a request names a repo id —
# even for an already-cached model — and if the network is unreachable the
# generation thread crashes and every later request queues forever (found
# during the phase-5 k3d bridge check). Serving from the local snapshot
# path avoids the hub entirely; fall back to the repo id when not cached.
resolve_local_snapshot() {
  local repo="$1"
  local dir="$HOME/.cache/huggingface/hub/models--${repo//\//--}/snapshots"
  [[ -d "$dir" ]] || return 1
  local snap
  snap="$(ls -d "$dir"/*/ 2>/dev/null | head -1)"
  [[ -n "$snap" ]] || return 1
  echo "$snap"
}

if [[ -d "$MODEL" ]]; then
  MODEL_PATH="$MODEL"
elif MODEL_PATH="$(resolve_local_snapshot "$MODEL")"; then
  echo "using local snapshot: $MODEL_PATH" >&2
else
  MODEL_PATH="$MODEL"
fi

echo "mlx-lm serving ${MODEL_PATH} (${MODEL}) on 127.0.0.1:${PORT}" >&2
# mlx-lm >= 0.25: the `python -m mlx_lm.server` entry point is deprecated;
# the supported CLI is `mlx_lm server`.
exec python3 -m mlx_lm server --model "${MODEL_PATH}" --port "${PORT}"
