#!/usr/bin/env bash
# MLX-LM worker — native macOS, Apple GPU via MLX.
# Uses the mlx-lm installed in the current Python environment.
# Env overrides: MODEL, PORT (default 9B 4-bit ≈ 5.5 GB; use MODEL=mlx-community/Qwen3.5-0.8B for smoke tests)
set -euo pipefail

PORT="${PORT:-8082}"
MODEL="${MODEL:-mlx-community/Qwen3.5-9B-4bit}"

echo "mlx-lm serving ${MODEL} on 127.0.0.1:${PORT}" >&2
exec python3 -m mlx_lm.server --model "${MODEL}" --port "${PORT}"
