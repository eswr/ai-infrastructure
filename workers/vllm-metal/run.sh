#!/usr/bin/env bash
# vLLM-Metal worker — vLLM scheduler/paged-KV concepts on Apple Silicon via MLX.
# Own uv project (Python 3.13; system Python is 3.14 which vllm-metal doesn't support yet).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8083}"
MODEL="${MODEL:-mlx-community/Qwen3.5-9B-4bit}"

if [[ ! -d .venv ]]; then
  echo "venv missing; run: cd workers/vllm-metal && uv sync" >&2
  exit 1
fi

exec uv run vllm-metal --model "${MODEL}" --host 127.0.0.1 --port "${PORT}"
