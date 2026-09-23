#!/usr/bin/env bash
# llama.cpp worker — native macOS, Metal backend by default.
# Resolves the cached Qwen3.5-0.8B GGUF; override with GGUF=/path/to/model.gguf.
# Env overrides: PORT, CTX (context size), PARALLEL (slots = continuous-batch capacity)
set -euo pipefail

PORT="${PORT:-8081}"
CTX="${CTX:-8192}"
PARALLEL="${PARALLEL:-4}"
DEFAULT_GGUF="$(find "${HOME}/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots" -name '*.gguf' ! -name 'mmproj*' 2>/dev/null | head -1 || true)"
GGUF="${GGUF:-${DEFAULT_GGUF}}"

if [[ -z "${GGUF}" ]]; then
  echo "no GGUF found; set GGUF=/path/to/model.gguf" >&2
  exit 1
fi

echo "llama-server serving ${GGUF} on 127.0.0.1:${PORT} (ctx=${CTX}, parallel=${PARALLEL})" >&2
exec llama-server \
  -m "${GGUF}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c "${CTX}" -np "${PARALLEL}" \
  --metrics
