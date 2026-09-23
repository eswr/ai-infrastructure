# vllm-metal worker

[vllm-metal](https://github.com/vllm-project/vllm-metal) brings upstream
vLLM's scheduler, paged block manager and continuous batching to Apple
Silicon (MLX + Metal underneath).

## Setup

```bash
cd workers/vllm-metal
uv sync          # creates a Python 3.13 venv (system python is 3.14, unsupported)
```

The base install is the MLX path (mlx, mlx-lm, mlx-vlm, transformers...).
The optional `vllm` extra pulls the full torch+vLLM stack (multi-GB) — add
it only if you want to run under upstream vLLM itself:

```bash
uv add "vllm-metal[vllm]"
```

## Run

```bash
../vllm-metal/run.sh          # or: make worker-vllm
# MODEL=mlx-community/Qwen3.5-0.8B ../vllm-metal/run.sh   # smoke-test size
```

Serves on `http://127.0.0.1:8083`.

## What to study in this codebase

scheduler integration · block allocation · paged KV cache · attention
implementation · prefill vs decode paths · continuous batching ·
prefix-cache lookup · speculative decoding · memory admission · Metal
kernels.
