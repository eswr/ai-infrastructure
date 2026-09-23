SHELL := /bin/bash
SCENARIO ?= smoke
MATRIX ?= mlx-prompt-concurrency

.PHONY: build test fmt clippy bench matrix analyze \
        worker-mlx worker-llama worker-vllm worker-mock gateway \
        up down obs-logs k8s

build:
	cargo build --workspace

test:
	cargo test --workspace

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

# ---------------------------------------------------------------- workers --
# Each worker serves an OpenAI-compatible API natively on macOS.
# (run each in its own terminal; they hold the foreground)

worker-mlx:      ## MLX-LM        :8082  (MODEL=... to override, default 9B-4bit)
	bash workers/mlx-lm/run.sh

worker-llama:    ## llama.cpp     :8081  (GGUF=... CTX=... to override)
	bash workers/llama-cpp/run.sh

worker-vllm:     ## vLLM-Metal    :8083  (needs: cd workers/vllm-metal && uv sync)
	bash workers/vllm-metal/run.sh

worker-mock:     ## mock GPU      :8090  (see workers/mock-gpu/README.md for knobs)
	python3 workers/mock-gpu/mock_worker.py

gateway:         ## Rust gateway  :8080  (WORKERS=name=url[,..] ROUTE_POLICY=lor|ect|prefix)
	cargo run --release -p gateway

# ---------------------------------------------------------- benchmarking --

bench: build     ## run benchmarks/scenarios/$(SCENARIO).toml
	cargo run --release -p loadgen -- --scenario benchmarks/scenarios/$(SCENARIO).toml

matrix: build     ## run benchmarks/matrices/$(MATRIX).toml (warmup + measured runs)
	cargo run --release -p loadgen -- --matrix benchmarks/matrices/$(MATRIX).toml

analyze:          ## tables + charts for make analyze STUDY=<results dir name>
	python3 benchmarks/analyze.py benchmarks/results/$(STUDY)/matrix.json

smoke:           ## start the cached 9B-4bit MLX model on :8082 (loads in seconds)
	bash workers/mlx-lm/run.sh

# -------------------------------------------------------- observability --

up:              ## start Prometheus/Grafana/Tempo/Loki/OTel-collector
	docker compose -f deploy/compose/docker-compose.yml up -d

down:
	docker compose -f deploy/compose/docker-compose.yml down

obs-logs:
	docker compose -f deploy/compose/docker-compose.yml logs -f --tail=50
