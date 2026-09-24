SHELL := /bin/bash
SCENARIO ?= smoke
MATRIX ?= mlx-prompt-concurrency

.PHONY: setup build build-release test fmt clippy bench matrix analyze failcheck \
        experiment-admission-saturation experiment-cross-runtime \
        worker-mlx worker-llama worker-vllm worker-mock gateway \
        up down obs-logs k8s

setup:            ## fresh-checkout bootstrap: toolchain checks + workspace build
	@command -v cargo >/dev/null 2>&1 || { echo "ERROR: rust not found — install via https://rustup.rs"; exit 1; }
	cargo --version
	python3 --version
	@docker info >/dev/null 2>&1 && echo "docker: ok" || echo "WARNING: docker not running — 'make up' will fail"
	@command -v llama-server >/dev/null 2>&1 && echo "llama.cpp: ok" || echo "note: llama.cpp not found (optional; brew install llama.cpp)"
	cargo build --workspace

build:
	cargo build --workspace

build-release: ## release profile — what failcheck and the experiments actually run
	cargo build --release --workspace

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

failcheck: build-release  ## automated failure-injection suite against the mock worker
	python3 benchmarks/failures/run_suite.py

experiment-admission-saturation: build-release ## the centerpiece A/B: unbounded vs admission-controlled 8K×c8 (~55 min, real GPU)
	bash experiments/admission/run.sh

experiment-cross-runtime: build-release ## matched mlx-lm vs llama.cpp vs vllm-metal, 3×3 grid (~2-3 h; RT=<runtime> for one leg)
	bash experiments/cross-runtime/run.sh $(RT)

# -------------------------------------------------------- observability --

up:              ## start Prometheus/Grafana/Tempo/Loki/OTel-collector
	docker compose -f deploy/compose/docker-compose.yml up -d

down:
	docker compose -f deploy/compose/docker-compose.yml down

obs-logs:
	docker compose -f deploy/compose/docker-compose.yml logs -f --tail=50
