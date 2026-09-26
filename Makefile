SHELL := /bin/bash
SCENARIO ?= smoke
MATRIX ?= mlx-prompt-concurrency

.PHONY: setup build build-release test fmt clippy bench matrix analyze failcheck lifecycle-check \
        experiment-admission-saturation experiment-cross-runtime experiment-prefix-cache \
        experiment-prefix-router experiment-serving-interactions \
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

lifecycle-check: build-release ## the six worker-lifecycle scenarios (wedge, connect, empty, stall, restart, drain)
	python3 benchmarks/lifecycle/run_lifecycle_check.py

experiment-admission-saturation: build-release ## the centerpiece A/B: unbounded vs admission-controlled 8K×c8 (~55 min, real GPU)
	bash experiments/admission/run.sh

experiment-cross-runtime: build-release ## matched mlx-lm vs llama.cpp vs vllm-metal, 3×3 grid (~2-3 h; RT=<runtime> for one leg)
	bash experiments/cross-runtime/run.sh $(RT)

experiment-prefix-cache: build-release ## phase-4B: cold vs warm prefix reuse on llama.cpp (~45 min, real GPU)
	python3 benchmarks/prefix_cache.py --runtime llama-cpp

experiment-prefix-router: build-release ## phase-4B.2: load-gated prefix routing, 4 policies x 4 cache topologies (~1.5 h, real GPU)
	python3 benchmarks/prefix_router.py --runtime llama-cpp --two-prefix

experiment-calibration: build-release ## phase-4B.3: calibrated cost model vs 4B.2 baseline, 8 legs + wave (~1 h, real GPU)
	python3 benchmarks/prefix_router.py --runtime llama-cpp --policies cache_aware_ect,least_work --topologies all-cold,one-warm,four-warm,all-warm --calibration-wave --study-name calibration-router

experiment-calibration-ab: build-release ## phase-4B.3 A/B: --no-calibration rerun for the same legs (seed-only baseline)
	python3 benchmarks/prefix_router.py --runtime llama-cpp --policies cache_aware_ect,least_work --topologies all-cold,one-warm,four-warm,all-warm --no-calibration --study-name calibration-router-nocal

experiment-serving-interactions: build-release ## phase-4C: contention wave + 16-cell matrix + mixed interactive workload (~3 h, real GPU)
	python3 benchmarks/serving_interactions.py --runtime llama-cpp --contention-wave --matrix --mixed

# -------------------------------------------------------- observability --

up:              ## start Prometheus/Grafana/Tempo/Loki/OTel-collector
	docker compose -f deploy/compose/docker-compose.yml up -d

down:
	docker compose -f deploy/compose/docker-compose.yml down

obs-logs:
	docker compose -f deploy/compose/docker-compose.yml logs -f --tail=50
