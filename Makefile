SHELL := /bin/bash
SCENARIO ?= smoke
MATRIX ?= mlx-prompt-concurrency
K3D_CLUSTER ?= inference-lab
K8S_NAMESPACE ?= inference-lab
GATEWAY_IMAGE ?= inference-lab/gateway:phase5
MOCK_IMAGE ?= inference-lab/mock-worker:phase5

.PHONY: setup build build-release test fmt clippy bench matrix analyze failcheck lifecycle-check \
        experiment-admission-saturation experiment-cross-runtime experiment-prefix-cache \
        experiment-prefix-router experiment-serving-interactions \
        worker-mlx worker-llama worker-vllm worker-mock gateway \
        up down obs-logs k8s k8s-images k8s-up k8s-deploy k8s-render k8s-check k8s-demo k8s-bridge-check \
        fleet-check k8s-status k8s-down slurm-up slurm-demo slurm-down \
        demo-metal demo-wedge demo-admission demo-k8s demo-placement demo-fleet-failure demo-slurm \
        console-install console-build console

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

experiment-serving-interactions: build-release ## phase-4C integration test: 4 cells (8K cold/warm x baseline/full, out=64) + mixed A/B (~40 min, real GPU)
	python3 benchmarks/serving_interactions.py --runtime llama-cpp --matrix-fast --mixed

experiment-serving-interactions-full: build-release ## phase-4C exhaustive 16-cell ablation (hours; run deliberately)
	python3 benchmarks/serving_interactions.py --runtime llama-cpp --matrix-full --mixed

# ------------------------------------------------------------- interview --
# Deterministic minute-scale demo flows for interviews (docs/interview-demo.md).
# Deliberately NO `demo-all`: an interviewer never watches a 20-minute script.
# Frozen fallback transcripts for every target: docs/results/interview-demo/.

demo-metal: build-release ## live: real MLX 9B-4bit measurement (TTFT/ITL/e2e) — ~1 min
	bash benchmarks/demo/demo_metal.sh

demo-wedge: build-release ## live: /health lies — silent wedge -> quarantine -> recovery — ~1 min
	python3 benchmarks/lifecycle/run_lifecycle_check.py silent-wedge

demo-admission: ## saved evidence: the 0/16 -> 16/16 overload incident (frozen artifacts)
	python3 benchmarks/demo/demo_admission.py

demo-k8s: ## live: the six-beat k3d control-plane demo — needs `make k8s-up` — ~2 min
	python3 benchmarks/k8s/run_demo.py

demo-placement: ## live: fleet discovery + placement matrix — needs `make k8s-up` — ~1 min
	python3 benchmarks/k8s/run_fleet_check.py discovery placement

demo-fleet-failure: ## live: XID / thermal / OOM drills — needs `make k8s-up` — ~1 min
	python3 benchmarks/k8s/run_fleet_check.py xid thermal oom

demo-slurm: ## live: Slurm GRES vocabulary — starts the compose lab if needed — ~1 min
	docker compose -f deploy/slurm/docker-compose.yml up -d
	@sleep 6
	bash experiments/slurm/exercises.sh

# --------------------------------------------------------------- console --
# Real-time control-plane visualization (DEV / DEMO ONLY, loopback only).
# The backend projects the gateway's /status + /metrics + /decisions and
# streams SSE; demo buttons run the SAME fixed scenario commands as the
# demo-* targets. See apps/console/README.md.

console-install:  ## once: install console frontend deps
	cd apps/console && npm install

console-build: console-install  ## build the SPA into apps/console/dist
	cd apps/console && npm run build

console:          ## serve the console on http://127.0.0.1:8099 (needs a gateway)
	cd apps/console && npm run build
	node apps/console/server.mjs

# ------------------------------------------------------- observability --

up:              ## start Prometheus/Grafana/Tempo/Loki/OTel-collector
	docker compose -f deploy/compose/docker-compose.yml up -d

down:
	docker compose -f deploy/compose/docker-compose.yml down

obs-logs:
	docker compose -f deploy/compose/docker-compose.yml logs -f --tail=50

# ------------------------------------------------------- k3d operations --
# Phase 5: the control plane under Kubernetes. Gateway + mock workers in
# k3d; the real Metal worker stays native on macOS (EXTERNAL_WORKERS via
# host.docker.internal). Every default here completes in minutes.

k8s: k8s-status  ## alias: kubectl get pods -n inference-lab (the demo view)

k8s-images:      ## build gateway + mock-worker container images
	docker build -f deploy/k8s/images/gateway/Dockerfile -t $(GATEWAY_IMAGE) .
	docker build -f deploy/k8s/images/mock-worker/Dockerfile -t $(MOCK_IMAGE) workers/mock-gpu
	docker build -q -f deploy/k8s/images/mock-dcgm/Dockerfile -t inference-lab/mock-dcgm:phase6 deploy/k8s/images/mock-dcgm >/dev/null
	k3d image import inference-lab/mock-dcgm:phase6 -c $(K3D_CLUSTER) 2>/dev/null || true

k8s-render:      ## render manifests (inject host.docker.internal IP + scaler script)
	bash deploy/k8s/render.sh

k8s-up: k8s-images k8s-render ## create the k3d cluster, import images, deploy the stack
	@k3d cluster get $(K3D_CLUSTER) >/dev/null 2>&1 || k3d cluster create $(K3D_CLUSTER) --config deploy/k8s/k3d-config.yaml
	k3d image import "$(GATEWAY_IMAGE)" "$(MOCK_IMAGE)" -c $(K3D_CLUSTER)
	kubectl apply -f deploy/k8s/rendered/
	@echo "waiting for the stack (gateway -> mock workers -> scaler)..."
	kubectl -n $(K8S_NAMESPACE) rollout status deploy/gateway --timeout=180s
	kubectl -n $(K8S_NAMESPACE) rollout status deploy/mock-workers --timeout=180s
	kubectl -n $(K8S_NAMESPACE) rollout status deploy/scaler --timeout=180s
	@echo "gateway (host) : http://localhost:8080"
	@$(MAKE) --no-print-directory k8s-status

k8s-deploy: k8s-render ## re-apply manifests to an existing cluster
	kubectl apply -f deploy/k8s/rendered/

k8s-check:       ## the six short k8s acceptance scenarios (minutes, mock workers only)
	python3 benchmarks/k8s/run_k8s_check.py

fleet-check:     ## phase-6 acceptance: heterogeneous fleet, placement, node lifecycle, simulated failures
	python3 benchmarks/k8s/run_fleet_check.py

k8s-demo:        ## the six-beat phase-5 demo (pods, autoscale, wedge, recover, drain, bridge) — under 5 min
	python3 benchmarks/k8s/run_demo.py

k8s-bridge-check: ## one native Metal worker + gateway in k3d + 4 concurrent short requests
	python3 benchmarks/k8s/run_bridge_check.py

k8s-status:
	@kubectl get pods -n $(K8S_NAMESPACE) 2>/dev/null || echo "cluster not up (make k8s-up)"

k8s-down:
	k3d cluster delete $(K3D_CLUSTER)

# ------------------------------------------------------------- slurm lab --
# Phase 6: the batch/HPC side of the datacenter story, on FAKE GPUs.
# Kubernetes = online serving; Slurm = queued batch. Vocabulary, not
# accelerators: sinfo/squeue/sbatch/srun/GRES/drain/resume.

slurm-up:         ## build + start the local Slurm lab (1 ctl + 2 nodes, fake A100 GRES)
	docker compose -f deploy/slurm/docker-compose.yml up -d --build
	@echo "waiting for node registration..."
	@sleep 6
	docker compose -f deploy/slurm/docker-compose.yml exec -T slurmctl sinfo

slurm-demo:       ## the Slurm vocabulary exercises (sbatch --gres, squeue, drain/resume)
	bash experiments/slurm/exercises.sh

slurm-down:
	docker compose -f deploy/slurm/docker-compose.yml down -v
