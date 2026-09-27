# mock-gpu worker

Fake OpenAI-compatible worker for router, streaming and failure-injection
experiments — no GPU, no dependencies (Python stdlib only).

```bash
make worker-mock                                  # :8090, ttft 80ms, 25ms/tok

# Failure injection
PORT=8091 FAIL_AFTER_REQUESTS=20 python3 workers/mock-gpu/mock_worker.py
PORT=8092 CRASH_STREAM_PCT=30 python3 workers/mock-gpu/mock_worker.py
PORT=8093 HANG_TTFT_PCT=50 TTFT_MS=900 python3 workers/mock-gpu/mock_worker.py

# The phase-4A failure signatures (what make lifecycle-check exercises)
PORT=8094 WEDGE_AFTER_REQUESTS=1 python3 workers/mock-gpu/mock_worker.py       # silent wedge: /health 200, POST hangs
PORT=8094 EMPTY_COMPLETION_PCT=100 python3 workers/mock-gpu/mock_worker.py     # HTTP 200, completion_tokens=0
PORT=8094 STALL_AFTER_CHUNKS=1 python3 workers/mock-gpu/mock_worker.py         # stream starts, stops producing
```

Endpoints: `/v1/chat/completions` (stream + non-stream), `/v1/models`,
`/health`, `/metrics` (Prometheus text, includes
`mock_wedged_total` / `mock_empty_completions_total` /
`mock_stalled_total` / `mock_draining` counters).

Phase-5 additions (k3d operations — see `deploy/k8s/README.md`):

```bash
# Readiness = a REAL mini-generation (tokens > 0), never "/health == 200";
# 503 while wedged/draining, and (when GATEWAY_URL is set) 503 unless the
# gateway's lifecycle controller says this pod is routable. This is what
# the pod readinessProbe hits.
curl -s localhost:8090/readyz

# Graceful drain (also runs on SIGTERM / kubernetes preStop): refuse new
# work, finish active streams, exit when active hits 0 (DRAIN_MAX_S cap).
curl -X POST localhost:8090/drain            # GET /drain = preStop form

# Runtime knobs — how the k8s scenarios break and heal a running pod
curl -s localhost:8090/admin/state
curl -s -XPOST -H 'Content-Type: application/json' \
     -d '{"wedge": true}' localhost:8090/admin/knob
curl -s -XPOST -H 'Content-Type: application/json' \
     -d '{"tok_interval_ms": 150, "empty_pct": 50}' localhost:8090/admin/knob
```

Bind address: `HOST` env (default 127.0.0.1; the container image sets
`HOST=0.0.0.0` so pods become kubernetes endpoints).

Use it to validate loadgen and gateway behaviour (cancellation propagation,
upstream failures, slow-decode routing) before spending GPU time.
