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
`mock_stalled_total` injection counters).

Use it to validate loadgen and gateway behaviour (cancellation propagation,
upstream failures, slow-decode routing) before spending GPU time.
