# mock-gpu worker

Fake OpenAI-compatible worker for router, streaming and failure-injection
experiments — no GPU, no dependencies (Python stdlib only).

```bash
make worker-mock                                  # :8090, ttft 80ms, 25ms/tok

# Failure injection
PORT=8091 FAIL_AFTER_REQUESTS=20 python3 workers/mock-gpu/mock_worker.py
PORT=8092 CRASH_STREAM_PCT=30 python3 workers/mock-gpu/mock_worker.py
PORT=8093 HANG_TTFT_PCT=50 TTFT_MS=900 python3 workers/mock-gpu/mock_worker.py
```

Endpoints: `/v1/chat/completions` (stream + non-stream), `/v1/models`,
`/health`, `/metrics` (Prometheus text).

Use it to validate loadgen and gateway behaviour (cancellation propagation,
upstream failures, slow-decode routing) before spending GPU time.
