# Failure testing

Goal: streaming, routing and cancellation behave correctly when things go
wrong — demonstrated, not asserted.

## Failure modes under test

| # | Failure | Injection |
| --- | --- | --- |
| 1 | Worker returns 5xx | `FAIL_AFTER_REQUESTS=20 make worker-mock`; loadgen must record failures, gateway must count `upstream_failures{reason="http"}` |
| 2 | Worker dies mid-stream | `CRASH_STREAM_PCT=30` mock worker; client sees truncated stream, gateway decrements outstanding on drop |
| 3 | Client disconnects mid-generation | `curl -N ... & kill` or loadgen timeout; gateway aborts upstream; worker stops generating (verify via worker metrics) |
| 4 | Slow client (backpressure) | `curl -N --limit-rate 1k`; gateway must not buffer unboundedly |
| 5 | Worker unreachable (conn refused) | kill worker; gateway returns 502, `reason="connect"` |
| 6 | Queue saturation | concurrency ≫ worker capacity; measure queue-induced TTFT inflation |
| 7 | Hang TTFT | `HANG_TTFT_PCT=50` mock; loadgen timeout captures it; router ECT penalizes slow worker |

## What "correct" means

- **outstanding counters return to zero** after every failure mode
  (`gateway_active_requests` settles at 0).
- **client cancellation reaches the worker**: after a disconnect, the worker's
  in-flight generation drops (check worker-side metrics / `mock_active_requests`).
- **loadgen reports failures as failures** — never as fast successful requests
  (a cancelled request has no TTFT, an error string, and no completion tokens).
- **no leaked tasks/connections**: repeated failure runs keep gateway RSS flat.

## Automated suite

`make failcheck` runs the six core failure modes end-to-end against the
mock worker: each `benchmarks/failures/*.toml` starts a fresh mock worker
(with injection knobs) and gateway (with policy env), drives HTTP traffic,
then asserts on responses, `/status` and `/metrics`:

| scenario | what it proves |
| --- | --- |
| `worker-crash` | 5xx → 502 to client; unhealthy on outage; automatic recovery |
| `first-token-timeout` | TTFT watchdog aborts a silent stream, frees the slot |
| `midstream-crash` | truncated stream → `upstream_error`, capacity released |
| `slow-stream` | degraded worker completes; ITL histogram populated |
| `queue-overflow` | bounded queue; 429 + `Retry-After` at the overload threshold |
| `client-cancel` | RST aborts upstream mid-generation, books saved compute |

Acceptance semantics: outstanding capacity (`inference_requests_active`,
`/status admission.active_requests`) returns to zero after every mode;
cancellations are counted by phase; the queued-cancel variant is reported
but not asserted (HTTP/1.1 disconnects are only observable at
response-write time — HTTP/2 resets propagate into the queue).

## Manual procedures

The mock worker (workers/mock-gpu) exists so none of these tests need GPU
time. Once green against mocks, repeat the client-disconnect test against a
real MLX-LM / llama.cpp worker and verify server-side generation actually
stops.
