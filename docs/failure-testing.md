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

## Harness

The mock worker (workers/mock-gpu) exists so none of these tests need GPU
time. Once green against mocks, repeat the client-disconnect test against a
real MLX-LM / llama.cpp worker and verify server-side generation actually
stops.
