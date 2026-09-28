# Inference Lab Console

Real-time visualization and replay of inference-control-plane decisions.
**DEV / DEMO ONLY** — the backend binds `127.0.0.1` only and exists to
narrate the lab, not to operate anything.

```text
┌───────────────────┬─────────────────────────────────────────────┐
│ Fleet             │ Request Flow                                │
│ nodes + workers   │ pipeline + latest scheduler decision table  │
│ lifecycle states  │ (candidates, reason, predicted vs actual)   │
├───────────────────┼──────────────────────┬──────────────────────┤
│ (fleet continues) │ Lifecycle timeline   │ Demo scenarios       │
│                   │ SSE transition feed  │ fixed-enum buttons   │
├───────────────────┴──────────────────────┴──────────────────────┤
│ TTFT p50 │ ITL p50 │ Queue est. │ Active │ Cache bypassed │ …  │
└─────────────────────────────────────────────────────────────────┘
```

## Run it

```bash
make console            # install (once) + build SPA + serve on http://127.0.0.1:8099
GATEWAY_URL=… make console   # point at any gateway (default http://127.0.0.1:8080)
```

Requires `make k8s-up` for the fleet/simulated-GPU view; the silent-wedge
and Real Metal scenarios run locally without a cluster.

## Architecture (deliberately tiny)

- `server.mjs` — zero-dependency Node backend. Polls the gateway's
  `/status`, `/metrics`, `/decisions` once per second, projects them
  (no PromQL in the browser), diffs transitions, and streams everything
  as SSE on `/api/events`. Demo scenarios are a **hardcoded enum**
  mapping to the same `benchmarks/` commands the Make targets run —
  never arbitrary shell execution, one scenario at a time.
- `src/` — SolidJS + Vite + TypeScript SPA, no router, no chart lib.
- The gateway's `GET /decisions` (cap-64 ring) is the one additive
  surface: the per-candidate routing table, the reason, and the measured
  TTFT backfilled at request completion next to the estimate.

## Scope

LIVE mode only in V0 (fleet, request flow with the live scheduler
decision, lifecycle timeline, metrics, demo buttons). REPLAY of frozen
benchmark artifacts, cache-slot map, and the calibration panel are V1 —
the numbers themselves stay in
[results/interview-scorecard.md](../../docs/results/interview-scorecard.md)
until then.
