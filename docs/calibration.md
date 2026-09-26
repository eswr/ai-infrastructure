# Runtime performance calibration

*Phase 4B.3. Why: the 4B.2 study measured its own cost model and found
TTFT prediction MAE of **170–360 s** with bias ≈ +MAE — every burst was
priced on startup seeds (25 ms/token decode) while loaded decode really
cost **10–13×** that on this machine, so the cache-aware load gate opened
too late. The model knew nothing between process restarts and had one
decode rate for all load levels. Calibration fixes both.*

## What it is

Each modeled server process carries one `RuntimeCalibration` — a shared,
persisting performance model with three parameter groups, all of them
EWMAs of this machine's own observations (no invented constants):

| Group | Parameters | Learned from |
|---|---|---|
| Prefill (solo regression) | `fixed_ms`, `ms_per_token` | bucket-1 (unloaded) first-token observations only — the online least-squares regression from phase 3 |
| Prefill contention | multiplier per bucket | loaded prefill observations: `ms/tok observed ÷ solo rate` |
| Decode | `ms_per_token` per bucket | decode-phase observations per concurrency bucket |

**Concurrency buckets** (`["1", "2", "3-4", "5-8"]`) count *server-wide
active streams* of the same phase, the dispatch included. Prefill and
decode are counted separately (`inference_active_prefills` /
`inference_active_decodes`): 8 requests prefilling and 8 requests
decoding have radically different consequences on this machine.

Prediction fallback when a bucket has no observations: exact bucket →
nearest **higher** bucket → nearest lower → physical seed. Over-pricing a
loaded queue is the safe direction (4B.2's failure was under-pricing
it). The admission controller and legacy formulas use the *solo* decode
rate instead — bucket 1 measured, or the seed — never a loaded rate.

## Where it lives

```text
calibration/
  apple-m5/
    gguf-qwen3.5-9b-q4_k_m.gguf/     # slugified model id
      llama-cpp-8slots.json          # <runtime>-<slot_count>slots.json
```

Key tuple: `machine × model × runtime × slot_count`. A profile records
`schema_version`, the deployment tuple, `updated_at_epoch_s`, sample
counts per parameter, and the three parameter groups above. Saves are
atomic (write-then-rename); a corrupt file is logged and treated as
absent — calibration must never wedge the gateway.

## Config (env)

| Variable | Default | Meaning |
|---|---|---|
| `CALIBRATION_DIR` | *(off)* | profile directory; empty/unset = persistence off |
| `CALIBRATION_MACHINE` | `local` | machine key inside the dir |
| `CALIBRATION_DISABLED` | `0` | `1` = ignore persistence entirely (4B.2 A/B baseline) |
| `CALIBRATION_SAVE_INTERVAL_S` | `5` | periodic save cadence when dirty; 0 = save only at shutdown |

Profiles load at gateway boot and save periodically while observations
land (plus once at graceful shutdown). Inspect live state:

```sh
curl -s localhost:8080/status | python3 -m json.tool | grep -A30 calibration
```

## How it learns (and how it is measured)

Observations ride the request's `DispatchTicket`, so the shared model
knows exactly which dispatch they belong to and how many same-phase
peers ran alongside:

- **Prefill** (response headers → first token): bucket-1 observations
  feed the regression; every observation updates its bucket's contention
  multiplier against the current solo rate. Feeding 8-way-parallel
  prefills into the solo regression is how a contention penalty becomes
  "the rate" — the exact one-rate-for-all-loads mistake this module
  removes.
- **Decode** (first token → stream end): ms/token into that decode
  bucket's EWMA (α=0.25 — buckets must converge within one burst).

Routing prices at the load the request would **join**: prefill contends
with active prefills + itself; decode contends with active decodes
**and** active prefills (they finish prefill before this request does).

Model quality is part of every benchmark run: each request reports its
TTFT and service-time prediction error to labeled histogram families

```text
scheduler_{ttft,service}_prediction_{,abs_}error_seconds{prompt,conc,cache}
```

split by prompt-length bucket (short/mid/long), the concurrency bucket
the estimate assumed, and cache hit/miss. `benchmarks/router_econ.py`
turns these into the per-leg *prediction quality by regime* table:
MAE, |err| p50/p95, signed bias, and relative error per split. Positive
bias = the model under-predicts = its load gate opens too late — the
4B.2 failure mode, now visible per regime.

## The calibration wave

First process on a machine starts from seeds, so
`benchmarks/prefix_router.py --calibration-wave` runs a cheap probe wave
before any measured leg: solo short + solo long prefills (the 2-point
regression + bucket-1 decode), then an 8-way cold burst (loaded buckets).
Every wave run carries a distinct seed — distinct prefix, honestly cold
prefill. Later legs (and every future gateway restart) boot from
measurement.

## Interacting with routing

- The minimum-savings floor (`SCHEDULER_MIN_CACHE_SAVINGS_MS`) compares
  against raw solo-rate savings, unchanged from 4B.2.
- Slot-cache state (`CacheInvestment`) is orthogonal and per-slot;
  calibration is per-server.
- `WORKERS` entries are deduplicated per `url#model` — two names for the
  same server share one calibration (and one stream counter).

## Known limits (deliberate, for 4B.3)

- Bucket classification uses the estimated prefill/decode split
  (`elapsed` vs `est_prefill_ms`) — a badly wrong estimate can
  mis-bucket an observation. Buckets are coarse; the error averages out.
- Decode rate during queue-ahead pricing uses the reservation-time
  bucket, not a time-integrated one.
- The contention multiplier for a *mixture* of prefill+decode load is
  approximated by the prefill-count bucket alone. Measured mixed-workload
  contention is a phase-4C question.
