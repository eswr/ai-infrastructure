# Prefix-cache efficiency — cold vs warm reuse, and the trap in between

*Phase 4B. Raw artifacts: `benchmarks/results/prefix-cache-reuse-20260925-222628/`
(six legs × loadgen records, 1 Hz vmmap memory CSV, llama-server log,
provenance in `lifecycle.json`); curated evidence in
[`prefix-cache-20260925-222628/`](prefix-cache-20260925-222628/). Economics
tables: [`prefix-cache-20260925-222628/cache-economics.md`](prefix-cache-20260925-222628/cache-economics.md).*

## Question

> **How much latency and compute does prefix caching save, what memory does
> it cost, and how does that change under concurrency?**

Not "does prefix caching work". The cache-OFF baseline stays **frozen** —
the phase-4A study *is* the all-miss baseline (its prompts were unique per
request, so observed reuse was zero by construction); nothing was rerun.

## Which runtimes can even answer the question

Phase 4A ranked the runtimes by throughput; phase 4B had to rank them by
something else first: **does a reusable-prefix mechanism exist, and can it
be observed?** Investigated before running anything:

| runtime | comparable reusable-prefix mechanism | evidence |
| --- | --- | --- |
| **llama.cpp** 0.4.1 | **yes** | slot-local KV caches, LCP slot selection, and per-request reuse telemetry — `timings.cache_n` on the final stream chunk, `usage.prompt_tokens_details.cached_tokens` on JSON answers. Verified live: identical 912-token prompt → cold `cache_n=0` / prompt 3.7 s, warm `cache_n=908` / prompt 141 ms. |
| **vllm-metal** 0.1.0 | **no** | The MLX backend prefills every request from a **fresh cache**: `v1/model_runner.py::_prefill_single` calls `make_prompt_cache(cache_model)` per request and processes the entire prompt. Server surface is 4 routes (`/health`, `/v1/models`, 2× completions) — no `/metrics`, no cache-reset endpoint, no hit counters. The vLLM name carries scheduler *concepts*, not paged-KV prefix reuse. Consistent with the 4A data: TTFT at c=1 is linear in prompt length with no warmup trend. |
| **mlx-lm** 0.31.3 | **no** | Stateless per request; every request reprefills (phase-2/4A prefill behavior is consistent with this throughout). |

So the experiment ran on **llama.cpp** — the one honest cell. Fabricating a
three-runtime cache comparison would have compared three different things
under one header; the 4A lesson (runtime semantics dominate) applies twice
over here.

## Method

Workload — the shape where prefix caching actually matters (shared system /
document / agent context, many small users on top):

```text
shared prefix     8192 nominal tokens (byte-identical across requests)
unique suffix      128 nominal tokens per request
output             256 tokens, streaming, burst arrivals
requests           8 per run, via the gateway (lor, admission off)
concurrency        1 / 4 / 8
```

Three cache states per concurrency:

* **off (frozen 4A)** — the phase-4A llama.cpp p8192 rows. Unique prompts ⇒
  zero reuse; identical model, sizes, output, arrival, gateway.
* **cold** — `reset_cache()` (process restart — the one honest reset:
  llama.cpp 0.4.1 ships no erase action, `POST /slots/N?action=erase`
  answers 501) → functional gate → measured run. First contact populates
  the cache mid-run.
* **warm** — reset → functional gate → **populate every slot** (an 8-request
  warmup with the same scenario seed, so the prefix is byte-identical —
  Python cannot reproduce the Rust `StdRng` word sequence) → measured run.
  Slot-by-slot warming is not optional: llama.cpp caches are **slot-local**
  (measured: two concurrent same-prefix requests → `cache_n=396` and `0` on
  different slots).

Every leg ended with the phase-4D machinery watching: a real-generation
functional probe plus the gateway's lifecycle verdict. **All six legs came
back healthy, zero restarts** — the cache experiments did not destabilize
the worker, so the performance numbers below are not buying reliability
with anything.

## Results

| c | Cache | TTFT p50 | TTFT p95 | Out tok/s | Max stall | Spread | Reused tok/req |
| -:|---|---|---|---|---|---| ---: |
| 1 | off (frozen 4A) | 51.9s | 54.2s | 3.2 | 1.5s | 10.3s | 0 |
| 1 | cold | 3.0s | 22.6s | 8.2 | 1.0s | 27.1s | 9740* |
| 1 | warm | 2.1s | 6.1s | 7.4 | 4.0s | 8.0s | 8384 |
| 4 | off (frozen 4A) | 10.0s | 15.7s | 9.4 | 2.4s | 25.8s | 0 |
| 4 | cold | 43.6s | 240.1s | 5.2 | 13.5s | 269.5s | 5564* |
| 4 | warm | 6.4s | 10.2s | 9.5 | 2.6s | 17.5s | 8384 |
| 8 | off (frozen 4A) | 19.0s | 19.0s | 13.1 | 2.5s | ~0 | 0 |
| 8 | cold | **253.0s** | 385.9s | 4.2 | 12.3s | 11.0s | 0 |
| 8 | warm | 5.1s | 5.3s | 20.8 | 1.1s | 66ms | 8384 |

\* cold means include the first-wave miss(es); `cache_n` counts cache
*positions* (template + decode offsets) and can exceed the 8384-token
server prompt — economics below cap it there. Raw counters in the curated
`cache_econ.json`. Reuse is **observed runtime telemetry**, never inferred
from TTFT.

## What the cache bought

| c | TTFT saved (cold−warm) | Warm reused tok/req | ms saved per cached token | Effective prefill (share of prompt) |
| -:|---|---|---|---|
| 1 | 0.9s | 8384 | 0.11 | ~0 (≈100% avoided) |
| 4 | 37.2s | 8384 | 4.43 | ~0 |
| 8 | 247.9s | 8384 | 29.57 | ~0 |

And what it costs: warm legs peak at **~10.1 GB** vmmap footprint vs
**~4.2–5.7 GB** cold — roughly **+4.4–6 GB** for eight resident slot
copies of the 11.4K-token prefix (llama.cpp duplicates the prefix per
slot; a shared/paged pool would not). On this 24 GB machine, the warm
state is ~42 % of system memory held to save, at c8, 4.1 minutes of
prefill per 8-request wave.

## Why — the four findings

**1. A warm shared prefix erases prefill entirely.** Reuse was the full
prompt in every warm request at every concurrency; TTFT collapses to the
decode-first-token floor (2.1 s at c1) and stops caring about prompt
length. Against the frozen all-miss baseline that is **×24.7 TTFT at c1**
(51.9 s → 2.1 s).

**2. The value of one cached token is a concurrency multiplier.**
`ms_saved_per_cached_token` grows 0.11 → 4.43 → 29.57 from c1 to c8 —
**270×**. The prefix is identical; what changes is the counterfactual. At
c1 an all-miss wave costs one prefill; at c8 it costs *eight concurrent
full prefills* plus batch interference. Cache savings are not a property
of the cache — they are a property of how much prefill contention it
removes. Aggregate throughput at c8 rose ×1.59 (13.1 → 20.8 tok/s) while
worst stall *fell* 2.5 s → 1.1 s and completion spread stayed at
milliseconds: warm cache made the system simultaneously faster **and**
fairer. This is the rare optimization that helps the tail most.

**3. The trap: cold same-prefix bursts serialize.** Cold c8 TTFT p50 is
253 s — **13× worse than the frozen all-miss baseline (19 s)** for the
same prompt sizes. Cause, visible in the slot-selection log
([excerpt](prefix-cache-20260925-222628/slot-selection-log-excerpt.txt)):
llama.cpp picks slots by LCP similarity, and a same-prefix request selects
the *busy* slot that is already prefilling that prefix
(`selected slot by LCP similarity, f_sim_best = 1.000`) rather than one of
seven idle empty slots. The burst queues behind the hottest slot and
prefill becomes a serial pipeline. 4A's unique-prompt bursts fanned out by
LRU and never hit this. So "cache exists" vs "cache helps" has a third
state nobody markets: **the population transient, where prefix affinity
turns parallel prefill into serial prefill**. Any production router that
does prefix-affinity placement inherits this failure mode at exactly the
moment a new context goes viral.

**4. Cold c1 is the good case, and it says why.** One miss (33 s), then
seven full hits (~3 s each): a single user walking into an empty cache
pays one prefill and every later request rides it. The pathology in
finding 3 needs *concurrency*; the economics in finding 2 need it too.
Cold-start behavior is benign alone and catastrophic stacked.

## Accounting notes

* OFF rows are the frozen 4A measurements (input 8192 nominal, unique
  suffixes) vs this study's 8320 nominal — identical all-miss semantics,
  prompt sizes within 2 %.
* `cache_reset_method = process_restart`;
  `cache_warm_method = loadgen warmup run (same seed), concurrency=8,
  direct-to-worker` — recorded per leg in `lifecycle.json`.
* Gateway steady-state probing disabled during measured runs
  (`FUNCTIONAL_PROBE_INTERVAL_READY_S=0`); startup gating and the post-leg
  functional probe stayed on.
* One GPU, one model (Qwen3.5-9B Q4_K_M, ctx 98304, 8 slots), speculative
  decoding off, sampling unchanged (temperature 1.0 from loadgen), no
  admission control — one systems variable changed per comparison.

## What this sets up

* **Phase 4B.2 — affinity routing:** finding 3 makes the routing question
  concrete *before* any multi-worker simulation exists: route by queue
  depth, or by cache locality? llama.cpp already answers "both, in that
  order, and locality-first during population is how you get 253-second
  TTFTs".
* **Eviction** stays a separate experiment (A/A/B/A/C/A access patterns)
  — with eight slot copies of one prefix costing ~6 GB, the eviction
  question on this machine is really "when does the KV pool evict vs the
  system swap".
* **vllm-metal** must gain a real prefix cache (or be replaced by a vLLM
  build with one) before it can re-enter cache comparisons. The negative
  result is recorded so nobody re-discovers it the expensive way.
