# Runtime Comparison on Apple M5 — MLX-LM vs llama.cpp vs vLLM-Metal

*Phase 4A. Raw artifacts: [`benchmarks/results/cross-runtime-20260925-004124/`](../../benchmarks/results/cross-runtime-20260925-004124/) (per-leg matrix runs, lifecycle records, 1 Hz memory CSVs, provenance manifests), merged view `cross-runtime.json`, tables and 13 charts in `analysis/`.*

## Question

Phases 2 and 3 established a performance pathology and then explained and
controlled it — but only on one runtime. Before building anything else
(prefix caching, continuous-batching studies, Kubernetes), the prerequisite
question is:

> **How much of the phase-2/3 behavior is MLX-LM-specific, and how much is
> fundamental to the machine and the workload?**

Not "which runtime wins" — a winner bar chart is the least portable thing
this study could produce. Three runtimes, three serving architectures, one
machine:

| leg | serving architecture | weights | server |
| --- | --- | --- | --- |
| mlx-lm | single model loop, **no continuous batching** (time-slicing) | MLX affine 4-bit, group 64 | `mlx_lm.server 0.31.3` |
| llama-cpp | slot server with **continuous batching** (`-np 8`) | GGUF **Q4_K_M** | `llama-server` (brew) 0.4.1, Metal |
| vllm-metal | **vLLM scheduler + paged KV** (v0.1.0, MLX backend) | same MLX 4-bit files as mlx-lm | `vllm-metal` (no tunable CLI) |

Headline before the detail:

* **llama.cpp absorbed the exact workload that silently wedges mlx-lm** —
  8K×c8 unbounded: 27/27 runs, 72/72 requests delivered, worst per-stream
  stall 2.7 s, functional after the burst, no restart. Continuous batching
  with real slots is the difference.
* **mlx-lm reproduced its phase-3 failure signature** under harness
  observation: after 8K×c8, `/health` and `/v1/models` both answered 200
  while a real generation timed out — verdict `silent_wedge`, restart
  required, healthy 4 s after restart.
* **vllm-metal 0.1.0 cannot stream** (every "streaming" request returns one
  JSON body), gives essentially **no aggregate throughput gain** from
  concurrency (×0.87–1.16), and occasionally returns **silent empty
  answers** (usage reports 0 completion tokens after 30–60 s).
* Prefill cost per nominal token is within ±25 % across all three
  (mlx ≈ 5.5, llama ≈ 6.3, vllm ≈ 4.1 ms/tok): **the machine's prefill
  ceiling is roughly runtime-independent; what differs radically is what
  each runtime does with concurrent work.**

## Hardware and software

* Apple MacBook Air M5, 10-core, Metal 4, 24 GB unified memory.
* macOS 27.0 (26A428). All inference servers ran natively on macOS; the
  gateway and observability stack in Docker (never Metal in containers).
* Gateway: this repo's Rust Axum proxy, one worker per leg, `lor` routing,
  **admission OFF** (each runtime's own scheduler is what saturates),
  `FIRST_TOKEN_TIMEOUT_S=900` — a silent worker aborts streams loudly at
  900 s instead of hanging to the 1800 s client timeout.
* Same loadgen (`target/release/loadgen --matrix`), same prompts, same
  seed 42, same arrival process for every leg.
* llama.cpp served on :8085 (:8081 is squatted by an unrelated Docker
  proxy on this machine). Context `-c 98304 -np 8` = 8 slots × 12,288:
  the "8192-token" scenario prompt is ~11.4k tokens server-side (see
  Methodology) and llama.cpp *refuses* oversized requests instead of
  truncating. Other flags are defaults; mlx-lm and vllm-metal run entirely
  at defaults, as in phases 2–3.

## Model equivalence

Someone will correctly challenge the fairness of comparing different
quantization formats, so this is recorded rather than hidden.

* **mlx-lm and vllm-metal load bit-identical weights**:
  `mlx-community/Qwen3.5-9B-4bit` (MLX affine quantization, 4 bits, group
  size 64, ~5.5 GB). Any mlx-vs-vllm difference is pure serving software.
* **llama.cpp uses `unsloth/Qwen3.5-9B-GGUF` Q4_K_M** (~5.9 GB): both are
  4-bit-class, but k-quants' super-block allocation ≠ MLX affine group
  quantization. Numerics differ; quality is "close", not identical. All
  llama.cpp rows carry that asterisk.
* Same model family everywhere: Qwen3.5-9B — 32 layers, hybrid
  full/linear attention, 4 KV heads × head_dim 256, 262k native context,
  **thinking model** (reasoning tokens stream as tokens and count as
  tokens; llama.cpp names the field `reasoning_content`, vLLM/MLX name it
  `reasoning` — the loadgen and gateway count both).
* Tokenizer is identical everywhere, so server-side prompt lengths match
  across legs for identical client prompts.

## Methodology

* **Client-side measurement** (the latency users feel): loadgen timestamps
  every streamed token per request — TTFT, ITL p50/p95/p99,
  **per-stream worst-gap stall**, completion time, absolute completion
  spread, slowest/fastest ratio. Server usage chunks capture truth token
  counts. A stream that ends "cleanly" with zero tokens is counted a
  failure (`delivered` = ok ∧ ≥1 token) — phase 4 learned this the hard
  way: watchdog-aborted starved prefills end cleanly and used to count
  as "ok".
* **Same gateway in front of every leg**, instrumented: TTFT
  decomposition + `inference_stream_inter_token_seconds`,
  `inference_stream_max_stall_seconds`,
  `inference_stream_completion_seconds`.
* **Lifecycle-aware benchmarking** — process health alone cannot be
  trusted (phase 3). Every leg walks
  `start → wait_live → wait_functional (real generation) → benchmark →
  functional_probe immediately after the worst workload → verdict
  (healthy | silent_wedge | loud_failure | dead) → restart if degraded →
  stop`, with 1 Hz memory sampling: process-tree RSS, vmmap physical
  footprint (RSS hides wired Metal buffers), system used/free, kernel
  pressure level.
* 9 measured runs per leg: prompt {128, 1024, 8192} × concurrency
  {1, 4, 8}, 8 requests per run (one closed-loop wave at c8), burst
  arrival, out = 256, streaming requested, temperature 1.0, per-request
  unique prompts, 10 s cooldown. 27 measured runs total.

## Workload matrix

| | c=1 | c=4 | c=8 |
| ---: | :-: | :-: | :-: |
| prompt 128 | ✓ | ✓ | ✓ |
| prompt 1024 | ✓ | ✓ | ✓ |
| prompt 8192 | ✓ | ✓ | ✓ |

## Single-request performance

The runtimes' happy path (p128, c=1; full table in
[`analysis/cross-runtime-tables.md`](../../benchmarks/results/cross-runtime-20260925-004124/analysis/cross-runtime-tables.md)):

| | TTFT p50 | ITL p50 | aggregate tok/s |
| --- | ---: | ---: | ---: |
| mlx-lm | 1.08 s | 50 ms | 12.9 |
| llama-cpp | 1.14 s | 77 ms | 10.2 |
| vllm-metal | 17.7 s * | — (no stream) | 8.7 |

\* vllm-metal's "TTFT" is its whole completion: it does not stream, so
first byte = last byte and the number includes all 256 decode steps.
Its decode-only rate (~14 tok/s) is in the same class as the other two.

At c=1 the two streaming runtimes are within noise of each other on the
happy path — TTFT ≈ 1.1 s, ITL 50–77 ms. Even mlx-lm's phase-2 story is
confirmed: nothing is wrong with mlx-lm *at c=1*.

## Prefill scaling

Fit at c=1: `TTFT ≈ fixed + rate × input_tokens` (nominal client tokens;
server-side Qwen tokens ≈ 1.4× nominal — divide rates by ~1.4 for
ms/real-token):

| runtime | fixed | rate | note |
| --- | ---: | ---: | --- |
| mlx-lm | −0.51 s | **5.46 ms/tok** | phase-2 measured 0.75 s + 5.35 ms/tok — reproduced |
| llama-cpp | 0.23 s | **6.30 ms/tok** | ~18 % slower prefill than MLX at c=1 (51.9 s vs 44.3 s at 8K) |
| vllm-metal | 13.54 s | **4.12 ms/tok** | intercept is the *entire decode* hidden in TTFT (non-streamed); the slope is a clean prefill estimate — the fastest of the three |

Two honest observations. First, prefill is remarkably portable: ~180–240
nominal-tok/s across three completely different kernel stacks. Second,
vllm-metal's fixed 13.5 s intercept is a *serving-contract* difference, not
a performance one: every response includes full decode before the first
byte arrives.

## Concurrency scaling

What c=8 buys and costs vs c=1 (×ratios; "—" = that leg collapsed, see
next sections):

| runtime | prompt | tok/s ×c8/c1 | TTFT ×c8/c1 |
| --- | ---: | ---: | ---: |
| mlx-lm | 128 | ×2.67 | ×4.73 |
| mlx-lm | 1024 | ×1.01 | ×12.48 |
| mlx-lm | 8192 | ×0 (0 delivered) | — |
| llama-cpp | 128 | ×2.14 | ×1.20 |
| llama-cpp | 1024 | ×2.27 | **×0.20** |
| llama-cpp | 8192 | **×4.08** | **×0.37** |
| vllm-metal | 128 | ×1.09 | ×3.26 |
| vllm-metal | 1024 | ×0.87 | ×4.90 |
| vllm-metal | 8192 | ×1.16 | ×3.53 |

* **llama.cpp is the only runtime where concurrency helps at long
  prompt**: TTFT at c8 is *lower* than c1 (19.0 s vs 51.9 s at 8K) because
  8 slots pipeline prefills and decode together; aggregate tok/s ×4.08.
  This is what continuous batching is *for*.
* **mlx-lm confirms phase 2**: time-slicing buys ×1.01 aggregate at
  1K×c8 while TTFT explodes ×12.5. The phase-2 knee (peak gain at c2–c4,
  moving left with prompt size) reproduces exactly.
* **vllm-metal's scheduler never aggregates** on this Metal build:
  tok/s ×0.87–1.16 everywhere. Concurrency only stretches every
  individual response (TTFT ×3.3–4.9). Whatever the vLLM codebase does
  with paged KV + continuous batching on CUDA, this Metal port does not
  yet deliver it as throughput.

## Stream fairness

The new per-stream metrics (`max_inter_token_gap_ms` per request;
completion spread and slowest/fastest across the run) exist because
aggregate throughput hides exactly this:

| run | worst per-stream stall | completion spread | slowest/fastest |
| --- | ---: | ---: | ---: |
| mlx-lm p8192-c4 | **193.0 s** | 140.7 s | ×1.43 |
| mlx-lm p8192-c8 | n/a (0 streams delivered) | 4.0 s | ×1.00 |
| llama-cpp p8192-c8 | **2.55 s** | ~0 s | ×1.00 |
| llama-cpp p1024-c8 | 2.73 s | ~0 s | ×1.00 |
| vllm-metal p8192-c8 | — (non-streamed) | **287.5 s** | **×10.93** |

* **mlx-lm starved streams for minutes**: at 8K×c4 a mid-stream gap of
  193 s — a user watching one of these streams sees the spinner freeze
  for 3+ minutes while the run's aggregate tok/s (2.6) looks merely
  "slow". This is the phase-2 pathology measured per-stream, and it is
  *worse* than the phase-2 aggregate view suggested.
* **llama.cpp is fair by construction**: everyone streams (≤2.7 s worst
  gap), everyone finishes together (spread ~0 at c8 — the whole wave
  completes as a batch).
* **vllm-metal's unfairness is completion-time unfairness**: some
  requests finish minutes before others (×10.9 spread ratio at 8K×c8).
* Per-stream ITL under batching: llama.cpp pays for fairness with ITL —
  313–476 ms p50 at c8 vs mlx's 137–212 ms. That is the honest
  batching trade: worse instantaneous per-stream cadence, radically
  better first-token and completion fairness.

## Memory behavior

Peaks across each leg (24 GB machine; footprint = vmmap physical
footprint, which counts wired Metal buffers that RSS hides):

| runtime | peak tree RSS | peak footprint | peak system used | pressure |
| --- | ---: | ---: | ---: | ---: |
| mlx-lm | 5.4 GB | **18.5 GB** | **21.4 GB (89 %)** | stayed 1 (normal) — barely |
| llama-cpp | 10.3 GB | 15.3 GB | 16.4 GB (68 %) | 1 |
| vllm-metal | 3.6 GB | (not captured) | 15.5 GB (65 %) | 1 |

* **mlx-lm ran the machine to 89 % system memory** during the 8K runs —
  the context in which its Metal GPU OOM wedged in phase 3. The wedge is
  not mysterious: unbounded per-request KV × 8 concurrent prefills
  approaches the wire limit, and this study got within one unlucky
  allocation of reproducing it under observation.
* **llama.cpp pays up front for headroom**: the 98k-token context pool
  costs ~10 GB RSS at load, but that *is* its robustness — the 8K×c8
  burst fits inside space it already owns. Memory reserved early buys
  latency-stable memory later.
* Process RSS is a misleading metric on Apple Silicon: mlx-lm's RSS
  *collapsed to ~270 MB* once weights moved to wired GPU memory, and
  without the vmmap footprint the model would look "tiny" while the
  system sat at 89 %. Footprint is the number that behaves.
* vllm-metal's footprint sample landed on its `uv` wrapper process
  (harness fix post-study: vmmap the tree's heaviest pid). Its RSS and
  the system counter show it as the lightest runtime.

## Saturation/failure behavior

Each runtime fails in its own signature way:

* **mlx-lm — connection drops, then stream starvation, then silent
  wedge.** At c8 even tiny prompts: 2/8 connections refused (fast 502s —
  the phase-3 rerun's accept-path behavior). At 8K×c8: 6 streams sat
  with **zero tokens** until the gateway's 900 s watchdog aborted them,
  2 more connections dropped, and the worker ended the run wedged
  (accepting GETs, never another POST). This is the phase-3 signature,
  now reproduced with per-stream instrumentation and a lifecycle verdict.
* **llama.cpp — stable degradation.** Nothing failed: ITL p50 degraded
  gracefully (77 → 476 ms), TTFT stayed ≤19 s at c8/8K, all 72 requests
  delivered. The failure mode you get is "slower", not "broken".
* **vllm-metal — queueing-style degradation + silent empty answers.**
  Every concurrent request's completion stretches (TTFT=e2e ×3.3–4.9),
  aggregate never improves, completion-time unfairness reaches ×10.9,
  and 3 of 72 responses arrived as complete JSON bodies reporting **0
  completion tokens** after 33–58 s — a silently empty answer that only
  the delivered-token check catches. Plus the architectural gap: it
  never streams at all.

## Functional recovery

The lifecycle table (this is the part phase 4D will productize):

| runtime | after 8K×c8 | `/health` said | real generation | restart needed |
| --- | --- | --- | --- | --- |
| mlx-lm | `silent_wedge` | 200, 200 | **timed out** | **yes** — healthy 4 s after restart |
| llama-cpp | `healthy` | 200 | ok (SSE) | no |
| vllm-metal | `healthy` | 200 | ok (JSON, mode recorded) | no |

 mlx-lm's post-wedge reality: `/health` and `/v1/models` both answered
200 while the functional probe timed out — the exact lie the phase-3
harness was built to catch, now caught automatically, with a restart
and post-restart probe proving recovery. llama.cpp's post-burst probe
took 38.4 s to first event (the last batch was still draining) but
succeeded — a drain delay, not damage.

## Runtime architecture interpretation

One sentence per runtime, earned above:

* **mlx-lm** is a research runtime with a production workload's dangers:
  excellent single-stream latency (50 ms ITL), fast prefill, and zero
  protection against concurrency — no batching, no queueing, no
  memory admission, and a failure mode (silent wedge) that *hides*
  from process health. Safe only behind an admission-controlled
  gateway (phase 3's result, now cross-validated).
* **llama.cpp** is the most production-shaped server on this machine:
  slots + continuous batching convert a killer workload into graceful
  ITL degradation, at the cost of permanent memory headroom and ~18 %
  slower c1 prefill. Its fairness is structural, not tunable.
* **vllm-metal** is a promise, not yet a server: the scheduler
  vocabulary (paged KV, continuous batching) is present, but on this
  build it neither streams, nor aggregates throughput, nor answers
  reliably (silent empties). It is the *architecture* to study — the
  codebase the roadmap points at — but not yet the runtime to serve
  from.

## Limitations

* **Quantization**: llama.cpp rows are Q4_K_M, not MLX 4-bit; close in
  size/quality, not equivalent. mlx↔vllm comparisons are bit-identical.
* **Tokenizer divergence**: nominal client tokens vs ~1.4× server-side
  tokens (known since phase 3). All runs share it, so cross-runtime
  comparisons hold; absolute ms/token values are per nominal token.
* **vllm-metal has no knobs** (no ctx/parallel/batch flags exposed), so
  its leg is whatever 0.1.0 defaults to; its non-streaming makes ITL and
  per-stream stall undefined by construction.
* **n = 8 per run**: percentile estimates at c=1 rest on 8 samples;
  enough for the ×2–12 effects reported, not for fine percentile work.
* The gateway-side fairness histogram counts were not captured into the
  lifecycle records (a truncated metrics scrape in the harness — fixed
  post-study); the fairness numbers above come from the loadgen's
  per-token timelines, which are complete.
* vllm-metal's vmmap footprint sampled its wrapper process; its true
  peak footprint is unknown (RSS + system counters recorded).
* One machine, one thermal state, no repetitions: effects at ×2 or
  larger are trustworthy; the 128-prompt c1 differences between mlx and
  llama (±10 %) are not.

## Conclusions

1. **The phase-2/3 pathology is mostly MLX-LM, not the machine.** The
   same hardware and workload, served by llama.cpp's slot scheduler,
   delivers 100 % of requests with 2.5 s worst stalls where mlx-lm
   starves streams for minutes and wedges silently. The machine can
   take 8K×c8; mlx-lm cannot.
2. **What transfers to production GPUs is the taxonomy, not the numbers**:
   time-slicing vs slots vs scheduler-first architectures; stable
   degradation vs starvation vs silent wedge; memory headroom bought
   early vs memory pressure discovered late. These are exactly the
   axes a Kubernetes-era worker state machine (phase 4D) must encode.
3. **Fairness metrics earned their place**: aggregate tok/s ranked
   mlx-lm *above* llama.cpp at 128×c8 (34.6 vs 21.8) while dropping 2/8
   connections; the per-stream stall and delivered counts invert that
   story. Every future benchmark in this lab reports them.
4. **The lifecycle contract is the study's second result.** `wait_live`
   plus `wait_functional` plus a post-worst-case probe turned "is it
   broken?" from a phase-3 investigation into a one-table answer — and
   it is now a permanent part of every experiment this lab runs.
5. **Next**: prefix caching (4B) and continuous-batching-deep-dive (4C)
   now have a clean cache=OFF baseline to measure against — and the
   phase-4D worker state machine has three concrete failure signatures
   (silent wedge, connection drop, silent empty answer) it must catch.
