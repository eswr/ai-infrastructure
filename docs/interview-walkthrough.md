# The 10-minute walkthrough — three incidents

A spoken walkthrough of inference-lab, structured as three incidents
instead of a commit tour. Each follows the same arc: **symptom →
measurement → solution → transferable lesson**. Total ~10 minutes; each
incident has one artifact to show and one number to say out loud.

---

## Incident 1 — the server that answered /health but answered nobody (3 min)

**Symptom.** Driving 8 concurrent 8K-context requests through mlx-lm on
the M5: 16 requests submitted, **0/16 answered**. The process stayed up,
answered `/health` with 200, and never served another completion. Same
shape reproduced on later mixed workloads: a 4.6-hour run where the
baseline wedge produced 0/8 completions while the server looked alive.

**Measurement.** A matrix runner (warmup + measured runs, per-stream
fairness metrics, 1 Hz memory sampling with `vmmap` footprint) showed the
signature: long-context concurrent prefill drove memory to ~89% of the
machine, TTFT stretched to hundreds of seconds, and the worker wedged
inside Metal allocation — *silently*. The failure class: HTTP-clean,
inference-dead.

**Solution.** Work-budgeted admission control at the gateway: every
request is priced (estimated prefill + decode from a calibrated cost
model, later budgeted against reusable prefix so warm requests aren't
over-charged), admitted against active-work budgets with
interactive-first queue priority, refused with Retry-After past an
overload ceiling. Cancellation releases the budget instantly.

**The number:** the same 8K×c8 saturation workload, unbounded vs
admission-controlled: **0/16 delivered → 16/16 delivered**, inter-token
latency p50 ≈ 60 ms controlled.

**Lesson:** the interesting overload failures are *silent*; a control
plane needs its own acceptance contract (did tokens actually flow?) and
the authority to say "not yet."

## Incident 2 — the cache that made things worse (3 min)

**Symptom.** After adding prefix-cache routing, a *cold* burst of 8
same-prefix requests had TTFT p50 of **253 s — versus 19 s when we
disabled cache affinity entirely**. The cache made a cold population 13×
worse: every request wanted the one warm slot, serialized behind it, and
the other slots sat idle.

**Measurement.** Per-decision tables exporting every candidate's
estimated TTFT, queue cost, and reusable-prefix match — plus runtime
reuse telemetry (`cache_n`) to separate predicted from actual hits.

**Solution, in two layers.** First, load-gated cache-aware routing:
price every candidate as *remaining work + (prompt − reusable prefix) ×
prefill rate*, take the argmin, and refuse affinity when queue/prefill
pressure says otherwise (with the refusal counted by reason). Second,
because bursts were decided on seed rates and the model was badly wrong
under load (decode under-priced 10–13×), a **persistent calibration**
layer: per-concurrency-bucket decode rates and prefill contention
multipliers learned from live dispatches, persisted, reloaded, and
audited via labeled prediction-error histograms.

**The numbers:** calibration cut pooled routing-prediction MAE
**250.6 s → 158.5 s (−37%)** and fixed a systematic bias; honest pricing
made even non-cache-aware policies spread correctly (all-warm p50 253 s
→ 26 s under least-work). The measured cache upside, when routing got it
right: warm TTFT 51.9 s → 2.1 s (×24.7) at c1, and reuse value growing
×270 from c1→c8.

**Lesson:** cache locality is a *trade*, not a rule — and a cost model
that is wrong in one regime is worse than no model, so the model must
observe its own error.

## Incident 3 — from false-healthy workers to a datacenter control plane (4 min)

**Symptom.** Incident 1's wedge had a nasty property: recovery existed,
but nothing distinguished a wedged worker from a healthy one — `/health`
was 200 in both. And the same blind spot scaled up: at fleet size, "which
workers exist, can they serve, should this request go there, and should
this node keep taking work?" had no answer.

**Measurement → design.** Four health dimensions (process/transport/
functional/semantic), with readiness defined by a tiny *real generation*
probe — `Reply with: OK`, acceptance requires tokens > 0 — because the
silent wedge answers `/health` for hours. Escalation by reason class
(immediate for probe timeouts; accumulating for one-off blips), recovery
by repeated passes, generation epochs so a wedged incarnation and its
healed successor are never the same worker in telemetry.

**Scale-up to Kubernetes.** Workers discovered from pods + endpoints with
capability labels (no static list); pod readiness points at the
*gateway's* lifecycle verdict, so a wedged worker leaves its Service in
seconds; draining finishes in-flight streams before termination;
autoscaling on the gateway's estimated queue seconds and interactive SLO
violations — never CPU. All of it re-verified in minutes: `k8s-check`,
six scenarios (discovery, wedge→readiness-false, recovery with epoch
bump, drain, rolling replacement with zero dropped streams, autoscale),
plus a native-Metal bridge run.

**Then one layer of abstraction.** A simulated heterogeneous GPU fleet
(2×H100 TP pair, L40S, A10 — `simulation="true"` everywhere), a placement
gate that refuses impossible dispatches *before* them (70B-fp16: nowhere
single-GPU; TP-2: the H100 pair; 400K context: KV alone exceeds every
budget), node lifecycle (provision/ready/cordon/drain/maintenance/failed/
recover), and differentiated failure drills: XID 48 fails a node;
thermal cordons it while traffic continues; OOM quarantines one worker.

**The number(s):** `k8s-check` 6/6 in ~3 min, `fleet-check` 6/6 in <1
min — and the rolling-replacement scenario's zero dropped requests while
a pod dies mid-stream.

**Lesson:** readiness must be earned by evidence, not asserted by
liveness; and the control plane's last question — "should the fleet
grow?" — belongs to queue signals, not CPU.

---

## Closing beat (30 s)

Six production systems — llm-d, Dynamo, vLLM, GPU Operator, dcgm-exporter,
Slurm — were then traced at source against these designs
([architecture-mapping.md](architecture-mapping.md)): where they matched,
where they diverged, and which simplifications were deliberate. Two of
their mechanisms were adopted back into the lab, tiny: exploration
against inferred cache state, and a scaling guard. The scope note in the
README says exactly what was measured on real hardware and what was
simulated — because the difference is the point.
