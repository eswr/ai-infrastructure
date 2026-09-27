# The interview demo kit

One message for the whole project:

> "I started from real inference measurements, found failure modes, built
> controls around them, then moved the same ideas into Kubernetes and a
> simulated datacenter fleet."

The narrative behind that arc is [interview-walkthrough.md](interview-walkthrough.md);
the one-page results slide is [results/interview-scorecard.md](results/interview-scorecard.md).
This page is the operational kit: what to run, how long it takes, what to
say, and what to fall back to if the machine misbehaves.

## The demo targets

Every target is deterministic, self-contained, and finishes in under three
minutes. There is deliberately **no `demo-all`** — an interviewer never
watches a 20-minute script.

| Target | Time | What it proves |
|---|---|---|
| `make demo-metal` | ~30 s | Real inference measurement: MLX 9B-4bit on Metal, client-side TTFT/ITL/e2e |
| `make demo-wedge` | ~10 s | `/health` lies: silent wedge → functional-probe quarantine → recovery, generation epoch |
| `make demo-admission` | instant | The 0/16 → 16/16 overload incident, frozen raw artifacts |
| `make demo-k8s` | ~2 min | The six-beat k3d control plane: discovery, queue-signal autoscale, wedge→readiness, recovery, drain, Metal bridge |
| `make demo-placement` | ~10 s | Heterogeneous fleet discovery + placement matrix (TP groups, VRAM, KV bounds) |
| `make demo-fleet-failure` | ~15 s | Differentiated failure drills: XID→fail→replace, thermal→degrade, OOM→quarantine |
| `make demo-slurm` | ~10 s | Slurm GRES vocabulary: sinfo / sbatch --gres / srun / drain-resume |

`demo-placement`, `demo-fleet-failure` and `demo-k8s` need the cluster:
`make k8s-up` first (~5 min, or bring it before the call). `demo-slurm`
starts its compose lab itself if needed.

## Three demo modes

**5 minutes — recruiter / hiring manager**

- 0:00 one diagram: `request → estimate → admission → scheduler → worker → stream`
- 0:45 `make demo-metal` — "real model, real numbers"
- 1:30 the silent-wedge story (`make demo-wedge`, then talk)
- 2:30 `make demo-k8s` beats 2/3/5 (autoscale, readiness, drain)
- 4:15 the measured / exercised / simulated boundary ([scorecard](results/interview-scorecard.md))

**15 minutes — technical screen**

- 2 min architecture diagram
- 3 min real inference + why TTFT and ITL are separate axes
- 3 min admission incident (`demo-admission` evidence + the budget model)
- 3 min cache-hotspot incident (frozen: 19 s → 253 s → calibrated routing, MAE −37%)
- 3 min `make demo-k8s` (lifecycle + autoscaling)
- 1 min production-system mapping ([architecture-mapping.md](architecture-mapping.md))

**30 minutes — deep dive**

First 10 minutes as above, then let THEM choose — scheduler, lifecycle,
Kubernetes, placement, observability, or Slurm. Do not pick for them:

> "Would you like to go deeper into the scheduler, runtime failure
> handling, or the Kubernetes/fleet side?"

## Talk tracks that carry each demo

**Demo-metal.** "Before building scheduling logic I wanted real
measurements — this is a real 9B model on Metal. The load generator
measures TTFT, inter-token latency, e2e, throughput, stalls and failures
per stream." If asked why TTFT *and* ITL: "TTFT exposes queueing and
prefill cost; ITL exposes decode quality and stream starvation. Aggregate
tokens/sec hides a terrible interactive experience, the pair doesn't."

**Demo-wedge.** "One of the real runtimes taught me that HTTP health is
not inference health — the process stayed up, `/health` stayed 200, and
generation was dead. So health is split into process, transport,
functional and semantic, and readiness is earned by a real tiny
generation, never by `/health`."

**Demo-k8s.** "Workers aren't hard-coded anymore — pods and their labels
are the registration. And I deliberately don't autoscale on CPU: I scale
on inference signals, estimated queue seconds and queued work."

**Cache incident (frozen).** "Naive cache affinity made a cold burst
13× worse — every request wanted the one warm slot while eight slots sat
idle. The cache wasn't the problem; unbounded cache affinity was." Price
every candidate: `est_ttft = queued_work + remaining_work + recompute_tokens × prefill_rate`, refuse affinity under pressure, then teach the cost
model from its own errors (MAE −37%).

**Admission incident (frozen).** "I stopped maximizing concurrency and
bounded how much estimated inference work could enter the accelerator:
same workload, 0/16 → 16/16." The model: `active_work + queued_work +
new_request_cost ≤ budget`, 429 + Retry-After past the ceiling.

**Placement/failure (live or frozen).** "These GPU capacities are
simulated — I don't claim NVIDIA measurements from this. What's exercised
is placement policy and differentiated response: XID fails a node,
thermal degrades it while traffic continues, OOM quarantines one worker.
Not every hardware signal should trigger the same response."

## Never rerun the expensive experiments

These stay frozen; show command + tag + artifact:

- 8K × c8 saturation A/B → `make experiment-admission-saturation` (~55 min) — frozen at `docs/results/run-A-console.txt` / `run-B-console.txt`
- full cache studies → `experiment-prefix-cache` / `experiment-prefix-router` / `experiment-calibration` (~45–90 min each) — curated dirs under `docs/results/`
- 16-cell 4C matrix + multi-hour mixed baseline → `experiment-serving-interactions-full` (hours) — `docs/results/serving-interactions-20260927-070425/`

The professional line: "This experiment takes hours, so I'm showing the
frozen raw artifacts rather than rerunning it."

## Failure-safe fallbacks

Frozen transcripts of every demo target, captured from real runs, live in
[results/interview-demo/](results/interview-demo/) — commit them. If
Docker Desktop or the GPU misbehaves mid-call:

> "I have the frozen transcript from this exact target right here."

and continue immediately. Never debug Docker during an interview.

| Fallback | Transcript |
|---|---|
| `demo-metal` | [results/interview-demo/metal.txt](results/interview-demo/metal.txt) |
| `demo-wedge` | [results/interview-demo/lifecycle-wedge.txt](results/interview-demo/lifecycle-wedge.txt) |
| `demo-admission` | [results/interview-demo/admission.txt](results/interview-demo/admission.txt) |
| `demo-k8s` | [results/interview-demo/k8s-demo.txt](results/interview-demo/k8s-demo.txt) |
| `demo-placement` | [results/interview-demo/placement.txt](results/interview-demo/placement.txt) |
| `demo-fleet-failure` | [results/interview-demo/fleet-failure.txt](results/interview-demo/fleet-failure.txt) |
| `demo-slurm` | [results/interview-demo/slurm.txt](results/interview-demo/slurm.txt) |
