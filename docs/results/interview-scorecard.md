# Interview scorecard — the eight numbers

One page, eight results. Everything here is FROZEN EVIDENCE (committed
reports/transcripts, or a minutes-reproducible acceptance suite) — nothing
on this page requires rerunning an experiment in front of an interviewer.
The expensive studies stay frozen; say so and show the artifact.

| #  | Finding                              | Result                                   | Evidence (committed)                                        | Tag |
|----|--------------------------------------|------------------------------------------|-------------------------------------------------------------|-----|
| 1  | MLX overload, same workload          | **0/16 → 16/16 delivered** with admission control | [admission-saturation.md](admission-saturation.md) · [run-A](run-A-console.txt) / [run-B](run-B-console.txt) | `phase-3-admission-control` |
| 2  | Prefix-cache warm TTFT (c1, 8K)      | **51.9 s → 2.1 s (×24.7)**               | [prefix-cache-efficiency.md](prefix-cache-efficiency.md)    | `phase-4b-prefix-cache` |
| 3  | Naive cold cache affinity            | **19 s → 253 s TTFT p50 (×13 worse)** — the trap that motivated load-gated routing | [prefix-router.md](prefix-router.md) | `phase-4b2-prefix-routing` |
| 4  | Routing cost-model calibration       | prediction MAE **250.6 s → 158.5 s (−37%)**, systematic bias gone | [calibration-4b3.md](calibration-4b3.md) | (4B.3 series) |
| 5  | Mixed interactive/batch overload     | baseline wedged the runtime (4.6 h, 0/8) → full stack **8/8 in 189 s** | [serving-interactions-4c.md](serving-interactions-4c.md)    | `phase-4c-serving-interactions` |
| 6  | Interactive TTFT under mixed load    | **118 s → 47.7 s (p50)**                  | [serving-interactions-4c.md](serving-interactions-4c.md)    | `phase-4c-serving-interactions` |
| 7  | Kubernetes failure suite             | **6/6** (~3 min: discovery, wedge→readiness-false, recovery, drain, zero-drop replacement, queue-signal autoscale) | `make k8s-check` → [benchmarks/k8s/run_k8s_check.py](../../benchmarks/k8s/run_k8s_check.py) | `phase-5-k3d-control-plane` |
| 8  | Fleet failure suite (simulated GPUs) | **6/6** (<1 min: placement, XID→fail→replace, thermal→degrade, OOM→quarantine) | `make fleet-check` → [benchmarks/k8s/run_fleet_check.py](../../benchmarks/k8s/run_fleet_check.py) | `phase-6-datacenter-control-plane` |

Scope boundary to say out loud (README has it too): tiers 1–6 are MEASURED
on real Apple-Silicon inference; tier 7–8 EXERCISE the real control plane
against mock workers and a `simulation="true"` GPU fleet — no NVIDIA
performance claims anywhere.

Live demo flows for these numbers: [docs/interview-demo.md](../interview-demo.md).
The spoken narrative: [docs/interview-walkthrough.md](../interview-walkthrough.md).
