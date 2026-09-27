# Phase-5 milestone demo — the k3d control plane

Recorded 2026-09-27 · commit `c5ec2ea` (tagged `phase-5-k3d-control-plane`)
· total runtime **1.8 min** (target < 5) · cluster: k3d `inference-lab` on
the M5, gateway + mock workers as pods, one native Metal beat at the end.

Reproduce with:

```bash
make k8s-up      # once
make k8s-demo    # the six beats, narrated and timed
```

Raw capture: [`demo-transcript.log`](./demo-transcript.log).

## The six beats

| # | beat | proves | time |
|---|------|--------|------|
| 1 | `kubectl get pods` | gateway + 2 workers READY; registry discovered from kubernetes, capabilities from pod labels — no hard-coded list | 1 s |
| 2 | queue pressure | 12 high-work requests cross `estimated_queue_seconds` threshold → autoscaler 2 → 3 (queue signal, never CPU) | 41 s |
| 3 | wedge one worker | `/health` still HTTP 200, functional probe fails → UNHEALTHY → pod readiness false → 6/6 chats served elsewhere, 0 on the wedged pod | 11 s |
| 4 | recover the worker | RECOVERING → WARMING → READY; generation epoch 3 → 5; pod Ready again | 21 s |
| 5 | delete a pod mid-stream | preStop drain: active stream COMPLETED, 5/5 concurrent chats, 0 dropped, pod leaves registry | 6 s |
| 6 | bridge to native MLX | k3d gateway → `host.docker.internal` → mlx-lm 9B-4bit on Metal: functional probe passes through the bridge, 4/4 concurrent real-model requests | 24 s |

## The one-line story

Application → scheduler → runtime → lifecycle → kubernetes all speak the
same truth: a worker receives traffic exactly when it has *proven* it can
generate, loses traffic the moment it can't, finishes what it started when
it's removed, and the pool grows only when the queue says so. The same
control plane reaches an accelerator that lives outside the cluster.
