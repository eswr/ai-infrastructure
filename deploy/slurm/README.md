# Slurm lab (phase 6) — the batch/HPC side of the datacenter story

Kubernetes (phase 5) runs **online serving**: low-latency, admission-controlled,
autoscaled. Slurm runs the other half of a real datacenter: **queued batch
work** — training jobs, batch inference, HPC workloads with partitions,
priorities, and accounting. This lab teaches that vocabulary on one laptop.

**The GPUs here are FAKE.** `slurm.conf` declares `Gres=gpu:a100:2` per node
and `gres.conf` backs it with no devices — scheduler-visible resources that
compute nothing. That is the point: `--gres` allocation, queue position, and
drain/resume are *scheduler* concepts, independent of what (if anything) is
behind the resource. Nothing in this lab is an NVIDIA measurement.

```bash
make slurm-up     # 1 controller + 2 compute nodes (docker compose)
make slurm-demo   # guided exercises, ~1 min
make slurm-down
```

## The exercises (`experiments/slurm/exercises.sh`)

| command | what it teaches | k8s counterpart |
|---|---|---|
| `sinfo` | partitions, node states (`idle*`, `drain`) | `kubectl get nodes` |
| `sbatch --gres=gpu:1` | resource-requested batch submission | (admission + scheduler pick) |
| `squeue` | the queue IS the interface | admission queue gauges |
| `srun --gres=gpu:a100:2` | interactive allocation of specific resources | `kubectl exec` + device plugin |
| `scontrol show node` | GRES inventory per node | `nvidia.com/gpu` allocatable |
| `scontrol update ... state=drain` | drain a node: queued elsewhere, running finishes | cordon + drain |
| `state=resume` | back to scheduling | uncordon |

## Concepts mapped back to inference-lab

* **GRES** ≈ the placement gate's `vram_gb` capability: a request declares a
  resource need; the scheduler refuses nodes that cannot satisfy it.
* **drain/resume** ≈ phase 6D node CORDONED/DRAINING/MAINTENANCE.
* **partitions** ≈ the service classes (interactive vs batch) — a queue
  boundary with its own policy.
* **backfill** ≈ ECT's smaller-job-fits-in-the-gap intuition, formalized.

## Topology

```
slurmctl (slurmctld :6817)  ← munge key shared via volume
  ├── node01 (slurmd, Gres=gpu:a100:2)
  └── node02 (slurmd, Gres=gpu:a100:2)
```
