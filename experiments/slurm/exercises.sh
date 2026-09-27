#!/usr/bin/env bash
# Slurm lab exercises (phase 6): the batch/HPC vocabulary, on FAKE GPUs.
# Run: make slurm-demo   (after make slurm-up)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SCTL="docker compose -f deploy/slurm/docker-compose.yml exec -T slurmctl"

echo "== sinfo: partitions and node states =="
$SCTL sinfo

echo
echo "== sbatch: submit a job that requests one FAKE A100 =="
$SCTL sbatch --wrap 'echo "hello from $(hostname); allocated GPUs: $CUDA_VISIBLE_DEVICES (fake)"; sleep 2' \
    --gres=gpu:1 -J gpu-hello -o /dev/stdout
sleep 3

echo
echo "== squeue: the queue (online inference has none of this) =="
$SCTL squeue

echo
echo "== srun: interactive allocation of both GPUs on node01 =="
$SCTL srun -N1 -w node01 --gres=gpu:a100:2 --pty hostname 2>/dev/null || \
    $SCTL srun -N1 -w node01 --gres=gpu:a100:2 hostname

echo
echo "== scontrol: inspect a node's GRES =="
$SCTL scontrol show node node01 | grep -E "NodeName|Gres|State|RealMemory"

echo
echo "== drain and resume node02 (the Slurm counterpart of k8s cordon/drain) =="
$SCTL scontrol update nodename=node02 state=drain reason="simulated-maintenance"
$SCTL sinfo -N
$SCTL scontrol update nodename=node02 state=resume
$SCTL sinfo -N

echo
echo "== sacct/accounting-free note: priorities are basic here; see README =="
echo "slurm-demo complete"
