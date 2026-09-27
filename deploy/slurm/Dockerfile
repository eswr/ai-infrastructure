# syntax=docker/dockerfile:1
# Slurm lab image (phase 6 — batch/HPC side of the datacenter story):
# controller and compute nodes share one image; roles differ by command.
# GRES "GPUs" here are FAKE (slurm.conf Gres=gpu:a100:2, no devices behind
# them) — the point is the scheduler vocabulary, not accelerators.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       slurm-wlm munge vim less procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /etc/slurm \
    && chown -R slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

COPY slurm.conf /etc/slurm/slurm.conf
COPY gres.conf /etc/slurm/gres.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6817 6818 6819
ENTRYPOINT ["/entrypoint.sh"]
