#!/usr/bin/env bash
# Shared entrypoint: munge bootstrap (the key is generated once into the
# shared volume; every node mounts the same secret), then the role command.
set -euo pipefail

# munged requires these exact paths/permissions and refuses root.
mkdir -p /run/munge /var/lib/munge /var/log/munge /etc/munge
chown munge:munge /run/munge /var/lib/munge /var/log/munge
chmod 755 /run/munge
chmod 700 /var/lib/munge /etc/munge

if [[ ! -s /var/lib/munge/munge.key ]]; then
  echo "[slurm-lab] generating munge key"
  dd if=/dev/urandom of=/var/lib/munge/munge.key bs=1 count=1024 2>/dev/null
  chown munge:munge /var/lib/munge/munge.key
  chmod 400 /var/lib/munge/munge.key
fi

su -s /bin/bash munge -c "munged" 
for i in $(seq 1 20); do
  [ -S /run/munge/munge.socket.2 ] && break
  sleep 0.5
done
if munge -n | remunge >/dev/null 2>&1; then
  echo "[slurm-lab] munge ok"
else
  echo "[slurm-lab] WARNING: munge self-check failed"
fi

exec "$@"
