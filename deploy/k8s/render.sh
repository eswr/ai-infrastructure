#!/usr/bin/env bash
# Render deploy/k8s/base -> deploy/k8s/rendered: inject the Docker Desktop
# host IP so in-cluster pods can resolve host.docker.internal (the native
# macOS Metal worker bridge), and inline the autoscaler script into its
# ConfigMap.
set -euo pipefail
cd "$(dirname "$0")/.."   # deploy/

HOST_IP=$(docker run --rm python:3.12-slim getent hosts host.docker.internal | awk '{print $1}' | head -1)
if [[ -z "${HOST_IP}" ]]; then
  echo "error: could not resolve host.docker.internal from inside docker" >&2
  exit 1
fi

rm -rf k8s/rendered
mkdir -p k8s/rendered

for f in k8s/base/*.yaml; do
  sed "s/__HOST_IP__/${HOST_IP}/g" "${f}" > "k8s/rendered/$(basename "${f}")"
done

# Inline scaler.py into the ConfigMap (4-space indent = valid YAML block).
python3 - <<EOF
script = open("k8s/scaler.py").read()
cm = open("k8s/rendered/30-scaler.yaml").read()
open("k8s/rendered/30-scaler.yaml", "w").write(cm.replace("__SCALER_SCRIPT__", "\n".join("    " + l if l.strip() else "" for l in script.splitlines())))
EOF

echo "rendered deploy/k8s/rendered (host.docker.internal -> ${HOST_IP})"
