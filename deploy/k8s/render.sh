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

# Inline scaler.py + mock_dcgm.py into their ConfigMaps (4-space indent =
# valid YAML block scalar).
python3 - <<EOF
def inline(cm_path, placeholder, script_path):
    script = open(script_path).read()
    cm = open(cm_path).read()
    indented = "\n".join("    " + l if l.strip() else "" for l in script.splitlines())
    open(cm_path, "w").write(cm.replace(placeholder, indented))

inline("k8s/rendered/30-scaler.yaml", "__SCALER_SCRIPT__", "k8s/scaler.py")
inline("k8s/rendered/40-dcgm.yaml", "__DCGM_SCRIPT__", "k8s/images/mock-dcgm/mock_dcgm.py")
EOF

echo "rendered deploy/k8s/rendered (host.docker.internal -> ${HOST_IP})"
