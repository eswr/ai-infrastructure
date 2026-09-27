# helm (deferred)

Intentionally empty for now. Phase 5 ships plain manifests
(`deploy/k8s/base/`) with a one-file render step (`render.sh`) — no chart
indirection was worth it for one namespace. If the deployment matrix grows
(multiple worker pools, per-pool models, real accelerators), wrap those
manifests in a chart modelled after vLLM Production Stack: values for the
worker pool, model registry, router policy, observability ServiceMonitors.
The custom queue-signal scaler stays a scaler; HPA/KEDA mapping is a later,
deliberate step.
