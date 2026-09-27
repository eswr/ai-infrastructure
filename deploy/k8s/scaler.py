#!/usr/bin/env python3
"""Queue-signal autoscaler for inference-lab mock workers (phase 5).

A deliberately hand-rolled scaling loop — no HPA, no CPU utilization.
Scaling decisions come from signals the GATEWAY already exports:

    inference_estimated_queue_seconds          calibrated queue-time estimate
    inference_requests_queued                  queued work (admission queue)
    inference_interactive_slo_violations_total interactive TTFT SLO misses

Policy (per the phase plan):

    if estimated_queue_seconds > threshold  -> desired += 1
    if queued >= threshold                  -> desired += 1
    if SLO violation rate in window > rate  -> desired += 1
    if queue == 0 for a full cooldown       -> desired -= 1

CPU teaches the wrong lesson for inference; queue time IS the load signal.
Run in-cluster with the inference-lab ServiceAccount (patch deployments/scale).
"""

import json
import os
import re
import ssl
import time
import urllib.request

NAMESPACE = os.environ.get("NAMESPACE", "inference-lab")
DEPLOYMENT = os.environ.get("DEPLOYMENT", "mock-workers")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway.inference-lab.svc.cluster.local")
TICK_S = float(os.environ.get("TICK_S", "3"))
SCALE_UP_QUEUE_S = float(os.environ.get("SCALE_UP_QUEUE_S", "0.5"))
SCALE_UP_QUEUED = int(os.environ.get("SCALE_UP_QUEUED", "2"))
SCALE_UP_SLO_RATE = float(os.environ.get("SCALE_UP_SLO_RATE", "0.2"))
SCALE_DOWN_IDLE_S = float(os.environ.get("SCALE_DOWN_IDLE_S", "20"))
MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "6"))

SA = "/var/run/secrets/kubernetes.io/serviceaccount"


def k8s_api(method: str, path: str, body=None, content_type="application/json"):
    token = open(f"{SA}/token").read().strip()
    ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt")
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    req = urllib.request.Request(
        f"https://{host}:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        return json.loads(r.read() or b"{}")


def current_replicas() -> int:
    doc = k8s_api("GET", f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments/{DEPLOYMENT}")
    return int(doc.get("spec", {}).get("replicas", 0))


def ready_replicas() -> int:
    doc = k8s_api("GET", f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments/{DEPLOYMENT}")
    return int(doc.get("status", {}).get("readyReplicas", 0))


def set_replicas(n: int) -> None:
    k8s_api(
        "PATCH",
        f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments/{DEPLOYMENT}/scale",
        {"spec": {"replicas": n}},
        content_type="application/strategic-merge-patch+json",
    )


def gateway_metrics() -> str:
    with urllib.request.urlopen(f"{GATEWAY_URL}/metrics", timeout=5) as r:
        return r.read().decode()


def metric_value(text: str, name: str) -> float:
    """Sum a labelless (or single-series) gauge/counter family."""
    total = 0.0
    for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.M):
        try:
            total += float(m.group(1))
        except ValueError:
            pass
    return total


def log(msg: str) -> None:
    print(f"[scaler] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    log(f"queue-signal autoscaler: {DEPLOYMENT} in {NAMESPACE}, gateway={GATEWAY_URL}")
    log(f"up: est_queue>{SCALE_UP_QUEUE_S}s | queued>={SCALE_UP_QUEUED} | "
        f"slo_rate>{SCALE_UP_SLO_RATE}/s   down: idle {SCALE_DOWN_IDLE_S}s   "
        f"bounds [{MIN_REPLICAS},{MAX_REPLICAS}]")
    idle_since: float | None = None
    last_slo: float | None = None
    last_slo_t: float | None = None
    desired = max(current_replicas(), MIN_REPLICAS)
    # External actors (kubectl scale, a scenario) may change replicas behind
    # our back. Adopt their value after it stays different from what we last
    # applied for a few ticks (replica count also LAGS our own PATCH, so a
    # single observation is not evidence of an external hand).
    applied = desired
    drift_ticks = 0

    while True:
        time.sleep(TICK_S)
        try:
            metrics = gateway_metrics()
            replicas = current_replicas()
        except Exception as e:  # gateway/API hiccup: hold state
            log(f"probe failed ({e}); holding")
            continue

        # External actors (kubectl scale, a scenario) may change replicas
        # behind our back. Adopt their value after it stays different from
        # what we last applied for a few ticks (replica count also LAGS our
        # own PATCH, so a single observation is not evidence of an external
        # hand). Until adoption we DO NOTHING — an external change wins
        # over our stale internal count.
        if replicas != applied:
            drift_ticks += 1
            if drift_ticks >= 3:
                log(f"adopting externally-set replica count: {replicas}")
                desired = max(min(replicas, MAX_REPLICAS), MIN_REPLICAS)
                applied = desired
                drift_ticks = 0
                idle_since = None
            continue
        drift_ticks = 0

        est_q = metric_value(metrics, "inference_estimated_queue_seconds")
        queued = metric_value(metrics, "inference_requests_queued")
        active = metric_value(metrics, "inference_requests_active")
        slo = metric_value(metrics, "inference_interactive_slo_violations_total")
        now = time.time()
        slo_rate = 0.0
        if last_slo is not None and now > (last_slo_t or now):
            slo_rate = max(0.0, (slo - last_slo) / (now - last_slo_t))
        last_slo, last_slo_t = slo, now

        # Phase-7 adoption (Dynamo planner): while a previous adjustment is
        # still starting (ready < desired), OBSERVE ONLY — stacking
        # scale-ups before the last ones register thrashes the pool, and
        # scale-downs would race a startup.
        if ready < desired:
            log(f"scaling in progress (ready={ready} < desired={desired}); observing only")
            continue

        reason = None
        if est_q > SCALE_UP_QUEUE_S:
            reason = f"estimated_queue_seconds={est_q:.2f}"
        elif queued >= SCALE_UP_QUEUED:
            reason = f"queued={queued:.0f}"
        elif slo_rate > SCALE_UP_SLO_RATE:
            reason = f"slo_violation_rate={slo_rate:.2f}/s"

        try:
            if reason and desired < MAX_REPLICAS:
                desired = min(desired + 1, MAX_REPLICAS)
                idle_since = None
                log(f"SCALE UP -> {desired} ({reason}; est_q={est_q:.2f} queued={queued:.0f} "
                    f"active={active:.0f} replicas={replicas})")
                set_replicas(desired)
                applied = desired
            elif est_q == 0 and queued == 0 and active < desired:
                idle_since = idle_since or now
                if now - idle_since >= SCALE_DOWN_IDLE_S and desired > MIN_REPLICAS:
                    desired = max(desired - 1, MIN_REPLICAS)
                    idle_since = now  # re-arm for the next step down
                    log(f"SCALE DOWN -> {desired} (queue empty {SCALE_DOWN_IDLE_S:.0f}s, "
                        f"active={active:.0f} replicas={replicas})")
                    set_replicas(desired)
                    applied = desired
            else:
                idle_since = None
        except Exception as e:
            log(f"scale write failed ({e}); will retry")
        # No unconditional write-back: replicas differing from `desired` is
        # either our own PATCH lagging or an external actor we must adopt.


if __name__ == "__main__":
    main()
