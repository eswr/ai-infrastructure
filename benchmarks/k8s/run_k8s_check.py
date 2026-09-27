#!/usr/bin/env python3
"""k8s acceptance suite (phase 5): six short scenarios against the live k3d
stack. Mock workers only — no real GPU, minutes not hours. Run `make k8s-up`
first; the suite drives the deployed gateway (via the k3d nodePort on
localhost:8080), the mock-worker pods (kubectl exec), and the scaler.

  1. worker discovery          registry = kubernetes pods, no static list
  2. readiness false on
     functional-probe failure  wedge -> UNHEALTHY -> pod NotReady, /health 200
  3. worker recovers and
     re-enters service         unwedge -> RECOVERING -> generation++ -> READY
  4. drain removes worker
     from new traffic          in-flight finishes, no new work, STOPPED
  5. rolling replacement
     preserves active stream   pod deleted mid-stream; stream completes
  6. autoscaler adds mock
     capacity under pressure   queue signal -> replicas up, idle -> down
"""

import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

NS = "inference-lab"
GW = "http://localhost:8080"
CHAT = {"role": "user", "content": "Reply with: OK"}


def log(msg: str) -> None:
    print(f"[k8s-check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers --

def kubectl(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def exec_curl(pod: str, method: str, path: str, body=None, timeout: int = 8) -> str:
    args = ["exec", pod, "-c", "worker", "--", "curl", "-s", "-m", str(timeout),
            "-X", method, f"http://localhost:8080{path}"]
    if body is not None:
        args += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    return kubectl(*args)


def http(method: str, url: str, body=None, timeout: int = 20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw[:1] in "{[" else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        return e.code, (json.loads(raw) if raw[:1] in "{[" else raw)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return None, f"{type(e).__name__}: {e}"


def chat(max_tokens: int = 4, service_class: str | None = None, timeout: int = 20):
    body = {"model": "mock-gpu", "messages": [CHAT],
            "max_tokens": max_tokens, "stream": False}
    if service_class:
        body["service_class"] = service_class
    return http("POST", f"{GW}/v1/chat/completions", body, timeout=timeout)


def status():
    s, body = http("GET", f"{GW}/status")
    if s != 200:
        raise RuntimeError(f"/status -> {s}")
    return body


def worker_status(st, wid):
    for w in st["workers"]:
        if w["id"] == wid:
            return w
    return None


def metrics_text() -> str:
    s, body = http("GET", f"{GW}/metrics")
    if s != 200:
        raise RuntimeError(f"/metrics -> {s}")
    return body


def metric_sum(name: str, **labels) -> float:
    total = 0.0
    pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", re.M)
    for m in pat.finditer(metrics_text()):
        if labels:
            lab = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
            if any(lab.get(k) != v for k, v in labels.items()):
                continue
        total += float(m.group(1))
    return total


def worker_request_count(pod: str) -> float:
    """mock_requests_total as reported by the pod itself (kubectl exec)."""
    text = exec_curl(pod, "GET", "/metrics")
    m = re.search(r"^mock_requests_total\s+(\S+)$", text, re.M)
    if not m:
        raise AssertionError(f"no mock_requests_total in {pod} /metrics: {text[:200]}")
    return float(m.group(1))


def worker_pods() -> list[str]:
    doc = json.loads(kubectl("get", "pods", "-l", "app=mock-worker", "-o", "json"))
    return sorted(p["metadata"]["name"] for p in doc["items"]
                  if p["metadata"].get("deletionTimestamp") is None)


def pod_ready(pod: str) -> bool:
    doc = json.loads(kubectl("get", "pod", pod, "-o", "json"))
    for c in doc["status"].get("conditions", []):
        if c["type"] == "Ready":
            return c["status"] == "True"
    return False


def replicas() -> int:
    return int(kubectl("get", "deploy", "mock-workers",
                       "-o", "jsonpath={.spec.replicas}"))


def scale(n: int) -> None:
    kubectl("scale", "deploy", "mock-workers", f"--replicas={n}")


FLEET_DEPLOYMENTS = ["fleet-node-gpu-01", "fleet-node-gpu-02", "fleet-node-gpu-03"]


def hard_reset() -> None:
    """Fresh worker pool: delete every mock pod (drops drained/wedged
    state), let the Deployment rebuild, wait for a clean registry. The
    phase-6 fleet Deployments are PARKED (replicas 0): this suite's
    assertions assume exactly the 2-replica phase-5 pool. `make fleet-check`
    brings the fleet back."""
    for d in FLEET_DEPLOYMENTS:
        kubectl("scale", "deploy", d, "--replicas=0")
    kubectl("delete", "pods", "-l", "app=mock-worker", "--wait=false")
    scale(2)
    wait_registry_ready(2)


def wait(pred, timeout: float, desc: str, interval: float = 0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(interval)
    raise AssertionError(f"timeout ({timeout:.0f}s): {desc}")


def reset_all_knobs() -> None:
    for pod in worker_pods():
        exec_curl(pod, "POST", "/admin/knob",
                  {"tok_interval_ms": 25, "ttft_ms": 80, "wedge": False})


def registry_ids():
    return {w["id"] for w in status()["workers"]}


def wait_registry_ready(n: int, timeout: float = 90) -> set[str]:
    """Registry has exactly the n live pods, all READY, pod Ready=True, and
    every worker is an in-service endpoint of its kubernetes Service."""
    def ok():
        pods = set(worker_pods())
        if len(pods) != n:
            return None
        st = status()
        ids = {w["id"] for w in st["workers"]}
        if ids != pods:
            return None
        if not all(w["state"] == "ready" for w in st["workers"]):
            return None
        if not all(w["in_service"] for w in st["workers"]):
            return None
        if not all(pod_ready(p) for p in pods):
            return None
        return pods
    return wait(ok, timeout, f"registry of {n} ready workers matching kubernetes pods")


# -------------------------------------------------------------- scenarios --

def scenario_discovery():
    """No hard-coded worker list: scale the Deployment, the registry follows
    the pods, and capabilities come from pod labels."""
    log("scenario 1: worker discovery")
    assert replicas() >= 2
    base = wait_registry_ready(2)
    for wid in base:
        w = worker_status(status(), wid)
        assert w["runtime"] == "mock" and w["model"] == "mock-gpu", \
            f"capabilities not picked up from labels: {w}"
        assert w["in_service"], f"{wid} not in service"
    scale(3)
    grown = wait_registry_ready(3)
    log(f"  registry follows the Deployment: {sorted(grown)}")
    scale(2)
    wait_registry_ready(2)
    log("  PASS: discovery (pods -> registry, labels -> capabilities)")


def scenario_readiness_failure():
    """Wedge a worker: /health keeps 200, functional generation fails,
    gateway lifecycle -> UNHEALTHY, pod readiness -> false, traffic moves."""
    log("scenario 2: readiness false on functional-probe failure")
    pods = wait_registry_ready(2)
    victim = sorted(pods)[1]
    r = exec_curl(victim, "POST", "/admin/knob", {"wedge": True})
    assert '"wedged":true' in r.replace(" ", ""), r

    st = status()
    w = worker_status(st, victim)
    wait(lambda: (worker_status(status(), victim) or {}).get("state") == "unhealthy",
         45, "gateway marks wedged worker UNHEALTHY (functional probe)")
    reason = worker_status(status(), victim)["state_reason"]
    assert reason == "functional_probe_timeout", reason
    log(f"  gateway lifecycle: ready -> unhealthy ({reason})")

    wait(lambda: pod_ready(victim) is False, 30,
         "kubernetes pod readiness -> false (gateway-authoritative probe)")
    log("  pod readiness: True -> False")

    health = exec_curl(victim, "GET", "/health")
    assert '"status":"ok"' in health.replace(" ", ""), health
    readyz_code = exec_curl(victim, "GET", "/readyz")
    assert '"ready":false' in readyz_code.replace(" ", ""), readyz_code
    log("  worker /health still 200 while /readyz (real generation) fails 503")

    s, _ = http("GET", f"{GW}/admin/workers/{victim}/readiness")
    assert s == 503, f"readiness endpoint -> {s}"
    s, _ = http("GET", f"{GW}/admin/workers/{[p for p in pods if p != victim][0]}/readiness")
    assert s == 200, f"healthy worker readiness -> {s}"

    reqs_before = worker_request_count(victim)
    for _ in range(6):
        code, body = chat()
        assert code == 200 and body["usage"]["completion_tokens"] > 0, (code, body)
    reqs_after = worker_request_count(victim)
    assert reqs_after == reqs_before, "traffic landed on a wedged worker"
    assert (worker_status(status(), victim) or {}).get("outstanding") == 0
    log("  traffic moved away: 6/6 chats served by the healthy worker")

    # clean up for the recovery scenario
    exec_curl(victim, "POST", "/admin/knob", {"wedge": False})
    log("  PASS: readiness mapping (lifecycle-authoritative, functional probe)")


def scenario_recovery():
    """Remove the wedge: UNHEALTHY -> RECOVERING -> generation++ -> READY,
    pod Ready again, back in service. Self-contained: wedges its own victim."""
    log("scenario 3: worker recovers and re-enters service")
    pods = wait_registry_ready(2)
    victim = sorted(pods)[0]
    exec_curl(victim, "POST", "/admin/knob", {"wedge": True})
    wait(lambda: (worker_status(status(), victim) or {}).get("state") == "unhealthy",
         45, "victim quarantined before recovery")
    gen_before = worker_status(status(), victim)["generation"]

    exec_curl(victim, "POST", "/admin/knob", {"wedge": False})
    states = []
    def track():
        w = worker_status(status(), victim)
        if w and (not states or states[-1] != w["state"]):
            states.append(w["state"])
    deadline = time.time() + 60
    while time.time() < deadline:
        track()
        w = worker_status(status(), victim)
        if w and w["state"] == "ready" and w["generation"] > gen_before:
            break
        time.sleep(0.5)
    else:
        raise AssertionError(f"no recovery to READY with new generation: {states}")
    track()
    log(f"  recovery path: {' -> '.join(states)}, generation {gen_before} -> "
        f"{worker_status(status(), victim)['generation']}")
    wait(lambda: pod_ready(victim), 30, "pod Ready again after recovery")
    code, body = chat()
    assert code == 200 and body["usage"]["completion_tokens"] > 0
    log("  PASS: recovery + re-entry (UNHEALTHY -> RECOVERING -> READY, epoch bump)")


def scenario_drain():
    """Admin drain while a stream is in flight: no new work, in-flight
    finishes, DRAINING -> STOPPED only at active == 0."""
    log("scenario 4: drain removes worker from new traffic")
    pods = wait_registry_ready(2)
    for p in pods:
        exec_curl(p, "POST", "/admin/knob", {"tok_interval_ms": 150})

    result = {}
    def slow_stream():
        result["r"] = http("POST", f"{GW}/v1/chat/completions",
                           {"model": "mock-gpu", "messages": [CHAT],
                            "max_tokens": 25, "stream": False},
                           timeout=30)
    t = threading.Thread(target=slow_stream)
    t.start()
    busy = wait(lambda: next((w["id"] for w in status()["workers"]
                              if w["outstanding"] > 0), None),
                10, "a worker picked up the stream")
    log(f"  stream landed on {busy}; draining it mid-flight")

    code, body = http("POST", f"{GW}/admin/workers/{busy}/drain")
    assert code == 200 and body["state"] == "draining", (code, body)

    # New work avoidance: the in-flight stream was already counted in
    # mock_requests_total before the drain, so a flat counter over these
    # chats proves none of them landed on the draining worker.
    reqs_before = worker_request_count(busy)
    for _ in range(5):
        c, b = chat(max_tokens=3)
        assert c == 200, (c, b)
    reqs_after = worker_request_count(busy)
    assert reqs_after == reqs_before, "new requests reached the drained worker"
    log("  new traffic avoided the draining worker (5/5 chats elsewhere)")

    t.join()
    c, b = result["r"]
    assert c == 200 and b["usage"]["completion_tokens"] == 25, (c, b)
    log("  in-flight stream finished on the draining worker")
    wait(lambda: (worker_status(status(), busy) or {}).get("state") == "stopped",
         30, "DRAINING -> STOPPED at active == 0")
    log("  drain completed: DRAINING -> STOPPED")

    kubectl("delete", "pod", busy, "--wait=false")
    wait_registry_ready(2)
    reset_all_knobs()
    log("  PASS: graceful drain")


def scenario_rolling_replacement():
    """Delete the pod holding an active stream: preStop drain lets the
    stream finish, the pod terminates cleanly, the replacement registers."""
    log("scenario 5: rolling replacement preserves active stream")
    pods = wait_registry_ready(2)
    for p in pods:
        exec_curl(p, "POST", "/admin/knob", {"tok_interval_ms": 150})

    result = {}
    def slow_stream():
        result["r"] = http("POST", f"{GW}/v1/chat/completions",
                           {"model": "mock-gpu", "messages": [CHAT],
                            "max_tokens": 25, "stream": False},
                           timeout=30)
    t = threading.Thread(target=slow_stream)
    t.start()
    busy = wait(lambda: next((w["id"] for w in status()["workers"]
                              if w["outstanding"] > 0), None),
                10, "a worker picked up the stream")

    chats_ok, chats_fail = [0], [0]
    def chat_loop():
        for _ in range(5):
            c, b = chat(max_tokens=3, timeout=30)
            if c == 200 and b.get("usage", {}).get("completion_tokens", 0) > 0:
                chats_ok[0] += 1
            else:
                chats_fail[0] += 1
    ct = threading.Thread(target=chat_loop)
    ct.start()

    log(f"  deleting {busy} mid-stream (preStop /drain, grace 60s)")
    kubectl("delete", "pod", busy, "--wait=false")

    t.join()
    c, b = result["r"]
    assert c == 200 and b["usage"]["completion_tokens"] == 25, \
        f"stream did not survive pod deletion: {(c, b)}"
    log("  active stream COMPLETED despite its pod being deleted")
    ct.join()
    assert chats_fail[0] == 0 and chats_ok[0] == 5, \
        f"client errors during rolling replacement: ok={chats_ok[0]} fail={chats_fail[0]}"
    log("  5/5 concurrent chats succeeded during the replacement")

    wait(lambda: busy not in registry_ids(), 30, "deleted pod left the registry")
    wait_registry_ready(2)
    reset_all_knobs()
    log("  PASS: rolling replacement (zero dropped requests)")


def scenario_autoscale():
    """Queue pressure -> the custom scaler adds replicas (from the gateway's
    estimated_queue_seconds, NOT CPU); idle -> cooldown -> back down.
    Twelve one-shot 512-token requests (≈2.6 s estimated work each) exceed
    the gateway's 4096-token active budget, so work QUEUES and
    estimated_queue_seconds crosses the scaler's 0.5 s threshold."""
    log("scenario 6: autoscaler adds mock capacity under queue pressure")
    wait_registry_ready(2)
    assert replicas() == 2, f"expected settled baseline of 2, got {replicas()}"

    results = []
    def pressure():
        results.append(chat(max_tokens=1000, timeout=90))
    threads = [threading.Thread(target=pressure, daemon=True) for _ in range(12)]
    log("  12 concurrent high-work requests exceeding the active-output budget")
    for t in threads:
        t.start()

    scaled_up = wait(lambda: (replicas() if replicas() > 2 else None),
                     75, "scaler adds replicas under queue pressure")
    log(f"  SCALE UP: 2 -> {scaled_up} replicas (queue-signal driven)")
    wait(lambda: len(worker_pods()) >= scaled_up, 60, "new pods created")

    for t in threads:
        t.join(timeout=90)
    ok = sum(1 for c, b in results if c == 200 and b.get("usage", {}).get("completion_tokens", 0) > 0)
    assert ok >= 10, f"pressure requests failed under admission: {ok}/12 ok"
    reset_all_knobs()
    log(f"  pressure absorbed ({ok}/12 completed); waiting out the idle cooldown")
    wait(lambda: (replicas() if replicas() == 2 else None), 180,
         "scaler returns to baseline after idle cooldown")
    wait_registry_ready(2)
    log("  SCALE DOWN: back to 2 replicas (queue empty for the cooldown)")
    log("  PASS: autoscaling on queue signals (no HPA, no CPU)")


SCENARIOS = [scenario_discovery, scenario_readiness_failure, scenario_recovery,
             scenario_drain, scenario_rolling_replacement, scenario_autoscale]


def main() -> int:
    t0 = time.time()
    s, _ = http("GET", f"{GW}/healthz", timeout=5)
    assert s == 200, f"gateway not reachable at {GW} — run `make k8s-up` first"
    log("gateway up; suite target: < 8 min, mock workers only")
    hard_reset()  # fresh worker pool whatever a previous run left behind
    log("")
    failed = []
    for sc in SCENARIOS:
        try:
            sc()
        except Exception as e:
            log(f"  FAIL: {sc.__name__}: {e}")
            failed.append(sc.__name__)
            # best-effort cleanup so the next scenario starts clean
            try:
                hard_reset()
            except Exception as e2:
                log(f"  cleanup failed: {e2}")
        print()
    mins = (time.time() - t0) / 60
    if failed:
        log(f"k8s-check: {len(SCENARIOS) - len(failed)}/{len(SCENARIOS)} passed "
            f"({mins:.1f} min); FAILED: {', '.join(failed)}")
        return 1
    log(f"k8s-check: {len(SCENARIOS)}/{len(SCENARIOS)} scenarios passed "
        f"in {mins:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
