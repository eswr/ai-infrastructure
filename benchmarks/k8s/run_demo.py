#!/usr/bin/env python3
"""Phase-5 milestone demo (target: under 5 minutes, mock workers + one
native Metal beat). The six-beat story of the k3d control plane:

  1. kubectl get pods                -> gateway + 2 workers READY
  2. queue pressure                  -> autoscaler 2 -> 3
  3. wedge one worker                -> /health 200, readiness false,
                                        traffic moves away
  4. recover the worker              -> RECOVERING -> WARMING -> READY,
                                        generation epoch increments
  5. delete a pod mid-stream         -> stream completes, no new work
  6. bridge to native MLX            -> 4/4 real-model requests

Run `make k8s-demo` (cluster must be up: `make k8s-up`). The transcript of
the milestone recording lives in docs/results/phase5-demo/.
"""

import subprocess
import sys
import threading
import time

from run_k8s_check import (
    GW, NS,
    chat, exec_curl, hard_reset, http, kubectl, pod_ready,
    replicas, scale, status, wait, wait_registry_ready, worker_pods,
    worker_request_count, worker_status,
)


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def beat(n: int, title: str) -> float:
    print(f"\n[{n}/6] {title}", flush=True)
    return time.time()


def done(t0: float, takeaway: str) -> None:
    print(f"  ✅ {takeaway}  ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ beats --

def beat_pods():
    t0 = beat(1, "the deployment — kubectl get pods")
    pods = wait_registry_ready(2)
    for line in kubectl("get", "pods").strip().splitlines():
        log(line)
    st = status()
    for w in st["workers"]:
        log(f"gateway registry: {w['id']} state={w['state']} "
            f"runtime={w['runtime']} model={w['model']} in_service={w['in_service']}")
    done(t0, f"gateway + {len(pods)} workers READY — discovered from "
             f"kubernetes, no hard-coded list")


def beat_autoscale():
    t0 = beat(2, "queue pressure — the autoscaler adds capacity")
    results = []
    def pressure():
        results.append(chat(max_tokens=1000, timeout=90))
    threads = [threading.Thread(target=pressure, daemon=True) for _ in range(12)]
    log("12 concurrent high-work requests exceed the active-output budget;")
    log("the gateway's estimated_queue_seconds crosses the scaler threshold")
    for t in threads:
        t.start()
    grown = wait(lambda: (replicas() if replicas() > 2 else None),
                 75, "autoscaler scales up")
    log(f"SCALE UP: 2 -> {grown} replicas (queue-signal driven, never CPU)")
    wait(lambda: len(worker_pods()) >= grown, 60, "new pods created")
    for t in threads:
        t.join(timeout=90)
    log("pressure absorbed; returning the pool to its 2-replica baseline")
    scale(2)
    wait(lambda: (replicas() if replicas() == 2 else None), 60,
         "scaler adopts the baseline")
    wait_registry_ready(2)
    done(t0, "the scaler watches the gateway's queue, not CPU")


def beat_wedge():
    t0 = beat(3, "break a worker — silent wedge")
    pods = wait_registry_ready(2)
    victim = sorted(pods)[0]
    log(f"wedge {victim} (answers /health, can no longer generate)")
    exec_curl(victim, "POST", "/admin/knob", {"wedge": True})
    wait(lambda: (worker_status(status(), victim) or {}).get("state") == "unhealthy",
         45, "lifecycle quarantines the wedged worker")
    log(f"gateway lifecycle: READY -> UNHEALTHY "
        f"({worker_status(status(), victim)['state_reason']})")
    wait(lambda: not pod_ready(victim), 30, "pod readiness flips false")
    log("kubernetes readiness: True -> False  (pod leaves its Service)")
    health = exec_curl(victim, "GET", "/health")
    assert '"status"' in health.replace(" ", ""), health
    log("worker /health during wedge: HTTP 200 — the lie the probe exists for")
    before = worker_request_count(victim)
    ok = sum(1 for _ in range(6) if chat()[0] == 200)
    assert ok == 6, "chats failed during wedge"
    assert worker_request_count(victim) == before, "traffic hit the wedged worker"
    log(f"traffic moved away: {ok}/6 chats, 0 landed on the wedged worker")
    exec_curl(victim, "POST", "/admin/knob", {"wedge": False})
    done(t0, "functional readiness drives kubernetes — /health alone lied")


def beat_recover():
    t0 = beat(4, "heal the worker — recovery with a new generation epoch")
    pods = wait_registry_ready(2)
    victim = sorted(pods)[0]
    gen_before = worker_status(status(), victim)["generation"]
    log(f"wedge again, then remove the fault; watch the recovery ladder")
    exec_curl(victim, "POST", "/admin/knob", {"wedge": True})
    wait(lambda: (worker_status(status(), victim) or {}).get("state") == "unhealthy",
         45, "re-quarantined")
    exec_curl(victim, "POST", "/admin/knob", {"wedge": False})
    states = []
    deadline = time.time() + 60
    while time.time() < deadline:
        w = worker_status(status(), victim)
        if w:
            if not states or states[-1] != w["state"]:
                states.append(w["state"])
            if w["state"] == "ready" and w["generation"] > gen_before:
                break
        time.sleep(0.4)
    else:
        raise AssertionError("no recovery")
    wait(lambda: (worker_status(status(), victim) or {}).get("state") == "ready"
         and worker_status(status(), victim)["generation"] > gen_before,
         60, "recovery completes")
    wait(lambda: pod_ready(victim), 30, "pod Ready again")
    log(f"recovery path: {' -> '.join(states)}")
    log(f"generation epoch: {gen_before} -> "
        f"{worker_status(status(), victim)['generation']} "
        f"(a wedged incarnation is never blended with its successor)")
    done(t0, "quarantine -> recovery hysteresis -> READY, re-entered service")


def beat_rolling_drain():
    t0 = beat(5, "delete a pod mid-stream — graceful draining")
    pods = wait_registry_ready(2)
    for p in pods:
        exec_curl(p, "POST", "/admin/knob", {"tok_interval_ms": 150})
    result = {}
    def slow_stream():
        result["r"] = http("POST", f"{GW}/v1/chat/completions",
                           {"model": "mock-gpu", "messages": [
                               {"role": "user", "content": "Reply with: OK"}],
                            "max_tokens": 25, "stream": False}, timeout=30)
    t = threading.Thread(target=slow_stream)
    t.start()
    busy = wait(lambda: next((w["id"] for w in status()["workers"]
                              if w["outstanding"] > 0), None),
                10, "a worker picked up the stream")
    chats = []
    ct = threading.Thread(target=lambda: chats.extend(
        chat(max_tokens=3, timeout=30) for _ in range(5)))
    ct.start()
    log(f"stream in flight on {busy}; kubectl delete pod (preStop /drain)")
    kubectl("delete", "pod", busy, "--wait=false")
    t.join()
    c, b = result["r"]
    assert c == 200 and b["usage"]["completion_tokens"] == 25, "stream lost!"
    log("active stream COMPLETED on the terminating pod")
    ct.join()
    assert all(c == 200 for c, _ in chats), "client errors during replacement"
    log("5/5 concurrent chats served during the replacement — 0 dropped")
    wait(lambda: busy not in {w["id"] for w in status()["workers"]}, 30,
         "deleted pod left the registry")
    wait_registry_ready(2)
    for p in worker_pods():
        exec_curl(p, "POST", "/admin/knob",
                  {"tok_interval_ms": 25, "wedge": False})
    done(t0, "drain lets work finish; termination waits for active == 0")


def beat_bridge():
    t0 = beat(6, "bridge to native MLX — k3d gateway, Metal on the host")
    log("(starting the native mlx-lm worker and bridging it via "
        "host.docker.internal — real generation through the cluster edge)")
    r = subprocess.run(
        [sys.executable, "benchmarks/k8s/run_bridge_check.py"],
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("[bridge") and ("native" in line or "gateway" in line
                                           or "request" in line or "4/4" in line
                                           or "BRIDGE" in line or "restored" in line):
            log(line.split("] ", 1)[-1])
    assert r.returncode == 0, f"bridge check failed:\n{r.stdout[-800:]}\n{r.stderr[-400:]}"
    done(t0, "the control plane reaches out-of-cluster accelerators unchanged")


BEATS = [beat_pods, beat_autoscale, beat_wedge, beat_recover,
         beat_rolling_drain, beat_bridge]


def main() -> int:
    t0 = time.time()
    print("=" * 74, flush=True)
    print("PHASE-5 DEMO — the k3d control plane", flush=True)
    print("discovery | readiness | recovery | draining | autoscaling | bridge",
          flush=True)
    print("=" * 74, flush=True)
    hard_reset()
    failed = None
    for b in BEATS:
        try:
            b()
        except Exception as e:
            print(f"  ❌ {b.__name__}: {e}", flush=True)
            failed = b.__name__
            break
    mins = (time.time() - t0) / 60
    print("\n" + "=" * 74, flush=True)
    if failed:
        print(f"DEMO FAILED at {failed} ({mins:.1f} min)", flush=True)
        return 1
    print(f"DEMO COMPLETE in {mins:.1f} min — one laptop, one cluster, "
          f"one GPU worker; the architecture transfers.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
