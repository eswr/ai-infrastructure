#!/usr/bin/env python3
# Worker lifecycle acceptance suite (phase 4D).
#
# Drives the six failure/recovery scenarios the lifecycle controller must
# handle, using the mock worker's injected failure modes to reproduce the
# three real signatures the phase-4A runtime study found:
#
#   1. silent_wedge       /health 200, POST hangs        -> READY -> UNHEALTHY
#                         (functional-probe timeout)     -> RECOVERING -> READY
#   2. connection_failure connect refused                -> READY -> DEGRADED
#                                                        -> UNHEALTHY
#   3. empty_completion   HTTP 200, 0 tokens             -> request failure
#                         recorded, DEGRADED, repeated   -> UNHEALTHY
#   4. midstream_stall    stream starts, stalls          -> watchdog trips,
#                         worker DEGRADED
#   5. restart_recovery   restart succeeds               -> generation++,
#                         functional probe required before READY
#   6. drain              drain with active request      -> no new work,
#                         in-flight finishes             -> STOPPED
#
# Every scenario runs its own mock worker (:8092) + gateway (:8096) and
# asserts against the gateway's /status and /metrics. stdlib only.
#
# Run:  make lifecycle-check
#       (or: python3 benchmarks/lifecycle/run_lifecycle_check.py [scenario])

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GATEWAY_BIN = REPO / "target/release/gateway"
WORKER_SCRIPT = REPO / "workers/mock-gpu/mock_worker.py"

WORKER_PORT = 8092
GATEWAY_PORT = 8096
GATEWAY = f"http://127.0.0.1:{GATEWAY_PORT}"

# Fast-enough knobs so scenarios run in seconds, not minutes.
BASE_GATEWAY_ENV = {
    "HEALTH_INTERVAL_S": "1",
    "FUNCTIONAL_PROBE_INTERVAL_S": "1",
    "FUNCTIONAL_PROBE_TIMEOUT_S": "3",
}

_PROCS = []  # everything to kill when a scenario ends


def log(msg):
    print(f"  [lifecycle {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Process + HTTP helpers
# ---------------------------------------------------------------------------


def spawn(cmd, env):
    e = dict(os.environ)
    e.update(env)
    p = subprocess.Popen(cmd, env=e,
                         stdout=Path(f"/tmp/lifecycle-{_CURRENT}.log").open("ab"),
                         stderr=subprocess.STDOUT)
    _PROCS.append(p)
    return p


def http(method, url, body=None, timeout=15, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}".encode()


def chat(stream=False, max_tokens=8, timeout=15):
    body = json.dumps({
        "model": "mock-gpu",
        "messages": [{"role": "user", "content": "inference"}],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    t0 = time.monotonic()
    status, raw = http(
        "POST", f"{GATEWAY}/v1/chat/completions", body, timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    return {
        "status": status,
        "elapsed": time.monotonic() - t0,
        "tokens": raw.count(b'"delta"') if stream else 0,
        "body": raw[:300].decode(errors="replace"),
    }


def worker_status():
    status, raw = http("GET", f"{GATEWAY}/status")
    assert status == 200, f"/status -> {status}"
    return json.loads(raw)["workers"][0]


def metrics_text():
    status, raw = http("GET", f"{GATEWAY}/metrics", timeout=10)
    assert status == 200, f"/metrics -> {status}"
    return raw.decode()


def metric_sum(name, **labels):
    """Sum of all series of `name` whose labels match (subset match)."""
    total = 0.0
    for line in metrics_text().splitlines():
        if not line.startswith(name) or line.startswith("#"):
            continue
        m = re.match(rf"{re.escape(name)}(\{{.*\}})?\s+(.+)$", line)
        if not m:
            continue
        series = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
        if any(series.get(k) != v for k, v in labels.items()):
            continue
        try:
            total += float(m.group(2))
        except ValueError:
            pass
    return total


def poll(fn, timeout, every=0.25, desc="condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return True, last
        except Exception as e:  # gateway briefly mid-restart, etc.
            last = f"{type(e).__name__}: {e}"
        time.sleep(every)
    return False, last


def wait_gateway_ready():
    ok, why = poll(lambda: http("GET", f"{GATEWAY}/healthz", timeout=2)[0] == 200, 15)
    assert ok, f"gateway never became ready: {why}"
    # The lifecycle contract: traffic only after a real generation probe
    # passes — /healthz being up is not enough.
    ok, why = poll(lambda: worker_status()["state"] == "ready", 30)
    assert ok, f"worker never reached READY after gateway start (last: {why})"


def start_worker(extra=None):
    env = {
        "PORT": str(WORKER_PORT),
        "MODEL_NAME": "mock-gpu",
        "TTFT_MS": "20",
        "TOK_INTERVAL_MS": "10",
    }
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return spawn([sys.executable, str(WORKER_SCRIPT)], env)


def start_gateway(extra=None):
    env = {"GATEWAY_ADDR": f"127.0.0.1:{GATEWAY_PORT}",
           "WORKERS": f"mock=http://127.0.0.1:{WORKER_PORT}#mock-gpu"}
    env.update(BASE_GATEWAY_ENV)
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return spawn([str(GATEWAY_BIN)], env)


def kill(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# The six scenarios
# ---------------------------------------------------------------------------


def scenario_silent_wedge():
    """1. /health 200, POST hangs: READY -> UNHEALTHY -> RECOVERING -> READY."""
    failures = []
    worker = start_worker({"WEDGE_AFTER_REQUESTS": 1})
    # READY-state cadence 2s: the prober must re-check a READY worker soon,
    # or a wedge would sit undetected for the default 30s.
    start_gateway({"FUNCTIONAL_PROBE_INTERVAL_READY_S": 2})
    wait_gateway_ready()

    # The boot probe consumed worker seq 1, so this chat is the wedge
    # trigger: it hangs at the worker. The client gives up first; the
    # gateway sees a client cancel and releases the slot. That is honest —
    # no watchdog here is configured to save this request.
    r = chat(max_tokens=4, timeout=5)
    if r["status"] is not None:
        failures.append(f"wedge-trigger chat unexpectedly answered: {r['status']}")

    # The wedge engages: every further POST hangs while /health lies.
    ok, last = poll(
        lambda: worker_status()["state"] == "unhealthy"
        and worker_status()["state_reason"] == "functional_probe_timeout",
        25, desc="UNHEALTHY(functional_probe_timeout)",
    )
    if not ok:
        failures.append(f"worker never quarantined on wedge (last: {last})")
    # The quarantine must be attributable to the *probe*, not /health —
    # transport evidence stayed green the whole time.
    st = worker_status()
    if st["dimensions"]["transport"] != "up":
        failures.append(f"transport dimension should stay up through a wedge: {st['dimensions']}")

    # No new traffic reaches the wedged worker: quick 503, no hang.
    t0 = time.monotonic()
    r = chat(max_tokens=4, timeout=10)
    if r["status"] != 503:
        failures.append(f"chat to quarantined pool got {r['status']} (want 503)")
    if time.monotonic() - t0 > 8:
        failures.append("503 was not fast — wedged worker may still be picked")

    n = metric_sum("inference_worker_state_transitions_total",
                   **{"from": "ready", "to": "unhealthy",
                      "reason": "functional_probe_timeout"})
    if n < 1:
        failures.append(f"missing transition ready->unhealthy(functional_probe_timeout): {n}")

    # External actor restarts the process (the gateway cannot).
    kill(worker)
    worker = start_worker()
    ok, last = poll(
        lambda: worker_status()["state"] == "ready"
        and worker_status()["generation"] >= 2,
        30, desc="READY at generation>=2",
    )
    if not ok:
        failures.append(f"restarted worker never reached READY gen>=2 (last: {last})")

    if metric_sum("inference_worker_state_transitions_total", **{"to": "warming"}) < 1:
        failures.append("recovery never passed through WARMING")
    if metric_sum("inference_worker_state_transitions_total",
                  **{"from": "warming", "to": "ready"}) < 1:
        failures.append("WARMING -> READY transition missing")
    if metric_sum("inference_worker_restarts_total") < 1:
        failures.append("restarts counter did not increment")
    if metric_sum("inference_worker_functional_probe_failures_total",
                  reason="functional_probe_timeout") < 1:
        failures.append("probe failures by reason missing the wedge")

    r = chat(max_tokens=4)
    if r["status"] != 200:
        failures.append(f"post-recovery chat got {r['status']}")
    return failures


def scenario_connection_failure():
    """2. connect refused: READY -> DEGRADED -> UNHEALTHY, then recovery."""
    failures = []
    worker = start_worker()
    start_gateway({"FUNCTIONAL_PROBE_INTERVAL_READY_S": 0})
    wait_gateway_ready()
    kill(worker)

    ok, last = poll(
        lambda: metric_sum("inference_worker_state_transitions_total",
                           **{"from": "ready", "to": "degraded",
                              "reason": "connect_failed"}) >= 1,
        15, desc="ready->degraded(connect_failed)",
    )
    if not ok:
        failures.append(f"first connect failure never degraded the worker: {last}")

    ok, last = poll(lambda: worker_status()["state"] == "unhealthy", 15, desc="UNHEALTHY")
    if not ok:
        failures.append(f"3 connect failures never quarantined: {last}")
    elif worker_status()["state_reason"] != "connect_failed":
        failures.append(f"wrong reason: {worker_status()['state_reason']}")

    r = chat(max_tokens=4, timeout=10)
    if r["status"] != 503:
        failures.append(f"chat to dead pool got {r['status']} (want 503)")

    start_worker()
    ok, last = poll(
        lambda: worker_status()["state"] == "ready"
        and worker_status()["generation"] >= 2,
        30, desc="READY gen>=2",
    )
    if not ok:
        failures.append(f"worker never recovered: {last}")
    r = chat(max_tokens=4)
    if r["status"] != 200:
        failures.append(f"post-recovery chat got {r['status']}")
    return failures


def scenario_empty_completion():
    """3. HTTP 200 with 0 tokens: invisible to HTTP, visible to lifecycle."""
    failures = []
    # EMPTY_AFTER_REQUESTS=1: the boot readiness probe (worker seq 1) is
    # clean — a worker that answers 100% empty can never pass the READY
    # gate, rightly — and everything after it is empty.
    start_worker({"EMPTY_COMPLETION_PCT": 100, "EMPTY_AFTER_REQUESTS": 1})
    start_gateway({"FUNCTIONAL_PROBE_INTERVAL_READY_S": 0})
    wait_gateway_ready()

    # The client sees a *clean* 200 — that is the whole point of this
    # failure class.
    r = chat(max_tokens=4)
    if r["status"] != 200:
        failures.append(f"empty-completion chat got {r['status']} (want 200)")

    ok, last = poll(
        lambda: metric_sum("inference_empty_completions_total") >= 1,
        10, desc="empty_completions_total",
    )
    if not ok:
        failures.append(f"empty completion never recorded: {last}")

    # 1st -> DEGRADED; probes and further requests repeat the signal;
    # 3 consecutive -> UNHEALTHY.
    ok, last = poll(
        lambda: worker_status()["state"] == "unhealthy"
        and worker_status()["state_reason"] == "empty_completion",
        25, desc="UNHEALTHY(empty_completion)",
    )
    if not ok:
        failures.append(f"repeated empty completions never quarantined: {last}")
    if metric_sum("inference_worker_state_transitions_total",
                  **{"from": "ready", "to": "degraded",
                     "reason": "empty_completion"}) < 1:
        failures.append("missing ready->degraded(empty_completion)")
    if metric_sum("inference_worker_functional_probe_failures_total",
                  reason="empty_completion") < 1:
        failures.append("functional probes did not also catch the empties")

    # Clean incarnation: semantic health recovers like anything else.
    kill(_PROCS[0])
    start_worker()
    ok, last = poll(lambda: worker_status()["state"] == "ready", 30, desc="READY")
    if not ok:
        failures.append(f"worker never recovered from empty-completion quarantine: {last}")
    r = chat(max_tokens=4)
    if r["status"] != 200:
        failures.append(f"post-recovery chat got {r['status']}")
    st = worker_status()
    if st["dimensions"]["semantic"] != "up":
        failures.append(f"semantic dimension should be up after recovery: {st['dimensions']}")
    return failures


def scenario_midstream_stall():
    """4. stream starts then stops: watchdog trips, capacity freed, DEGRADED."""
    failures = []
    start_worker({"STALL_AFTER_CHUNKS": 1, "MAX_OUTPUT": 32})
    start_gateway({"STREAM_STALL_TIMEOUT_S": 2, "FUNCTIONAL_PROBE_INTERVAL_READY_S": 0})
    wait_gateway_ready()

    r = chat(stream=True, max_tokens=8, timeout=20)
    if r["status"] != 200:
        failures.append(f"stalled stream got HTTP {r['status']} (headers already sent: want 200)")
    if r["elapsed"] > 8:
        failures.append(f"stall watchdog was slow: {r['elapsed']:.1f}s")

    ok, last = poll(
        lambda: metric_sum("inference_requests_total", outcome="stream_stalled") >= 1,
        10, desc="stream_stalled outcome",
    )
    if not ok:
        failures.append(f"stalled stream never recorded the outcome: {last}")

    ok, last = poll(
        lambda: metric_sum("inference_worker_state_transitions_total",
                           **{"from": "ready", "to": "degraded",
                              "reason": "stream_stalled"}) >= 1,
        10, desc="ready->degraded(stream_stalled)",
    )
    if not ok:
        failures.append(f"stall never degraded the worker: {last}")

    # The watchdog must release admission capacity, not hold the slot.
    ok, last = poll(lambda: worker_status()["outstanding"] == 0, 10, desc="capacity freed")
    if not ok:
        failures.append(f"watchdog did not free the worker slot: {last}")

    # A clean incarnation recovers; also proves escalation stayed
    # proportional (a single stall degraded, it did not brick anything).
    kill(_PROCS[0])
    start_worker()
    ok, last = poll(lambda: worker_status()["state"] == "ready", 30, desc="READY")
    if not ok:
        failures.append(f"worker never recovered: {last}")
    return failures


def scenario_restart_recovery():
    """5. restart: generation increments, probe gate required before READY."""
    failures = []
    start_worker()
    start_gateway({"FUNCTIONAL_PROBE_INTERVAL_READY_S": 0})
    wait_gateway_ready()
    if worker_status()["generation"] != 1:
        failures.append(f"first incarnation must be generation 1, got {worker_status()['generation']}")

    kill(_PROCS[0])
    ok, _ = poll(lambda: worker_status()["state"] == "unhealthy", 15, desc="UNHEALTHY")
    start_worker()

    ok, last = poll(
        lambda: worker_status()["state"] == "ready"
        and worker_status()["generation"] == 2,
        30, desc="READY generation==2",
    )
    if not ok:
        failures.append(f"restart never produced READY gen 2: {last}")

    # The path must have been RECOVERING -> WARMING -> READY: probes are
    # required before traffic, not just /health.
    if metric_sum("inference_worker_state_transitions_total", **{"to": "recovering"}) < 1:
        failures.append("restart never entered RECOVERING")
    if metric_sum("inference_worker_state_transitions_total", **{"to": "warming"}) < 1:
        failures.append("restart never passed WARMING")
    if metric_sum("inference_worker_state_transitions_total",
                  **{"from": "warming", "to": "ready"}) < 1:
        failures.append("WARMING -> READY missing")

    r = chat(max_tokens=4)
    if r["status"] != 200:
        failures.append(f"chat after restart got {r['status']}")
    return failures


def scenario_drain():
    """6. drain: no new assignments, in-flight finishes, then STOPPED."""
    failures = []
    start_worker({"TTFT_MS": 400, "TOK_INTERVAL_MS": 80})
    start_gateway({"FUNCTIONAL_PROBE_INTERVAL_READY_S": 0})
    wait_gateway_ready()

    result = {}

    def slow_chat():
        result["r"] = chat(stream=True, max_tokens=24, timeout=30)

    t = threading.Thread(target=slow_chat)
    t.start()
    ok, _ = poll(lambda: worker_status()["outstanding"] >= 1, 10, desc="in-flight request")
    if not ok:
        failures.append("slow request never went in-flight")

    status, raw = http("POST", f"{GATEWAY}/admin/workers/mock/drain", b"", timeout=10)
    if status != 200:
        failures.append(f"drain endpoint got {status}: {raw[:200]}")
    if worker_status()["state"] != "draining":
        failures.append(f"state after drain: {worker_status()['state']} (want draining)")

    # No new assignments while draining.
    r = chat(max_tokens=4, timeout=10)
    if r["status"] != 503:
        failures.append(f"chat while draining got {r['status']} (want 503)")

    # The in-flight request finishes cleanly.
    t.join(timeout=30)
    r = result.get("r")
    if not r or r["status"] != 200 or r["tokens"] < 2:
        failures.append(f"in-flight request did not complete: {r}")

    ok, last = poll(lambda: worker_status()["state"] == "stopped", 10, desc="STOPPED")
    if not ok:
        failures.append(f"drain never completed: {last}")

    if metric_sum("inference_worker_state_transitions_total", **{"to": "draining"}) < 1:
        failures.append("missing ->draining transition")
    if metric_sum("inference_worker_state_transitions_total",
                  **{"from": "draining", "to": "stopped"}) < 1:
        failures.append("missing draining->stopped transition")
    return failures


SCENARIOS = [
    ("silent-wedge", scenario_silent_wedge),
    ("connection-failure", scenario_connection_failure),
    ("empty-completion", scenario_empty_completion),
    ("midstream-stall", scenario_midstream_stall),
    ("restart-recovery", scenario_restart_recovery),
    ("drain", scenario_drain),
]

_CURRENT = "run"


def run(name, fn):
    global _CURRENT
    _CURRENT = name
    print(f"\n=== {name}")
    _PROCS.clear()
    try:
        try:
            failures = fn()
        except Exception as e:
            failures = [f"scenario raised {type(e).__name__}: {e}"]
    finally:
        for p in _PROCS:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
    for f in failures:
        print(f"    [FAIL] {f}")
    if failures:
        log_path = Path(f"/tmp/lifecycle-{name}.log")
        print(f"    [diag] process log: {log_path}")
        print("    [diag] " + log_path.read_text(errors="replace")[-1500:].replace("\n", "\n    [diag] "))
    return not failures


def main():
    if not GATEWAY_BIN.exists():
        sys.exit("gateway binary missing — run: cargo build --release -p gateway")
    wanted = sys.argv[1:] or [name for name, _ in SCENARIOS]
    passed = failed = 0
    for name, fn in SCENARIOS:
        if name not in wanted:
            continue
        ok = run(name, fn)
        passed += ok
        failed += not ok
        print(f"  => {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nlifecycle-check: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
