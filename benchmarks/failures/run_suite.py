# Suite runner: automated failure-injection scenarios against the gateway.
#
# Each benchmarks/failures/*.toml describes one experiment: mock-worker
# injection knobs, gateway env, a sequence of steps, and post-run asserts
# evaluated against the last chat response, gateway /status and /metrics.
# stdlib only (python 3.11+, uses tomllib).
#
# Run:  make failcheck          (or: python3 benchmarks/failures/run_suite.py)

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GATEWAY_BIN = REPO / "target/release/gateway"
WORKER_SCRIPT = REPO / "workers/mock-gpu/mock_worker.py"

WORKER_PORT = 8091
GATEWAY_PORT = 8095
GATEWAY = f"http://127.0.0.1:{GATEWAY_PORT}"


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def spawn(cmd, env, log):
    e = dict(os.environ)
    e.update(env)
    return subprocess.Popen(
        cmd, env=e,
        stdout=log, stderr=subprocess.STDOUT,
    )


def http(method, url, body=None, headers=None, timeout=15):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def wait_ready(url, tries=60):
    for _ in range(tries):
        try:
            if http("GET", url, timeout=2)[0] == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def prompt_words(n):
    return " ".join(["inference"] * n)


def do_chat(step):
    """One chat completion; returns a result dict. count>1 fires concurrently."""
    n = int(step.get("count", 1))
    body = json.dumps({
        "model": "mock-gpu",
        "messages": [{"role": "user", "content": prompt_words(step.get("prompt_words", 128))}],
        "stream": bool(step.get("stream", False)),
        "max_tokens": int(step.get("max_tokens", 16)),
        "temperature": 0.1,
    }).encode()
    results = []

    def one():
        t0 = time.monotonic()
        status, headers, raw = http(
            "POST", f"{GATEWAY}/v1/chat/completions", body,
            {"Content-Type": "application/json"}, timeout=int(step.get("timeout_s", 60)),
        )
        elapsed = time.monotonic() - t0
        tokens = raw.count(b'delta"') if step.get("stream") else 0
        results.append({
            "status": status, "elapsed_s": elapsed, "tokens": tokens,
            "retry_after": headers.get("retry-after"),
            "body": raw[:200].decode(errors="replace"),
        })

    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"kind": "chat", "results": results, "last": results[-1] if results else None}


def do_cancel_active(step):
    """Open a streaming request, consume `read_chunks` SSE chunks, then slam
    the socket shut — the gateway must abort upstream and free capacity."""
    body = json.dumps({
        "model": "mock-gpu",
        "messages": [{"role": "user", "content": prompt_words(step.get("prompt_words", 128))}],
        "stream": True,
        "max_tokens": int(step.get("max_tokens", 64)),
    }).encode()
    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {GATEWAY_PORT}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    s = socket.create_connection(("127.0.0.1", GATEWAY_PORT), timeout=10)
    s.sendall(req)
    seen = 0
    want = int(step.get("read_chunks", 2))
    try:
        buf = b""
        while seen < want:
            data = s.recv(4096)
            if not data:
                break
            buf += data
            seen = buf.count(b"data:")
    finally:
        # SO_LINGER 0 forces an immediate RST regardless of unread data, so
        # the gateway's next body write fails deterministically and the
        # in-flight stream is dropped (a graceful FIN would let the gateway
        # buffer the rest of the response and "complete" normally).
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()
    return {"kind": "cancel_active", "chunks_seen": seen}


def do_cancel_queued(step):
    """Occupy the budget with a filler, open a second request that must
    queue, then close its socket while it waits. HTTP/1.1 caveat: the
    gateway only notices once the response would be written, so this is
    best-effort (reported, not asserted)."""
    filler_body = json.dumps({
        "model": "mock-gpu",
        "messages": [{"role": "user", "content": prompt_words(step.get("prompt_words", 128))}],
        "stream": False,
        "max_tokens": int(step.get("filler_tokens", 64)),
    }).encode()
    victim_body = filler_body

    def filler():
        http("POST", f"{GATEWAY}/v1/chat/completions", filler_body,
             {"Content-Type": "application/json"}, timeout=120)

    t = threading.Thread(target=filler)
    t.start()
    time.sleep(0.3)  # let the filler take the budget

    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {GATEWAY_PORT}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(victim_body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + victim_body
    s = socket.create_connection(("127.0.0.1", GATEWAY_PORT), timeout=10)
    s.sendall(req)
    time.sleep(float(step.get("queued_for_s", 1.0)))
    status = json_status()
    queued_before_close = status["admission"]["queued_requests"]
    s.close()  # victim "disconnects" while queued
    t.join()
    return {"kind": "cancel_queued", "queued_before_close": queued_before_close}


def json_status():
    _, _, raw = http("GET", f"{GATEWAY}/status")
    return json.loads(raw)


def metrics_text():
    return http("GET", f"{GATEWAY}/metrics")[2].decode()


def dig(obj, path):
    cur = obj
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


# ---------------------------------------------------------------------------
# Asserts
# ---------------------------------------------------------------------------

def check(assert_, ctx, results):
    kind = assert_["kind"]
    if kind == "status":
        got = results["last"]["status"]
        want = assert_["equals"]
        return got == want, f"status {got} != {want}"
    if kind == "status_set":
        got = sorted(r["status"] for r in results["results"])
        want = sorted(assert_["contains"])
        missing = [w for w in want if w not in got]
        return not missing, f"statuses {got} missing {missing}"
    if kind == "tokens_le":
        got = results["last"]["tokens"]
        return got <= assert_["value"], f"tokens {got} > {assert_['value']}"
    if kind == "tokens_eq":
        got = results["last"]["tokens"]
        return got == assert_["value"], f"tokens {got} != {assert_['value']}"
    if kind == "min_elapsed_s":
        got = results["last"]["elapsed_s"]
        return got >= assert_["value"], f"elapsed {got:.2f}s < {assert_['value']}s"
    if kind == "max_elapsed_s":
        got = results["last"]["elapsed_s"]
        return got <= assert_["value"], f"elapsed {got:.2f}s > {assert_['value']}s"
    if kind == "has_retry_after":
        rejected = [r for r in results["results"] if r["status"] == 429]
        if not rejected:
            return False, "no 429 responses to check"
        missing = [r for r in rejected if not r["retry_after"]]
        return not missing, f"{len(missing)}/{len(rejected)} 429s missing Retry-After"
    if kind == "status_path":
        got = dig(json_status(), assert_["path"])
        want = assert_["equals"]
        ok = (got == want) if not isinstance(want, bool) else (bool(got) == want)
        return ok, f"{assert_['path']} = {got!r} != {want!r}"
    if kind == "poll_status_path":
        deadline = time.monotonic() + float(assert_.get("timeout_s", 10))
        last = None
        while time.monotonic() < deadline:
            try:
                last = dig(json_status(), assert_["path"])
                want = assert_["equals"]
                ok = (last == want) if not isinstance(want, bool) else (bool(last) == want)
                if ok:
                    return True, f"{assert_['path']} = {last!r}"
            except Exception:
                pass
            time.sleep(0.25)
        return False, f"{assert_['path']} never reached {assert_['equals']!r} (last {last!r})"
    if kind == "metric_contains":
        text = metrics_text()
        ok = assert_["substring"] in text
        return ok, f"metrics missing {assert_['substring']!r}"
    if kind == "metric_value":
        name, op, want = assert_["metric"], assert_["op"], assert_["value"]
        vals = []
        for line in metrics_text().splitlines():
            if line.startswith(name) and not line.startswith("#"):
                try:
                    vals.append(float(line.rsplit(None, 1)[1]))
                except (IndexError, ValueError):
                    pass
        got = max(vals) if vals else None
        if got is None:
            return False, f"metric {name} not found"
        ok = got >= want if op == "ge" else got <= want if op == "le" else got > want
        return ok, f"{name} max {got} not {op} {want}"
    if kind == "active_le":
        got = json_status()["admission"]["active_requests"]
        return got <= assert_["value"], f"active {got} > {assert_['value']}"
    return False, f"unknown assert kind {kind}"


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

def run_scenario(path):
    conf = tomllib.loads(path.read_text())
    name = conf["name"]
    print(f"\n=== {name}: {conf.get('description', '')}")
    worker = None
    gateway = None
    try:
        wenv = {"PORT": str(WORKER_PORT), "MODEL_NAME": "mock-gpu"}
        wenv.update({k: str(v) for k, v in conf.get("worker", {}).items()})
        log_path = Path(f"/tmp/suite-{name}.log")
        log = log_path.open("wb")
        worker = spawn([sys.executable, str(WORKER_SCRIPT)], wenv, log)
        if not wait_ready(f"http://127.0.0.1:{WORKER_PORT}/health"):
            return name, False, ["mock worker never became ready"]

        genv = {
            "GATEWAY_ADDR": f"127.0.0.1:{GATEWAY_PORT}",
            "WORKERS": f"mock=http://127.0.0.1:{WORKER_PORT}#mock-gpu",
            # Fast lifecycle so scenarios don't wait on probe cadences.
            "HEALTH_INTERVAL_S": "1",
            "FUNCTIONAL_PROBE_INTERVAL_S": "1",
        }
        genv.update({k: str(v) for k, v in conf.get("gateway", {}).items()})
        gateway = spawn([str(GATEWAY_BIN)], genv, log)

        if not wait_ready(f"{GATEWAY}/healthz"):
            return name, False, ["gateway never became ready"]

        # Phase-4D contract: the gateway routes traffic only after the
        # worker passed a functional probe (READY). /healthz alone lies.
        ready = False
        for _ in range(120):
            try:
                if json_status()["workers"][0]["state"] == "ready":
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ready:
            return name, False, ["worker never reached READY (functional probe)"]

        failures = []
        results = None
        for step in conf.get("steps", []):
            kind = step["kind"]
            if kind == "chat":
                results = do_chat(step)
            elif kind == "cancel_active":
                r = do_cancel_active(step)
                print(f"    cancel_active: chunks_seen={r['chunks_seen']}")
            elif kind == "cancel_queued":
                r = do_cancel_queued(step)
                print(f"    cancel_queued: queued_before_close={r['queued_before_close']}")
            elif kind == "sleep":
                time.sleep(float(step["seconds"]))
            elif kind == "restart_worker":
                worker.terminate()
                worker.wait(timeout=5)
                time.sleep(float(step.get("down_s", 0)))  # let health probes fail
                # The gateway must have noticed the outage...
                flipped = False
                for _ in range(20):
                    try:
                        if json_status()["workers"][0]["healthy"] is False:
                            flipped = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.25)
                if not flipped:
                    failures.append("restart_worker: gateway never marked worker unhealthy")
                # ...and must see it healthy again after the respawn.
                wenv2 = {"PORT": str(WORKER_PORT), "MODEL_NAME": "mock-gpu"}
                worker = spawn([sys.executable, str(WORKER_SCRIPT)], wenv2, log)
                wait_ready(f"http://127.0.0.1:{WORKER_PORT}/health")
                back = False
                for _ in range(40):
                    try:
                        if json_status()["workers"][0]["healthy"] is True:
                            back = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.25)
                if not back:
                    failures.append("restart_worker: gateway never recovered the worker")
            else:
                failures.append(f"unknown step kind {kind}")

        for a in conf.get("asserts", []):
            ok, why = check(a, conf, results or {"last": None, "results": []})
            print(f"    [{'PASS' if ok else 'FAIL'}] {a['kind']}: {why}")
            if not ok:
                failures.append(f"{a['kind']}: {why}")

        if failures:
            log.flush()
            print(f"    [diag] process log: {log_path}")
            print("    [diag] " + log_path.read_text(errors="replace")[-1200:].replace("\n", "\n    [diag] "))
        return name, not failures, failures
    finally:
        for p in (gateway, worker):
            if p:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        log.close()


def main():
    if not GATEWAY_BIN.exists():
        sys.exit("gateway binary missing — run: cargo build --release -p gateway")
    scenarios = sorted(HERE.glob("*.toml"))
    if len(sys.argv) > 1:
        scenarios = [HERE / f"{sys.argv[1]}.toml"]
    passed, failed = 0, 0
    for path in scenarios:
        name, ok, failures = run_scenario(path)
        passed += ok
        failed += not ok
        print(f"  => {name}: {'PASS' if ok else 'FAIL ' + str(failures)}")
    print(f"\nfailure suite: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
