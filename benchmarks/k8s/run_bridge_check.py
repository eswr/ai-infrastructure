#!/usr/bin/env python3
"""Bridge validation (phase 5 finale): the NATIVE Metal worker stays on
macOS while the gateway runs in k3d, proving the host <-> cluster path.

    native macOS (mlx-lm, Qwen3.5-0.8B, Metal)
      ^ host.docker.internal (render.sh-injected hostAlias)
      |
    k3d gateway (WORKER_DISCOVERY=kubernetes + EXTERNAL_WORKERS)

Steps: start the native worker if needed -> bridge it into the in-cluster
gateway via EXTERNAL_WORKERS -> wait for the gateway's REAL functional
generation probe to pass through the bridge (Metal generation observed from
inside a Linux container) -> 4 concurrent short requests -> restore the
deployment and the host to how we found them.

Deliberately small: this is a bridge proof, not a benchmark.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

NS = "inference-lab"
GW = "http://localhost:8080"
MODEL = os.environ.get("BRIDGE_MODEL", "mlx-community/Qwen3.5-9B-4bit")
PORT = os.environ.get("BRIDGE_PORT", "8082")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mlx-lm >= 0.31 is a multi-model server: its `--model` default is keyed
# `default_model`, and a request naming anything else triggers a per-request
# Hugging Face resolve (which 404s without network — found live). Register
# the worker and address it under that key.
SERVE_MODEL = "default_model"
EXT = f"mlx-native=http://host.docker.internal:{PORT}#{SERVE_MODEL}"


def log(msg: str) -> None:
    print(f"[bridge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kubectl(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def http(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw[:1] in "{[" else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def wait(pred, timeout, desc, interval=1.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            v = pred()
            if v:
                return v
            last = v
        except Exception as e:
            last = e
        time.sleep(interval)
    raise AssertionError(f"timeout ({timeout:.0f}s): {desc} (last: {last})")


def main() -> int:
    t0 = time.time()
    worker_proc = None
    env_patched = False
    try:
        # 1. native worker (start it only if nothing is on the port)
        s, body = http("GET", f"http://localhost:{PORT}/v1/models", timeout=3)
        if s == 200:
            log(f"native worker already serving on :{PORT}")
        else:
            log(f"starting native mlx-lm worker on :{PORT} ({MODEL})")
            worker_proc = subprocess.Popen(
                ["bash", "workers/mlx-lm/run.sh"],
                cwd=REPO, env=dict(os.environ, MODEL=MODEL, PORT=PORT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait(lambda: http("GET", f"http://localhost:{PORT}/v1/models", timeout=3)[0] == 200,
                 180, "native mlx-lm worker live")
        log("native worker live (Metal, host macOS)")

        # 2. bridge it into the in-cluster gateway
        kubectl("set", "env", "deploy/gateway", f"EXTERNAL_WORKERS={EXT}")
        env_patched = True
        kubectl("rollout", "restart", "deploy/gateway")
        kubectl("rollout", "status", "deploy/gateway", "--timeout=120s")

        # 3. the gateway's functional probe must pass THROUGH the bridge —
        #    a real Metal generation observed from inside the cluster.
        def native_ready():
            s, st = http("GET", f"{GW}/status")
            for w in (st.get("workers", []) if s == 200 else []):
                if w["id"] == "mlx-native" and w["state"] == "ready":
                    return w
            return None
        w = wait(native_ready, 90, "mlx-native READY via functional probe through the bridge")
        log(f"gateway lifecycle: mlx-native state=ready generation={w['generation']} "
            f"(probe = real generation via host.docker.internal)")

        # 4. four concurrent short requests through the full path
        results = []
        def one(i):
            t = time.time()
            c, b = http("POST", f"{GW}/v1/chat/completions",
                        {"model": SERVE_MODEL, "messages": [{"role": "user",
                         "content": "Reply with: OK"}], "max_tokens": 16},
                        timeout=120)
            results.append((i, c, b, time.time() - t))
        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = [r for r in results if r[1] == 200 and r[2]["usage"]["completion_tokens"] > 0]
        assert len(ok) == 4, f"bridge validation failed: {results}"
        for i, c, b, dt in sorted(results):
            log(f"  request {i}: 200, {b['usage']['completion_tokens']} tokens, "
                f"client e2e {dt:.2f}s")
        log("4/4 concurrent requests served: k3d gateway -> host.docker.internal "
            "-> native Metal -> client")

        # 5. restore
        kubectl("set", "env", "deploy/gateway", "EXTERNAL_WORKERS-")
        env_patched = False
        kubectl("rollout", "status", "deploy/gateway", "--timeout=120s")
        log("gateway restored (EXTERNAL_WORKERS unset)")
        log(f"BRIDGE OK in {time.time() - t0:.0f}s")
        return 0
    finally:
        if env_patched:
            try:
                kubectl("set", "env", "deploy/gateway", "EXTERNAL_WORKERS-")
                kubectl("rollout", "restart", "deploy/gateway")
            except Exception:
                pass
        if worker_proc is not None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=30)  # graceful: finishes any stragglers
            except subprocess.TimeoutExpired:
                worker_proc.kill()
            log("native worker stopped")


if __name__ == "__main__":
    sys.exit(main())
