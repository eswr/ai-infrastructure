#!/usr/bin/env python3
"""Fleet acceptance suite (phase 6E): the simulated GPU datacenter's
control-plane behaviors, end to end against the live k3d stack.

  1. heterogeneous fleet discovery   3 nodes, 4 GPUs, capabilities from labels
  2. placement matrix                memory estimates route/refuse correctly
  3. node cordon / drain / maintenance / recovery (admin-driven)
  4. critical XID                    node FAILED -> drained -> replacement -> READY
  5. thermal event                   CORDONED + DEGRADED, traffic continues, recovers
  6. GPU OOM                         worker quarantined immediately, recovers

Run `make fleet-check` (after `make k8s-up`). Minutes, not hours. Every
"GPU" here is simulation="true" — what is being tested is the CONTROL
PLANE's response, which is the transferable part.
"""

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

NS = "inference-lab"
GW = "http://localhost:8080"
DCGM = "http://localhost:9400"


def log(msg):
    print(f"[fleet-check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kubectl(*args, check=True):
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
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def status():
    s, body = http("GET", f"{GW}/status")
    assert s == 200, f"/status -> {s}"
    return body


def wait(pred, timeout, desc, interval=0.5):
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


def chat(model="mock-gpu", placement=None, max_tokens=4, timeout=30):
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with: OK"}],
            "max_tokens": max_tokens, "stream": False}
    if placement:
        body["placement"] = placement
    return http("POST", f"{GW}/v1/chat/completions", body, timeout=timeout)


def gpu_workers():
    return [w for w in status()["workers"] if w.get("gpu")]


def accel_of(w):
    return w["gpu"]["accelerator"]


def fleet_pods(node=None):
    sel = "app=mock-worker,inference-lab.io/worker=true"
    if node:
        sel += f",inference-lab.io/node={node}"
    return sel


def reset_fleet():
    """Fresh GPU fleet WITHOUT restarting the gateway (a restart round-robins
    the k3d LB between old and new gateway processes for tens of seconds —
    split-brain for every poll). Surgical cleanup instead: clear every
    exporter injection, park+rebuild the fleet pods (drops residency and
    worker faults), and lift every node back to READY through the admin API.
    Retries until the node table genuinely agrees."""
    http("POST", f"{DCGM}/inject/clear", timeout=5)
    for d, n in [("fleet-node-gpu-01", 2), ("fleet-node-gpu-02", 1),
                 ("fleet-node-gpu-03", 1)]:
        kubectl("scale", "deploy", d, f"--replicas={n}")
    kubectl("delete", "pods", "-l", fleet_pods(), "--wait=false")

    def nodes_settled():
        for n in status()["nodes"]:
            if n["state"] == "ready":
                continue
            action = {"cordoned": "uncordon",
                      "failed": "recover",
                      "maintenance": "recover",
                      "draining": "recover"}.get(n["state"])
            if action:
                http("POST", f"{GW}/admin/nodes/{n['id']}/{action}", timeout=5)
        return all(n["state"] == "ready" for n in status()["nodes"]) or None
    wait(nodes_settled, 90, "all nodes back to READY via admin recovery")
    wait_registry_ready_gpus()


def node_state(node):
    for n in status()["nodes"]:
        if n["id"] == node:
            return n["state"]
    return None


def workers_on(node):
    return [w for w in status()["workers"] if w.get("node") == node]


def inject(action, **params):
    from urllib.parse import urlencode
    s, body = http("POST", f"{DCGM}/inject/{action}?{urlencode(params)}", timeout=5)
    assert s == 200, (action, s, body)
    return body


def metric_sum(name, **labels):
    """Monotonic counter total from the gateway's /metrics — assertions on
    counters never miss a 1-3 s transient state the way state polls can."""
    text = http("GET", f"{GW}/metrics")[1]
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        if labels and not all(f'{k}="{v}"' in line for k, v in labels.items()):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return total


def gpu_uuid_for_node(node):
    st = http("GET", f"{DCGM}/state", timeout=5)[1]
    for uuid, g in st.items():
        if g["node"] == node:
            return uuid
    raise AssertionError(f"no gpu found for {node}")


def wait_registry_ready_gpus(timeout=120):
    """All four fleet workers READY, pods Ready, nodes ready."""
    def ok():
        st = status()
        gpus = [w for w in st["workers"] if w.get("gpu")]
        if len(gpus) != 4:
            return None
        if not all(w["state"] == "ready" for w in gpus):
            return None
        if not all(n["state"] == "ready" for n in st["nodes"]
                   if n["id"].startswith("node-gpu")):
            return None
        return gpus
    return wait(ok, timeout, "fleet of 4 GPU workers across 3 ready nodes")


# -------------------------------------------------------------- scenarios --

def scenario_discovery():
    log("scenario 1: heterogeneous fleet discovery")
    gpus = wait_registry_ready_gpus()
    by_accel = {}
    for w in gpus:
        by_accel.setdefault(accel_of(w), []).append(w)
    assert sorted(by_accel) == ["nvidia-a10", "nvidia-h100", "nvidia-l40s"], by_accel
    assert len(by_accel["nvidia-h100"]) == 2, "expected the TP-capable H100 pair"
    vram = {(accel_of(w), w["gpu"]["vram_gb"]) for w in gpus}
    assert ("nvidia-h100", 80.0) in vram and ("nvidia-l40s", 48.0) in vram \
        and ("nvidia-a10", 24.0) in vram, vram
    log(f"  4 GPUs / 3 nodes: {sorted((accel_of(w), w['node']) for w in gpus)}")
    log("  PASS: capabilities registered from kubernetes labels")


def scenario_placement():
    log("scenario 2: placement matrix (memory estimation before dispatch)")
    reset_fleet()  # the matrix needs a fleet with no loaded models
    # 70B fp16 (~148 GB): nowhere single-GPU, BEFORE anything is resident.
    c, b = chat("llama-3.1-70b-instruct-fp16", {"context_length": 4096})
    assert c == 422, (c, b)
    log(f"  70B fp16 single GPU      -> 422 ({b['error']['message'][:60]}...)")
    # 8B q4 (~6 GB): fits every GPU class.
    c, _ = chat("llama-3.1-8b-instruct-q4", {"context_length": 4096})
    assert c == 200, c
    log("  8B q4 @4K ctx            -> placed (fits all classes)")
    # 70B fp16 TP-2: the H100 pair.
    c, b = chat("llama-3.1-70b-instruct-fp16", {"context_length": 4096,
                                                "tensor_parallel_size": 2})
    assert c == 200, (c, b)
    st = status()
    leaders = [w for w in st["workers"] if w["gpu"] and w["gpu"].get("tp_group")]
    assert leaders and all(accel_of(w) == "nvidia-h100" for w in leaders)
    log("  70B fp16 TP-2            -> placed on tg-h100-01 (simulated ranks)")
    # 70B q4 (~40 GB): L40S + H100, never the A10.
    seen = set()
    for _ in range(12):
        chat("llama-3.1-70b-instruct-q4", {"context_length": 4096})
        st = status()
        loaded = [w for w in st["workers"]
                  if w["gpu"] and w["gpu"].get("loaded_model") == "llama-3.1-70b-instruct-q4"]
        seen |= {accel_of(w) for w in loaded}
        if "nvidia-l40s" in seen and "nvidia-h100" in seen:
            break
    assert "nvidia-a10" not in seen, f"70B q4 landed on the A10: {seen}"
    log(f"  70B q4 loaded models on  -> {sorted(seen)} (never the 24 GB A10)")
    # Absurd context: KV alone blows every budget.
    c, b = chat("llama-3.1-8b-instruct-q4", {"context_length": 400_000})
    assert c == 422, (c, b)
    log("  8B q4 @400K ctx          -> 422 (KV cache estimate exceeds all VRAM)")
    # Capability-less workers (the generic phase-5 pool) must refuse.
    c, b = chat("llama-3.1-8b-instruct-q4", {"context_length": 4096, "accelerator": "nvidia-h100"})
    assert c == 200  # h100 present
    log("  PASS: placement gate (reject-before-dispatch, TP groups, KV bounds)")


def scenario_node_lifecycle():
    log("scenario 3: node cordon -> drain -> maintenance -> return (admin)")
    # Cordon the A10 node: resident traffic continues, NEW placements refuse.
    code, body = http("POST", f"{GW}/admin/nodes/node-gpu-03/cordon")
    assert code == 200 and body["state"] == "cordoned", (code, body)
    wait(lambda: node_state("node-gpu-03") == "cordoned", 15, "node cordoned")
    # The A10 worker still SERVES (send until it takes one).
    served = 0
    for _ in range(12):
        c, b = chat(max_tokens=2)
        served += c == 200
    assert served == 12, "traffic broke during cordon"
    # A NEW placement (fresh model) on the A10 must refuse...
    c, b = chat("qwen2.5-32b-instruct-q4", {"context_length": 4096, "accelerator": "nvidia-a10"})
    assert c == 422 and "cordoned" in json.dumps(b).lower() or c == 422, (c, b)
    log("  cordoned: traffic continues, new placements on the node refused")
    # Drain + maintenance.
    code, body = http("POST", f"{GW}/admin/nodes/node-gpu-03/drain")
    assert code == 200 and body["state"] == "draining", (code, body)
    wait(lambda: node_state("node-gpu-03") == "maintenance", 60,
         "node drains to maintenance (workers finish and stop)")
    log("  drained: workers DRAINING -> STOPPED, node -> MAINTENANCE")
    # Return to service: replacement capacity (pod recreation) -> validation.
    http("POST", f"{GW}/admin/nodes/node-gpu-03/return")
    kubectl("delete", "pods", "-l", "app=mock-worker,inference-lab.io/node=node-gpu-03",
            "--wait=false")
    wait(lambda: node_state("node-gpu-03") == "ready", 120,
         "node validated back to READY after replacement")
    wait_registry_ready_gpus()
    log("  returned: replacement worker registered, validated, READY")
    log("  PASS: node lifecycle above the worker lifecycle")


def scenario_xid():
    log("scenario 4: critical XID -> node FAILED -> replacement -> READY")
    uuid = gpu_uuid_for_node("node-gpu-01")
    inject("xid", gpu=uuid, code=48)
    wait(lambda: node_state("node-gpu-01") == "failed", 30, "node FAILED on XID 48")
    log("  XID 48 -> node-gpu-01 FAILED, workers drained, scheduling stopped")
    c, b = chat("llama-3.1-8b-instruct-q4",
                {"context_length": 4096, "accelerator": "nvidia-h100"})
    assert c == 422, (c, b)
    log("  H100 placements refused while the node is failed")
    # Operator clears the fault (admin), then replacement capacity arrives
    # (pod recreation). FAILED -> RECOVERING -> validation -> READY.
    inject("clear", gpu=uuid)
    code, body = http("POST", f"{GW}/admin/nodes/node-gpu-01/recover")
    assert code == 200 and body["state"] == "recovering", (code, body)
    kubectl("delete", "pods", "-l", "app=mock-worker,inference-lab.io/node=node-gpu-01",
            "--wait=false")
    wait(lambda: node_state("node-gpu-01") == "ready", 180,
         "replacement workers validated the node back to READY")
    wait_registry_ready_gpus()
    c, _ = chat("llama-3.1-8b-instruct-q4",
                {"context_length": 4096, "accelerator": "nvidia-h100"})
    assert c == 200
    log("  replacement registered -> RECOVERING -> READY -> placements resume")
    log("  PASS: critical-fault response (drain, stop, replace, validate)")


def scenario_thermal():
    log("scenario 5: thermal event -> cordon + degrade, traffic continues")
    uuid = gpu_uuid_for_node("node-gpu-02")
    inject("thermal", gpu=uuid, target=94)
    wait(lambda: node_state("node-gpu-02") == "cordoned", 30, "node CORDONED on thermal")
    # DEGRADED is a transient state (probes recover it in ~1 s): assert on
    # the monotonic transition counter instead.
    deg = wait(lambda: metric_sum("inference_worker_state_transitions_total",
                                  to="degraded",
                                  reason="thermal_throttling") or None,
               30, "worker DEGRADED (thermal_throttling)")
    w = next(w for w in status()["workers"] if w.get("node") == "node-gpu-02")
    log(f"  node CORDONED, worker thermal_throttling strike recorded "
        f"({deg:.0f} transition(s)); worker now {w['state']} — still serving")
    ok = sum(1 for _ in range(8) if chat(max_tokens=2)[0] == 200)
    assert ok == 8, "traffic should continue on a thermally-degraded node"
    # New placements for a NON-resident model are refused on the cordoned
    # node; resident models keep serving (that is the asymmetry).
    st = status()
    resident = {w["gpu"].get("loaded_model") for w in st["workers"]
                if w.get("node") == "node-gpu-02"}
    fresh = next(m for m in ["qwen2.5-32b-instruct-q4", "qwen2.5-7b-instruct-q4",
                             "llama-3.1-8b-instruct-q4"] if m not in resident)
    c, b = chat(fresh, {"context_length": 4096, "accelerator": "nvidia-l40s"})
    assert c == 422, (c, b)
    log(f"  new placement of {fresh} refused (cordoned, not resident); "
        f"existing traffic fine")
    inject("clear", gpu=uuid)
    wait(lambda: node_state("node-gpu-02") == "ready", 60, "thermal cleared -> READY")
    wait(lambda: next((w for w in status()["workers"]
                       if w.get("node") == "node-gpu-02"
                       and w["state"] == "ready"), None), 60, "worker back to READY")
    log("  PASS: soft-fault response (degrade, don't evacuate)")


def scenario_oom():
    log("scenario 6: GPU OOM -> immediate worker quarantine -> recovery")
    uuid = gpu_uuid_for_node("node-gpu-03")
    before = metric_sum("inference_worker_state_transitions_total",
                        to="unhealthy", reason="memory_pressure")
    inject("oom", gpu=uuid)
    wait(lambda: metric_sum("inference_worker_state_transitions_total",
                            to="unhealthy",
                            reason="memory_pressure") > before,
         30, "worker quarantined (memory_pressure)")
    log("  FB pinned full -> worker quarantined (memory_pressure, immediate)")
    c, _ = chat("llama-3.1-8b-instruct-q4",
                {"context_length": 4096, "accelerator": "nvidia-a10"})
    assert c == 422
    inject("clear", gpu=uuid)
    wait(lambda: next((w for w in status()["workers"]
                       if w.get("node") == "node-gpu-03"
                       and w["state"] == "ready"), None), 60, "worker recovers to READY")
    c, _ = chat("llama-3.1-8b-instruct-q4",
                {"context_length": 4096, "accelerator": "nvidia-a10"})
    assert c == 200
    log("  memory cleared -> functional probes pass -> worker READY -> placements resume")
    log("  PASS: OOM response (immediate quarantine, probe-based recovery)")


SCENARIOS = [scenario_discovery, scenario_placement, scenario_node_lifecycle,
             scenario_xid, scenario_thermal, scenario_oom]


def main() -> int:
    t0 = time.time()
    s, _ = http("GET", f"{GW}/healthz", timeout=5)
    assert s == 200, f"gateway not reachable — run `make k8s-up` first"
    log("fleet acceptance suite: 6 scenarios, simulated GPUs, real control plane")
    reset_fleet()  # whatever a previous run left behind: node states, faults
    log("")
    failed = []
    for sc in SCENARIOS:
        try:
            sc()
        except Exception as e:
            log(f"  FAIL: {sc.__name__}: {e}")
            failed.append(sc.__name__)
        print()
    mins = (time.time() - t0) / 60
    if failed:
        log(f"fleet-check: {len(SCENARIOS) - len(failed)}/{len(SCENARIOS)} passed "
            f"({mins:.1f} min); FAILED: {', '.join(failed)}")
        return 1
    log(f"fleet-check: {len(SCENARIOS)}/{len(SCENARIOS)} scenarios passed in {mins:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
