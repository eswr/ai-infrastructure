#!/usr/bin/env python3
"""prefix_router.py — phase-4B.2 load-gated prefix-aware routing study.

Phase 4B found the trap: llama.cpp's native LCP slot selection gives cache
locality absolute priority, so a cold same-prefix burst serializes on the
busy hot slot (cold-c8 TTFT p50 253 s vs 19 s all-miss). This study asks
the follow-up question: when the GATEWAY owns slot assignment, which
routing policy survives the pathological workloads?

One llama-server with 8 slots is modeled by the gateway as 8 schedulable
cache domains (WORKER_SLOTS=8). Four policies x four cache topologies:

    policies    native          llama.cpp keeps choosing (no id_slot)
                least_work      gateway: lowest estimated work, cache ignored
                prefix_only     gateway: largest reusable prefix wins
                cache_aware_ect gateway: argmin estimated TTFT
                                (queue-ahead + prefill-cost-after-reuse)
    topologies  all-cold        fresh server, no prefix anywhere
                one-warm        slot 0 holds the shared prefix
                four-warm       slots 0-3 hold it
                all-warm        every slot holds it

Each leg: cache reset (process restart — the only honest reset) ->
functional gate -> [pinned slot warmups through the gateway, so the
router's cache model learns them] -> measured run (8 x [8192 shared + 128
unique] tokens, out=256, burst, c8, streaming) -> post-run probe ->
/metrics scrape + llama.cpp /slots occupancy trace + routing-decision log
extract -> teardown.

Warmups are loadgen one-shots with --pin-slot <i> carrying the SAME seed
as the measured run (Rust StdRng prefix synthesis — Python cannot
reproduce the byte-identical prefix text). They ride THROUGH the gateway:
slot policies honor the pin and mark the slot cache RESIDENT from the
runtime's own cache_n telemetry; native passes id_slot through untouched.

Artifacts per leg dir: leg.toml, matrix.json (+records), warm/ (populate
runs), slots_occupancy.csv (1 Hz is_processing per slot), slots_snapshot.json,
gateway_metrics.json, routing_decisions.log, gateway.log. Plus study-wide
lifecycle.json and memory.csv. Run router_econ.py on the study dir after.

Usage:
    python3 benchmarks/prefix_router.py --runtime llama-cpp \
        [--policies native,cache_aware_ect] [--topologies all-cold,one-warm]
        [--two-prefix] [--smoke]
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_harness import (  # noqa: E402
    FUNCTIONAL_TIMEOUT_S,
    LIVE_TIMEOUT_S,
    MemorySampler,
    http_get,
    kill_leftovers,
    log,
    make_runtime,
)

REPO = Path(__file__).resolve().parent.parent

SHARED_TOKENS = 8192
UNIQUE_TOKENS = 128
OUTPUT_TOKENS = 256
NUM_REQUESTS = 8
CONCURRENCY = 8
SEED = 42

POLICIES = ["native", "least_work", "prefix_only", "cache_aware_ect"]
TOPOLOGIES = {"all-cold": 0, "one-warm": 1, "four-warm": 4, "all-warm": 8}
# ROUTE_POLICY env spelling per policy name.
POLICY_ENV = {
    "native": "native",
    "least_work": "least-work",
    "prefix_only": "prefix-only",
    "cache_aware_ect": "cache-aware-ect",
}

# llama.cpp service-model seeds on this machine, from the 4A/4B studies
# (estimated-token units; the gateway refits them online from observations):
# ~6.2 ms/est-tok prefill (51.9 s for a full 8.4K-est-tok prompt at c1),
# ~25 ms/est-tok decode, ~250 ms fixed overhead.
PROFILE = {"PROFILE_FIXED_MS": "250", "PROFILE_PREFILL_MS_PER_TOK": "6.2",
           "PROFILE_DECODE_MS_PER_TOK": "25"}


def gateway_slot_env(policy: str, model: str, port: int, slots: int) -> dict:
    return {
        "GATEWAY_ADDR": "127.0.0.1:8080",
        "WORKERS": f"llama=http://127.0.0.1:{port}#{model}",
        "ROUTE_POLICY": POLICY_ENV[policy],
        # native models the whole server as ONE worker (llama.cpp's own LCP
        # slot selection); slot policies model each -np cell separately.
        "WORKER_SLOTS": str(0 if policy == "native" else slots),
        "FIRST_TOKEN_TIMEOUT_S": "900",
        # A probe queued behind an 8K slot prefill cannot answer in 10 s:
        # the default timeout false-quarantines busy slots (observed live:
        # 14 quarantine/recovery generations during one 170 s populate).
        # 120 s tolerates a slot busy with a full-size prefill.
        "FUNCTIONAL_PROBE_TIMEOUT_S": "120",
        # Steady-state probes are GPU work (and would evict slot prefixes);
        # keep measured runs quiet.
        "FUNCTIONAL_PROBE_INTERVAL_READY_S": "0",
        "HEALTH_INTERVAL_S": "2",
        "FUNCTIONAL_PROBE_INTERVAL_S": "2",
        **PROFILE,
    }


def wait_gateway_all_ready(slots: int, timeout_s: float = 90.0) -> dict:
    """Wait until every modeled worker (1 for native, N for slots) is READY
    — the 4D gate, per slot."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, body = http_get("http://127.0.0.1:8080/status")
        if status == 200:
            try:
                workers = json.loads(body)["workers"]
                if len(workers) == (1 if slots == 0 else slots) and \
                        all(w["state"] == "ready" for w in workers):
                    return {"ready": True,
                            "workers": [{"id": w["id"], "slot": w.get("slot"),
                                         "state": w["state"]} for w in workers]}
            except (KeyError, json.JSONDecodeError):
                pass
        time.sleep(0.3)
    raise RuntimeError("gateway workers never reached READY")


def write_leg_matrix(path: Path, leg: str, description: str, gateway_endpoint: str,
                     model: str, out_tokens: int, c: int, n: int) -> None:
    """One matrix = the measured run only; topology warmup is driven
    separately as pinned one-shot runs (see warm_slots)."""
    path.write_text(f"""
study = "prefix-router-{leg}"
description = "{description}"
cooldown_s = 2.0
[defaults]
endpoint = "{gateway_endpoint}"
model = "{model}"
output_tokens = {out_tokens}
stream = true
arrival = "burst"
timeout_s = 900
seed = {SEED}
shared_prefix_tokens = {SHARED_TOKENS}

[[runs]]
name = "{leg}"
input_tokens = {SHARED_TOKENS + UNIQUE_TOKENS}
concurrency = {c}
num_requests = {n}
""")


def warm_slots(gateway_bin: Path, loadgen_bin: Path, leg_dir: Path, model: str,
               warm_n: int, seeds: list[int] | None = None) -> dict:
    """Populate `warm_n` slots' caches: parallel pinned loadgen one-shots
    THROUGH the gateway, same seed as the measured run so the shared
    prefix is byte-identical. Slot policies learn the warm topology from
    these completions (id_slot pin + observed cache_n). `seeds` overrides
    the seed per slot (the two-prefix experiment warms slot0 with A's
    seed and slot1 with B's)."""
    out_dir = leg_dir / "warm"
    out_dir.mkdir(exist_ok=True)
    seeds = seeds or [SEED] * warm_n
    procs = []
    t0 = time.time()
    for i in range(warm_n):
        cmd = [str(loadgen_bin),
               "--endpoint", "http://127.0.0.1:8080",
               "--model", model,
               "--input-tokens", str(SHARED_TOKENS + UNIQUE_TOKENS),
               "--shared-prefix-tokens", str(SHARED_TOKENS),
               "--output-tokens", "4",
               "--num-requests", "1",
               "--concurrency", "1",
               "--stream", "true",
               "--seed", str(seeds[i]),
               "--pin-slot", str(i),
               "--timeout-s", "900",
               "--out-dir", str(out_dir / f"slot{i}")]
        procs.append((i, subprocess.Popen(cmd, cwd=REPO,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, text=True)))
    results = []
    for i, p in procs:
        _, err = p.communicate()
        results.append({"slot": i, "exit": p.returncode,
                        "err": (err or "")[-200:] if p.returncode else None})
    wall = round(time.time() - t0, 1)
    ok = all(r["exit"] == 0 for r in results)
    log(f"warm: {warm_n} pinned populate run(s) in {wall}s, ok={ok}")
    return {"requested": warm_n, "ok": ok, "wall_s": wall, "results": results}


class SlotsOccupancySampler(threading.Thread):
    """1 Hz llama.cpp GET /slots -> per-slot is_processing CSV. This is the
    direct evidence of WHERE requests actually ran: native should show a
    hotspot on one slot under the pathological burst, pinned policies a
    spread (or a deliberate queue on the warm slot)."""

    def __init__(self, port: int, csv_path: Path):
        super().__init__(daemon=True)
        self.url = f"http://127.0.0.1:{port}/slots"
        self.csv_path = csv_path
        self._stop = threading.Event()

    def run(self) -> None:
        with self.csv_path.open("w") as fh:
            fh.write("ts," + ",".join(f"slot{i}" for i in range(8)) + "\n")
            while not self._stop.is_set():
                status, body = http_get(self.url, timeout_s=2.0)
                if status == 200:
                    try:
                        slots = json.loads(body)
                        busy = {s["id"]: 1 if s.get("is_processing") else 0
                                for s in slots}
                        row = [str(busy.get(i, "")) for i in range(8)]
                        fh.write(f"{time.time():.1f}," + ",".join(row) + "\n")
                        fh.flush()
                    except (json.JSONDecodeError, KeyError):
                        pass
                self._stop.wait(1.0)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=3)


def parse_prom(text: str) -> dict:
    """Minimal Prometheus text parser: {(name, labels-tuple): value}."""
    out = {}
    pat = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+e0-9.]+)$")
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        m = pat.match(ln.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        out[(name, labels)] = out.get((name, labels), 0.0) + v
    return out


def scrape_gateway_metrics(base_path: Path) -> None:
    """Capture /metrics twice: raw text (`gateway_metrics.txt`, the full
    evidence) and a parsed JSON of the scheduler/inference families the
    analyzer consumes (`gateway_metrics.json`)."""
    status, body = http_get("http://127.0.0.1:8080/metrics")
    base_path.with_suffix(".txt").write_text(body if status == 200 else "")
    if status != 200:
        base_path.write_text("{}")
        return
    parsed = {f"{name}{labels}": v for (name, labels), v in parse_prom(body).items()}
    base_path.write_text(json.dumps(
        {k: v for k, v in parsed.items()
         if k.startswith("scheduler_") or k.startswith("inference_")}, indent=2))


def capture_slots_snapshot(rt, path: Path) -> None:
    status, body = http_get(f"http://127.0.0.1:{rt.port}/slots")
    path.write_text(body if status == 200 else json.dumps({"available": False}))


def extract_routing_log(gateway_log: Path, dest: Path) -> int:
    if not gateway_log.exists():
        return 0
    lines = [ln for ln in gateway_log.read_text(errors="replace").splitlines()
             if "routing decision" in ln or "routing_decision" in ln]
    dest.write_text("\n".join(lines) + "\n")
    return len(lines)


def run_leg(rt, gateway_bin: Path, loadgen_bin: Path, study_dir: Path,
            policy: str, topo: str, warm_n: int, sampler: MemorySampler,
            leg_params: dict) -> dict:
    """One leg: reset -> functional gate -> gateway(policy) -> [warm] ->
    measured run -> post-run health + evidence capture. Returns the record."""
    leg = f"{topo}-{policy}"
    leg_dir = study_dir / leg
    leg_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"leg": leg, "policy": policy, "topology": topo,
                    "warm_slots": warm_n}

    # -- reset: process restart IS the cache reset ---------------------------
    sampler.phase = "reset"
    if not rt.reset_cache():
        record["verdict"] = "restart_failed"
        return record
    probe = rt.wait_functional(FUNCTIONAL_TIMEOUT_S)
    record["post_reset_probe"] = {"ok": probe.ok, "tokens": probe.tokens,
                                  "error": probe.error}
    if not probe.ok:
        record["verdict"] = "not_functional_after_reset"
        return record

    model = rt.model
    sampler.phase = "idle"

    with (leg_dir / "gateway.log").open("wb") as gw_log:
        gw = subprocess.Popen([str(gateway_bin)],
                              stdout=gw_log, stderr=subprocess.STDOUT,
                              env={**os.environ,
                                   **gateway_slot_env(policy, model, rt.port, leg_params["slots"])},
                              cwd=REPO, start_new_session=True)
        try:
            effective_slots = 0 if policy == "native" else leg_params["slots"]
            record["gateway_at_start"] = wait_gateway_all_ready(effective_slots)

            # -- topology warmup (pinned, through the gateway) ---------------
            sampler.phase = f"{leg}-warm"
            if warm_n > 0:
                record["warm"] = warm_slots(gateway_bin, loadgen_bin, leg_dir,
                                            model, warm_n)
                capture_slots_snapshot(rt, leg_dir / "slots_snapshot.json")
                status, body = http_get("http://127.0.0.1:8080/status")
                if status == 200:
                    (leg_dir / "gateway_status_after_warm.json").write_text(body)
                if not record["warm"]["ok"]:
                    # Never analyze a mistopologied leg: the measured run
                    # would silently measure the wrong cache state.
                    record["verdict"] = "warm_failed"
                    return record
                time.sleep(1.0)  # let the gateway's completion bookkeeping land

            # -- measured run -------------------------------------------------
            sampler.phase = leg
            write_leg_matrix(leg_dir / "leg.toml", leg,
                             f"phase-4B.2 {leg}: {SHARED_TOKENS}t shared prefix, "
                             f"{leg_params['n']} reqs, c{leg_params['c']}, out={leg_params['out']}",
                             "http://127.0.0.1:8080", model, leg_params["out"],
                             leg_params["c"], leg_params["n"])
            occupancy = SlotsOccupancySampler(rt.port, leg_dir / "slots_occupancy.csv")
            occupancy.start()
            t0 = time.time()
            rc = subprocess.run(
                [str(loadgen_bin), "--matrix", str(leg_dir / "leg.toml"),
                 "--out-dir", str(leg_dir)],
                cwd=REPO, capture_output=True, text=True)
            record["run_exit"] = rc.returncode
            record["run_wall_s"] = round(time.time() - t0, 1)
            if rc.returncode != 0:
                (leg_dir / "run.log").write_text(rc.stdout[-4000:] + rc.stderr[-4000:])
            occupancy.stop()

            # -- evidence capture ---------------------------------------------
            time.sleep(2)
            sampler.phase = "post_probe"
            scrape_gateway_metrics(leg_dir / "gateway_metrics.json")
            n = extract_routing_log(leg_dir / "gateway.log", leg_dir / "routing_decisions.log")
            record["routing_decision_lines"] = n

            post = rt.functional_probe(phase=f"post_{leg}")
            record["post_run_probe"] = {"ok": post.ok, "tokens": post.tokens,
                                        "error": post.error, "total_ms": post.total_ms}
            record["verdict"] = ("healthy" if post.ok and record["run_exit"] == 0
                                 else "degraded")
        finally:
            if gw.poll() is None:
                os.killpg(gw.pid, 15)
                try:
                    gw.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(gw.pid, 9)
    return record


def two_prefix_experiment(rt, gateway_bin: Path, loadgen_bin: Path,
                          study_dir: Path, sampler: MemorySampler) -> dict:
    """Miniature multi-tenant cluster: A and B prefixes warm on slots 0/1,
    then 4 A-requests and 4 B-requests arrive interleaved (two concurrent
    loadgen processes). The router should keep each prefix on its warm
    capacity yet spill rather than starve when a warm slot is saturated."""
    record: dict = {"leg": "two-prefix", "policy": "cache_aware_ect"}
    leg_dir = study_dir / "two-prefix"
    leg_dir.mkdir(parents=True, exist_ok=True)

    if not rt.reset_cache():
        record["verdict"] = "restart_failed"
        return record
    probe = rt.wait_functional(FUNCTIONAL_TIMEOUT_S)
    if not probe.ok:
        record["verdict"] = "not_functional_after_reset"
        return record

    model = rt.model
    with (leg_dir / "gateway.log").open("wb") as gw_log:
        gw = subprocess.Popen([str(gateway_bin)],
                              stdout=gw_log, stderr=subprocess.STDOUT,
                              env={**os.environ,
                                   **gateway_slot_env("cache_aware_ect", model, rt.port, 8)},
                              cwd=REPO, start_new_session=True)
        try:
            record["gateway_at_start"] = wait_gateway_all_ready(8)

            sampler.phase = "two-prefix-warm"
            # slot0 <- prefix A (seed 42), slot1 <- prefix B (seed 43).
            record["warm"] = warm_slots(gateway_bin, loadgen_bin, leg_dir, model, 2,
                                        seeds=[42, 43])

            sampler.phase = "two-prefix"
            base = dict(input_tokens=SHARED_TOKENS + UNIQUE_TOKENS,
                        shared_prefix_tokens=SHARED_TOKENS,
                        output_tokens=OUTPUT_TOKENS, stream=True,
                        arrival="burst", timeout_s=900, concurrency=4,
                        num_requests=4, model=model,
                        endpoint="http://127.0.0.1:8080")
            variants = {"a": 42, "b": 43}
            procs = []
            t0 = time.time()
            for v, seed in variants.items():
                toml = leg_dir / f"{v}.toml"
                toml.write_text(f"""
study = "prefix-router-two-prefix-{v}"
description = "two-prefix {v}: seed {seed}, 4 interleaved requests"
cooldown_s = 0.0
[defaults]
endpoint = "{base['endpoint']}"
model = "{model}"
output_tokens = {base['output_tokens']}
stream = true
arrival = "burst"
timeout_s = 900
seed = {seed}
shared_prefix_tokens = {SHARED_TOKENS}

[[runs]]
name = "two-prefix-{v}"
input_tokens = {base['input_tokens']}
concurrency = 4
num_requests = 4
""")
                procs.append((v, subprocess.Popen(
                    [str(loadgen_bin), "--matrix", str(toml),
                     "--out-dir", str(leg_dir)],
                    cwd=REPO, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True)))
            for v, p in procs:
                out, err = p.communicate()
                record[f"{v}_exit"] = p.returncode
                if p.returncode != 0:
                    (leg_dir / f"{v}.log").write_text((out or "")[-2000:] + (err or "")[-2000:])
            record["run_wall_s"] = round(time.time() - t0, 1)

            time.sleep(2)
            sampler.phase = "post_probe"
            scrape_gateway_metrics(leg_dir / "gateway_metrics.json")
            record["routing_decision_lines"] = extract_routing_log(
                leg_dir / "gateway.log", leg_dir / "routing_decisions.log")
            post = rt.functional_probe(phase="post_two_prefix")
            record["post_run_probe"] = {"ok": post.ok, "tokens": post.tokens,
                                        "error": post.error}
            record["verdict"] = ("healthy" if post.ok
                                 and all(record.get(f"{v}_exit") == 0 for v in variants)
                                 else "degraded")
        finally:
            if gw.poll() is None:
                os.killpg(gw.pid, 15)
                try:
                    gw.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(gw.pid, 9)
    return record


def main() -> int:
    global SHARED_TOKENS, UNIQUE_TOKENS, OUTPUT_TOKENS, NUM_REQUESTS, CONCURRENCY
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="llama-cpp",
                    choices=["llama-cpp", "mlx-lm", "vllm-metal"])
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--topologies", default=",".join(TOPOLOGIES))
    ap.add_argument("--cs", default=f"{CONCURRENCY}", help="concurrency (comma list adds legs)")
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks/results")
    ap.add_argument("--gateway", type=Path, default=REPO / "target/release/gateway")
    ap.add_argument("--study-name", default="prefix-router")
    ap.add_argument("--two-prefix", action="store_true",
                    help="append the A/B two-prefix interleaved experiment")
    ap.add_argument("--skip-matrix", action="store_true",
                    help="skip the 16-leg matrix (use with --two-prefix)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny workload (1K prefix, 4 requests, out=32) to "
                         "validate the pipeline; not comparable data")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    topos = [t.strip() for t in args.topologies.split(",") if t.strip()]
    cs = [int(x) for x in args.cs.split(",")]
    for p in policies:
        if p not in POLICIES:
            sys.exit(f"unknown policy {p!r} (one of {POLICIES})")
    for t in topos:
        if t not in TOPOLOGIES:
            sys.exit(f"unknown topology {t!r} (one of {list(TOPOLOGIES)})")
    if args.smoke:
        SHARED_TOKENS, UNIQUE_TOKENS = 1024, 128
        OUTPUT_TOKENS, NUM_REQUESTS, CONCURRENCY = 32, 4, 4
        policies = [p for p in policies if p in ("native", "cache_aware_ect")] or ["native", "cache_aware_ect"]
        topos = [t for t in topos if t in ("all-cold", "one-warm")] or ["all-cold", "one-warm"]

    if not args.gateway.exists():
        sys.exit("gateway binary missing — cargo build --release -p gateway")
    loadgen_bin = args.gateway.parent / "loadgen"
    if not loadgen_bin.exists():
        sys.exit("loadgen binary missing — cargo build --release --workspace")

    ts = time.strftime("%Y%m%d-%H%M%S")
    study_dir = args.out / f"{args.study_name}-{ts}"
    study_dir.mkdir(parents=True)
    print(f"study dir: {study_dir}")

    rt = make_runtime(args.runtime)
    legs: list[dict] = []
    sampler = MemorySampler(lambda: rt.proc.pid if rt.proc else None,
                            study_dir / "memory.csv")
    sampler.start()
    try:
        kill_leftovers(rt.name)
        rt.start(study_dir / "runtime.log")
        if not rt.wait_live(LIVE_TIMEOUT_S):
            sys.exit(f"{rt.name} never went live")
        time.sleep(2)
        sampler.phase = "after_load"

        leg_params = {"slots": 8, "c": CONCURRENCY, "n": NUM_REQUESTS, "out": OUTPUT_TOKENS}
        if not args.skip_matrix:
            for topo in topos:
                warm_n = TOPOLOGIES[topo]
                for policy in policies:
                    for c in cs:
                        leg_params = {"slots": 8, "c": c, "n": NUM_REQUESTS, "out": OUTPUT_TOKENS}
                        leg = f"{topo}-{policy}" + (f"-c{c}" if len(cs) > 1 else "")
                        log(f"=== leg {leg}")
                        rec = run_leg(rt, args.gateway, loadgen_bin, study_dir,
                                      policy, topo, warm_n, sampler, leg_params)
                        if len(cs) > 1:
                            rec["leg"] = leg
                        legs.append(rec)
                        log(f"=== leg {leg}: {rec.get('verdict')} "
                            f"(exit={rec.get('run_exit')}, wall={rec.get('run_wall_s')}s)")

        if args.two_prefix:
            log("=== two-prefix A/B interleaved experiment")
            rec = two_prefix_experiment(rt, args.gateway, loadgen_bin, study_dir,
                                        sampler)
            legs.append(rec)
            log(f"=== two-prefix: {rec.get('verdict')}")
    finally:
        rt.stop()
        sampler.stop()
        lifecycle = {
            "study": args.study_name,
            "created": ts,
            "runtime": rt.name,
            "model": rt.model,
            "question": (
                "phase-4B showed llama.cpp's native LCP slot selection makes cache "
                "locality absolute and cold same-prefix bursts serialize (253 s vs 19 s "
                "TTFT p50). Which routing policy reuses cache when it pays and spreads "
                "when it does not?"
            ),
            "workload": {
                "shared_prefix_tokens": SHARED_TOKENS,
                "unique_suffix_tokens": UNIQUE_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "num_requests": NUM_REQUESTS,
                "concurrency": CONCURRENCY,
                "seed": SEED,
                "stream": True,
                "arrival": "burst",
                "via_gateway": True,
            },
            "policies": policies,
            "topologies": {t: TOPOLOGIES[t] for t in topos},
            "cache_reset_method": rt.cache_reset_method(),
            "cache_warm_method": (
                "parallel pinned loadgen one-shots THROUGH the gateway "
                "(--pin-slot i, same seed -> byte-identical shared prefix); slot "
                "policies learn the topology from these completions"
            ),
            "profile_seeds": PROFILE,
            "legs": legs,
            "memory_peak_rss_mb": round(sampler.peak_rss_mb, 1),
            "memory_peak_footprint_mb": round(sampler.peak_footprint_mb, 1),
        }
        (study_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2))
        log(f"lifecycle written: {study_dir / 'lifecycle.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
