#!/usr/bin/env python3
"""serving_interactions.py — phase-4C serving interactions study driver.

The phase question: given limited accelerator capacity, should a request
reuse cached KV, join concurrent execution, spill to another slot, or wait?
4B.3 left the cost model with measured solo rates but a DEGENERATE
contention curve (multipliers clamped at the 0.5 floor; anti-scaling not
captured). 4C therefore starts by MEASURING contention as a first-class
resource before letting routing/admission consume it.

Three controlled wave phases, all cold (distinct seed -> distinct prefix,
never a reused byte), all through the gateway so the shared calibration
learns every observation in its concurrency bucket:

  prefill contention   L in {1,2,4,8} identical-size 8K prompts, out=1:
                       per-request prefill ms/token, aggregate prefill
                       tok/s, TTFT. The point where adding another prefill
                       stops raising aggregate throughput is the PREFILL
                       SATURATION POINT.
  decode scaling       L in {1,2,4,8} short-prompt long-decode streams:
                       decode ms/token per active-stream level.
  decode interference  ONE live interactive decode stream, then inject
                       N in {1,2,4,8} cold 8K prefills: the stream's ITL
                       p50/p95/p99 and max stall show what prefill work
                       does to an already-interactive user.

Everything lands in the study's calibration dir, so later legs (and later
gateway restarts) boot from the measured curve instead of seeds.

Usage:
    python3 benchmarks/serving_interactions.py --runtime llama-cpp \
        --contention-wave [--rounds 2] [--smoke]
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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

PREFILL_TOKENS = 8192
UNIQUE_TOKENS = 128
OUTPUT_TOKENS = 256
PREFILL_LEVELS = [1, 2, 4, 8]
DECODE_LEVELS = [1, 2, 4, 8]
DECODE_OUT = 384
INTERFERE_OUT = 192

# llama.cpp service-model seeds on this machine (4A/4B studies). The wave
# re-measures everything; these only shape decisions until it does.
PROFILE = {"PROFILE_FIXED_MS": "250", "PROFILE_PREFILL_MS_PER_TOK": "6.2",
           "PROFILE_DECODE_MS_PER_TOK": "25"}


def gateway_env(model: str, port: int, cal_dir: str, *,
                admission: bool = False, gate: bool = False) -> dict:
    env = {
        "GATEWAY_ADDR": "127.0.0.1:8080",
        "WORKERS": f"llama=http://127.0.0.1:{port}#{model}",
        "ROUTE_POLICY": "cache-aware-ect",
        "WORKER_SLOTS": "8",
        "FIRST_TOKEN_TIMEOUT_S": "900",
        # 4B.2 lesson: probes queued behind 8K slot prefills false-quarantine
        # busy slots at the 10 s default; 120 s tolerates a full prefill.
        "FUNCTIONAL_PROBE_TIMEOUT_S": "120",
        # Steady-state probes are GPU work; keep measured phases quiet.
        "FUNCTIONAL_PROBE_INTERVAL_READY_S": "0",
        "HEALTH_INTERVAL_S": "2",
        "FUNCTIONAL_PROBE_INTERVAL_S": "2",
        **PROFILE,
        "CALIBRATION_DIR": cal_dir,
        "CALIBRATION_MACHINE": "apple-m5",
        "CALIBRATION_SAVE_INTERVAL_S": "2",
    }
    # The 4C knobs. The input budget (2 x 8K prompts) is the binding one:
    # the output budget sits at exactly the c8 demand and never bites.
    if admission:
        env["ADMIT_MAX_ACTIVE_INPUT_TOKENS"] = "16384"
        env["ADMIT_MAX_ACTIVE_OUTPUT_TOKENS"] = "2048"
    if gate:
        env["ADMIT_MARGINAL_COST"] = "1"
    return env


def wait_gateway_all_ready(slots: int, timeout_s: float = 90.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, body = http_get("http://127.0.0.1:8080/status")
        if status == 200:
            try:
                workers = json.loads(body)["workers"]
                if len(workers) == slots and \
                        all(w["state"] == "ready" for w in workers):
                    return {"ready": True}
            except (KeyError, json.JSONDecodeError):
                pass
        time.sleep(0.3)
    raise RuntimeError("gateway workers never reached READY")


def run_loadgen(loadgen_bin: Path, toml_path: Path, out_dir: Path,
                log_path: Path | None = None) -> dict:
    rc = subprocess.run(
        [str(loadgen_bin), "--matrix", str(toml_path), "--out-dir", str(out_dir)],
        cwd=REPO, capture_output=True, text=True)
    if rc.returncode != 0:
        text = (rc.stdout or "")[-3000:] + (rc.stderr or "")[-3000:]
        if log_path is not None:
            log_path.write_text(text)
    return {"exit": rc.returncode}


def run_loadgen_monitored(loadgen_bin: Path, toml_path: Path, out_dir: Path,
                          log_path: Path | None = None, *,
                          budget_s: float = 900.0, stall_s: float = 90.0) -> dict:
    """Run one measured leg with EARLY TERMINATION (phase 4C): a collapsed
    leg is better evidence recorded than another 20 minutes burned.

    Aborts the loadgen when any of these fire:
      * worker_unhealthy — the gateway quarantined a worker (lifecycle);
      * no_progress     — no new completed request for `stall_s` while work
                          is in flight. Armed only after the leg's FIRST
                          completion: a cold c8 anti-scaling crawl
                          legitimately completes nothing for minutes;
      * time_budget     — the leg exceeded `budget_s` (the collapse backstop).
    Terminating the loadgen disconnects its clients, which cancels upstream
    generation and releases the workers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (log_path or (out_dir / "run.log")).open("wb") as fh:
        proc = subprocess.Popen(
            [str(loadgen_bin), "--matrix", str(toml_path), "--out-dir", str(out_dir)],
            cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        last_ok = 0.0
        last_progress = time.time()
        termination = None
        while proc.poll() is None:
            time.sleep(5)
            status, body = http_get("http://127.0.0.1:8080/status", timeout_s=2.0)
            if status == 200:
                try:
                    if any(not w["healthy"] for w in json.loads(body)["workers"]):
                        termination = "worker_unhealthy"
                        break
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            mstatus, mbody = http_get("http://127.0.0.1:8080/metrics", timeout_s=2.0)
            if mstatus == 200:
                prom = parse_prom_text(mbody)
                ok = sum(v for (n, l), v in prom.items()
                         if n == "inference_requests_total" and 'outcome="ok"' in l)
                in_flight = sum(v for (n, _l), v in prom.items()
                                if n == "inference_requests_in_flight")
                if ok > last_ok:
                    last_ok = ok
                    last_progress = time.time()
                elif in_flight > 0 and time.time() - last_progress > stall_s:
                    termination = "no_progress"
                    break
            if time.time() - t0 > budget_s:
                termination = "time_budget"
                break
        if termination and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    return {"exit": proc.returncode if not termination else 0,
            "raw_exit": proc.returncode,
            "terminated_early": termination is not None,
            "termination_reason": termination}


def write_matrix(path: Path, study: str, description: str, model: str, *,
                 input_tokens: int, shared: int, out_tokens: int,
                 concurrency: int, num_requests: int, seed: int) -> None:
    path.write_text(f"""
study = "{study}"
description = "{description}"
cooldown_s = 1.0
[defaults]
endpoint = "http://127.0.0.1:8080"
model = "{model}"
output_tokens = {out_tokens}
stream = true
arrival = "burst"
timeout_s = 900
seed = {seed}
shared_prefix_tokens = {shared}

[[runs]]
name = "{path.stem}"
input_tokens = {input_tokens}
concurrency = {concurrency}
num_requests = {num_requests}
""")


def read_records(out_dir: Path, name: str) -> tuple[dict, list[dict]]:
    """Latest <name>-<ts>.json under out_dir -> (summary, records).

    loadgen matrix mode nests artifacts one level down in a
    <study>-<timestamp>/ directory (per-run files carry no extra stamp),
    so match recursively and accept both spellings."""
    matches = sorted(set(out_dir.glob(f"**/{name}.json"))
                     | set(out_dir.glob(f"**/{name}-*.json")))
    if not matches:
        return {}, []
    data = json.loads(matches[-1].read_text())
    return data.get("summary", {}), data.get("records", [])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]


def calibration_status() -> dict | None:
    status, body = http_get("http://127.0.0.1:8080/status")
    if status != 200:
        return None
    try:
        return json.loads(body).get("calibration")
    except json.JSONDecodeError:
        return None


def start_gateway(gateway_bin: Path, log_path: Path, model: str, port: int,
                  cal_dir: str):
    gw = subprocess.Popen([str(gateway_bin)], stdout=log_path,
                          stderr=subprocess.STDOUT,
                          env={**os.environ, **gateway_env(model, port, cal_dir)},
                          cwd=REPO, start_new_session=True)
    return gw


def stop_gateway(gw) -> None:
    if gw.poll() is None:
        os.killpg(gw.pid, 15)
        try:
            gw.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(gw.pid, 9)


# ---------------------------------------------------------------------------
# Wave phases
# ---------------------------------------------------------------------------

def prefill_contention_phase(loadgen_bin: Path, wave_dir: Path, model: str,
                             rounds: int, seeds, pf_tokens: int,
                             levels=None) -> list[dict]:
    """L concurrent cold prefills of `pf_tokens` tokens (out=1), L in
    levels. Client-side TTFT ≈ prefill at out=1: individual ms/token =
    TTFT/tokens, aggregate tok/s = L × tokens / max(TTFT)."""
    levels = levels or PREFILL_LEVELS
    phase_dir = wave_dir / "prefill-contention"
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in levels:
        for r in range(rounds):
            seed = next(seeds)
            name = f"pf-L{level}-r{r}"
            write_matrix(phase_dir / f"{name}.toml", "contention-wave",
                         f"prefill contention L={level} round={r}", model,
                         input_tokens=pf_tokens + UNIQUE_TOKENS,
                         shared=0, out_tokens=1, concurrency=level,
                         num_requests=level, seed=seed)
            t0 = time.time()
            rec = run_loadgen(loadgen_bin, phase_dir / f"{name}.toml",
                              phase_dir, phase_dir / f"{name}.log")
            wall = round(time.time() - t0, 1)
            summary, records = read_records(phase_dir, name)
            ttfts = [x.get("ttft_ms") for x in records if x.get("ok")]
            per_tok = [x["ttft_ms"] / (pf_tokens + UNIQUE_TOKENS)
                       for x in records if x.get("ok") and x.get("ttft_ms")]
            agg_tok_s = None
            if ttfts:
                agg_tok_s = level * (pf_tokens + UNIQUE_TOKENS) / max(ttfts) * 1000.0
            rows.append({
                "level": level, "round": r, "seed": seed, "wall_s": wall,
                "exit": rec["exit"], "ok": len(ttfts), "total": level,
                "ttft_p50_s": round(percentile(ttfts, 0.5) / 1000.0, 2) if ttfts else None,
                "ttft_max_s": round(max(ttfts) / 1000.0, 2) if ttfts else None,
                "prefill_ms_per_tok_mean": round(sum(per_tok) / len(per_tok), 3) if per_tok else None,
                "aggregate_prefill_tok_s": round(agg_tok_s, 1) if agg_tok_s else None,
            })
            log(f"  prefill L={level} r={r}: p50={rows[-1]['ttft_p50_s']}s "
                f"ms/tok={rows[-1]['prefill_ms_per_tok_mean']} "
                f"agg={rows[-1]['aggregate_prefill_tok_s']} tok/s")
    return rows


def decode_scaling_phase(loadgen_bin: Path, wave_dir: Path, model: str,
                         rounds: int, seeds, levels=None) -> list[dict]:
    """L concurrent short-prompt long-decode streams: the decode contention
    curve (per-stream decode ms/token by active-stream level)."""
    levels = levels or DECODE_LEVELS
    phase_dir = wave_dir / "decode-scaling"
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in levels:
        for r in range(rounds):
            seed = next(seeds)
            name = f"dec-L{level}-r{r}"
            write_matrix(phase_dir / f"{name}.toml", "contention-wave",
                         f"decode scaling L={level} round={r}", model,
                         input_tokens=128, shared=0, out_tokens=DECODE_OUT,
                         concurrency=level, num_requests=level, seed=seed)
            rec = run_loadgen(loadgen_bin, phase_dir / f"{name}.toml",
                              phase_dir, phase_dir / f"{name}.log")
            _, records = read_records(phase_dir, name)
            decode_ms_tok = []
            stalls = []
            for x in records:
                if not x.get("ok") or not x.get("e2e_ms") or not x.get("ttft_ms"):
                    continue
                toks = max(1, x.get("completion_tokens") or 0)
                decode_ms_tok.append((x["e2e_ms"] - x["ttft_ms"]) / toks)
                if x.get("max_inter_token_gap_ms") is not None:
                    stalls.append(x["max_inter_token_gap_ms"])
            rows.append({
                "level": level, "round": r, "seed": seed,
                "exit": rec["exit"],
                "decode_ms_per_tok_mean": round(sum(decode_ms_tok) / len(decode_ms_tok), 1)
                if decode_ms_tok else None,
                "decode_ms_per_tok_p50": round(percentile(decode_ms_tok, 0.5), 1)
                if decode_ms_tok else None,
                "max_stall_s": round(max(stalls) / 1000.0, 2) if stalls else None,
            })
            log(f"  decode L={level} r={r}: {rows[-1]['decode_ms_per_tok_mean']} ms/tok")
    return rows


def decode_interference_phase(loadgen_bin: Path, wave_dir: Path, model: str,
                              seeds, levels=None) -> list[dict]:
    """One live interactive decode stream per level; inject N cold 8K
    prefills mid-stream; report the stream's ITL/stall damage."""
    levels = levels or PREFILL_LEVELS
    phase_dir = wave_dir / "decode-interference"
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in levels:
        seed = next(seeds)
        a_name = f"inj-L{level}-interactive"
        write_matrix(phase_dir / f"{a_name}.toml", "contention-wave",
                     f"interactive stream under L={level} injected prefills",
                     model, input_tokens=128, shared=0,
                     out_tokens=INTERFERE_OUT, concurrency=1, num_requests=1,
                     seed=seed)
        b_name = f"inj-L{level}-prefills"
        write_matrix(phase_dir / f"{b_name}.toml", "contention-wave",
                     f"injected burst L={level}", model,
                     input_tokens=PREFILL_TOKENS + UNIQUE_TOKENS, shared=0,
                     out_tokens=1, concurrency=level, num_requests=level,
                     seed=seed + 1)
        a = subprocess.Popen(
            [str(loadgen_bin), "--matrix", str(phase_dir / f"{a_name}.toml"),
             "--out-dir", str(phase_dir)],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True)
        # Let the interactive stream reach its decode phase before injecting.
        time.sleep(6)
        t0 = time.time()
        b = run_loadgen(loadgen_bin, phase_dir / f"{b_name}.toml", phase_dir,
                        phase_dir / f"{b_name}.log")
        inject_wall = round(time.time() - t0, 1)
        a.communicate()
        a_exit = a.returncode
        _, a_records = read_records(phase_dir, a_name)
        _, b_records = read_records(phase_dir, b_name)
        a_rec = next((x for x in a_records if x.get("ok")), {})
        b_ttfts = [x.get("ttft_ms") for x in b_records if x.get("ok") and x.get("ttft_ms")]
        rows.append({
            "level": level, "interactive_exit": a_exit,
            "inject_exit": b["exit"], "inject_wall_s": inject_wall,
            "itl_p50_ms": a_rec.get("itl_p50_ms"),
            "itl_p95_ms": a_rec.get("itl_p95_ms"),
            "itl_p99_ms": a_rec.get("itl_p99_ms"),
            "max_stall_s": round(a_rec["max_inter_token_gap_ms"] / 1000.0, 2)
            if a_rec.get("max_inter_token_gap_ms") is not None else None,
            "interactive_ttft_s": round(a_rec["ttft_ms"] / 1000.0, 2)
            if a_rec.get("ttft_ms") else None,
            "injected_ttft_p50_s": round(percentile(b_ttfts, 0.5) / 1000.0, 2)
            if b_ttfts else None,
        })
        log(f"  interference L={level}: itl p95={rows[-1]['itl_p95_ms']}ms "
            f"max stall={rows[-1]['max_stall_s']}s")
    return rows


def contention_wave(gateway_bin: Path, loadgen_bin: Path, study_dir: Path,
                    model: str, rt, sampler: MemorySampler, cal_dir: str,
                    rounds: int, smoke: bool) -> dict:
    leg_dir = study_dir / "contention-wave"
    leg_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"leg": "contention-wave", "rounds": rounds, "smoke": smoke}

    if not rt.reset_cache():
        record["verdict"] = "restart_failed"
        return record
    if not rt.wait_functional(FUNCTIONAL_TIMEOUT_S).ok:
        record["verdict"] = "not_functional_after_reset"
        return record

    seed_counter = {"n": 9000}

    def seeds():
        while True:
            seed_counter["n"] += 7
            yield seed_counter["n"]

    seed_iter = seeds()
    pf_levels = [1, 2] if smoke else PREFILL_LEVELS
    dec_levels = [1, 2] if smoke else DECODE_LEVELS
    wave_rounds = 1 if smoke else rounds
    pf_tokens = 1024 - UNIQUE_TOKENS if smoke else PREFILL_TOKENS
    with (leg_dir / "gateway.log").open("wb") as gw_log:
        gw = start_gateway(gateway_bin, gw_log, model, rt.port, cal_dir)
        try:
            record["gateway_at_start"] = wait_gateway_all_ready(8)
            sampler.phase = "wave-prefill"
            t0 = time.time()
            record["prefill_contention"] = prefill_contention_phase(
                loadgen_bin, leg_dir, model, wave_rounds, seed_iter,
                pf_tokens, pf_levels)
            record["calibration_after_prefill"] = calibration_status()
            sampler.phase = "wave-decode"
            record["decode_scaling"] = decode_scaling_phase(
                loadgen_bin, leg_dir, model, wave_rounds, seed_iter, dec_levels)
            record["calibration_after_decode"] = calibration_status()
            if not smoke:
                sampler.phase = "wave-interfere"
                record["decode_interference"] = decode_interference_phase(
                    loadgen_bin, leg_dir, model, seed_iter)
            record["wall_s"] = round(time.time() - t0, 1)
            record["calibration_after"] = calibration_status()

            # Saturation point: the first level whose aggregate prefill
            # throughput stops improving on the previous level.
            agg = {}
            for row in record["prefill_contention"]:
                agg.setdefault(row["level"], []).append(row["aggregate_prefill_tok_s"] or 0.0)
            means = {lv: sum(v) / len(v) for lv, v in agg.items() if v}
            sat = None
            ordered = sorted(means)
            for prev, cur in zip(ordered, ordered[1:]):
                if means[cur] <= means[prev]:
                    sat = cur
                    break
            record["aggregate_prefill_tok_s_by_level"] = {
                str(k): round(v, 1) for k, v in means.items()}
            record["prefill_saturation_level"] = sat
            ok = all(row["exit"] == 0 for row in record["prefill_contention"]) and \
                all(row["exit"] == 0 for row in record["decode_scaling"])
            record["verdict"] = "healthy" if ok else "degraded"
        finally:
            stop_gateway(gw)
    return record


# ---------------------------------------------------------------------------
# The 16-cell serving-interaction matrix and the mixed interactive workload
# ---------------------------------------------------------------------------

WARM_SEED = 42          # the shared-prefix seed: warmups and measured runs
SIZES = {"short": 1024, "long": 8192}
UNIQUE = 128            # per-request unique suffix tokens
MIXED_SLO_INTERACTIVE_TTFT_S = 5.0


def parse_prom_text(text: str) -> dict:
    """Minimal Prometheus text parser: {(name, labels): value}."""
    import re
    out = {}
    pat = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+e0-9.]+)$")
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        m = pat.match(ln.strip())
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[(m.group(1), m.group(2) or "")] = out.get((m.group(1), m.group(2) or ""), 0.0) + v
    return out


def scrape_evidence(leg_dir: Path) -> None:
    status, body = http_get("http://127.0.0.1:8080/metrics")
    raw = leg_dir / "gateway_metrics.txt"
    js = leg_dir / "gateway_metrics.json"
    if status != 200:
        raw.write_text("")
        js.write_text("{}")
        return
    raw.write_text(body)
    parsed = {f"{name}{labels}": v for (name, labels), v in parse_prom_text(body).items()}
    js.write_text(json.dumps(
        {k: v for k, v in parsed.items()
         if k.startswith("scheduler_") or k.startswith("inference_")}, indent=2))
    status, body = http_get("http://127.0.0.1:8080/status")
    if status == 200:
        (leg_dir / "gateway_status.json").write_text(body)


def extract_routing_log(gateway_log: Path, dest: Path) -> int:
    if not gateway_log.exists():
        return 0
    lines = [ln for ln in gateway_log.read_text(errors="replace").splitlines()
             if "routing decision" in ln or "contention gate" in ln]
    dest.write_text("\n".join(lines) + "\n")
    return len(lines)


def warm_slots(loadgen_bin: Path, leg_dir: Path, model: str, size: int, n: int = 8) -> dict:
    """Populate every slot's cache with the shared prefix: parallel pinned
    one-shot loadgen runs THROUGH the gateway, same seed as the measured
    run so the prefix is byte-identical."""
    out_dir = leg_dir / "warm"
    out_dir.mkdir(exist_ok=True)
    procs = []
    t0 = time.time()
    for i in range(n):
        cmd = [str(loadgen_bin),
               "--endpoint", "http://127.0.0.1:8080",
               "--model", model,
               "--input-tokens", str(size),
               "--shared-prefix-tokens", str(size - UNIQUE),
               "--output-tokens", "4",
               "--num-requests", "1",
               "--concurrency", "1",
               "--stream", "true",
               "--seed", str(WARM_SEED),
               "--pin-slot", str(i),
               "--timeout-s", "900",
               "--out-dir", str(out_dir / f"slot{i}")]
        procs.append((i, subprocess.Popen(cmd, cwd=REPO,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, text=True)))
    results = []
    for i, proc in procs:
        _, err = proc.communicate()
        results.append({"slot": i, "exit": proc.returncode,
                        "err": (err or "")[-200:] if proc.returncode else None})
    wall = round(time.time() - t0, 1)
    ok = all(r["exit"] == 0 for r in results)
    log(f"warm: {n} pinned populate run(s) in {wall}s, ok={ok}")
    return {"requested": n, "ok": ok, "wall_s": wall, "results": results}


def seed_cal_dir(src: Path, dst: Path) -> None:
    """Give a leg its own copy of the calibrated profile: every leg (and
    both sides of an A/B) starts from the SAME measured machine truth and
    evolves independently - no cross-leg calibration drift."""
    if src and src.exists():
        import shutil
        shutil.copytree(src, dst, dirs_exist_ok=True)


def matrix_leg(gateway_bin: Path, loadgen_bin: Path, study_dir: Path,
               model: str, rt, sampler: MemorySampler, cal_dir: str,
               size_name: str, cache: str, admission: bool, gate: bool,
               out_tokens: int, n: int, c: int, smoke: bool,
               budget_s: float = 900.0) -> dict:
    """One cell of the 4C matrix: size x cache x admission x gate.
    Cache state is rebuilt honestly per leg (process restart = cold; the
    restart IS the only cache reset llama.cpp 0.4.1 offers)."""
    leg = f"{size_name}-{cache}-admit{int(admission)}-gate{int(gate)}"
    leg_dir = study_dir / leg
    leg_dir.mkdir(parents=True, exist_ok=True)
    size = SIZES[size_name]
    shared = size - UNIQUE
    record: dict = {"leg": leg, "size": size_name, "cache": cache,
                    "admission": admission, "gate": gate}

    sampler.phase = "reset"
    if not rt.reset_cache():
        record["verdict"] = "restart_failed"
        return record
    probe = rt.wait_functional(FUNCTIONAL_TIMEOUT_S)
    record["post_reset_probe"] = {"ok": probe.ok, "error": probe.error}
    if not probe.ok:
        record["verdict"] = "not_functional_after_reset"
        return record

    with (leg_dir / "gateway.log").open("wb") as gw_log:
        gw = subprocess.Popen(
            [str(gateway_bin)], stdout=gw_log, stderr=subprocess.STDOUT,
            env={**os.environ, **gateway_env(model, rt.port, cal_dir,
                                             admission=admission, gate=gate)},
            cwd=REPO, start_new_session=True)
        try:
            record["gateway_at_start"] = wait_gateway_all_ready(8)

            if cache == "warm":
                sampler.phase = f"{leg}-warm"
                record["warm"] = warm_slots(loadgen_bin, leg_dir, model, size)
                if not record["warm"]["ok"]:
                    record["verdict"] = "warm_failed"
                    return record
                time.sleep(1.0)

            sampler.phase = leg
            write_matrix(leg_dir / "leg.toml", f"serving-interactions-{leg}",
                         f"4C {leg}: {size}t prompts ({shared}+{UNIQUE}), "
                         f"n={n} c={c} out={out_tokens}",
                         model, input_tokens=size, shared=shared,
                         out_tokens=out_tokens, concurrency=c,
                         num_requests=n, seed=WARM_SEED)
            t0 = time.time()
            rec = run_loadgen_monitored(loadgen_bin, leg_dir / "leg.toml",
                                        leg_dir, leg_dir / "run.log",
                                        budget_s=budget_s)
            record.update(rec)
            record["run_exit"] = rec["exit"]
            record["run_wall_s"] = round(time.time() - t0, 1)

            time.sleep(2)
            sampler.phase = "post_probe"
            scrape_evidence(leg_dir)
            record["routing_decision_lines"] = extract_routing_log(
                leg_dir / "gateway.log", leg_dir / "routing_decisions.log")

            post = rt.functional_probe(phase=f"post_{leg}")
            record["post_run_probe"] = {"ok": post.ok, "tokens": post.tokens,
                                        "error": post.error}
            collapsed = (record.get("terminated_early")
                         and record.get("termination_reason") != "worker_unhealthy")
            record["verdict"] = ("collapsed" if collapsed
                                 else "healthy"
                                 if post.ok and record["run_exit"] == 0
                                 else "degraded")
        finally:
            stop_gateway(gw)
    return record


# The integration-test cells (phase 4C revised scope): does the full stack
# (calibrated contention gate + admission) help when prefill is expensive
# (cold), and does it avoid hurting when cache reuse makes prefill cheap
# (warm)? Baseline = both off; full = both on. out=64: the question is
# TTFT / contention / queueing, not decode volume.
FAST_CELLS = [("long", "cold", False, False), ("long", "cold", True, True),
              ("long", "warm", False, False), ("long", "warm", True, True)]


def cache_matrix(gateway_bin: Path, loadgen_bin: Path, study_dir: Path,
                 model: str, rt, sampler: MemorySampler, cal_dir: str,
                 smoke: bool, full: bool, seed_cal: Path | None,
                 budget_s: float) -> list[dict]:
    """full=True: the exhaustive 2x2x2x2 = 16-cell ablation (out=256).
    full=False (the default integration test): FAST_CELLS with out=64.
    Each leg gets its own calibration copy seeded from `seed_cal` so the
    A/B sides start from identical machine knowledge."""
    out_tokens = 32 if smoke else (OUTPUT_TOKENS if full else 64)
    n = c = 4 if smoke else 8
    if full:
        cells = [(s, cache, adm, gate)
                 for s in (["short"] if smoke else list(SIZES))
                 for cache in ("cold", "warm")
                 for adm in (False, True)
                 for gate in (False, True)]
    else:
        cells = ([(s, cache, adm, gate)
                  for (s, cache, adm, gate) in FAST_CELLS]
                 if not smoke else
                 [("short", "cold", False, False), ("short", "cold", True, True)])
    legs = []
    for (size_name, cache, adm, gate) in cells:
        leg = f"{size_name}-{cache}-admit{int(adm)}-gate{int(gate)}"
        log(f"=== cell {leg}")
        # Per-leg calibration: a fresh copy of the seeded profile per leg
        # when seeding is on; otherwise the shared study dir (legacy).
        leg_cal = cal_dir
        if seed_cal:
            leg_cal = str(study_dir / leg / "calibration")
            seed_cal_dir(seed_cal, Path(leg_cal))
        rec = matrix_leg(gateway_bin, loadgen_bin, study_dir, model, rt,
                         sampler, leg_cal, size_name, cache, adm, gate,
                         out_tokens, n, c, smoke, budget_s=budget_s)
        legs.append(rec)
        log(f"=== cell {rec['leg']}: {rec.get('verdict')} "
            f"(exit={rec.get('run_exit')}, wall={rec.get('run_wall_s')}s"
            + (f", early={rec.get('termination_reason')}"
               if rec.get("terminated_early") else "") + ")")
    return legs


def mixed_arrival(loadgen_bin: Path, phase_dir: Path, model: str, tag: str,
                  *, n: int, c: int, input_tokens: int, shared: int,
                  out_tokens: int, seed: int, service_class: str) -> subprocess.Popen:
    """One class-tagged concurrent burst process of the mixed workload."""
    toml = phase_dir / f"{tag}.toml"
    write_matrix(toml, f"mixed-{tag}", f"mixed workload {tag} ({service_class})",
                 model, input_tokens=input_tokens, shared=shared,
                 out_tokens=out_tokens, concurrency=c, num_requests=n,
                 seed=seed)
    # The class tag rides the matrix defaults onto every request.
    text = toml.read_text()
    text = text.replace("[defaults]\n", f"[defaults]\nservice_class = \"{service_class}\"\n", 1)
    toml.write_text(text)
    return subprocess.Popen(
        [str(loadgen_bin), "--matrix", str(toml), "--out-dir", str(phase_dir)],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def mixed_workload(gateway_bin: Path, loadgen_bin: Path, study_dir: Path,
                   model: str, rt, sampler: MemorySampler, cal_dir: str,
                   seed_cal: Path | None = None, budget_s: float = 900.0) -> dict:
    """The 4C centerpiece: 2x8K warm-prefix + 2x8K cold + 4x1K interactive,
    all arriving over one short burst, run twice - baseline (admission off,
    gate off) vs full stack (admission on, contention gate on). Warm-cache
    state before each config, classes on the wire, interactive SLO 5 s.
    Each config gets its own calibration copy (same seed) so the A/B is
    not confounded by cross-config calibration drift."""
    record: dict = {"leg": "mixed", "slo_interactive_ttft_s": MIXED_SLO_INTERACTIVE_TTFT_S}
    record["configs"] = []
    for config, admission, gate in (("baseline", False, False), ("full", True, True)):
        cfg_dir = study_dir / f"mixed-{config}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg: dict = {"config": config, "admission": admission, "gate": gate}
        leg_cal = cal_dir
        if seed_cal:
            leg_cal = str(cfg_dir / "calibration")
            seed_cal_dir(seed_cal, Path(leg_cal))

        if not rt.reset_cache():
            cfg["verdict"] = "restart_failed"
            record["configs"].append(cfg)
            continue
        if not rt.wait_functional(FUNCTIONAL_TIMEOUT_S).ok:
            cfg["verdict"] = "not_functional_after_reset"
            record["configs"].append(cfg)
            continue

        with (cfg_dir / "gateway.log").open("wb") as gw_log:
            gw = subprocess.Popen(
                [str(gateway_bin)], stdout=gw_log, stderr=subprocess.STDOUT,
                env={**os.environ, **gateway_env(model, rt.port, leg_cal,
                                                 admission=admission, gate=gate)},
                cwd=REPO, start_new_session=True)
            try:
                cfg["gateway_at_start"] = wait_gateway_all_ready(8)
                sampler.phase = f"mixed-{config}-warm"
                cfg["warm"] = warm_slots(loadgen_bin, cfg_dir, model,
                                         SIZES["long"])
                if not cfg["warm"]["ok"]:
                    cfg["verdict"] = "warm_failed"
                    record["configs"].append(cfg)
                    continue
                time.sleep(1.0)

                # One short burst: 2 warm 8K (long_context/standard),
                # 2 cold 8K (batch), 4 1K (interactive). Distinct seeds for
                # the cold and interactive classes (honestly cold), the
                # warm class reuses the warm prefix seed.
                sampler.phase = f"mixed-{config}"
                procs = [
                    ("long-context", mixed_arrival(
                        loadgen_bin, cfg_dir, model, "long-context", n=2, c=2,
                        input_tokens=SIZES["long"], shared=SIZES["long"] - UNIQUE,
                        out_tokens=OUTPUT_TOKENS, seed=WARM_SEED,
                        service_class="standard")),
                    ("batch", mixed_arrival(
                        loadgen_bin, cfg_dir, model, "batch", n=2, c=2,
                        input_tokens=SIZES["long"], shared=0,
                        out_tokens=OUTPUT_TOKENS, seed=777,
                        service_class="batch")),
                    ("interactive", mixed_arrival(
                        loadgen_bin, cfg_dir, model, "interactive", n=4, c=4,
                        input_tokens=SIZES["short"], shared=0,
                        out_tokens=OUTPUT_TOKENS, seed=555,
                        service_class="interactive")),
                ]
                t0 = time.time()
                for tag, proc in procs:
                    _, err = proc.communicate()
                    cfg[f"{tag}_exit"] = proc.returncode
                    if proc.returncode != 0:
                        (cfg_dir / f"{tag}.log").write_text((err or "")[-2000:])
                cfg["run_wall_s"] = round(time.time() - t0, 1)

                time.sleep(2)
                sampler.phase = "post_probe"
                scrape_evidence(cfg_dir)
                cfg["routing_decision_lines"] = extract_routing_log(
                    cfg_dir / "gateway.log", cfg_dir / "routing_decisions.log")
                post = rt.functional_probe(phase=f"post_mixed_{config}")
                cfg["post_run_probe"] = {"ok": post.ok, "error": post.error}
                exits_ok = all(cfg.get(f"{t}_exit") == 0
                               for t in ("long-context", "batch", "interactive"))
                cfg["verdict"] = ("healthy" if post.ok and exits_ok else "degraded")
            finally:
                stop_gateway(gw)
        record["configs"].append(cfg)
        log(f"=== mixed/{config}: {cfg.get('verdict')} (wall={cfg.get('run_wall_s')}s)")
    return record


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="llama-cpp",
                    choices=["llama-cpp", "mlx-lm", "vllm-metal"])
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks/results")
    ap.add_argument("--gateway", type=Path, default=REPO / "target/release/gateway")
    ap.add_argument("--study-name", default="serving-interactions-4c")
    ap.add_argument("--cal-dir", type=Path, default=None,
                    help="persistent calibration dir (default: <study>/calibration)")
    ap.add_argument("--contention-wave", action="store_true",
                    help="run the controlled contention wave "
                         "(prefill levels, decode scaling, decode interference)")
    ap.add_argument("--matrix-fast", action="store_true", dest="matrix_fast",
                    help="the 4-cell integration test: {cold,warm} x "
                         "{baseline, full stack} on 8K prompts, out=64")
    ap.add_argument("--matrix-full", action="store_true", dest="matrix_full",
                    help="the exhaustive 16-cell ablation (publication-grade; "
                         "NOT the default - it runs for hours)")
    ap.add_argument("--mixed", action="store_true",
                    help="run the mixed interactive workload (baseline vs full stack)")
    ap.add_argument("--seed-cal", type=Path, default=None, dest="seed_cal",
                    help="calibration DIR to copy per leg (e.g. a prior "
                         "study's wave-taught profile): every leg and both "
                         "A/B sides start from the same machine truth")
    ap.add_argument("--leg-budget-s", type=float, default=900.0,
                    help="early-termination wall budget per measured leg "
                         "(default 900)")
    ap.add_argument("--rounds", type=int, default=2,
                    help="rounds per contention level (default 2)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny wave (1K prompts, 2 levels, 1 round) to "
                         "validate the pipeline; not comparable data")
    args = ap.parse_args()

    if not (args.contention_wave or args.matrix_fast or args.matrix_full or args.mixed):
        log("nothing to do: pass --contention-wave, --matrix-fast, "
            "--matrix-full and/or --mixed")
        return 2
    if not args.gateway.exists():
        sys.exit("gateway binary missing — cargo build --release -p gateway")
    loadgen_bin = args.gateway.parent / "loadgen"
    if not loadgen_bin.exists():
        sys.exit("loadgen binary missing — cargo build --release --workspace")

    ts = time.strftime("%Y%m%d-%H%M%S")
    study_dir = args.out / f"{args.study_name}-{ts}"
    study_dir.mkdir(parents=True)
    print(f"study dir: {study_dir}")
    cal_dir = str(args.cal_dir or (study_dir / "calibration"))

    rt = make_runtime(args.runtime)
    sampler = MemorySampler(lambda: rt.proc.pid if rt.proc else None,
                            study_dir / "memory.csv")
    sampler.start()
    record = {}
    try:
        kill_leftovers(rt.name)
        rt.start(study_dir / "runtime.log")
        if not rt.wait_live(LIVE_TIMEOUT_S):
            sys.exit(f"{rt.name} never went live")
        time.sleep(2)
        sampler.phase = "after_load"
        if args.contention_wave:
            log("=== contention wave (controlled prefill/decode levels)")
            record["wave"] = contention_wave(args.gateway, loadgen_bin, study_dir,
                                             rt.model, rt, sampler, cal_dir,
                                             rounds=args.rounds, smoke=args.smoke)
            wave = record["wave"]
            log(f"=== contention wave: {wave.get('verdict')} "
                f"(saturation level={wave.get('prefill_saturation_level')}, "
                f"wall={wave.get('wall_s')}s)")
        if args.matrix_fast or args.matrix_full:
            log("=== serving-interaction matrix (%s)"
                % ("16-cell full" if args.matrix_full else "fast integration test"))
            record["matrix"] = cache_matrix(args.gateway, loadgen_bin, study_dir,
                                            rt.model, rt, sampler, cal_dir,
                                            smoke=args.smoke,
                                            full=args.matrix_full,
                                            seed_cal=args.seed_cal,
                                            budget_s=args.leg_budget_s)
            good = sum(1 for leg in record["matrix"]
                       if leg.get("verdict") in ("healthy", "collapsed"))
            log(f"=== matrix: {good}/{len(record['matrix'])} legs finished")
        if args.mixed:
            log("=== mixed interactive workload (baseline vs full stack)")
            record["mixed"] = mixed_workload(args.gateway, loadgen_bin, study_dir,
                                             rt.model, rt, sampler, cal_dir,
                                             seed_cal=args.seed_cal,
                                             budget_s=args.leg_budget_s)
    finally:
        rt.stop()
        sampler.stop()
        if cal_dir:
            for prof in sorted(Path(cal_dir).glob("*/*/*.json")):
                try:
                    (study_dir / "calibration_profile_final.json").write_text(prof.read_text())
                except OSError:
                    pass
        lifecycle = {
            "study": args.study_name,
            "created": ts,
            "runtime": rt.name,
            "model": rt.model,
            "question": (
                "phase-4C: measure accelerator contention as a first-class "
                "resource — prefill slowdown vs aggregate throughput per "
                "concurrent level, decode scaling, and prefill->decode "
                "interference — before routing/admission consume the curve."
            ),
            "smoke": args.smoke,
            "rounds": args.rounds,
            "stages": record,
            "memory_peak_rss_mb": round(sampler.peak_rss_mb, 1),
            "memory_peak_footprint_mb": round(sampler.peak_footprint_mb, 1),
        }
        (study_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2))
        log(f"lifecycle written: {study_dir / 'lifecycle.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
