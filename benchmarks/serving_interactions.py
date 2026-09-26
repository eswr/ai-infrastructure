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


def gateway_env(model: str, port: int, cal_dir: str) -> dict:
    return {
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
    """Latest <name>-<ts>.json under out_dir -> (summary, records)."""
    matches = sorted(out_dir.glob(f"{name}-*.json"))
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
    ap.add_argument("--rounds", type=int, default=2,
                    help="rounds per contention level (default 2)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny wave (1K prompts, 2 levels, 1 round) to "
                         "validate the pipeline; not comparable data")
    args = ap.parse_args()

    if not args.contention_wave:
        log("nothing to do: pass --contention-wave (matrix/mixed land in a later commit)")
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
        log("=== contention wave (controlled prefill/decode levels)")
        record = contention_wave(args.gateway, loadgen_bin, study_dir,
                                 rt.model, rt, sampler, cal_dir,
                                 rounds=args.rounds, smoke=args.smoke)
        log(f"=== contention wave: {record.get('verdict')} "
            f"(saturation level={record.get('prefill_saturation_level')}, "
            f"wall={record.get('wall_s')}s)")
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
            "wave": record,
            "memory_peak_rss_mb": round(sampler.peak_rss_mb, 1),
            "memory_peak_footprint_mb": round(sampler.peak_footprint_mb, 1),
        }
        (study_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2))
        log(f"lifecycle written: {study_dir / 'lifecycle.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
