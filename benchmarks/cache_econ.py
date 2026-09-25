#!/usr/bin/env python3
"""cache_econ.py — cache economics for the phase-4B prefix-cache study.

Reads a study directory of legs (one matrix.json per leg dir, named
`<state>-c<c>`, state in {cold, warm}) produced by
benchmarks/prefix_cache.py, plus the harness memory CSV, and emits:

  analysis/cache-economics.md   the central table + derived economics
  analysis/cache_econ.json      the same numbers machine-readable

The economics layer, per concurrency c:

  ttft_saved_ms          = TTFT_p50(cold) - TTFT_p50(warm)
  ms_saved_per_cached_token = ttft_saved_ms / mean_reused_tokens(warm)
  effective_prefill_tokens  = prompt_tokens - reused_tokens  (per request,
                             from the runtime's OWN telemetry — never
                             inferred from latency)
  effective_prefill_tok_s   = effective_prefill_tokens / TTFT
  footprint per checkpoint  from the 1 Hz vmmap-sampled memory CSV
                            (RSS lies once weights are wired into Metal
                            memory — phase-4A lesson)

Run:  python3 benchmarks/cache_econ.py <study-dir> \
          [--frozen-off benchmarks/results/cross-runtime-<ts>/cross-runtime.json]
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1000:.1f}s" if v >= 10_000 else f"{v:.0f}"


def load_leg(leg_dir: Path) -> dict | None:
    mj_files = sorted(leg_dir.glob("*/matrix.json"))
    if not mj_files:
        return None
    data = json.loads(mj_files[-1].read_text())
    rows = [r for r in data.get("rows", []) if not r.get("warmup")]
    if not rows:
        return None
    row = rows[-1]
    p = row["params"]
    s = row["summary"]
    # Observed reuse from the measured run's own summary — warm legs also
    # contain a populate warmup run whose records would dilute the mean.
    n_obs = s.get("cache_observed_requests")
    reused_per_req = None
    if s.get("reused_tokens_total") is not None and n_obs:
        reused_per_req = s["reused_tokens_total"] / n_obs
    prompt_per_req = (s.get("prompt_tokens_total") / s["total_requests"]
                      if s.get("prompt_tokens_total") else None)
    return {
        "leg": leg_dir.name,
        "state": leg_dir.name.rsplit("-c", 1)[0],
        "c": int(leg_dir.name.rsplit("-c", 1)[1]),
        "ttft_p50_ms": (s.get("ttft_ms") or {}).get("p50"),
        "ttft_p95_ms": (s.get("ttft_ms") or {}).get("p95"),
        "out_tok_s": s.get("output_tok_s"),
        "stall_max_ms": (s.get("stream_max_stall_ms") or {}).get("max"),
        "spread_ms": s.get("stream_completion_spread_ms"),
        "ok": s.get("successful"),
        "total": s.get("total_requests"),
        "input_tokens": p.get("input_tokens"),
        "prompt_tokens_server": prompt_per_req,
        "shared_prefix_tokens": p.get("shared_prefix_tokens"),
        "prefix_fingerprint": p.get("prefix_fingerprint"),
        "reused_total": s.get("reused_tokens_total"),
        "reused_n": n_obs,
        "reused_per_req": reused_per_req,
    }


def frozen_off_rows(path: Path, runtime: str, prompt: int) -> dict[int, dict]:
    """The frozen cache-OFF baseline: phase-4A rows for `runtime` at
    `prompt` nominal tokens. Those runs used per-request-unique prompts
    (shared_prefix_tokens = 0), so observed reuse was zero by construction
    — the honest OFF semantics without rerunning anything."""
    d = json.loads(path.read_text())
    entry = d.get("runtimes", {}).get(runtime, {})
    out: dict[int, dict] = {}
    for row in entry.get("rows", []):
        p = row.get("params", {})
        if p.get("input_tokens") != prompt or p.get("shared_prefix_tokens", 0):
            continue
        s = row["summary"]
        c = p.get("concurrency")
        out[c] = {
            "c": c,
            "state": "off (frozen 4A)",
            "ttft_p50_ms": (s.get("ttft_ms") or {}).get("p50"),
            "ttft_p95_ms": (s.get("ttft_ms") or {}).get("p95"),
            "out_tok_s": s.get("output_tok_s"),
            "stall_max_ms": (s.get("stream_max_stall_ms") or {}).get("max"),
            "spread_ms": s.get("stream_completion_spread_ms"),
            "ok": s.get("successful"),
            "total": s.get("total_requests"),
            "input_tokens": p.get("input_tokens"),
            "prompt_tokens_server": (s.get("prompt_tokens_total") / s["total_requests"]
                                     if s.get("prompt_tokens_total") else None),
            "shared_prefix_tokens": 0,
            "reused_per_req": 0,
            "reused_total": 0,
        }
    return out


def footprint_phases(mem_csv: Path) -> dict[str, dict]:
    """Peak RSS / footprint per sampled phase. vmmap footprint (every 5th
    sample) is the number that counts on unified memory; RSS underreports
    wired GPU buffers."""
    if not mem_csv.exists():
        return {}
    phases: dict[str, dict] = {}
    lines = mem_csv.read_text().splitlines()[1:]
    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 7:
            continue
        phase = parts[1]
        try:
            rss = float(parts[2]) if parts[2] else None
            fp = float(parts[3]) if parts[3] else None
        except ValueError:
            continue
        ph = phases.setdefault(phase, {"peak_rss": 0.0, "peak_fp": 0.0, "n": 0})
        ph["n"] += 1
        if rss:
            ph["peak_rss"] = max(ph["peak_rss"], rss)
        if fp:
            ph["peak_fp"] = max(ph["peak_fp"], fp)
    return phases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir", type=Path)
    ap.add_argument("--frozen-off", type=Path, default=None,
                    help="cross-runtime.json with the frozen 4A OFF rows")
    ap.add_argument("--frozen-runtime", default="llama-cpp")
    ap.add_argument("--frozen-prompt", type=int, default=8192)
    args = ap.parse_args()

    legs = [load_leg(d) for d in sorted(args.study_dir.iterdir()) if d.is_dir()]
    legs = [l for l in legs if l]
    if not legs:
        sys.exit(f"no legs (matrix.json) under {args.study_dir}")

    if args.frozen_off:
        off = frozen_off_rows(args.frozen_off, args.frozen_runtime, args.frozen_prompt)
        legs = [off[k] for k in sorted(off)] + legs

    legs.sort(key=lambda l: (l["c"], 0 if l["state"].startswith("off") else
                             1 if l["state"] == "cold" else 2))

    cold_by_c = {l["c"]: l for l in legs if l["state"] == "cold"}
    warm_by_c = {l["c"]: l for l in legs if l["state"] == "warm"}

    econ: dict[int, dict] = {}
    for c, cold in cold_by_c.items():
        warm = warm_by_c.get(c)
        if not warm:
            continue
        saved = None
        if cold["ttft_p50_ms"] is not None and warm["ttft_p50_ms"] is not None:
            saved = cold["ttft_p50_ms"] - warm["ttft_p50_ms"]
        per_tok = None
        eff_prefill = None
        # cache_n counts cache POSITIONS (chat template + decode offsets),
        # so it can exceed the API prompt size; economics caps observed
        # reuse at the server-reported prompt. Raw counters stay in legs.
        prompt_real = warm["prompt_tokens_server"] or warm["input_tokens"]
        reused_eff = min(warm["reused_per_req"], prompt_real) if (
            warm["reused_per_req"] is not None and prompt_real) else None
        if saved is not None and reused_eff:
            per_tok = saved / reused_eff
        if reused_eff is not None:
            eff_prefill = prompt_real - reused_eff
        econ[c] = {
            "ttft_saved_ms": saved,
            "warm_mean_reused_tokens": reused_eff,
            "warm_mean_reused_raw_cache_n": warm["reused_per_req"],
            "server_prompt_tokens": prompt_real,
            "ms_saved_per_cached_token": per_tok,
            "effective_prefill_tokens_per_req": eff_prefill,
            "effective_prefill_fraction": (
                eff_prefill / prompt_real
                if eff_prefill is not None and prompt_real else None
            ),
        }

    phases = footprint_phases(args.study_dir / "memory.csv")

    # -- render -------------------------------------------------------------
    out_dir = args.study_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    md = ["# Prefix-cache economics — " + args.study_dir.name, ""]
    md += ["## Central table", "",
           "| c | Cache | TTFT p50 | TTFT p95 | Out tok/s | Max stall | Spread | Reused tok/req |",
           "| -:" + "|---" * 6 + "| ---: |"]
    for l in legs:
        reused = "0" if l["state"].startswith("off") else (
            f"{l['reused_per_req']:.0f}" if l["reused_per_req"] is not None else "n/a")
        md.append(
            f"| {l['c']} | {l['state']} | {fmt_ms(l['ttft_p50_ms'])} | {fmt_ms(l['ttft_p95_ms'])} "
            f"| {l['out_tok_s']:.1f} | {fmt_ms(l['stall_max_ms'])} "
            f"| {fmt_ms(l['spread_ms'])} | {reused} |")
    md += ["", "## What the cache bought", "",
           "| c | TTFT saved (cold−warm) | Warm reused tok/req | ms saved per cached token "
           "| Effective prefill tok/req (share of server prompt) |",
           "| -:" + "|---" * 4 + "|"]
    for c in sorted(econ):
        e = econ[c]
        saved = fmt_ms(e["ttft_saved_ms"]) if e["ttft_saved_ms"] is not None else "n/a"
        per_tok = f"{e['ms_saved_per_cached_token']:.2f}" if e["ms_saved_per_cached_token"] is not None else "n/a"
        eff = e["effective_prefill_tokens_per_req"]
        eff_s = "n/a" if eff is None else f"{eff:.0f} ({e['effective_prefill_fraction'] * 100:.1f}%)"
        md.append(f"| {c} | {saved} | {e['warm_mean_reused_tokens']:.0f} | {per_tok} | {eff_s} |")

    if phases:
        md += ["", "## Memory footprint by checkpoint (vmmap, MB)", "",
               "| phase | samples | peak footprint | peak RSS |", "|---|---:|---:|---:|"]
        for ph, v in phases.items():
            fp = f"{v['peak_fp']:.0f}" if v["peak_fp"] else "n/a"
            md.append(f"| {ph} | {v['n']} | {fp} | {v['peak_rss']:.0f} |")

    md += ["", "Reused tokens come from the runtime's own telemetry "
           "(llama.cpp timings.cache_n / usage cached_tokens); absence of "
           "telemetry is reported as n/a, never inferred from TTFT."]

    (out_dir / "cache-economics.md").write_text("\n".join(md) + "\n")
    (out_dir / "cache_econ.json").write_text(json.dumps(
        {"legs": legs, "economics": econ, "memory_phases": phases}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_dir}/cache-economics.md and cache_econ.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
