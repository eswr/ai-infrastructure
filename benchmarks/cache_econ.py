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
    mj = leg_dir / "matrix.json"
    if not mj.exists():
        return None
    data = json.loads(mj.read_text())
    rows = [r for r in data.get("rows", []) if not r.get("warmup")]
    if not rows:
        return None
    row = rows[-1]
    recs_path = sorted(leg_dir.glob("*.records.json")) or sorted(leg_dir.glob("*-*.json"))
    # records live in <leg>.json written by loadgen (summary+records)
    reused = []
    for jf in leg_dir.glob("*.json"):
        if jf.name == "matrix.json":
            continue
        try:
            d = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        for rec in d.get("records", []):
            if rec.get("reused_tokens") is not None and rec.get("ok"):
                reused.append(rec["reused_tokens"])
    p = row["params"]
    s = row["summary"]
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
        "shared_prefix_tokens": p.get("shared_prefix_tokens"),
        "prefix_fingerprint": p.get("prefix_fingerprint"),
        "reused_total": s.get("reused_tokens_total"),
        "reused_n": s.get("cache_observed_requests"),
        "reused_per_req": (sum(reused) / len(reused)) if reused else None,
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
        if saved is not None and warm["reused_per_req"]:
            per_tok = saved / warm["reused_per_req"]
        if warm["reused_per_req"] and warm["input_tokens"]:
            eff_prefill = warm["input_tokens"] - warm["reused_per_req"]
        econ[c] = {
            "ttft_saved_ms": saved,
            "warm_mean_reused_tokens": warm["reused_per_req"],
            "ms_saved_per_cached_token": per_tok,
            "effective_prefill_tokens_per_req": eff_prefill,
            "effective_prefill_fraction": (
                eff_prefill / warm["input_tokens"]
                if eff_prefill is not None and warm["input_tokens"] else None
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
           "| Effective prefill tok/req | Effective prefill share |",
           "| -:" + "|---" * 5 + "|"]
    for c in sorted(econ):
        e = econ[c]
        saved = fmt_ms(e["ttft_saved_ms"]) if e["ttft_saved_ms"] is not None else "n/a"
        per_tok = f"{e['ms_saved_per_cached_token']:.2f}" if e["ms_saved_per_cached_token"] else "n/a"
        eff = f"{e['effective_prefill_tokens_per_req']:.0f}" if e["effective_prefill_tokens_per_req"] else "n/a"
        share = f"{e['effective_prefill_fraction'] * 100:.1f}%" if e["effective_prefill_fraction"] else "n/a"
        md.append(f"| {c} | {saved} | {e['warm_mean_reused_tokens']:.0f} | {per_tok} | {eff} | {share} |")

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
