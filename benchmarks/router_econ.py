#!/usr/bin/env python3
"""router_econ.py — routing economics for the phase-4B.2 prefix-router study.

Reads a study directory produced by benchmarks/prefix_router.py (leg dirs
`<topology>-<policy>` with matrix.json + gateway_metrics.json + slots
occupancy CSVs, plus optional two-prefix/) and emits:

  analysis/router-economics.md   the policy x topology matrix + the three
                                 questions that matter
  analysis/router_econ.json      the same numbers machine-readable

The three questions, per cell:

  1. Did the policy reuse the cache when reuse was available?
     (observed reused tokens/req from the runtime's own cache_n telemetry)
  2. Did it refuse reuse when load made reuse the slower choice — and did
     it say so? (scheduler_cache_affinity_bypassed_total by reason vs
     scheduler_affinity_opportunities_total: "locality was available for
     X% of requests; the router deliberately ignored it for Y%")
  3. Was the model honest? (scheduler_prediction_abs_error_seconds
     sum/count = MAE, buckets = p50/p95; signed sum/count = bias)

plus the slot-occupancy trace (who actually ran where) and the A/B
two-prefix interleaved experiment when present.

Run:  python3 benchmarks/router_econ.py <study-dir>
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

POLICY_ORDER = ["native", "least_work", "prefix_only", "cache_aware_ect"]
TOPO_ORDER = ["all-cold", "one-warm", "four-warm", "all-warm"]


def fmt_s(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}"


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1000:.1f}s" if v >= 10_000 else f"{v:.0f}"


def parse_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def m(metrics: dict, name: str, labels: str = "") -> float | None:
    return metrics.get(f"{name}{labels}")


def bypasses_by_reason(metrics: dict) -> dict:
    out = {}
    for reason in ("queue_pressure", "prefill_pressure", "worker_degraded",
                   "insufficient_benefit"):
        v = m(metrics, "scheduler_cache_affinity_bypassed_total",
              f'{{reason="{reason}"}}')
        if v:
            out[reason] = v
    return out


def hist_stats(metrics: dict, name: str) -> dict:
    """sum/count plus bucket-interpolated p50/p95 from one histogram."""
    count = m(metrics, f"{name}_count")
    total = m(metrics, f"{name}_sum")
    if not count:
        return {}
    buckets = []
    for key, v in metrics.items():
        if not key.startswith(f"{name}_bucket{{"):
            continue
        mm = re.search(r'le="([^"]+)"', key)
        if mm:
            le = float("inf") if mm.group(1) == "+Inf" else float(mm.group(1))
            buckets.append((le, v))
    buckets.sort()

    def q(qq: float) -> float | None:
        if not buckets or not buckets[-1][1]:
            return None
        target = qq * buckets[-1][1]
        prev_le, prev_c = 0.0, 0.0
        for le, c in buckets:
            if c >= target:
                if le == float("inf") or c == prev_c:
                    return prev_le
                return prev_le + (target - prev_c) / (c - prev_c) * (le - prev_le)
            prev_le, prev_c = le, c
        return None

    return {"count": count, "sum": total or 0.0,
            "mean": (total or 0.0) / count,
            "p50": q(0.50), "p95": q(0.95)}


def occupancy_summary(csv_path: Path) -> dict | None:
    """Per-slot busy-sample share from the 1 Hz /slots trace — who ran."""
    if not csv_path.exists():
        return None
    lines = csv_path.read_text().splitlines()[1:]
    if not lines:
        return None
    n = len(lines)
    busy: dict[int, int] = {}
    for ln in lines:
        parts = ln.split(",")[1:]
        for i, v in enumerate(parts):
            if v == "1":
                busy[i] = busy.get(i, 0) + 1
    return {"samples": n,
            "busy_pct": {str(i): round(100.0 * c / n, 1) for i, c in sorted(busy.items())}}


def decisions_per_slot(metrics: dict) -> dict:
    out = {}
    for key, v in metrics.items():
        if key.startswith("scheduler_decisions_total{"):
            mm = re.search(r'worker="([^"]+)"', key)
            if mm:
                out[mm.group(1)] = v
    return out


def load_leg(leg_dir: Path) -> dict | None:
    mj_files = sorted(leg_dir.glob("matrix.json")) or sorted(leg_dir.glob("*/matrix.json"))
    if not mj_files:
        return None
    data = json.loads(mj_files[0].read_text())
    rows = [r for r in data.get("rows", []) if not r.get("warmup")]
    if not rows:
        return None
    row = rows[-1]
    p, s = row["params"], row["summary"]
    n_obs = s.get("cache_observed_requests")
    reused_per_req = (s["reused_tokens_total"] / n_obs
                      if s.get("reused_tokens_total") is not None and n_obs else None)
    metrics = parse_metrics_json(leg_dir / "gateway_metrics.json")
    pred = hist_stats(metrics, "scheduler_prediction_abs_error_seconds")
    signed = hist_stats(metrics, "scheduler_prediction_error_seconds")
    est = hist_stats(metrics, "scheduler_estimated_ttft_seconds")
    return {
        "leg": leg_dir.name,
        "topology": leg_dir.name,
        "policy": leg_dir.name,
        "ttft_p50_ms": (s.get("ttft_ms") or {}).get("p50"),
        "ttft_p95_ms": (s.get("ttft_ms") or {}).get("p95"),
        "itl_p50_ms": (s.get("itl_ms") or {}).get("p50"),
        "itl_p99_ms": (s.get("itl_ms") or {}).get("p99"),
        "out_tok_s": s.get("output_tok_s"),
        "stall_max_ms": (s.get("stream_max_stall_ms") or {}).get("max"),
        "spread_ms": s.get("stream_completion_spread_ms"),
        "ok": s.get("successful"),
        "total": s.get("total_requests"),
        "reused_per_req": reused_per_req,
        "reused_total": s.get("reused_tokens_total"),
        "cache_observed_requests": n_obs,
        "prompt_tokens_server": (s.get("prompt_tokens_total") / s["total_requests"]
                                 if s.get("prompt_tokens_total") else None),
        "bypasses": bypasses_by_reason(metrics),
        "opportunities": m(metrics, "scheduler_affinity_opportunities_total") or 0.0,
        "reused_tokens_observed_total": m(metrics, "scheduler_observed_reused_tokens_total") or 0.0,
        "est_ttft": est,
        "prediction": pred,
        "bias": signed.get("mean"),
        "decisions_per_slot": decisions_per_slot(metrics),
        "occupancy": occupancy_summary(leg_dir / "slots_occupancy.csv"),
    }


def split_leg(name: str) -> tuple[str, str]:
    """'one-warm-cache_aware_ect-c4' -> ('one-warm', 'cache_aware_ect')."""
    for topo in TOPO_ORDER:
        if name.startswith(topo + "-"):
            rest = name[len(topo) + 1:]
            m = re.match(r"^(.*)-c(\d+)$", rest)
            if m:
                rest = m.group(1)
            return topo, rest
    return name, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir", type=Path)
    args = ap.parse_args()

    legs = [load_leg(d) for d in sorted(args.study_dir.iterdir()) if d.is_dir()]
    legs = [l for l in legs if l]
    if not legs:
        sys.exit(f"no legs (matrix.json) under {args.study_dir}")

    for l in legs:
        topo, policy = split_leg(l["leg"])
        l["topology"], l["policy"] = topo, policy

    matrix = [l for l in legs if l["policy"] in POLICY_ORDER]
    two = next((l for l in legs if l["leg"].startswith("two-prefix")), None)
    matrix.sort(key=lambda l: (TOPO_ORDER.index(l["topology"]) if l["topology"] in TOPO_ORDER else 99,
                               POLICY_ORDER.index(l["policy"]) if l["policy"] in POLICY_ORDER else 99))

    out_dir = args.study_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    md: list[str] = ["# Prefix-router economics — " + args.study_dir.name, ""]

    # -- the 4x4 matrix ------------------------------------------------------
    md += ["## Policy x topology matrix (c8, 8x [8192 shared + 128 unique], out=256)", "",
           "| Topology | Policy | OK | TTFT p50 | TTFT p95 | ITL p50/p99 | Out tok/s | Max stall | Spread | Reused tok/req |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l in matrix:
        itl = f"{fmt_ms(l['itl_p50_ms'])}/{fmt_ms(l['itl_p99_ms'])}"
        reused = "n/a" if l["reused_per_req"] is None else f"{l['reused_per_req']:.0f}"
        md.append(
            f"| {l['topology']} | {l['policy']} | {l['ok']}/{l['total']} "
            f"| {fmt_ms(l['ttft_p50_ms'])} | {fmt_ms(l['ttft_p95_ms'])} | {itl} "
            f"| {l['out_tok_s']:.1f} | {fmt_ms(l['stall_max_ms'])} "
            f"| {fmt_ms(l['spread_ms'])} | {reused} |")

    # -- locality vs deliberate bypass ----------------------------------------
    md += ["", "## Locality vs load: was reuse available, and who refused it?", "",
           "| Topology | Policy | Affinity opportunities | Bypassed (by reason) | Observed reused tokens (total) |",
           "|---|---|---:|---|---:|"]
    for l in matrix:
        b = ", ".join(f"{k}: {v:.0f}" for k, v in l["bypasses"].items()) or "0"
        md.append(f"| {l['topology']} | {l['policy']} | {l['opportunities']:.0f} | {b} "
                  f"| {l['reused_tokens_observed_total']:.0f} |")

    # -- prediction quality ----------------------------------------------------
    md += ["", "## Was the cost model honest? (estimated vs actual TTFT)", "",
           "| Topology | Policy | Est TTFT mean | MAE | |err| p50 | |err| p95 | Bias (signed mean) | n |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for l in matrix:
        pred, est = l["prediction"], l["est_ttft"]
        if not pred:
            md.append(f"| {l['topology']} | {l['policy']} | n/a | n/a | n/a | n/a | n/a | 0 |")
            continue
        md.append(
            f"| {l['topology']} | {l['policy']} | {fmt_s(est.get('mean'))} "
            f"| {fmt_s(pred['mean'])} | {fmt_s(pred.get('p50'))} | {fmt_s(pred.get('p95'))} "
            f"| {fmt_s(l['bias'])} | {pred['count']:.0f} |")

    # -- slot occupancy ----------------------------------------------------------
    md += ["", "## Who actually ran? (slot busy share, 1 Hz /slots trace)", "",
           "| Topology | Policy | busy% per slot |", "|---|---|---|"]
    for l in matrix:
        occ = l.get("occupancy")
        if not occ:
            md.append(f"| {l['topology']} | {l['policy']} | n/a |")
            continue
        cells = ", ".join(f"s{k}: {v:.0f}%" for k, v in occ["busy_pct"].items())
        md.append(f"| {l['topology']} | {l['policy']} | {cells} |")

    if two:
        md += ["", "## Two-prefix A/B interleaved (mini multi-tenant cluster)", ""]
        md += ["| Variant | OK | TTFTs (s, by request id) | Reused tok/req | Mean reused |",
               "|---|---:|---|---:|---:|"]
        for v in ("a", "b"):
            files = sorted((args.study_dir / "two-prefix").glob(f"*/two-prefix-{v}.json"))
            if not files:
                continue
            recs = json.loads(files[0].read_text())["records"]
            tts = [round(r["ttft_ms"] / 1000, 1) for r in sorted(recs, key=lambda r: r["id"])]
            reuse = [r["reused_tokens"] for r in sorted(recs, key=lambda r: r["id"])]
            ok = sum(1 for r in recs if r["ok"])
            md.append(f"| {v} (seed {42 if v == 'a' else 43}) | {ok}/{len(recs)} | {tts} "
                      f"| {reuse} | {sum(reuse)/max(len(reuse),1):.0f} |")
        md += ["",
               "Warm slots: 0←A, 1←B (pinned, own seeds). Affinity = each prefix riding "
               "its warm slot; spill = a saturated warm slot's traffic rerouted cold "
               "(attributed bypasses) instead of queueing indefinitely."]

    (out_dir / "router-economics.md").write_text("\n".join(md) + "\n")
    (out_dir / "router_econ.json").write_text(json.dumps(
        {"legs": legs, "two_prefix": two}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_dir}/router-economics.md and router_econ.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
