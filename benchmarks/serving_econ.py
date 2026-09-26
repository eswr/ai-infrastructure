#!/usr/bin/env python3
"""serving_econ.py — phase-4C serving-interactions analyzer.

Renders one study dir from benchmarks/serving_interactions.py into the
phase's economics tables:

  * contention wave — prefill slowdown vs aggregate throughput per
    concurrent level (the saturation point), decode scaling, and what
    injected prefills do to a live interactive stream's ITL/stall;
  * the learned curve — the persisted calibration profile the wave (and
    legs) taught the gateway, next to the machine truth that measured it;
  * the 16-cell matrix — size x cache x admission x contention-gate, with
    per-cell latency, throughput, reuse, and gate activity;
  * the mixed workload — per service class latency vs the interactive
    SLO (TTFT <= 5 s), baseline vs full stack.

Usage:  python3 benchmarks/serving_econ.py <study_dir>
Stdlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUCKETS = ["1", "2", "3-4", "5-8"]
SLO_INTERACTIVE_TTFT_S = 5.0


def fmt(v, nd=1, suffix=""):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}{suffix}"


def ms(v):
    return fmt(v / 1000.0, 1, "s") if v is not None else "n/a"


def load_lifecycle(study: Path) -> dict:
    path = study / "lifecycle.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_profile(study: Path) -> dict | None:
    path = study / "calibration_profile_final.json"
    return json.loads(path.read_text()) if path.exists() else None


def leg_summary(leg_dir: Path) -> dict | None:
    """The measured run's summary from a leg's matrix.json."""
    for name in ("matrix.json",):
        p = leg_dir / name
        if p.exists():
            rows = [r for r in json.loads(p.read_text()).get("rows", [])
                    if not r.get("warmup")]
            if rows:
                return rows[-1].get("summary", {})
    return None


def metric(metrics: dict, name: str) -> float | None:
    """Total of a metric family across label sets."""
    total = 0.0
    found = False
    for (n, _labels), v in metrics.items():
        if n == name:
            total += v
            found = True
    return total if found else None


def load_metrics(leg_dir: Path) -> dict:
    p = leg_dir / "gateway_metrics.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    # keys are "name{labels}" strings -> parse back to (name, labels)
    out = {}
    for k, v in data.items():
        if "{" in k:
            name = k[:k.index("{")]
            out[(name, k[k.index("{"):])] = v
        else:
            out[(k, "")] = v
    return out


def pct(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


# ---------------------------------------------------------------------------
# Wave
# ---------------------------------------------------------------------------

def wave_tables(wave: dict, md: list[str]) -> None:
    md += ["## Contention wave — the machine, measured", ""]
    pf = wave.get("prefill_contention") or []
    if pf:
        by_level: dict[int, list[dict]] = {}
        for row in pf:
            by_level.setdefault(row["level"], []).append(row)
        md += ["### Prefill contention (cold identical prompts, out=1)", "",
               "| Active prefills | Runs | TTFT p50 | ms/token (individual) | "
               "Aggregate tok/s |",
               "|---:|---:|---:|---:|---:|"]
        for level in sorted(by_level):
            rows = by_level[level]
            ms_tok = [r["prefill_ms_per_tok_mean"] for r in rows
                      if r.get("prefill_ms_per_tok_mean") is not None]
            agg = [r["aggregate_prefill_tok_s"] for r in rows
                   if r.get("aggregate_prefill_tok_s") is not None]
            p50 = [r["ttft_p50_s"] for r in rows if r.get("ttft_p50_s") is not None]
            mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
            md.append(
                f"| {level} | {len(rows)} | {fmt(mean(p50), 1, 's')} "
                f"| {fmt(mean(ms_tok), 2)} | {fmt(mean(agg))} |")
        md += ["",
               f"**Prefill saturation point: L={wave.get('prefill_saturation_level')}** "
               "(first level whose aggregate tok/s stops improving on the previous).", ""]

    dec = wave.get("decode_scaling") or []
    if dec:
        by_level: dict[int, list[dict]] = {}
        for row in dec:
            by_level.setdefault(row["level"], []).append(row)
        md += ["### Decode scaling (short prompts, long decode)", "",
               "| Active decodes | Runs | decode ms/token (mean) | decode ms/token p50 |",
               "|---:|---:|---:|---:|"]
        for level in sorted(by_level):
            rows = by_level[level]
            mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
            m1 = [r["decode_ms_per_tok_mean"] for r in rows if r.get("decode_ms_per_tok_mean")]
            m2 = [r["decode_ms_per_tok_p50"] for r in rows if r.get("decode_ms_per_tok_p50")]
            md.append(f"| {level} | {len(rows)} | {fmt(mean(m1))} | {fmt(mean(m2))} |")
        md.append("")

    inj = wave.get("decode_interference") or []
    if inj:
        md += ["### Prefill → decode interference (one live interactive stream)",
               "",
               "| Injected prefills | ITL p50 | ITL p95 | ITL p99 | Max stall | "
               "Interactive TTFT | Injected TTFT p50 |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in inj:
            md.append(
                f"| {row['level']} | {fmt(row.get('itl_p50_ms'))}ms "
                f"| {fmt(row.get('itl_p95_ms'))}ms | {fmt(row.get('itl_p99_ms'))}ms "
                f"| {fmt(row.get('max_stall_s'), 2, 's')} "
                f"| {fmt(row.get('interactive_ttft_s'), 2, 's')} "
                f"| {fmt(row.get('injected_ttft_p50_s'), 2, 's')} |")
        md.append("")


def curve_table(study: Path, wave: dict, md: list[str]) -> None:
    prof = load_profile(study)
    if not prof:
        return
    md += ["## The learned curve (persisted calibration profile)", "",
           "```text",
           f"prefill solo: fixed {prof['prefill']['fixed_ms']:.0f} ms, "
           f"{prof['prefill']['ms_per_token']:.2f} ms/est-tok "
           f"({prof['prefill']['samples']} samples)",
           "bucket            " + "  ".join(f"{b:>6}" for b in BUCKETS),
           "prefill mult      " + "  ".join(f"{m:6.2f}" for m in prof["prefill"]["contention_multiplier"]),
           "  samples         " + "  ".join(f"{s:6d}" for s in prof["prefill"]["contention_samples"]),
           "decode ms/tok     " + "  ".join(f"{m:6.0f}" for m in prof["decode"]["ms_per_token"]),
           "  samples         " + "  ".join(f"{s:6d}" for s in prof["decode"]["samples"]),
           "```", ""]
    if wave.get("aggregate_prefill_tok_s_by_level"):
        md += ["Client-side aggregate tok/s by level (wave): `" +
               json.dumps(wave["aggregate_prefill_tok_s_by_level"]) + "`", ""]


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

def matrix_table(study: Path, legs: list[dict], md: list[str]) -> None:
    if not legs:
        return
    md += ["## The serving-interaction matrix (cache x admission x contention gate)", "",
           "| Cell | Verdict | Term | OK | TTFT p50 | TTFT p95 | ITL p50 | Out tok/s "
           "| Reused tok/req | Gate holds | Spills |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for leg in legs:
        leg_dir = study / leg["leg"]
        s = leg_summary(leg_dir)
        if not s:
            continue
        metrics = load_metrics(leg_dir)
        holds = metric(metrics, "scheduler_contention_gate_holds_total")
        spills = metric(metrics, "scheduler_contention_gate_spills_total")
        n_obs = s.get("cache_observed_requests")
        reused = (s.get("reused_tokens_total") / n_obs
                  if s.get("reused_tokens_total") is not None and n_obs else None)
        term = ("-" if not leg.get("terminated_early")
                else leg.get("termination_reason") or "early")
        md.append(
            f"| {leg['leg']} | {leg.get('verdict')} | {term} "
            f"| {s.get('successful', '?')}/{s.get('total_requests', '?')} "
            f"| {ms(s.get('ttft_ms', {}).get('p50'))} "
            f"| {ms(s.get('ttft_ms', {}).get('p95'))} "
            f"| {ms(s.get('itl_ms', {}).get('p50'))} "
            f"| {fmt(s.get('output_tok_s'))} "
            f"| {fmt(reused, 0)} "
            f"| {fmt(holds, 0)} | {fmt(spills, 0)} |")
    md.append("")


# ---------------------------------------------------------------------------
# Mixed workload
# ---------------------------------------------------------------------------

def class_stats(cfg_dir: Path, tag: str) -> dict | None:
    """Per-request stats for one mixed-workload class (its single-run JSON)."""
    matches = sorted(set(cfg_dir.glob(f"**/{tag}.json"))
                     | set(cfg_dir.glob(f"**/{tag}-*.json")))
    if not matches:
        return None
    data = json.loads(matches[-1].read_text())
    records = [r for r in data.get("records", []) if r.get("ok")]
    ttfts = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    e2es = [r["e2e_ms"] for r in records if r.get("e2e_ms") is not None]
    total = data.get("summary", {}).get("total_requests", len(records))
    violations = sum(1 for t in ttfts if t > SLO_INTERACTIVE_TTFT_S * 1000.0)
    out_tok = sum(r.get("completion_tokens") or 0 for r in records)
    return {
        "ok": len(records), "total": total,
        "ttft_p50_ms": pct(ttfts, 0.5), "ttft_p95_ms": pct(ttfts, 0.95),
        "e2e_p50_ms": pct(e2es, 0.5),
        "slo_violations": violations, "slo_rate": violations / total if total else None,
        "out_tok_s": (out_tok / (max(e2es) / 1000.0)) if e2es else None,
    }


def mixed_table(study: Path, mixed: dict, md: list[str]) -> None:
    if not mixed:
        return
    md += ["## Mixed interactive workload", "",
           f"Interactive SLO: TTFT <= {SLO_INTERACTIVE_TTFT_S:.0f} s. "
           "2x8K warm-prefix (long_context/standard) + 2x8K cold (batch) "
           "+ 4x1K (interactive) over one short burst.", "",
           "| Config | Class | OK | TTFT p50 | TTFT p95 | E2E p50 | SLO violations "
           "| Out tok/s | Wall |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in mixed.get("configs", []):
        cfg_dir = study / f"mixed-{cfg['config']}"
        for tag in ("long-context", "batch", "interactive"):
            stats = class_stats(cfg_dir, tag)
            if not stats:
                continue
            md.append(
                f"| {cfg['config']} | {tag} | {stats['ok']}/{stats['total']} "
                f"| {ms(stats['ttft_p50_ms'])} | {ms(stats['ttft_p95_ms'])} "
                f"| {ms(stats['e2e_p50_ms'])} "
                f"| {stats['slo_violations']}/{stats['total']} "
                f"({fmt(100 * stats['slo_rate'])}%) "
                f"| {fmt(stats['out_tok_s'])} |")
        md.append("")
    # Verdict comparison block
    md += ["### Baseline vs full stack", ""]
    for cfg in mixed.get("configs", []):
        cfg_dir = study / f"mixed-{cfg['config']}"
        s = class_stats(cfg_dir, "interactive")
        wall = cfg.get("run_wall_s")
        md.append(f"- **{cfg['config']}** (admission={cfg['admission']}, "
                  f"gate={cfg['gate']}): interactive TTFT p50 "
                  f"{ms(s['ttft_p50_ms']) if s else 'n/a'}, p95 "
                  f"{ms(s['ttft_p95_ms']) if s else 'n/a'}, SLO violations "
                  f"{fmt(100 * s['slo_rate']) if s and s['slo_rate'] is not None else 'n/a'}%, "
                  f"wall {fmt(wall)}s")
    md.append("")


# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: serving_econ.py <study_dir>")
    study = Path(sys.argv[1])
    lc = load_lifecycle(study)
    if not lc:
        sys.exit(f"no lifecycle.json under {study}")
    stages = lc.get("stages", {})

    md: list[str] = [
        f"# Serving interactions — {study.name}", "",
        f"Phase 4C study on {lc.get('runtime')} / {lc.get('model')}, "
        f"created {lc.get('created')}. Question: {lc.get('question')}", "",
    ]

    wave = stages.get("wave") or {}
    wave_tables(wave, md)
    curve_table(study, wave, md)
    matrix_table(study, stages.get("matrix") or [], md)
    mixed_table(study, stages.get("mixed") or {}, md)

    out_dir = study / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "serving-economics.md").write_text("\n".join(md))
    (out_dir / "serving_econ.json").write_text(json.dumps(
        {"study": study.name, "lifecycle": lc}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_dir / 'serving-economics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
