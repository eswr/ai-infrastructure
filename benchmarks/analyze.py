#!/usr/bin/env python3
"""analyze.py — aggregate a loadgen matrix run into tables and charts.

Usage:
    python3 benchmarks/analyze.py benchmarks/results/<study>-<ts>/matrix.json

Reads the matrix.json written by `loadgen --matrix` (plus the per-run
record JSONs next to it) and writes into `<study dir>/analysis/`:
  * tables.md  — markdown tables (TTFT / ITL / E2E / throughput / knee)
  * *.svg      — line charts for the report

Stdlib only, so it runs on the bare system Python.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------- helpers --

PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea",
    "#ea580c", "#0891b2", "#4b5563", "#be185d",
]


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v >= 10_000:
        return f"{v / 1000:.1f}s"
    if v >= 1_000:
        return f"{v / 1000:.2f}s"
    return f"{v:.0f}"


def fmt_num(v: float | None, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def pctl(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    rank = min(max(p, 0.0), 1.0) * (len(s) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] * (1 - (rank - lo)) + s[hi] * (rank - lo)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo: float, hi: float, linear: bool) -> list[float]:
    if linear:
        raw = [lo + (hi - lo) * i / 4 for i in range(5)]
        step = (hi - lo) / 4
        mag = 10 ** math.floor(math.log10(step)) if step > 0 else 1
        return [round(t / mag) * mag for t in raw]
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    return [10.0**e for e in range(lo_exp, hi_exp + 1)]


def tick_label(v: float) -> str:
    if v >= 1000:
        for div, suffix in ((1_000_000, "M"), (1_000, "k")):
            if v >= div and abs(v / div - round(v / div)) < 1e-9:
                return f"{round(v / div)}{suffix}"
        return f"{v:.0f}"
    return f"{v:g}"


def line_chart(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, dict[float, float | None]]],
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """series: (label, {x: y}) — y None marks a gap (not plotted)."""
    W, H, M = 780, 470, 62  # canvas + left/bottom margin
    rx, rw, ry, rh = M, W - M - 150, 24, H - M - 30

    xs = [x for _, pts in series for x in pts]
    ys = [
        y for _, pts in series for y in pts.values()
        if y is not None and (y > 0 if log_y else y >= 0)
    ]
    if not xs or not ys:
        path.write_text(f"<!-- no data for {title} -->\n")
        return

    def project_x(x: float) -> float:
        if log_x:
            lx, hx = math.log10(min(xs)), math.log10(max(xs))
            return rx if hx == lx else rx + (math.log10(x) - lx) / (hx - lx) * rw
        lx, hx = min(xs), max(xs)
        return rx if hx == lx else rx + (x - lx) / (hx - lx) * rw

    # A measured 0 (e.g. a fully-timed-out run) must sit ON the axis, not
    # outside the plot: include it in the range on linear scales.
    ymin = 0.0 if (not log_y and min(ys) <= 0) else min(ys) * (0.5 if log_y else 0.9)
    ymax = max(ys) * (2.0 if log_y else 1.08)

    def project_y(y: float) -> float:
        if log_y:
            ly, hy = math.log10(ymin), math.log10(ymax)
            return ry + rh - (math.log10(max(y, ymin)) - ly) / (hy - ly) * rh
        return ry + rh - (y - ymin) / (ymax - ymin) * rh

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f"<text x='{rx}' y='16' font-size='15' font-weight='bold'>{esc(title)}</text>",
    ]

    xticks = sorted(set(xs))
    if log_x and len(xticks) > 6:
        xticks = nice_ticks(min(xs), max(xs), linear=False)
    for t in xticks:
        px = project_x(t)
        parts.append(f"<line x1='{px:.1f}' y1='{ry}' x2='{px:.1f}' y2='{ry + rh}' "
                     f"stroke='#e5e7eb' stroke-width='1'/>")
        parts.append(f"<text x='{px:.1f}' y='{ry + rh + 18}' font-size='11' "
                     f"text-anchor='middle' fill='#374151'>{tick_label(t)}</text>")
    for t in nice_ticks(ymin, ymax, linear=not log_y):
        if not (ymin <= t <= ymax):
            continue
        py = project_y(t)
        parts.append(f"<line x1='{rx}' y1='{py:.1f}' x2='{rx + rw}' y2='{py:.1f}' "
                     f"stroke='#f3f4f6' stroke-width='1'/>")
        parts.append(f"<text x='{rx - 8}' y='{py + 4:.1f}' font-size='11' "
                     f"text-anchor='end' fill='#374151'>{tick_label(t)}</text>")

    parts.append(f"<line x1='{rx}' y1='{ry + rh}' x2='{rx + rw}' y2='{ry + rh}' stroke='#9ca3af'/>")
    parts.append(f"<line x1='{rx}' y1='{ry}' x2='{rx}' y2='{ry + rh}' stroke='#9ca3af'/>")
    parts.append(f"<text x='{rx + rw / 2:.0f}' y='{H - 6}' font-size='12' "
                 f"text-anchor='middle' fill='#111827'>{esc(xlabel)}</text>")
    parts.append(f"<text x='16' y='{ry + rh / 2:.0f}' font-size='12' text-anchor='middle' "
                 f"fill='#111827' transform='rotate(-90 16 {ry + rh / 2:.0f})'>{esc(ylabel)}</text>")

    for i, (label, pts) in enumerate(series):
        color = PALETTE[i % len(PALETTE)]
        poly = " ".join(
            f"{project_x(x):.1f},{project_y(y):.1f}"
            for x, y in sorted(pts.items())
            if y is not None
        )
        if poly:
            parts.append(f"<polyline points='{poly}' fill='none' stroke='{color}' "
                         f"stroke-width='2'/>")
        for x, y in sorted(pts.items()):
            if y is not None:
                parts.append(f"<circle cx='{project_x(x):.1f}' cy='{project_y(y):.1f}' "
                             f"r='3.5' fill='{color}'/>")
        ly = ry + 18 + i * 18
        parts.append(f"<rect x='{rx + rw + 14}' y='{ly - 9}' width='11' height='11' "
                     f"fill='{color}' rx='2'/>")
        parts.append(f"<text x='{rx + rw + 31}' y='{ly + 1}' font-size='12' "
                     f"fill='#111827'>{esc(label)}</text>")

    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


# ------------------------------------------------------------------ model --

class Run:
    def __init__(self, row: dict, study_dir: Path):
        self.name: str = row["name"]
        p = row["params"]
        self.concurrency: int = p["concurrency"]
        self.input_tokens: int = p["input_tokens"]
        self.output_tokens: int = p["output_tokens"]
        self.num_requests: int = p["num_requests"]
        self.stream: bool = p.get("stream", True)
        s = row["summary"]
        self.ok, self.total = s["successful"], s["total_requests"]
        self.ttft = s.get("ttft_ms") or {}
        self.itl = s.get("itl_ms") or {}
        self.tpot = s.get("tpot_ms") or {}
        self.e2e = s.get("e2e_ms") or {}
        self.stall = s.get("stream_max_stall_ms") or {}
        self.spread_ms = s.get("stream_completion_spread_ms")
        self.slowfast = s.get("stream_slowest_fastest_ratio")
        self.req_s = s.get("request_throughput", 0.0)
        self.tok_s = s.get("output_tok_s", 0.0)
        self.delivered: int = 0  # ok AND ≥1 token; set with stream_tok_s below
        self.stream_tok_s = self._per_stream_tok_s(study_dir / f"{self.name}.json")

    def _per_stream_tok_s(self, records_path: Path) -> float | None:
        """p50 of per-request output tok/s (completion_tokens / e2e)."""
        try:
            data = json.loads(records_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        vals = []
        delivered = 0
        for r in data.get("records", []):
            # A stream can end "cleanly" with zero tokens when a gateway
            # watchdog aborts a starved prefill — ok alone lies; delivered
            # (ok AND ≥1 token) is the honest count.
            if r.get("ok") and r.get("completion_tokens", 0) > 0:
                delivered += 1
            if r.get("ok") and r.get("completion_tokens", 0) >= 2 and r.get("e2e_ms"):
                vals.append(r["completion_tokens"] / (r["e2e_ms"] / 1000.0))
        self.delivered: int = delivered
        return pctl(vals, 0.5)

    def get(self, metric: str, stat: str = "p50") -> float | None:
        block = {
            "ttft": self.ttft, "itl": self.itl, "tpot": self.tpot,
            "e2e": self.e2e, "stall": self.stall,
        }[metric]
        v = block.get(stat)
        return float(v) if v is not None else None


def fairness_table(runs: list[Run]) -> str:
    """Completion spread (s) and slowest/fastest ratio per cell."""
    prompts = sorted({r.input_tokens for r in runs})
    concs = sorted({r.concurrency for r in runs})
    by = {(r.input_tokens, r.concurrency): r for r in runs}
    lines = [
        "| c\\prompt | " + " | ".join(f"{tick_label(p)} spread s / ×sf" for p in prompts) + " |",
        "| --- | " + " | ".join("---:" for _ in prompts) + " |",
    ]
    for c in concs:
        cells = []
        for p in prompts:
            r = by.get((p, c))
            if r is None:
                cells.append("n/a")
                continue
            sp = f"{r.spread_ms / 1000:.1f}" if r.spread_ms is not None else "n/a"
            sf = f"×{r.slowfast:.2f}" if r.slowfast is not None else "n/a"
            cells.append(f"{sp} / {sf}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table(runs: list[Run], metric: str, stat: str = "p50", unit: str = "ms") -> str:
    prompts = sorted({r.input_tokens for r in runs})
    concs = sorted({r.concurrency for r in runs})
    by = {(r.input_tokens, r.concurrency): r for r in runs}
    head = "| c\\prompt | " + " | ".join(tick_label(p) for p in prompts) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in prompts) + " |"
    lines = [head, sep]
    for c in concs:
        cells = []
        for p in prompts:
            r = by.get((p, c))
            v = r.get(metric, stat) if r else None
            cells.append(fmt_ms(v) if unit == "ms" else fmt_num(v))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------- cross-runtime --

def linfit(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares (intercept, slope) — the TTFT ≈ fixed + rate×prompt fit."""
    n = len(pts)
    if n < 2:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    slope = (n * sxy - sx * sy) / den
    return (sy - slope * sx) / n, slope


def short_health(probe: dict | None) -> str:
    """'/health: 200 /v1/models: 200' from a probe record's health dict."""
    if not probe:
        return "n/a"
    parts = []
    for path, h in probe.get("health", {}).items():
        seg = path.rsplit("/", 1)[-1] or path
        parts.append(f"{seg}: {h.get('status')}")
    mode = probe.get("mode")
    return " ".join(parts) + (f" (mode={mode})" if mode else "")


def compare(study_path: Path, out_dir: Path | None) -> int:
    """Analyze a merged cross-runtime.json (runtime_harness.py merge)."""
    data = json.loads(study_path.read_text())
    study_dir = study_path.parent
    out = out_dir or study_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    legs: dict[str, list[Run]] = {}
    for name, entry in data.get("runtimes", {}).items():
        if "rows" not in entry:
            continue
        leg_dir = Path(entry["matrix_json"]).parent if entry.get("matrix_json") else study_dir / name
        legs[name] = [Run(row, leg_dir) for row in entry["rows"] if not row.get("warmup")]
    if not legs:
        print("no legs with rows in", study_path, file=sys.stderr)
        return 1

    md: list[str] = [
        f"# Cross-runtime comparison — {data.get('study', '?')}",
        "",
        f"- created: `{data.get('created', '?')}` | legs: {', '.join(legs)}",
        "",
        "## Per-run results",
        "",
        "| runtime | prompt | c | ok | delivered | ttft p50 | itl p50 | stall max | spread s | ×sf | tok/s |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, runs in legs.items():
        for r in sorted(runs, key=lambda r: (r.input_tokens, r.concurrency)):
            stall = r.get("stall", "max")
            md.append(
                f"| {name} | {tick_label(r.input_tokens)} | {r.concurrency} "
                f"| {r.ok}/{r.total} | {r.delivered} "
                f"| {fmt_ms(r.get('ttft'))} | {fmt_ms(r.get('itl'))} "
                f"| {fmt_ms(stall)} | {fmt_num(r.spread_ms and r.spread_ms / 1000)} "
                f"| {fmt_num(r.slowfast, 2)} | {fmt_num(r.tok_s)} |"
            )

    # Prefill scaling: TTFT ≈ fixed + rate × prompt, fit at c=1.
    md += ["", "## Prefill scaling (c=1 fit: TTFT ≈ fixed + rate × prompt)", "",
           "| runtime | fixed (s) | rate (ms/tok) | points |", "| --- | ---: | ---: | --- |"]
    fits: dict[str, tuple[float, float] | None] = {}
    for name, runs in legs.items():
        pts = [(float(r.input_tokens), r.get("ttft")) for r in runs
               if r.concurrency == 1 and r.get("ttft") is not None]
        fits[name] = linfit(pts)
        fixed, rate = fits[name] if fits[name] else (None, None)
        md.append(f"| {name} | {fmt_num(fixed and fixed / 1000, 2)} "
                  f"| {fmt_num(rate, 2)} | {len(pts)} |")

    # Concurrency scaling: what c=8 buys and costs vs c=1.
    md += ["", "## Concurrency scaling (c8 vs c1 per prompt)", "",
           "| runtime | prompt | tok/s ×c8/c1 | TTFT ×c8/c1 | stall max c8 |",]
    for name, runs in legs.items():
        by = {(r.input_tokens, r.concurrency): r for r in runs}
        for p in sorted({r.input_tokens for r in runs}):
            r1, r8 = by.get((p, 1)), by.get((p, 8))
            if not r1 or not r8:
                continue
            tps = (r8.tok_s / r1.tok_s) if r1.tok_s else None
            t1, t8 = r1.get("ttft"), r8.get("ttft")
            ttft = (t8 / t1) if (t1 and t8) else None
            md.append(f"| {name} | {tick_label(p)} | {fmt_num(tps, 2)} "
                      f"| {fmt_num(ttft, 2)} | {fmt_ms(r8.get('stall', 'max'))} |")

    # Lifecycle / recovery / memory.
    md += ["", "## Lifecycle, recovery and memory (from runtime_harness.py)", "",
           "| runtime | startup gen | post-benchmark verdict | /health said | restart needed "
           "| peak tree RSS | peak footprint | peak sys used |",
           "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for name, entry in data.get("runtimes", {}).items():
        lc = entry.get("lifecycle")
        if not lc:
            md.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        phases = lc.get("phases", [])
        startup = next((p for p in phases if p["phase"].startswith("wait_functional")), None)
        post = next((p for p in phases if p["phase"] == "post_benchmark"), None)
        # The post-benchmark verdict is the recovery question ("did the
        # worst workload break it?"); phases[-1] would show the
        # post-restart state instead and hide a silent wedge.
        verdict = (post or phases[-1] if phases else {}).get("verdict", "?")
        md.append(
            f"| {name} | {'ok' if startup and startup.get('ok') else 'FAILED'} "
            f"| {verdict} | {short_health(post)} "
            f"| {lc.get('restart_needed', '?')} "
            f"| {fmt_num(lc.get('memory_peak_rss_mb'), 0)} MB "
            f"| {fmt_num(lc.get('memory_peak_footprint_mb'), 0)} MB "
            f"| {fmt_num(lc.get('memory_peak_sys_used_mb'), 0)} MB |"
        )

    # Charts: the money chart is throughput vs concurrency, one per prompt,
    # with a runtime per series; plus TTFT knee, ITL and stall views.
    prompts = sorted({r.input_tokens for runs in legs.values() for r in runs})
    for p in prompts:
        line_chart(out / f"throughput-vs-concurrency-p{p}.svg",
                   f"Aggregate output tok/s vs concurrency — prompt {tick_label(p)}",
                   "concurrency", "output tok/s",
                   [(name, {float(r.concurrency): r.tok_s for r in runs if r.input_tokens == p})
                    for name, runs in legs.items()])
        line_chart(out / f"ttft-vs-concurrency-p{p}.svg",
                   f"TTFT p50 vs concurrency — prompt {tick_label(p)}",
                   "concurrency", "TTFT p50 (ms, log)",
                   [(name, {float(r.concurrency): r.get("ttft") for r in runs if r.input_tokens == p})
                    for name, runs in legs.items()], log_y=True)
        line_chart(out / f"itl-vs-concurrency-p{p}.svg",
                   f"ITL p50 vs concurrency — prompt {tick_label(p)}",
                   "concurrency", "ITL p50 (ms)",
                   [(name, {float(r.concurrency): r.get("itl") for r in runs if r.input_tokens == p})
                    for name, runs in legs.items()])
        line_chart(out / f"max-stall-vs-concurrency-p{p}.svg",
                   f"Worst per-stream stall vs concurrency — prompt {tick_label(p)}",
                   "concurrency", "max stall (ms, log)",
                   [(name, {float(r.concurrency): r.get("stall", "max") for r in runs if r.input_tokens == p})
                    for name, runs in legs.items()], log_y=True)
    line_chart(out / "ttft-vs-prompt-c1.svg",
               "Prefill scaling — TTFT p50 vs prompt at c=1",
               "prompt tokens (log)", "TTFT p50 (ms, log)",
               [(name, {float(r.input_tokens): r.get("ttft") for r in runs if r.concurrency == 1})
                for name, runs in legs.items()], log_x=True, log_y=True)

    (out / "cross-runtime-tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/cross-runtime-tables.md and {len(list(out.glob('*.svg')))} charts")
    print("\n".join(md[:14]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix_json", type=Path, nargs="?",
                    help="matrix.json from `loadgen --matrix`")
    ap.add_argument("--compare", type=Path, default=None,
                    help="merged cross-runtime.json: compare legs instead")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <study dir>/analysis")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare, args.out_dir)
    if args.matrix_json is None:
        ap.error("need a matrix_json path or --compare cross-runtime.json")

    data = json.loads(args.matrix_json.read_text())
    study_dir = args.matrix_json.parent
    out: Path = args.out_dir or study_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs = [Run(row, study_dir) for row in data["rows"] if not row.get("warmup")]
    if not runs:
        print("no measured runs in matrix.json", file=sys.stderr)
        return 1
    meta = data.get("meta", {})
    first_row = next(row for row in data["rows"] if not row.get("warmup"))

    prompts = sorted({r.input_tokens for r in runs})
    concs = sorted({r.concurrency for r in runs})
    by = {(r.input_tokens, r.concurrency): r for r in runs}

    md = [
        f"# Analysis — {meta.get('study', '?')}",
        "",
        f"- created: `{meta.get('created', '?')}`  | manifest: `{meta.get('manifest', '?')}`",
        f"- model: `{first_row['params'].get('model', 'n/a')}` via "
        f"`{first_row['params'].get('endpoint', 'n/a')}`",
        f"- measured runs: {len(runs)} (warmup excluded); failed requests per run are "
        f"counted, not hidden",
        "",
        "## TTFT p50 (ms)",
        "",
        table(runs, "ttft"),
        "",
        "## TTFT p95 (ms)",
        "",
        table(runs, "ttft", "p95"),
        "",
        "## ITL p50 (ms)",
        "",
        table(runs, "itl"),
        "",
        "## E2E p50 (ms)",
        "",
        table(runs, "e2e"),
        "",
        "## Max per-stream stall (ms) — worst gap any single stream sat through",
        "",
        table(runs, "stall"),
        "",
        "## Stream fairness — completion spread and slowest/fastest ratio",
        "",
        fairness_table(runs),
        "",
        "## Throughput",
        "",
        "| c\\prompt | " + " | ".join(
            f"{tick_label(p)} tok/s" for p in prompts) + " |",
        "| --- | " + " | ".join("---:" for _ in prompts) + " |",
    ]
    for c in concs:
        cells = [fmt_num(by[(p, c)].tok_s if (p, c) in by else None) for p in prompts]
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    md += [
        "",
        "## Saturation knee (aggregate output tok/s and TTFT p50, relative to c=1)",
        "",
        "| prompt | c | tok/s | ×c1 | TTFT p50 | ×c1 | stream tok/s p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in prompts:
        base = by.get((p, 1))
        for c in concs:
            r = by.get((p, c))
            if not r:
                continue
            rt, bt = r.get("ttft"), base.get("ttft") if base else None
            t_mult = (rt / bt) if rt is not None and bt else None
            s_mult = (r.tok_s / base.tok_s) if base and base.tok_s else None
            md.append(
                f"| {tick_label(p)} | {c} | {fmt_num(r.tok_s)} | "
                f"{fmt_num(s_mult, 2)} | {fmt_ms(r.get('ttft'))} | "
                f"{fmt_num(t_mult, 2)} | {fmt_num(r.stream_tok_s)} |"
            )

    # Charts -------------------------------------------------------------
    chart_series = []
    for p in prompts:
        chart_series.append((
            f"prompt {tick_label(p)}",
            {float(r.concurrency): r.get("ttft") for r in runs if r.input_tokens == p},
        ))
    line_chart(out / "ttft-vs-concurrency.svg",
               "TTFT p50 vs concurrency", "concurrency", "TTFT p50 (ms, log)",
               chart_series, log_y=True)
    line_chart(out / "throughput-vs-concurrency.svg",
               "Aggregate output tok/s vs concurrency", "concurrency", "output tok/s",
               [(f"prompt {tick_label(p)}",
                 {float(r.concurrency): r.tok_s for r in runs if r.input_tokens == p})
                for p in prompts])
    line_chart(out / "itl-vs-concurrency.svg",
               "ITL p50 vs concurrency", "concurrency", "ITL p50 (ms)",
               [(f"prompt {tick_label(p)}",
                 {float(r.concurrency): r.get("itl") for r in runs if r.input_tokens == p})
                for p in prompts])
    line_chart(out / "ttft-vs-prompt.svg",
               "TTFT p50 vs prompt length", "prompt tokens (log)", "TTFT p50 (ms, log)",
               [(f"c={c}",
                 {float(r.input_tokens): r.get("ttft") for r in runs if r.concurrency == c})
                for c in concs],
               log_x=True, log_y=True)

    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/tables.md and {len(list(out.glob('*.svg')))} charts")
    print("\n".join(md[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
