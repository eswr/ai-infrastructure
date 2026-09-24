#!/usr/bin/env python3
"""Experiment provenance manifests — the environment a result was produced in.

Every experiment writes one `manifest.json` next to its raw artifacts so a
result can be independently tied to the exact code, machine, model, and
configuration that produced it. Stdlib-only (system python, like analyze.py).

Usage (from the repo root, called by experiment scripts):

    manifest.py write   --out DIR [--fragment experiment.json] [--experiment NAME]
    manifest.py finalize --out DIR

`write` snapshots the environment (git, OS, machine, python, mlx, model
quantization, toolchain) at experiment start and merges an optional JSON
fragment with experiment-specific args. `finalize` (idempotent, safe in a
trap) stamps `ended_at` and attaches the `summary` block of every loadgen
run JSON found in DIR.
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

RUNE = "benchmarks/results"  # informational only; paths come from --out


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sh(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_facts() -> dict:
    def g(*args):
        return sh(["git", *args])
    dirty = bool(g("status", "--porcelain"))
    return {
        "commit": g("rev-parse", "HEAD"),
        "abbrev": g("rev-parse", "--short", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": g("describe", "--tags", "--always"),
        "dirty": dirty,
    }


def machine_facts() -> dict:
    def s(name):
        return sh(["sysctl", "-n", name])
    mem = s("hw.memsize")
    return {
        "model_id": s("hw.model"),
        "chip": s("machdep.cpu.brand_string"),
        "ram_bytes": int(mem) if mem and mem.isdigit() else None,
        "os_version": sh(["sw_vers", "-productVersion"]),
        "os_build": sh(["sw_vers", "-buildVersion"]),
        "uname": platform.platform(),
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def model_facts(model: str | None) -> dict:
    facts = {"name": model, "quantization": None, "model_type": None}
    if not model:
        return facts
    import os
    hub = os.path.expanduser(
        os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
    repo = "models--" + model.replace("/", "--")
    for cfg in sorted(Path(hub, repo).glob("snapshots/*/config.json")):
        try:
            conf = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        facts["quantization"] = conf.get("quantization")
        facts["model_type"] = conf.get("model_type")
        break
    return facts


def env_snapshot(fragment: dict | None, experiment: str | None) -> dict:
    manifest = {
        "schema": "inference-lab/manifest@1",
        "experiment": experiment,
        "started_at": utcnow(),
        "git": git_facts(),
        "machine": machine_facts(),
        "toolchain": {
            "python": platform.python_version(),
            "rustc": sh(["rustc", "--version"]),
            "cargo": sh(["cargo", "--version"]),
        },
        "packages": {
            "mlx": package_version("mlx"),
            "mlx-lm": package_version("mlx-lm"),
        },
        "model": model_facts((fragment or {}).pop("model", None)),
    }
    if fragment:
        manifest.update(fragment)
    return manifest


def run_summaries(out_dir: Path) -> dict:
    """Pull the `summary` block out of every loadgen run JSON in DIR."""
    runs = {}
    for path in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "summary" in data:
            runs[path.stem] = data["summary"]
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="snapshot environment at experiment start")
    w.add_argument("--out", required=True, help="experiment artifacts dir")
    w.add_argument("--experiment", default=None)
    w.add_argument("--fragment", default=None,
                   help="JSON file with experiment-specific args to merge")

    f = sub.add_parser("finalize", help="stamp end time + run summaries")
    f.add_argument("--out", required=True)

    args = ap.parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_dir():
        print(f"manifest: no such directory {out_dir}", file=sys.stderr)
        return 1

    if args.cmd == "write":
        fragment = None
        if args.fragment:
            try:
                fragment = json.loads(Path(args.fragment).read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"manifest: bad fragment {args.fragment}: {e}", file=sys.stderr)
                return 1
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(env_snapshot(fragment, args.experiment),
                                   indent=2) + "\n")
        print(f"manifest: wrote {path}")
        return 0

    path = out_dir / "manifest.json"
    if not path.exists():
        print(f"manifest: {path} missing — run `write` first", file=sys.stderr)
        return 1
    manifest = json.loads(path.read_text())
    manifest["ended_at"] = utcnow()
    manifest["runs"] = run_summaries(out_dir)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: finalized {path} ({len(manifest['runs'])} run summaries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
