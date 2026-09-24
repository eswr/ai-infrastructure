#!/usr/bin/env python3
"""runtime_harness.py — lifecycle-aware benchmarking of native LLM runtimes.

Phase 3 taught us that process health alone cannot be trusted: a wedged
mlx_lm.server kept answering /health while never serving another POST.
Every runtime in a benchmark therefore walks the same lifecycle:

    start() -> wait_live() -> wait_functional() -> benchmark() ->
    functional_probe() -> [restart if needed] -> stop()

The `Runtime` base class fixes that contract; one adapter per runtime
(mlx-lm, llama-cpp, vllm-metal) supplies commands and endpoint quirks.
The harness drives a runtime through the contract, runs the loadgen
matrix for it behind the same gateway the other studies used, samples
process + system memory the whole time, and writes a lifecycle record
that answers, per runtime: what did /health say, did a real generation
actually work, and was a restart needed after the worst workload?

Usage:
    python3 benchmarks/runtime_harness.py run   --runtime mlx-lm \
        --matrix experiments/cross-runtime/mlx-lm.toml \
        --out benchmarks/results/cross-runtime-<ts> [--gateway PATH]
    python3 benchmarks/runtime_harness.py merge --out benchmarks/results/cross-runtime-<ts>

Stdlib only, like the rest of benchmarks/*.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Generous because model load + first generation can be slow; small enough
# that a truly dead runtime doesn't stall the study for many minutes.
LIVE_TIMEOUT_S = 360.0
FUNCTIONAL_TIMEOUT_S = 360.0
PROBE_TIMEOUT_S = 120.0


def log(msg: str) -> None:
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(url: str, timeout_s: float = 5.0) -> tuple[int | None, str]:
    """(status, body) or (None, error) — never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return r.status, r.read(2048).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """One observation of runtime state: health + a real generation."""

    phase: str
    ts: float = field(default_factory=time.time)
    health: dict = field(default_factory=dict)  # path -> {"status": int|None, "body": str}
    # Functional probe results:
    ok: bool = False
    error: str | None = None
    first_event_ms: float | None = None  # request sent -> first SSE data event
    tokens: int = 0
    total_ms: float | None = None

    def to_json(self) -> dict:
        return {
            "phase": self.phase,
            "ts": self.ts,
            "health": self.health,
            "ok": self.ok,
            "error": self.error,
            "first_event_ms": self.first_event_ms,
            "tokens": self.tokens,
            "total_ms": self.total_ms,
        }


def classify(probe: Probe, process_alive: bool) -> str:
    """The four recovery states that matter after a worst-case workload."""
    if probe.ok:
        return "healthy"
    if not process_alive:
        return "dead"
    health_ok = any(h.get("status") == 200 for h in probe.health.values())
    # The phase-3 signature: /health lies, POST never returns a token.
    return "silent_wedge" if health_ok else "loud_failure"


# ---------------------------------------------------------------------------
# Runtime contract + adapters
# ---------------------------------------------------------------------------


class Runtime(ABC):
    """Lifecycle contract every benchmarked runtime must satisfy."""

    def __init__(self, name: str, port: int, model: str):
        self.name = name
        self.port = port
        self.model = model
        self.proc: subprocess.Popen | None = None
        self.log_fh = None

    # -- adapter-supplied --------------------------------------------------

    @abstractmethod
    def start_argv(self) -> list[str]:
        """Command to exec (a single process; child trees are handled)."""

    @abstractmethod
    def start_env(self) -> dict:
        """Extra environment for the start command."""

    @abstractmethod
    def live_url(self) -> str:
        """URL that returns 200 once the server is *up* (not yet truthful)."""

    @abstractmethod
    def health_urls(self) -> list[str]:
        """Endpoints to interrogate and record, in order.

        All of them get recorded verbatim — the point is to capture what
        each claims, including when they disagree with the POST probe.
        """

    def metrics_url(self) -> str | None:
        """Runtime-reported cache/state metrics, if the runtime exposes any."""
        return None

    # -- shared lifecycle --------------------------------------------------

    def start(self, log_path: Path) -> None:
        self.log_fh = log_path.open("ab")
        env = {**os.environ, **self.start_env()}
        self.proc = subprocess.Popen(
            self.start_argv(),
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO,
            start_new_session=True,  # own process group: clean tree teardown
        )
        log(f"{self.name}: started pid {self.proc.pid}")

    def wait_live(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status, _ = http_get(self.live_url())
            if status == 200:
                return True
            if self.proc and self.proc.poll() is not None:
                return False
            time.sleep(1.0)
        return False

    def functional_probe(self, phase: str, max_tokens: int = 4) -> Probe:
        """Tiny real generation over SSE — the only health signal that
        can't lie. Any streamed data event counts as a token; a reasoning
        delta is a token (Qwen3.5 thinks before it speaks)."""
        p = Probe(phase=phase)
        for path in self.health_urls():
            status, body = http_get(path)
            p.health[path] = {"status": status, "body": body[:200]}
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": "Readiness probe. Reply with the single word: ready"}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
                if resp.status != 200:
                    p.error = f"http {resp.status}"
                    p.total_ms = (time.time() - t0) * 1000
                    return p
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    if p.first_event_ms is None:
                        p.first_event_ms = (time.time() - t0) * 1000
                    p.tokens += 1
            p.ok = p.tokens > 0
            if not p.ok:
                p.error = "stream ended without any data event"
        except Exception as e:
            p.error = f"{type(e).__name__}: {e}"
        p.total_ms = (time.time() - t0) * 1000
        return p

    def wait_functional(self, timeout_s: float) -> Probe:
        deadline = time.time() + timeout_s
        attempt = 0
        while True:
            attempt += 1
            p = self.functional_probe(phase=f"wait_functional#{attempt}")
            if p.ok:
                return p
            if self.proc and self.proc.poll() is not None:
                p.error = p.error or "runtime process exited"
                return p
            if time.time() > deadline:
                p.error = p.error or f"not functional after {timeout_s}s"
                return p
            time.sleep(3.0)

    def metrics_snapshot(self) -> dict:
        url = self.metrics_url()
        if not url:
            return {"available": False}
        status, body = http_get(url)
        if status != 200:
            return {"available": False, "status": status}
        keep = [
            ln for ln in body.splitlines()
            if re.search(r"kv|cache|prompt|generation|inference|request", ln, re.I)
            and not ln.startswith("#")
        ]
        return {"available": True, "excerpt": keep[:40]}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log(f"{self.name}: SIGTERM ignored, sending SIGKILL")
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        self.proc = None
        if self.log_fh:
            self.log_fh.close()
            self.log_fh = None
        log(f"{self.name}: stopped")

    def info(self) -> dict:
        return {"name": self.name, "port": self.port, "model": self.model}


class MlxLmRuntime(Runtime):
    def __init__(self, model: str = "mlx-community/Qwen3.5-9B-4bit", port: int = 8082):
        super().__init__("mlx-lm", port, model)

    def start_argv(self) -> list[str]:
        return ["bash", "workers/mlx-lm/run.sh"]

    def start_env(self) -> dict:
        return {"MODEL": self.model, "PORT": str(self.port)}

    def live_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/models"

    def health_urls(self) -> list[str]:
        return [
            f"http://127.0.0.1:{self.port}/health",
            f"http://127.0.0.1:{self.port}/v1/models",
        ]


class LlamaCppRuntime(Runtime):
    def __init__(self, gguf: str | None = None, port: int = 8085, ctx: int = 73_728, parallel: int = 8):
        self.gguf = gguf or find_qwen35_gguf()
        if not self.gguf:
            raise SystemExit("no Qwen3.5-9B GGUF in the HF cache; run: hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf")
        super().__init__("llama-cpp", port, Path(self.gguf).name)
        self.ctx = ctx
        self.parallel = parallel

    def start_argv(self) -> list[str]:
        return ["bash", "workers/llama-cpp/run.sh"]

    def start_env(self) -> dict:
        return {
            "GGUF": self.gguf,
            "PORT": str(self.port),
            "CTX": str(self.ctx),
            "PARALLEL": str(self.parallel),
        }

    def live_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def health_urls(self) -> list[str]:
        return [
            f"http://127.0.0.1:{self.port}/health",
            f"http://127.0.0.1:{self.port}/v1/models",
        ]

    def metrics_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}/metrics"

    def info(self) -> dict:
        return {**super().info(), "gguf": self.gguf, "ctx": self.ctx, "parallel_slots": self.parallel}


class VllmMetalRuntime(Runtime):
    def __init__(self, model: str = "mlx-community/Qwen3.5-9B-4bit", port: int = 8083):
        super().__init__("vllm-metal", port, model)

    def start_argv(self) -> list[str]:
        return ["bash", "workers/vllm-metal/run.sh"]

    def start_env(self) -> dict:
        return {"MODEL": self.model, "PORT": str(self.port)}

    def live_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def health_urls(self) -> list[str]:
        return [
            f"http://127.0.0.1:{self.port}/health",
            f"http://127.0.0.1:{self.port}/v1/models",
        ]

    def metrics_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}/metrics"


def find_qwen35_gguf() -> str | None:
    """Newest Qwen3.5-9B Q4_K_M GGUF in the HF cache (any repo name)."""
    root = Path.home() / ".cache/huggingface/hub"
    hits = sorted(root.glob("models--*Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf"))
    return str(hits[-1]) if hits else None


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------


class MemorySampler(threading.Thread):
    """Samples the runtime process tree and system memory ~1 Hz into a CSV.

    Two process numbers: `ps` RSS (fast, underestimates — Metal GPU
    buffers are wired, not resident) and every 5th sample `vmmap`
    physical footprint (slow but closer to Activity Monitor's number).
    System-wide: vm_stat page classes -> used ≈ active + wired +
    compressor, plus the kernel pressure level (1 normal / 2 warning /
    3 critical).
    """

    PHASES = ("loading", "after_load", "benchmark", "recovery", "post_restart")

    def __init__(self, pid_fn, csv_path: Path):
        super().__init__(daemon=True)
        self.pid_fn = pid_fn  # callable -> root pid (or None)
        self.phase = "loading"
        self._stop = threading.Event()
        self._fh = csv_path.open("w")
        self._fh.write("ts,phase,rss_mb,footprint_mb,sys_used_mb,sys_free_mb,pressure_level\n")
        self.peak_rss_mb = 0.0
        self.peak_footprint_mb = 0.0
        self.peak_sys_used_mb = 0.0
        self._i = 0

    def run(self) -> None:
        page = 16384  # re-parsed from vm_stat anyway
        while not self._stop.is_set():
            ts = time.time()
            rss_mb = self._tree_rss_mb()
            fp_mb = ""
            if self._i % 5 == 0 and rss_mb:
                fp_mb = self._footprint_mb()
            sys_free, sys_used, page = self._system_mb()
            level = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True, text=True,
            ).stdout.strip()
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb or 0)
            self.peak_footprint_mb = max(self.peak_footprint_mb, self._to_f(fp_mb))
            self.peak_sys_used_mb = max(self.peak_sys_used_mb, sys_used or 0)
            self._fh.write(
                f"{ts:.1f},{self.phase},{rss_mb or ''},{fp_mb},"
                f"{sys_used or ''},{sys_free or ''},{level}\n"
            )
            self._fh.flush()
            self._i += 1
            self._stop.wait(1.0)

    @staticmethod
    def _to_f(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _tree_rss_mb(self) -> float | None:
        root = self.pid_fn()
        if root is None:
            return None
        try:
            out = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10
            ).stdout
        except subprocess.SubprocessError:
            return None
        parents: dict[int, int] = {}
        rss: dict[int, float] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            pid, ppid, r = (int(x) for x in parts)
            parents[pid] = ppid
            rss[pid] = r
        total, stack = 0.0, [root]
        seen: set[int] = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            total += rss.get(pid, 0)
            stack.extend(p for p, pp in parents.items() if pp == pid)
        return total / 1024.0 if total else None

    def _footprint_mb(self) -> str:
        """Physical footprint via vmmap — slower than ps, but counts the
        wired Metal buffers that RSS underreports. Empty string on failure."""
        root = self.pid_fn()
        if root is None:
            return ""
        try:
            out = subprocess.run(
                ["vmmap", "--summary", str(root)],
                capture_output=True, text=True, timeout=25,
            ).stdout
        except subprocess.SubprocessError:
            return ""
        m = re.search(r"Physical footprint:\s*([0-9.]+)([KMG])", out)
        if not m:
            return ""
        val, unit = float(m.group(1)), m.group(2)
        mult = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[unit]
        return f"{val * mult:.1f}"

    def _system_mb(self) -> tuple[float | None, float | None, int]:
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        except subprocess.SubprocessError:
            return None, None, 16384
        m = re.search(r"page size of (\d+) bytes", out)
        page = int(m.group(1)) if m else 16384
        pages: dict[str, float] = {}
        for line in out.splitlines():
            mm = re.match(r"Pages ([a-z ]+?):\s+(\d+)\.", line)
            if mm:
                pages[mm.group(1).strip()] = int(mm.group(2))
        free = pages.get("free", 0) + pages.get("speculative", 0)
        used = pages.get("active", 0) + pages.get("wired down", 0) + pages.get("compressed", 0)
        mb = page / (1024.0 * 1024.0)
        return free * mb, used * mb, page

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)
        self._fh.close()


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------


def make_runtime(name: str) -> Runtime:
    if name == "mlx-lm":
        return MlxLmRuntime()
    if name == "llama-cpp":
        return LlamaCppRuntime()
    if name == "vllm-metal":
        return VllmMetalRuntime()
    raise SystemExit(f"unknown runtime {name!r} (mlx-lm | llama-cpp | vllm-metal)")


def runtime_versions(name: str) -> dict:
    def py(code: str) -> str:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
        return r.stdout.strip() or r.stderr.strip()[:200]

    if name == "mlx-lm":
        return {"mlx_lm": py("import mlx_lm;print(mlx_lm.__version__)")}
    if name == "llama-cpp":
        r = subprocess.run(["llama-server", "--version"], capture_output=True, text=True)
        return {"llama_server": (r.stdout or r.stderr).strip().splitlines()[0]}
    if name == "vllm-metal":
        r = subprocess.run(
            ["uv", "run", "python", "-c",
             "import importlib.metadata as m;print(m.version('vllm-metal'))"],
            cwd=REPO / "workers/vllm-metal", capture_output=True, text=True,
        )
        return {"vllm_metal": r.stdout.strip() or r.stderr.strip()[:200]}
    return {}


def kill_leftovers(name: str) -> None:
    """Nothing else may own the port: kill strays of the same runtime.
    Patterns are server-invocation-specific — "vllm-metal" alone would
    match this harness's own argv (--runtime vllm-metal) and kill the study."""
    pattern = {
        "mlx-lm": "mlx_lm.server",
        "llama-cpp": "llama-server",
        "vllm-metal": "vllm-metal --model",
    }[name]
    subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(2)


def run_study(rt: Runtime, matrix: Path, out_dir: Path, gateway_bin: Path) -> int:
    rt_dir = out_dir / rt.name
    rt_dir.mkdir(parents=True, exist_ok=True)
    sampler = MemorySampler(lambda: rt.proc.pid if rt.proc else None, rt_dir / "memory.csv")
    sampler.start()
    lifecycle: dict = {"runtime": rt.name, "gateway_first_token_timeout_s": 900, "phases": []}

    def record(probe: Probe, extra: dict | None = None) -> None:
        entry = probe.to_json() | {"verdict": classify(probe, rt.alive())} | (extra or {})
        lifecycle["phases"].append(entry)
        log(f"{rt.name} [{probe.phase}]: {entry['verdict']} "
            f"(health={ {k: v['status'] for k, v in probe.health.items()} }, "
            f"tokens={probe.tokens}, error={probe.error})")

    try:
        # -- start -> live -> functional ------------------------------------
        kill_leftovers(rt.name)
        rt.start(rt_dir / "runtime.log")
        if not rt.wait_live(LIVE_TIMEOUT_S):
            p = Probe(phase="startup", error=f"live endpoint never returned 200 in {LIVE_TIMEOUT_S}s")
            record(p)
            lifecycle["verdict"] = "startup_failed"
            return 1
        record(rt.wait_functional(FUNCTIONAL_TIMEOUT_S))

        # model resident; capture the runtime's own cache/KV view
        sampler.phase = "after_load"
        time.sleep(3)
        rt_info = rt.info() | runtime_versions(rt.name) | {"metrics": rt.metrics_snapshot()}
        (rt_dir / "runtime_info.json").write_text(json.dumps(rt_info, indent=2))

        # -- benchmark behind the same gateway ------------------------------
        sampler.phase = "benchmark"
        gw_env = {
            **os.environ,
            "GATEWAY_ADDR": "127.0.0.1:8080",
            "WORKERS": f"{rt.name}=http://127.0.0.1:{rt.port}#{rt.model}",
            "ROUTE_POLICY": "lor",
            "FIRST_TOKEN_TIMEOUT_S": "900",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
        }
        with (rt_dir / "gateway.log").open("wb") as gw_log:
            gw = subprocess.Popen(
                [str(gateway_bin)], stdout=gw_log, stderr=subprocess.STDOUT,
                env=gw_env, cwd=REPO, start_new_session=True,
            )
            try:
                for _ in range(50):
                    if http_get("http://127.0.0.1:8080/healthz")[0] == 200:
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError("gateway never became ready")
                cmd = [str(gateway_bin.parent / "loadgen"), "--matrix", str(matrix), "--out-dir", str(rt_dir)]
                log(f"{rt.name}: loadgen {' '.join(str(c) for c in cmd[1:])}")
                rc = subprocess.run(cmd, cwd=REPO).returncode
                lifecycle["loadgen_exit"] = rc
            finally:
                # Give in-flight streams a moment to unwind, then take the
                # recovery probe while the gateway is still up.
                time.sleep(2)
                sampler.phase = "recovery"
                record(rt.functional_probe(phase="post_benchmark"), extra={"gateway_metrics": gateway_fairness()})
                os.killpg(gw.pid, signal.SIGTERM)
                gw.wait(timeout=15)

        # -- restart if the worst workload degraded the runtime --------------
        final = lifecycle["phases"][-1]
        if final["verdict"] == "healthy":
            lifecycle["restart_needed"] = False
        else:
            lifecycle["restart_needed"] = True
            log(f"{rt.name}: degraded after benchmark — restarting")
            rt.stop()
            kill_leftovers(rt.name)
            sampler.phase = "post_restart"
            rt.start(rt_dir / "runtime-restart.log")
            if rt.wait_live(LIVE_TIMEOUT_S):
                record(rt.wait_functional(FUNCTIONAL_TIMEOUT_S))
            else:
                record(Probe(phase="post_restart", error="restarted runtime never went live"))
        lifecycle["verdict"] = lifecycle["phases"][-1]["verdict"]
        return 0
    finally:
        rt.stop()
        sampler.stop()
        lifecycle["memory_peak_rss_mb"] = round(sampler.peak_rss_mb, 1)
        lifecycle["memory_peak_footprint_mb"] = round(sampler.peak_footprint_mb, 1)
        lifecycle["memory_peak_sys_used_mb"] = round(sampler.peak_sys_used_mb, 1)
        (rt_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2))
        log(f"{rt.name}: lifecycle written, peak tree RSS {sampler.peak_rss_mb:.0f} MB")


def gateway_fairness() -> dict:
    """Fairness counters scraped from the just-finished benchmark."""
    status, body = http_get("http://127.0.0.1:8080/metrics")
    if status != 200:
        return {}
    want = (
        "inference_stream_inter_token_seconds",
        "inference_stream_max_stall_seconds",
        "inference_stream_completion_seconds",
        "inference_ttft_seconds",
    )
    out: dict = {}
    for ln in body.splitlines():
        for w in want:
            if ln.startswith(w + "_count"):
                try:
                    out[w] = {"count": float(ln.rsplit(" ", 1)[1])}
                except ValueError:
                    continue
    return out


def find_matrix_json(rt_dir: Path) -> Path | None:
    hits = sorted(rt_dir.glob("*/matrix.json"))
    return hits[-1] if hits else None


def merge(out_dir: Path) -> int:
    study = {"study": "cross-runtime-apple-silicon", "created": time.strftime("%Y%m%d-%H%M%S"),
             "runtimes": {}}
    for name in ("mlx-lm", "llama-cpp", "vllm-metal"):
        rt_dir = out_dir / name
        if not rt_dir.exists():
            log(f"merge: {rt_dir} missing, skipping {name}")
            continue
        mj = find_matrix_json(rt_dir)
        entry: dict = {"dir": str(rt_dir)}
        if mj:
            data = json.loads(mj.read_text())
            entry |= {"matrix_json": str(mj), "meta": data.get("meta"), "rows": data.get("rows")}
        lp = rt_dir / "lifecycle.json"
        if lp.exists():
            entry["lifecycle"] = json.loads(lp.read_text())
        ip = rt_dir / "runtime_info.json"
        if ip.exists():
            entry["runtime_info"] = json.loads(ip.read_text())
        study["runtimes"][name] = entry
    path = out_dir / "cross-runtime.json"
    path.write_text(json.dumps(study, indent=2))
    log(f"merged → {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="walk one runtime through the full lifecycle")
    run.add_argument("--runtime", required=True, choices=["mlx-lm", "llama-cpp", "vllm-metal"])
    run.add_argument("--matrix", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--gateway", type=Path, default=REPO / "target/release/gateway")

    m = sub.add_parser("merge", help="combine per-runtime artifacts into cross-runtime.json")
    m.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "run":
        return run_study(make_runtime(args.runtime), args.matrix, args.out, args.gateway)
    return merge(args.out)


if __name__ == "__main__":
    sys.exit(main())
