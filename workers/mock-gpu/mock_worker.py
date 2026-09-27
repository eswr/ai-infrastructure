#!/usr/bin/env python3
"""Mock GPU worker: a fake OpenAI-compatible streaming server.

Serves /v1/chat/completions (stream + non-stream), /v1/models, /health and
/metrics with zero dependencies. Used to exercise the gateway/router and
failure-injection experiments without occupying the real GPU.

Knobs (env):
  HOST                 bind address                   (default 127.0.0.1;
                       containers/k3d set HOST=0.0.0.0)
  PORT                 listen port                    (default 8090)
  MODEL_NAME           reported model name            (default mock-gpu)
  TTFT_MS              simulated time-to-first-token  (default 80)
  TOK_INTERVAL_MS      simulated inter-token latency  (default 25)
  MAX_OUTPUT           default max_tokens cap         (default 64)
  FAIL_AFTER_REQUESTS  start returning 503 after N requests (-1 = never)
  CRASH_STREAM_PCT     % of requests aborted mid-stream (default 0)
  HANG_TTFT_PCT        % of requests with 30s TTFT    (default 0)
  HANG_STREAM_ONLY     when 1, the 30s TTFT hang applies only to stream:true
                       requests — non-streaming requests (like the gateway's
                       functional probe) answer normally (default 0)
  WEDGE_AFTER_REQUESTS after N requests: silent wedge — /health stays 200,
                       every POST hangs forever (the mlx-lm phase-4A
                       signature; -1 = never, default)
  EMPTY_COMPLETION_PCT % of requests answered HTTP 200 with a valid payload
                       but zero completion tokens (the vllm-metal phase-4A
                       signature; default 0)
  EMPTY_AFTER_REQUESTS only empty requests with seq > N (default -1 = the
                       percentage applies from the first request). Lets a
                       worker pass its readiness probe before going bad.
  STALL_AFTER_CHUNKS   streams normally until chunk k, then stops producing
                       while holding the connection open (default -1 = never)
  DRAIN_MAX_S          hard cap on the graceful-drain wait after SIGTERM or
                       POST /drain (default 20; the process exits even with
                       active requests still held — Kubernetes would
                       SIGKILL here)

The last three reproduce the real failure classes the phase-4D worker
lifecycle controller must detect and escalate on; `make lifecycle-check`
drives all of them.

Phase-5 additions (k3d operations):
  GET  /readyz         functional READINESS probe, the conjunction of:
                         (a) a real mini-generation (tokens > 0), same
                             acceptance contract as the gateway's probe —
                             never "/health == 200";
                         (b) the GATEWAY's lifecycle verdict for this pod
                             (READY/DEGRADED -> routable), when GATEWAY_URL
                             is configured — the lifecycle controller stays
                             authoritative even though kubelet probes the
                             pod locally (httpGet cannot target another pod
                             by name, and /health stays 200 through a
                             wedge by design).
                       503 while draining or wedged.
  POST /drain          graceful drain: 503 new chat requests, finish active
                       ones, exit when active hits 0 (or DRAIN_MAX_S).
                       SIGTERM runs the same path (the preStop hook).
  GET  /admin/state    snapshot: draining/wedged flags, active, seq.
  POST /admin/knob     adjust knobs at runtime — {"wedge": true/false,
                       "empty_pct": N, "empty_after": N, "ttft_ms": N,
                       "tok_interval_ms": N, "max_output": N,
                       "fail_after": N}. How the k8s scenarios break and
                       heal a running pod without a restart.
"""

import json
import os
import random
import signal
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8090"))
# Kubernetes sets the container hostname to the pod name; POD_NAME is set
# explicitly in the pod spec. GATEWAY_URL enables the gateway-authoritative
# readiness conjunction (empty = local mode, self-check only).
POD_NAME = os.environ.get("POD_NAME", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "mock-gpu")
TTFT_MS = float(os.environ.get("TTFT_MS", "80"))
TOK_INTERVAL_MS = float(os.environ.get("TOK_INTERVAL_MS", "25"))
MAX_OUTPUT = int(os.environ.get("MAX_OUTPUT", "64"))
FAIL_AFTER = int(os.environ.get("FAIL_AFTER_REQUESTS", "-1"))
CRASH_STREAM_PCT = float(os.environ.get("CRASH_STREAM_PCT", "0"))
HANG_TTFT_PCT = float(os.environ.get("HANG_TTFT_PCT", "0"))
HANG_STREAM_ONLY = os.environ.get("HANG_STREAM_ONLY", "0") == "1"
WEDGE_AFTER = int(os.environ.get("WEDGE_AFTER_REQUESTS", "-1"))
EMPTY_COMPLETION_PCT = float(os.environ.get("EMPTY_COMPLETION_PCT", "0"))
EMPTY_AFTER = int(os.environ.get("EMPTY_AFTER_REQUESTS", "-1"))
STALL_AFTER_CHUNKS = int(os.environ.get("STALL_AFTER_CHUNKS", "-1"))
DRAIN_MAX_S = float(os.environ.get("DRAIN_MAX_S", "20"))

_LOCK = threading.Lock()
_STATS = {"requests": 0, "tokens": 0, "failures": 0, "active": 0,
          "wedged": 0, "empty": 0, "stalled": 0}
# Runtime-adjustable state (POST /admin/knob) — the env knobs seed these.
_KNOBS = {
    "ttft_ms": TTFT_MS,
    "tok_interval_ms": TOK_INTERVAL_MS,
    "max_output": MAX_OUTPUT,
    "fail_after": FAIL_AFTER,
    "empty_pct": EMPTY_COMPLETION_PCT,
    "empty_after": EMPTY_AFTER,
    "wedged": False,
}
# Set once the wedge latch trips (by sequence or by knob): /health keeps
# answering, generation never does. /readyz reports 503 immediately.
_SERVER = {"draining": False}
_SERVER_REF: dict = {}


def _gateway_readiness() -> tuple[bool, str]:
    """Ask the gateway's lifecycle controller whether this pod is routable.
    200 = READY/DEGRADED; anything else (or unreachable) = not routable."""
    if not GATEWAY_URL or not POD_NAME:
        return True, "local-mode"  # no control plane configured: self-check only
    try:
        req = urllib.request.Request(
            f"{GATEWAY_URL}/admin/workers/{POD_NAME}/readiness")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200, f"gateway:{r.status}"
    except urllib.error.HTTPError as e:
        return False, f"gateway:{e.code}"
    except Exception as e:
        return False, f"gateway-unreachable:{type(e).__name__}"


def _forever() -> None:
    """Hold the connection (and this handler thread) without answering."""
    while True:
        time.sleep(3600)


def _emptying_now(seq: int) -> bool:
    pct = float(_KNOBS["empty_pct"])
    if pct <= 0:
        return False
    after = int(_KNOBS["empty_after"])
    return after < 0 or seq > after


def _begin_drain(reason: str) -> None:
    """Graceful drain: refuse new work, finish active, then exit."""
    with _LOCK:
        if _SERVER["draining"]:
            return
        _SERVER["draining"] = True
    print(f"draining ({reason}): refusing new work, finishing "
          f"{_STATS['active']} active", flush=True)
    server = _SERVER_REF.get("server")

    def _exit_when_quiet():
        deadline = time.time() + DRAIN_MAX_S
        while time.time() < deadline:
            if _STATS["active"] <= 0:
                break
            time.sleep(0.05)
        if _STATS["active"] > 0:
            print(f"drain cap {DRAIN_MAX_S}s reached with "
                  f"{_STATS['active']} still active — exiting anyway",
                  flush=True)
        if server is not None:
            # shutdown() must come from a non-serve_forever thread.
            threading.Thread(target=server.shutdown, daemon=True).start()

    threading.Thread(target=_exit_when_quiet, daemon=True).start()


WORDS = (
    "inference scheduler kernel latency throughput batching prefill decode "
    "memory cache router gateway streaming quantization tokens attention "
    "worker queue pressure system buffer engine pipeline server request context"
).split()


def _gen_text(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(max(n, 1)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # ---- helpers -------------------------------------------------------

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_chunk(self, req_id: str, content: str, finish=None) -> bytes:
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": finish}
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    def _usage_chunk(self, req_id: str, completion_tokens: int) -> bytes:
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": completion_tokens,
                "total_tokens": 32 + completion_tokens,
            },
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    def _write_chunked(self, payload: bytes) -> None:
        """HTTP/1.1 chunked-transfer framing for streaming bodies."""
        self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    # ---- routes --------------------------------------------------------

    def _consume_body(self) -> None:
        """Read+discard an unread request body: responding 503/404 over a
        body still in the socket buffer gets the connection RST before the
        client ever sees the status."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)
        except (ValueError, OSError):
            pass

    def do_GET(self):
        if self.path in ("/drain", "/admin/drain"):
            # GET form exists for the Kubernetes preStop httpGet hook.
            _begin_drain(f"GET {self.path}")
            self._json(200, {"draining": True, "active": _STATS["active"]})
            return
        if self.path == "/health":
            # Liveness: process is up. Never a readiness signal — the
            # silent wedge answers this for hours.
            self._json(200, {"status": "ok", "model": MODEL_NAME,
                             "draining": _SERVER["draining"]})
        elif self.path == "/readyz":
            # Readiness = a REAL mini-generation (the gateway probe's
            # acceptance contract: tokens > 0) AND, when configured, the
            # gateway lifecycle verdict. Fast-failing on the latched wedge.
            if _SERVER["draining"]:
                self._json(503, {"ready": False, "reason": "draining"})
                return
            if _KNOBS["wedged"]:
                self._json(503, {"ready": False, "reason": "wedged"})
                return
            seq = _STATS["requests"]
            if _emptying_now(seq):
                self._json(503, {"ready": False, "reason": "empty_completion"})
                return
            ok, via = _gateway_readiness()
            if not ok:
                self._json(503, {"ready": False, "reason": via,
                                 "pod": POD_NAME or None})
                return
            time.sleep(min(float(_KNOBS["ttft_ms"]), 300) / 1000.0)
            self._json(200, {"ready": True, "model": MODEL_NAME,
                             "probe_tokens": 2, "via": via})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        elif self.path == "/metrics":
            with _LOCK:
                s = dict(_STATS)
            body = (
                "# TYPE mock_requests_total counter\n"
                f"mock_requests_total {s['requests']}\n"
                "# TYPE mock_tokens_total counter\n"
                f"mock_tokens_total {s['tokens']}\n"
                "# TYPE mock_failures_total counter\n"
                f"mock_failures_total {s['failures']}\n"
                "# TYPE mock_active_requests gauge\n"
                f"mock_active_requests {s['active']}\n"
                "# TYPE mock_wedged_total counter\n"
                f"mock_wedged_total {s['wedged']}\n"
                "# TYPE mock_empty_completions_total counter\n"
                f"mock_empty_completions_total {s['empty']}\n"
                "# TYPE mock_stalled_total counter\n"
                f"mock_stalled_total {s['stalled']}\n"
                "# TYPE mock_draining gauge\n"
                f"mock_draining {1 if _SERVER['draining'] else 0}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/scheduler_state":
            # Phase-7 adoption: the runtime scheduler-state interface
            # contract (see docs/architecture-mapping.md, vLLM session).
            # A production runtime (vLLM) reports num_requests_running,
            # num_requests_waiting, KV utilization and preemption counts
            # here — the ground truth a control plane can only estimate.
            # The mock is honest about what it cannot know: its queue is
            # unbounded (no waiting state) and it has no KV cache.
            with _LOCK:
                s = dict(_STATS)
            self._json(200, {
                "running": s["active"],
                "waiting": 0,
                "queue_mode": "unbounded",
                "kv_cache_usage": None,
                "num_preemptions": 0,
                "requests_total": s["requests"],
                "model": MODEL_NAME,
            })
        elif self.path == "/admin/state":
            with _LOCK:
                s = dict(_STATS)
            self._json(200, {
                "model": MODEL_NAME,
                "draining": _SERVER["draining"],
                "wedged": bool(_KNOBS["wedged"]),
                "knobs": {k: v for k, v in _KNOBS.items()},
                "active": s["active"],
                "requests": s["requests"],
                "tokens": s["tokens"],
            })
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path in ("/drain", "/admin/drain"):
            self._consume_body()
            _begin_drain(f"POST {self.path}")
            self._json(200, {"draining": True, "active": _STATS["active"]})
            return
        if self.path == "/admin/knob":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                patch = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "bad json"}})
                return
            applied = {}
            if "wedge" in patch:
                _KNOBS["wedged"] = bool(patch["wedge"])
                applied["wedged"] = _KNOBS["wedged"]
                if _KNOBS["wedged"]:
                    with _LOCK:
                        _STATS["wedged"] += 1
            for key, knob in (("empty_pct", "empty_pct"),
                              ("empty_after", "empty_after"),
                              ("ttft_ms", "ttft_ms"),
                              ("tok_interval_ms", "tok_interval_ms"),
                              ("max_output", "max_output"),
                              ("fail_after", "fail_after")):
                if key in patch:
                    _KNOBS[knob] = patch[key]
                    applied[knob] = patch[key]
            self._json(200, {"applied": applied})
            return
        if self.path != "/v1/chat/completions":
            self._consume_body()
            self._json(404, {"error": {"message": "not found"}})
            return
        if _SERVER["draining"]:
            # Draining: in-flight streams finish; new work is refused.
            self._consume_body()
            self.close_connection = True
            self._json(503, {"error": {"message": "draining",
                                       "type": "mock_draining"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "bad json"}})
            return

        with _LOCK:
            _STATS["requests"] += 1
            seq = _STATS["requests"]
            _STATS["active"] += 1

        try:
            if _KNOBS["fail_after"] >= 0 and seq > _KNOBS["fail_after"]:
                with _LOCK:
                    _STATS["failures"] += 1
                self._json(503, {"error": {"message": "injected failure", "type": "mock"}})
                return

            if _KNOBS["wedged"] or (WEDGE_AFTER >= 0 and seq > WEDGE_AFTER):
                # Silent wedge: keep answering /health, never answer POST.
                if not _KNOBS["wedged"]:
                    _KNOBS["wedged"] = True  # latch: /readyz must fail fast
                with _LOCK:
                    _STATS["wedged"] += 1
                _forever()
                return

            n = min(int(req.get("max_tokens") or _KNOBS["max_output"]),
                    int(_KNOBS["max_output"]))
            req_id = f"chatcmpl-mock-{seq}"
            rng = random.Random(seq)
            hang = rng.random() * 100 < HANG_TTFT_PCT
            if HANG_STREAM_ONLY and not req.get("stream"):
                hang = False
            ttft = 30_000.0 if hang else float(_KNOBS["ttft_ms"])
            empty = rng.random() * 100 < float(_KNOBS["empty_pct"])
            if _KNOBS["empty_after"] >= 0 and seq <= _KNOBS["empty_after"]:
                empty = False
            if empty:
                with _LOCK:
                    _STATS["empty"] += 1

            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                time.sleep(ttft / 1000.0)
                if not empty:
                    self._write_chunked(self._sse_chunk(req_id, _gen_text(1, seq)))
                crash_at = (
                    rng.randrange(1, max(n - 1, 2))
                    if rng.random() * 100 < CRASH_STREAM_PCT and n > 2
                    else None
                )
                for i in range(1, n):
                    time.sleep(float(_KNOBS["tok_interval_ms"]) / 1000.0)
                    if STALL_AFTER_CHUNKS > 0 and i >= STALL_AFTER_CHUNKS:
                        # Mid-stream stall: stop producing, keep holding the
                        # connection open — the client sees a frozen stream.
                        with _LOCK:
                            _STATS["stalled"] += 1
                        _forever()
                    if crash_at is not None and i >= crash_at:
                        # Abort mid-stream: just close the connection.
                        self.close_connection = True
                        return
                    self._write_chunked(self._sse_chunk(req_id, _gen_text(1, seq + i)))
                if empty:
                    # A usage-only chunk with completion_tokens = 0: the
                    # vllm-metal silent-empty-answer shape.
                    self._write_chunked(self._usage_chunk(req_id, 0))
                self._write_chunked(self._sse_chunk(req_id, "", finish="stop"))
                self._write_chunked(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")  # terminal chunk
                self.wfile.flush()
                with _LOCK:
                    _STATS["tokens"] += 0 if empty else n
            else:
                time.sleep(ttft / 1000.0 + (n - 1) * float(_KNOBS["tok_interval_ms"]) / 1000.0)
                text = "" if empty else _gen_text(n, seq)
                self._json(
                    200,
                    {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": 0 if empty else n,
                            "total_tokens": 32 + (0 if empty else n),
                        },
                    },
                )
                with _LOCK:
                    _STATS["tokens"] += 0 if empty else n
        finally:
            with _LOCK:
                _STATS["active"] -= 1


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    _SERVER_REF["server"] = server

    def _on_signal(signum, _frame):
        _begin_drain(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    print(
        f"mock-gpu worker on {HOST}:{PORT}  model={MODEL_NAME}  ttft={TTFT_MS}ms  "
        f"itl={TOK_INTERVAL_MS}ms  fail_after={FAIL_AFTER}  crash%={CRASH_STREAM_PCT}  "
        f"hang%={HANG_TTFT_PCT}  wedge_after={WEDGE_AFTER}  "
        f"empty%={EMPTY_COMPLETION_PCT}  stall_after_chunks={STALL_AFTER_CHUNKS}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        print("mock-gpu worker stopped", flush=True)


if __name__ == "__main__":
    main()
