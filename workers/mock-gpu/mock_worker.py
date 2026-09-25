#!/usr/bin/env python3
"""Mock GPU worker: a fake OpenAI-compatible streaming server.

Serves /v1/chat/completions (stream + non-stream), /v1/models, /health and
/metrics with zero dependencies. Used to exercise the gateway/router and
failure-injection experiments without occupying the real GPU.

Knobs (env):
  PORT                 listen port                     (default 8090)
  MODEL_NAME           reported model name             (default mock-gpu)
  TTFT_MS              simulated time-to-first-token    (default 80)
  TOK_INTERVAL_MS      simulated inter-token latency   (default 25)
  MAX_OUTPUT           default max_tokens cap          (default 64)
  FAIL_AFTER_REQUESTS  start returning 503 after N requests (-1 = never)
  CRASH_STREAM_PCT     % of requests aborted mid-stream (default 0)
  HANG_TTFT_PCT        % of requests with 30s TTFT     (default 0)
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

The last three reproduce the real failure classes the phase-4D worker
lifecycle controller must detect and escalate on; `make lifecycle-check`
drives all of them.
"""

import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8090"))
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

_LOCK = threading.Lock()
_STATS = {"requests": 0, "tokens": 0, "failures": 0, "active": 0,
          "wedged": 0, "empty": 0, "stalled": 0}


def _forever() -> None:
    """Hold the connection (and this handler thread) without answering."""
    while True:
        time.sleep(3600)

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

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": MODEL_NAME})
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
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
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
            if FAIL_AFTER >= 0 and seq > FAIL_AFTER:
                with _LOCK:
                    _STATS["failures"] += 1
                self._json(503, {"error": {"message": "injected failure", "type": "mock"}})
                return

            if WEDGE_AFTER >= 0 and seq > WEDGE_AFTER:
                # Silent wedge: keep answering /health, never answer POST.
                with _LOCK:
                    _STATS["wedged"] += 1
                _forever()
                return

            n = min(int(req.get("max_tokens") or MAX_OUTPUT), MAX_OUTPUT)
            req_id = f"chatcmpl-mock-{seq}"
            rng = random.Random(seq)
            hang = rng.random() * 100 < HANG_TTFT_PCT
            if HANG_STREAM_ONLY and not req.get("stream"):
                hang = False
            ttft = 30_000.0 if hang else TTFT_MS
            empty = rng.random() * 100 < EMPTY_COMPLETION_PCT
            if EMPTY_AFTER >= 0 and seq <= EMPTY_AFTER:
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
                    time.sleep(TOK_INTERVAL_MS / 1000.0)
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
                time.sleep(ttft / 1000.0 + (n - 1) * TOK_INTERVAL_MS / 1000.0)
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


if __name__ == "__main__":
    print(
        f"mock-gpu worker on :{PORT}  model={MODEL_NAME}  ttft={TTFT_MS}ms  "
        f"itl={TOK_INTERVAL_MS}ms  fail_after={FAIL_AFTER}  crash%={CRASH_STREAM_PCT}  "
        f"hang%={HANG_TTFT_PCT}  wedge_after={WEDGE_AFTER}  "
        f"empty%={EMPTY_COMPLETION_PCT}  stall_after_chunks={STALL_AFTER_CHUNKS}"
    )
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
