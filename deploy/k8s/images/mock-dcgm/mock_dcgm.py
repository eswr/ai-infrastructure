#!/usr/bin/env python3
"""Mock DCGM exporter (phase 6B): DCGM-shaped telemetry for the SIMULATED
fleet. Every series carries simulation="true" — this is observability
wiring, not GPU measurement (see deploy/k8s/README.md scope note).

Consumers and the contract each one fixes:

  gateway fleet.rs   scrapes FLEET_DCGM_URL (…:9400/metrics) and correlates
                     series on the UUID="…" label:
                       DCGM_FI_DEV_XID_ERRORS  latest-code gauge; a CHANGE to
                                               48/74/79/95 fails the node
                       DCGM_FI_DEV_GPU_TEMP    °C; >= FLEET_THERMAL_C (87)
                                               cordons, < _CLEAR_C (80) ends
                                               the episode
                       DCGM_FI_DEV_FB_USED     MiB (DCGM reports framebuffer
                                               in MiB; capabilities are in
                                               GB — the gateway divides by
                                               1024); >= 0.98*VRAM quarantines
                                               the worker, < 0.90*VRAM ends
                                               the episode (hysteresis)
  run_fleet_check.py GET /state -> {uuid: {..., "node": …}}; POST
                     /inject/{xid,thermal,oom,clear}
  Prometheus job dcgm-sim + Grafana gpu-fleet-simulation: also
                     DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_POWER_USAGE.

Catalog provenance: names/labels/units follow dcgm-exporter's
etc/default-counters.csv — XID is a latest-error-code gauge, FB_USED is MiB,
and PCIe traffic is the computed PROF_*_BYTES families (bytes/s), matching
the real exporter's default catalog so a real dcgm-exporter drops in without
changing a dashboard.

Inventory is derived from the gateway /status every second: the control
plane decides what exists; the exporter only mirrors it.

Fault semantics (why the pins are where they are):
  thermal  holds temp at the requested target until cleared.
  xid      holds the code until cleared (the gateway fires on change).
  oom      pins FB_USED at 0.99*VRAM — ABOVE the gateway's 0.98 trigger.
           Baseline FB never exceeds 0.5*VRAM (a loaded-model-proportional
           walk crosses the 0.90 hysteresis floor and latches forever), and
           the injected pin must clear the 0.98 line decisively.
"""

import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.request

PORT = int(os.environ.get("PORT", "9400"))
GATEWAY_URL = os.environ.get("GATEWAY_URL",
                             "http://gateway.inference-lab.svc.cluster.local")
INVENTORY_POLL_S = float(os.environ.get("INVENTORY_POLL_S", "1.0"))

MODEL_NAMES = {
    "nvidia-h100": "NVIDIA H100 80GB HBM3",
    "nvidia-l40s": "NVIDIA L40S",
    "nvidia-a10": "NVIDIA A10",
}
IDLE_WATTS = {"nvidia-h100": 70.0, "nvidia-l40s": 46.0, "nvidia-a10": 35.0}


class Gpu:
    def __init__(self, uuid, node, accelerator, vram_gb):
        self.uuid = uuid
        self.node = node
        self.accelerator = accelerator
        self.vram_gb = float(vram_gb)
        # Fault state (all cleared by /inject/clear)
        self.xid = 0
        self.thermal_target = None
        self.oom_hold = False
        # Plausible idle telemetry; kept well inside the safe band.
        self.temp = random.uniform(41.0, 49.0)
        self.util = random.uniform(0.0, 4.0)
        self.loaded = False

    @property
    def model_name(self):
        return MODEL_NAMES.get(self.accelerator, self.accelerator)

    def fb_used_mib(self):
        if self.oom_hold:
            return self.vram_gb * 0.99 * 1024.0
        # Idle + a modest constant bump while any model is resident. NOT
        # proportional to the model (see module docstring).
        return self.vram_gb * (0.06 + (0.10 if self.loaded else 0.0)) * 1024.0

    def gpu_temp(self):
        if self.thermal_target is not None:
            return self.thermal_target
        # Idles in the low 40s; the thermal-clear threshold is 80 C.
        return self.temp


class Exporter:
    def __init__(self):
        self.lock = threading.Lock()
        self.inventory: dict[str, Gpu] = {}
        self.last_sync = 0.0

    # ------------------------------------------------------- inventory --
    def sync_inventory(self):
        try:
            with urllib.request.urlopen(f"{GATEWAY_URL}/status", timeout=5) as r:
                status = json.loads(r.read().decode())
        except Exception:
            return  # keep the last known inventory
        gpus = {}
        for w in status.get("workers", []):
            g = w.get("gpu") or {}
            uuid = g.get("uuid")
            if not uuid:
                continue
            gpus[uuid] = Gpu(uuid, w.get("node") or "", g.get("accelerator") or "",
                             g.get("vram_gb") or 0.0)
            gpus[uuid].loaded = bool(g.get("loaded_model"))
        with self.lock:
            # Preserve injected fault state and telemetry across re-syncs of
            # the same GPU so values do not twitch every poll.
            for uuid, g in gpus.items():
                old = self.inventory.get(uuid)
                if old:
                    g.xid, g.thermal_target, g.oom_hold = (
                        old.xid, old.thermal_target, old.oom_hold)
                    g.temp, g.util, g.loaded = old.temp, old.util, old.loaded
            self.inventory = gpus
            self.last_sync = time.time()

    def loop(self):
        while True:
            self.sync_inventory()
            time.sleep(INVENTORY_POLL_S)

    # ---------------------------------------------------------- output --
    def snapshot(self):
        with self.lock:
            return {u: g for u, g in self.inventory.items()}

    def state_json(self):
        return {u: {
            "node": g.node,
            "accelerator": g.accelerator,
            "vram_gb": g.vram_gb,
            "loaded": g.loaded,
            "temp_c": round(g.gpu_temp(), 1),
            "fb_used_mib": round(g.fb_used_mib(), 1),
            "xid": g.xid,
            "faults": {"thermal": g.thermal_target is not None,
                       "oom": g.oom_hold, "xid": g.xid},
        } for u, g in self.snapshot().items()}

    def metrics_text(self):
        gpus = sorted(self.snapshot().values(), key=lambda g: g.uuid)
        lines = []

        def emit(name, help_text, mtype, value_of):
            # Units live in the HELP string, as in dcgm-exporter's catalog.
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            for i, g in enumerate(gpus):
                labels = (f'gpu="{i}",UUID="{g.uuid}",device="nvidia{i}",'
                          f'modelName="{g.model_name}",Hostname="{g.node}",'
                          f'simulation="true"')
                lines.append(f"{name}{{{labels}}} {value_of(g)}")

        emit("DCGM_FI_DEV_GPU_TEMP",
             "GPU temperature (in C) [simulation=true]", "gauge",
             lambda g: f"{g.gpu_temp():.1f}")
        emit("DCGM_FI_DEV_FB_USED",
             "Framebuffer memory used (in MiB) [simulation=true]", "gauge",
             lambda g: f"{g.fb_used_mib():.1f}")
        emit("DCGM_FI_DEV_XID_ERRORS",
             "Latest XID error code encountered (0 = healthy) [simulation=true]",
             "gauge", lambda g: f"{g.xid}")
        emit("DCGM_FI_DEV_GPU_UTIL",
             "GPU utilization (in %) [simulation=true]", "gauge",
             lambda g: f"{g.util:.1f}")
        emit("DCGM_FI_DEV_POWER_USAGE",
             "GPU power draw (in W) [simulation=true]", "gauge",
             lambda g: f"{IDLE_WATTS.get(g.accelerator, 40.0) + g.util:.2f}")
        emit("DCGM_FI_PROF_PCIE_TX_BYTES",
             "Total number of bytes transmitted through PCIe (in bytes/s) "
             "[simulation=true]", "gauge",
             lambda g: f"{(1 + g.util) * 1e6:.0f}")
        emit("DCGM_FI_PROF_PCIE_RX_BYTES",
             "Total number of bytes received through PCIe (in bytes/s) "
             "[simulation=true]", "gauge",
             lambda g: f"{(1 + g.util) * 1.2e6:.0f}")
        lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------- inject --
    def inject(self, action, params):
        uuid = (params.get("gpu") or [None])[0]
        with self.lock:
            targets = ([self.inventory[uuid]] if uuid and uuid in self.inventory
                       else list(self.inventory.values()))
            if not targets:
                return 404, {"error": f"no gpu for {uuid or '*'}"}
            for g in targets:
                if action == "xid":
                    g.xid = int((params.get("code") or ["48"])[0])
                elif action == "thermal":
                    g.thermal_target = float((params.get("target") or ["94"])[0])
                elif action == "oom":
                    g.oom_hold = True
                elif action == "clear":
                    g.xid = 0
                    g.thermal_target = None
                    g.oom_hold = False
                else:
                    return 404, {"error": f"unknown inject {action}"}
        return 200, {"injected": action, "gpus": [g.uuid for g in targets]}


EXPORTER = Exporter()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # keep pod logs quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/metrics":
            self._send(200, EXPORTER.metrics_text().encode(), "text/plain; version=0.0.4; charset=utf-8")
        elif path == "/state":
            self._send(200, EXPORTER.state_json())
        elif path in ("/healthz", "/"):
            self._send(200, {"ok": True, "gpus": len(EXPORTER.snapshot()),
                             "last_gateway_sync_s": round(time.time() - EXPORTER.last_sync, 1)})
        else:
            self._send(404, {"error": f"no route {path}"})

    def do_POST(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]  # /inject/<action>
        action = parts[1] if len(parts) == 2 and parts[0] == "inject" else None
        if action is None:
            self._send(404, {"error": f"no route {u.path}"})
            return
        code, body = EXPORTER.inject(action, parse_qs(u.query))
        self._send(code, body)


def main():
    threading.Thread(target=EXPORTER.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[mock-dcgm] serving DCGM-shaped metrics on :{PORT} "
          f"(inventory from {GATEWAY_URL}/status, simulation=\"true\")", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
