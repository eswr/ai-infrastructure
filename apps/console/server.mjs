#!/usr/bin/env node
/**
 * Inference Lab Console — local-only backend.
 *
 *   DEV / DEMO ONLY. Binds 127.0.0.1. Not for production use.
 *
 * A storytelling projection of the control plane, not a second gateway:
 *
 *   GET  /api/snapshot         one-shot projection (fleet, metrics, decisions)
 *   GET  /api/events           SSE: snapshot ticks + transitions + decisions
 *   POST /api/scenarios/<name> run a FIXED demo scenario (hard enum, no
 *                              arbitrary commands), streaming its output
 *   GET  /api/scenarios        the enum + current run state
 *   GET  /healthz              liveness
 *
 * The gateway (and its /status, /metrics, /decisions) stays the source of
 * truth; this process only projects and diff-polls it. Scenario execution
 * shells out to the same Make targets the docs use — one code path for
 * demos, never a second implementation.
 *
 * Zero dependencies on purpose: node:http, node:child_process, fetch.
 */

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const DIST = path.join(REPO, "apps", "console", "dist");
const HOST = "127.0.0.1";                       // never expose beyond loopback
const PORT = Number(process.env.CONSOLE_PORT || 8099);
const GATEWAY_URL = process.env.GATEWAY_URL || "http://127.0.0.1:8080";
const POLL_MS = Number(process.env.CONSOLE_POLL_MS || 1000);

// ---------------------------------------------------------------------------
// Fixed demo-scenario enum → known argv. The ONLY execution path; requests
// never interpolate into these arrays.
// ---------------------------------------------------------------------------
const PY = process.execPath.includes("node") ? "python3" : "python3";
const SCENARIOS = {
  wedge:     { argv: ["python3", "benchmarks/k8s/run_k8s_check.py", "readiness_failure"], needs: "k3d cluster (make k8s-up) — visible live in the fleet view" },
  metal:     { argv: ["bash", "benchmarks/demo/demo_metal.sh"], needs: "local-only (real Metal GPU)" },
  autoscale: { argv: ["python3", "benchmarks/k8s/run_k8s_check.py", "autoscale"], needs: "k3d cluster (make k8s-up)" },
  drain:     { argv: ["python3", "benchmarks/k8s/run_k8s_check.py", "drain"], needs: "k3d cluster (make k8s-up)" },
  placement: { argv: ["python3", "benchmarks/k8s/run_fleet_check.py", "discovery", "placement"], needs: "k3d cluster (make k8s-up)" },
  xid:       { argv: ["python3", "benchmarks/k8s/run_fleet_check.py", "xid"], needs: "k3d cluster (make k8s-up)" },
  thermal:   { argv: ["python3", "benchmarks/k8s/run_fleet_check.py", "thermal"], needs: "k3d cluster (make k8s-up)" },
  oom:       { argv: ["python3", "benchmarks/k8s/run_fleet_check.py", "oom"], needs: "k3d cluster (make k8s-up)" },
};

// ---------------------------------------------------------------------------
// Small Prometheus text-format helpers (projection-grade, not a parser lib).
// ---------------------------------------------------------------------------

function parseSamples(text) {
  const samples = [];
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)\s*$/);
    if (!m) continue;
    const labels = {};
    if (m[2]) {
      for (const pair of m[2].matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"/g)) {
        labels[pair[1]] = pair[2];
      }
    }
    const value = Number(m[3]);
    if (Number.isFinite(value)) samples.push({ name: m[1], labels, value });
  }
  return samples;
}

function sumMetric(samples, name, labelFilter = {}) {
  return samples
    .filter((s) => s.name === name &&
      Object.entries(labelFilter).every(([k, v]) => s.labels[k] === v))
    .reduce((a, s) => a + s.value, 0);
}

/** Histogram quantile from cumulative _bucket series (le values are upper bounds). */
function histogramQuantile(samples, name, q) {
  const buckets = samples.filter((s) => s.name === `${name}_bucket`);
  if (buckets.length === 0) return null;
  const inf = buckets.find((b) => b.labels.le === "+Inf");
  const total = inf ? inf.value : Math.max(...buckets.map((b) => b.value));
  if (!(total > 0)) return null;
  const sorted = buckets
    .filter((b) => b.labels.le !== "+Inf")
    .map((b) => ({ le: Number(b.labels.le), value: b.value }))
    .sort((a, b) => a.le - b.le);
  for (const b of sorted) {
    if (b.value >= q * total) return b.le;
  }
  return sorted.length ? sorted[sorted.length - 1].le : null;
}

// ---------------------------------------------------------------------------
// Projection: gateway /status + /metrics + /decisions → console snapshot.
// ---------------------------------------------------------------------------

async function fetchJson(pathname, timeoutMs = 4000) {
  const res = await fetch(GATEWAY_URL + pathname, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`${pathname} -> ${res.status}`);
  return res.json();
}

async function fetchText(pathname, timeoutMs = 4000) {
  const res = await fetch(GATEWAY_URL + pathname, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`${pathname} -> ${res.status}`);
  return res.text();
}

function projectMetrics(samples) {
  const outcomes = {};
  for (const s of samples) {
    if (s.name === "inference_requests_total") {
      const k = s.labels.outcome || "?";
      outcomes[k] = (outcomes[k] || 0) + s.value;
    }
  }
  const bypass = {};
  for (const s of samples) {
    if (s.name === "inference_scheduler_cache_affinity_bypassed_total") {
      bypass[s.labels.reason || "?"] = (bypass[s.labels.reason || "?"] || 0) + s.value;
    }
  }
  return {
    queue_seconds: sumMetric(samples, "inference_estimated_queue_seconds"),
    requests_active: sumMetric(samples, "inference_requests_active"),
    requests_queued: sumMetric(samples, "inference_requests_queued"),
    active_prefills: sumMetric(samples, "inference_active_prefills"),
    active_decodes: sumMetric(samples, "inference_active_decodes"),
    output_tokens_total: sumMetric(samples, "inference_output_tokens_total"),
    interactive_slo_violations: sumMetric(samples, "inference_interactive_slo_violations_total"),
    placement_rejections: sumMetric(samples, "inference_placement_rejections_total"),
    requests_by_outcome: outcomes,
    cache_affinity_bypassed: bypass,
    ttft_p50_s: histogramQuantile(samples, "inference_ttft_seconds", 0.5),
    ttft_p95_s: histogramQuantile(samples, "inference_ttft_seconds", 0.95),
    itl_p50_ms:
      histogramQuantile(samples, "inference_stream_inter_token_seconds", 0.5) != null
        ? histogramQuantile(samples, "inference_stream_inter_token_seconds", 0.5) * 1000
        : null,
  };
}

/** SIMULATED vs MEASURED vs MOCK: nvidia-* accelerators are the phase-6
 * simulated fleet (simulation="true" in every exported series); Apple
 * Metal is the machine the benchmarks actually measured; the mock pool
 * is deterministic failure-injection hardware (exercised, not measured). */
function scopeOf(worker) {
  const accel = worker.gpu?.accelerator || "";
  if (accel.startsWith("nvidia-")) return "SIMULATED";
  if (accel.startsWith("apple") || accel === "") {
    return worker.runtime === "mock" || worker.model === "mock-gpu" ? "MOCK" : "MEASURED";
  }
  return worker.runtime === "mock" || worker.model === "mock-gpu" ? "MOCK" : "MEASURED";
}

function projectWorker(w) {
  return {
    id: w.id,
    runtime: w.runtime,
    model: w.model,
    slot: w.slot,
    state: w.state,
    state_reason: w.state_reason || null,
    generation: w.generation,
    healthy: w.healthy,
    dimensions: w.dimensions || {},
    outstanding: w.outstanding,
    node: w.node || null,
    node_state: w.node_state || null,
    placement_gate: w.placement_gate || null,
    gpu: w.gpu || null,
    scope: scopeOf(w),
  };
}

async function buildSnapshot(previousDecisions) {
  const status = await fetchJson("/status");
  let metricsText = "";
  try {
    metricsText = await fetchText("/metrics");
  } catch {
    /* metrics are projection sugar; snapshot survives without them */
  }
  const samples = parseSamples(metricsText);
  const workers = (status.workers || []).map(projectWorker);
  const nodes = (status.nodes || []).map((n) => ({
    id: n.id,
    state: n.state,
    allows_new_placements: n.allows_new_placements,
    serves_traffic: n.serves_traffic,
    transitions_total: n.transitions_total,
  }));

  let decisions = [];
  try {
    decisions = (await fetchJson("/decisions")).decisions || [];
  } catch {
    /* older gateway image without /decisions */
  }
  const newDecisionIds = decisions
    .filter((d) => !previousDecisions.has(d.id))
    .map((d) => d.id);
  const newlyResolved = decisions.filter(
    (d) => d.actual_ttft_ms != null && !previousDecisions.has(`ok:${d.id}`),
  );
  for (const d of decisions) {
    previousDecisions.add(d.id);
    if (d.actual_ttft_ms != null) previousDecisions.add(`ok:${d.id}`);
  }

  return {
    ts: Date.now(),
    gateway_url: GATEWAY_URL,
    workers,
    nodes,
    metrics: projectMetrics(samples),
    decisions,
    calibration: status.calibration || status.calibrations || null,
    scenario: scenarioState(),
    newDecisionIds,
    newlyResolvedIds: newlyResolved.map((d) => d.id),
  };
}

// ---------------------------------------------------------------------------
// Scenario runner: one at a time, fixed argv, output streamed to clients.
// ---------------------------------------------------------------------------

let currentScenario = null; // { name, proc, startedAt }

function scenarioState() {
  if (!currentScenario) return { name: null, state: "idle", started_at: null };
  return {
    name: currentScenario.name,
    state: "running",
    started_at: currentScenario.startedAt,
  };
}

function runScenario(name) {
  const spec = SCENARIOS[name];
  if (!spec) return { error: `unknown scenario '${name}'` };
  if (currentScenario) return { error: `scenario '${currentScenario.name}' already running`, busy: true };
  const proc = spawn(spec.argv[0], spec.argv.slice(1), {
    cwd: REPO,
    env: process.env,
  });
  currentScenario = { name, proc, startedAt: Date.now() };
  broadcast("scenario.start", { name, needs: spec.needs });
  const forward = (buf) => {
    for (const line of buf.toString().split("\n")) {
      if (line.trim()) broadcast("scenario.log", { name, line });
    }
  };
  proc.stdout.on("data", forward);
  proc.stderr.on("data", forward);
  proc.on("close", (code) => {
    currentScenario = null;
    broadcast("scenario.exit", { name, code });
  });
  return { started: name };
}

// ---------------------------------------------------------------------------
// SSE clients + the 1 s diff-poll loop.
// ---------------------------------------------------------------------------

const clients = new Set();
const seenDecisions = new Set();

function broadcast(event, data) {
  const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) res.write(frame);
}

async function tick() {
  let snap;
  try {
    snap = await buildSnapshot(seenDecisions);
  } catch (e) {
    broadcast("gateway.down", { error: String(e.message || e) });
    return;
  }
  broadcast("snapshot", snap);
  for (const id of snap.newDecisionIds) {
    const d = snap.decisions.find((x) => x.id === id);
    if (d) broadcast("scheduler.decision", d);
  }
  for (const id of snap.newlyResolvedIds) {
    const d = snap.decisions.find((x) => x.id === id);
    if (d) broadcast("scheduler.outcome", d);
  }
  // Transitions: diff worker/node (state, generation) against the previous tick.
  const prev = tick.prev;
  if (prev) {
    for (const w of snap.workers) {
      const p = prev.workers.find((x) => x.id === w.id);
      if (!p) {
        broadcast("worker.transition", { worker: w.id, from: null, to: w.state, reason: w.state_reason, generation: w.generation });
      } else if (p.state !== w.state || p.generation !== w.generation) {
        broadcast("worker.transition", {
          worker: w.id, from: p.state, to: w.state, reason: w.state_reason,
          generation: w.generation, prev_generation: p.generation,
          node: w.node,
        });
      }
    }
    for (const n of snap.nodes) {
      const p = prev.nodes.find((x) => x.id === n.id);
      if (p && p.state !== n.state) {
        broadcast("node.transition", { node: n.id, from: p.state, to: n.state });
      }
    }
  }
  tick.prev = { workers: snap.workers, nodes: snap.nodes };
}

// ---------------------------------------------------------------------------
// HTTP: static SPA + API.
// ---------------------------------------------------------------------------

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".tsx": "text/plain",
};

async function serveStatic(res, urlPath) {
  let rel = urlPath === "/" ? "/index.html" : urlPath;
  try {
    const file = await readFile(path.join(DIST, rel));
    res.writeHead(200, { "Content-Type": MIME[path.extname(rel)] || "application/octet-stream" });
    res.end(file);
    return;
  } catch {
    /* SPA fallback below */
  }
  try {
    const file = await readFile(path.join(DIST, "index.html"));
    res.writeHead(200, { "Content-Type": MIME[".html"] });
    res.end(file);
  } catch {
    res.writeHead(503, { "Content-Type": "text/plain" });
    res.end("console frontend not built — run: make console-build");
  }
}

function json(res, code, body) {
  res.writeHead(code, { "Content-Type": "application/json", "X-Console-Scope": "DEV-DEMO-ONLY" });
  res.end(JSON.stringify(body));
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);

  if (url.pathname === "/api/events" && req.method === "GET") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write(`retry: 2000\n\n`);
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }

  if (url.pathname === "/api/snapshot" && req.method === "GET") {
    try {
      json(res, 200, await buildSnapshot(seenDecisions));
    } catch (e) {
      json(res, 503, { error: `gateway unreachable: ${e.message}` });
    }
    return;
  }

  if (url.pathname === "/api/scenarios" && req.method === "GET") {
    json(res, 200, {
      scenarios: Object.fromEntries(
        Object.entries(SCENARIOS).map(([k, v]) => [k, { needs: v.needs }]),
      ),
      current: scenarioState(),
    });
    return;
  }

  const scenarioMatch = url.pathname.match(/^\/api\/scenarios\/([a-z]+)$/);
  if (scenarioMatch && req.method === "POST") {
    const result = runScenario(scenarioMatch[1]);
    json(res, result.error ? (result.busy ? 409 : 400) : 202, result);
    return;
  }

  if (url.pathname === "/healthz") {
    json(res, 200, { ok: true, gateway: GATEWAY_URL });
    return;
  }

  if (req.method === "GET") return serveStatic(res, url.pathname);
  res.writeHead(405).end();
});

server.listen(PORT, HOST, () => {
  console.log(`[console] DEV / DEMO ONLY — http://${HOST}:${PORT} (gateway: ${GATEWAY_URL})`);
  console.log(`[console] scenarios: ${Object.keys(SCENARIOS).join(", ")}`);
  setInterval(() => tick().catch(() => {}), POLL_MS);
  tick().catch(() => {});
});
