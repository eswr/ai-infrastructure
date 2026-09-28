// EventSource wiring + one-shot snapshot fetch. The server diffs the
// gateway and pushes full snapshots; transitions are derived server-side.

import {
  setSnapshot,
  pushFeed,
  setScenarioLog,
  setConnected,
  setFlash,
  type Snapshot,
} from "./state";

export function connect() {
  const es = new EventSource("/api/events");

  es.addEventListener("open", () => setConnected(true));
  es.addEventListener("error", () => setConnected(false));

  es.addEventListener("snapshot", (e) => {
    setConnected(true);
    setSnapshot(JSON.parse((e as MessageEvent).data) as Snapshot);
  });

  es.addEventListener("worker.transition", (e) => {
    const d = JSON.parse((e as MessageEvent).data);
    const gen =
      d.generation != null && d.prev_generation != null && d.generation !== d.prev_generation
        ? ` · generation ${d.prev_generation}→${d.generation}`
        : "";
    const reason = d.reason ? ` (${d.reason})` : "";
    pushFeed({
      ts: Date.now(),
      kind: "worker",
      text: `${d.worker}: ${d.from ?? "?"} → ${d.to}${reason}${gen}`,
      state: d.to,
    });
  });

  es.addEventListener("node.transition", (e) => {
    const d = JSON.parse((e as MessageEvent).data);
    pushFeed({
      ts: Date.now(),
      kind: "node",
      text: `${d.node}: ${d.from ?? "?"} → ${d.to}`,
      state: d.to,
    });
  });

  es.addEventListener("scheduler.decision", () => setFlash((n) => n + 1));

  es.addEventListener("scenario.log", (e) => {
    const d = JSON.parse((e as MessageEvent).data);
    setScenarioLog((lines) => [...lines.slice(-160), `[${d.name}] ${d.line}`]);
  });

  es.addEventListener("scenario.exit", (e) => {
    const d = JSON.parse((e as MessageEvent).data);
    setScenarioLog((lines) => [...lines.slice(-160), `[${d.name}] exited with code ${d.code}`]);
  });

  es.addEventListener("gateway.down", () => setConnected(false));
}

export async function runScenario(name: string): Promise<string> {
  const res = await fetch(`/api/scenarios/${name}`, { method: "POST" });
  const body = await res.json();
  if (!res.ok) return body.error || `HTTP ${res.status}`;
  return `started ${name}`;
}
