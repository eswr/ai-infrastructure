import { For, Show, createMemo } from "solid-js";
import { snapshot, STATE_CLASS, type Worker } from "../state";

function WorkerRow(w: Worker) {
  const cls = () => STATE_CLASS[w.state] ?? "idle";
  const dims = () =>
    w.state === "ready" || w.state === "degraded"
      ? ""
      : ` · fn=${w.dimensions.functional ?? "?"} tr=${w.dimensions.transport ?? "?"}`;
  return (
    <div class="worker-row">
      <span class={`dot ${cls()}`} />
      <span class="worker-id" title={w.runtime + (w.slot != null ? ` slot ${w.slot}` : "")}>
        {w.id}
      </span>
      <span class={`state-text ${cls()}`}>{w.state}</span>
      <span class="worker-meta">
        gen {w.generation}
        {w.outstanding > 0 ? ` · ${w.outstanding} active` : ""}
        {dims()}
      </span>
      <span class={`badge ${w.scope === "SIMULATED" ? "sim" : w.scope === "MOCK" ? "mock" : "meas"}`}>{w.scope}</span>
    </div>
  );
}

export default function FleetMap() {
  const groups = createMemo(() => {
    const snap = snapshot();
    if (!snap) return [];
    const byNode = new Map<string, Worker[]>();
    for (const w of snap.workers) {
      const key = w.node || "worker pool";
      byNode.set(key, [...(byNode.get(key) ?? []), w]);
    }
    return [...byNode.entries()].map(([node, workers]) => {
      const nodeState = snap.nodes.find((n) => n.id === node);
      const accel = workers.find((x) => x.gpu)?.gpu;
      const vram = workers.reduce((a, x) => a + (x.gpu?.vram_gb ?? 0), 0);
      return { node, workers, nodeState, accel, vram };
    });
  });

  return (
    <section class="panel fleet">
      <h2>Fleet</h2>
      <Show when={snapshot()} fallback={<p class="muted">waiting for gateway…</p>}>
        <div class="fleet-cards">
          <For each={groups()}>
            {(g) => (
              <div class="node-card">
                <div class="node-head">
                  <span class="node-id">{g.node}</span>
                  <Show when={g.nodeState}>
                    {(ns) => (
                      <span class={`state-text ${STATE_CLASS[ns().state] ?? "idle"}`}>
                        {ns().state}
                      </span>
                    )}
                  </Show>
                </div>
                <Show when={g.accel} fallback={<p class="muted">mock workers · labels are the registration</p>}>
                  <p class="node-sub">
                    {g.accel!.accelerator} ×{g.workers.length}
                    {g.vram > 0 ? ` · ${g.vram} GB` : ""}
                    {g.accel!.tp_group ? ` · ${g.accel!.tp_group}` : ""}
                  </p>
                </Show>
                <For each={g.workers}>{(w) => WorkerRow(w)}</For>
                {g.nodeState && (
                  <p class="node-foot">
                    placement {g.nodeState.allows_new_placements ? "OPEN" : "CLOSED"} · traffic{" "}
                    {g.nodeState.serves_traffic ? "serving" : "stopped"} · {g.nodeState.transitions_total} transitions
                  </p>
                )}
              </div>
            )}
          </For>
        </div>
      </Show>
    </section>
  );
}
