import { For, Show, createMemo, createSignal } from "solid-js";
import { snapshot, scenarioLog, setScenarioLog, connected } from "../state";
import { runScenario } from "../api";

const BUTTONS: { name: string; label: string; hint: string }[] = [
  { name: "wedge", label: "Silent wedge", hint: "mock worker hangs while /health stays 200 → probe quarantine → recovery (local, ~10 s)" },
  { name: "metal", label: "Real Metal", hint: "start MLX 9B on this Mac, measure TTFT/ITL with loadgen (~40 s, real GPU)" },
  { name: "autoscale", label: "Autoscale", hint: "queue pressure → scaler 2→3→2 (needs k3d, ~1.5 min)" },
  { name: "drain", label: "Drain", hint: "graceful drain with in-flight request (needs k3d)" },
  { name: "placement", label: "Placement", hint: "fleet discovery + VRAM/KV/TP placement matrix (needs k3d)" },
  { name: "xid", label: "XID failure", hint: "critical XID → node FAILED → replace → READY (needs k3d)" },
  { name: "thermal", label: "Thermal", hint: "thermal → CORDONED/DEGRADED, traffic continues (needs k3d)" },
  { name: "oom", label: "GPU OOM", hint: "FB full → immediate worker quarantine → recovery (needs k3d)" },
];

export default function DemoControls() {
  const [busy, setBusy] = createSignal<string | null>(null);
  const scenario = createMemo(() => snapshot()?.scenario);

  const click = async (name: string) => {
    setBusy(name);
    const msg = await runScenario(name);
    setScenarioLog((lines: string[]) => [...lines.slice(-160), `[${name}] ${msg}`]);
    // busy state clears when the scenario state flips back to idle (polled)
    const check = setInterval(() => {
      const s = scenario();
      if (!s || s.state !== "running" || s.name !== name) {
        clearInterval(check);
        setBusy(null);
      }
    }, 1000);
  };

  return (
    <section class="panel controls">
      <h2>Demo scenarios</h2>
      <div class="controls-grid">
        <For each={BUTTONS}>
          {(b) => (
            <button
              disabled={busy() !== null || scenario()?.state === "running"}
              title={b.hint}
              onClick={() => click(b.name)}
            >
              {b.label}
            </button>
          )}
        </For>
      </div>
      <Show when={scenario()?.state === "running"}>
        <p class="running">running: {scenario()?.name} — output below</p>
      </Show>
      <pre class="scen-log">{scenarioLog().join("\n")}</pre>
      <p class="muted footer-note">
        DEV / DEMO ONLY · fixed scenario enum, no arbitrary commands · loopback only
        <Show when={!connected()}>
          <span class="state-text bad"> · event stream disconnected</span>
        </Show>
      </p>
    </section>
  );
}
