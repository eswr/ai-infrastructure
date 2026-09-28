import { Show, createMemo } from "solid-js";
import { snapshot } from "../state";

/** The bottom strip: the numbers you say out loud. */
export default function MetricsBar() {
  const snap = createMemo(() => snapshot());
  const m = () => snap()?.metrics;
  const healthy = createMemo(() => (snap()?.workers ?? []).filter((w) => w.healthy).length);
  const total = createMemo(() => (snap()?.workers ?? []).length);

  return (
    <footer class="metrics-bar">
      <Metric label="TTFT p50" value={m()?.ttft_p50_s != null ? `${m()!.ttft_p50_s!.toFixed(2)} s` : "—"} />
      <Metric label="ITL p50" value={m()?.itl_p50_ms != null ? `${m()!.itl_p50_ms!.toFixed(0)} ms` : "—"} />
      <Metric label="Queue est." value={m()?.queue_seconds != null ? `${m()!.queue_seconds.toFixed(2)} s` : "—"} />
      <Metric label="Active" value={String(m()?.requests_active ?? 0)} />
      <Metric
        label="Cache bypassed"
        value={Object.entries(m()?.cache_affinity_bypassed ?? {})
          .map(([k, v]) => `${k}:${v}`)
          .join(" ") || "0"}
      />
      <Metric label="Healthy workers" value={`${healthy()}/${total()}`} />
      <Show when={snap()}>
        <span class="muted gateway">{snap()!.gateway_url}</span>
      </Show>
    </footer>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div class="metric">
      <span class="metric-label">{props.label}</span>
      <span class="metric-value">{props.value}</span>
    </div>
  );
}
