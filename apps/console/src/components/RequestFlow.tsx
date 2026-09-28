import { For, Show, createMemo } from "solid-js";
import { snapshot, fmtS, flash } from "../state";

/** The live request path with real counts at each stage, plus the most
 * recent scheduler decision (the per-candidate table, the reason, and
 * predicted-vs-actual once the request closes). */
export default function RequestFlow() {
  const snap = createMemo(() => snapshot());
  const m = () => snap()?.metrics;
  const decision = createMemo(() => {
    const d = snap()?.decisions ?? [];
    return d.length ? d[d.length - 1] : null;
  });

  const stages = () => [
    { name: "CLIENT", value: null, hint: "OpenAI-compatible edge" },
    { name: "ADMISSION", value: `${m()?.requests_queued ?? 0} queued`, hint: "estimated-work budget" },
    { name: "PLACEMENT", value: `${m()?.placement_rejections ?? 0} refused`, hint: "VRAM + KV + TP gate" },
    { name: "ROUTER", value: `${snap()?.decisions.length ?? 0} decisions`, hint: "cache-aware ECT" },
    { name: "WORKER", value: `${m()?.requests_active ?? 0} active`, hint: `${m()?.active_prefills ?? 0} prefill · ${m()?.active_decodes ?? 0} decode` },
    { name: "TOKENS", value: `${Math.round(m()?.output_tokens_total ?? 0)}`, hint: `ITL p50 ${m()?.itl_p50_ms?.toFixed(0) ?? "—"} ms` },
  ];

  return (
    <section class="panel flow">
      <h2>Request Flow</h2>
      <div class="pipeline">
        <For each={stages()}>
          {(s, i) => (
            <>
              <Show when={i() > 0}>
                <span class="pipe">→</span>
              </Show>
              <div class="stage" title={s.hint}>
                <div class="stage-name">{s.name}</div>
                <div class="stage-value">{s.value ?? ""}</div>
              </div>
            </>
          )}
        </For>
      </div>

      <Show when={decision()} fallback={<p class="muted">no routing decisions yet — send a request through the gateway</p>}>
        {(d) => (
          <div class="decision" data-flash={flash()}>
            <div class="decision-head">
              <span class="mono">#{d().id}</span>
              <span class="muted">{d().policy}</span>
              <span class="chosen-line">
                → {d().worker}
                {d().slot != null ? ` (slot ${d().slot})` : ""}
              </span>
              <span class="state-text info">{d().reason}</span>
            </div>
            <Show when={d().bypass}>
              {(b) => (
                <p class="bypass">
                  bypassed affinity on {b().worker_id}: {b().reason} (forgone saving {fmtS(b().forgone_savings_ms)})
                </p>
              )}
            </Show>
            <Show when={d().candidates.length > 0}>
              <table class="cand">
              <thead>
                <tr>
                  <th>slot</th>
                  <th>reuse</th>
                  <th>cache save</th>
                  <th>queue</th>
                  <th>est TTFT</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                <For each={d().candidates}>
                  {(c) => (
                    <tr class={c.chosen ? "chosen" : ""}>
                      <td>{c.label}</td>
                      <td>{c.reusable_tokens > 0 ? `${c.reusable_tokens} tok` : "0"}</td>
                      <td>{fmtS(c.cache_savings_ms)}</td>
                      <td>{fmtS(c.queue_ms)}</td>
                      <td>{fmtS(c.estimated_ttft_ms)}</td>
                      <td>{c.chosen ? "← CHOSEN" : ""}</td>
                    </tr>
                  )}
                </For>
              </tbody>
              </table>
            </Show>
            <div class="pred-actual">
              <span>predicted {fmtS(d().estimated_ttft_ms)}</span>
              <span>actual {fmtS(d().actual_ttft_ms)}</span>
              {d().actual_ttft_ms != null && d().estimated_ttft_ms != null && (
                <span class="muted">error {fmtS(d().actual_ttft_ms! - d().estimated_ttft_ms!)}</span>
              )}
            </div>
          </div>
        )}
      </Show>
    </section>
  );
}
