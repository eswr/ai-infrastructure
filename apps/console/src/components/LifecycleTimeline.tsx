import { For, createMemo } from "solid-js";
import { feed, STATE_CLASS } from "../state";

function hhmmss(ts: number) {
  const d = new Date(ts);
  return d.toTimeString().slice(0, 8);
}

/** Compact event timeline: every worker/node transition the console saw,
 * newest first. This is the visual half of the lifecycle story. */
export default function LifecycleTimeline() {
  const entries = createMemo(() => feed());
  return (
    <section class="panel timeline">
      <h2>Lifecycle timeline</h2>
      <ul class="feed">
        <For each={entries()}>
          {(e) => (
            <li>
              <span class="ts">{hhmmss(e.ts)}</span>
              <span class={`state-text ${STATE_CLASS[e.state] ?? "idle"}`}>{e.text}</span>
            </li>
          )}
        </For>
      </ul>
      <ShowEmpty />
    </section>
  );
}

function ShowEmpty() {
  const empty = createMemo(() => feed().length === 0);
  return empty() ? <p class="muted">transitions appear here as workers change state</p> : null;
}
