// Global console state: one EventSource feeds everything.

import { createSignal } from "solid-js";

export interface Dimensions {
  process?: string;
  transport?: string;
  functional?: string;
  semantic?: string;
}

export interface Worker {
  id: string;
  runtime: string;
  model: string;
  slot: number | null;
  state: string;
  state_reason: string | null;
  generation: number;
  healthy: boolean;
  dimensions: Dimensions;
  outstanding: number;
  node: string | null;
  node_state: string | null;
  placement_gate: string | null;
  gpu: { accelerator: string; vram_gb: number; uuid: string; tp_group?: string | null; loaded_model?: string | null } | null;
  scope: "SIMULATED" | "MEASURED" | "MOCK";
}

export interface NodeState {
  id: string;
  state: string;
  allows_new_placements: boolean;
  serves_traffic: boolean;
  transitions_total: number;
}

export interface Candidate {
  label: string;
  worker_id: string;
  slot: number | null;
  reusable_tokens: number;
  cache_savings_ms: number;
  queue_ms: number;
  prefill_ms: number;
  estimated_ttft_ms: number;
  cache_state: string;
  chosen: boolean;
}

export interface Decision {
  id: number;
  ts_ms: number;
  policy: string;
  worker: string;
  slot: number | null;
  reason: string;
  bypass: { reason: string; worker_id: string; forgone_savings_ms: number; forgone_match_tokens: number } | null;
  prefix_hash: string;
  estimated_ttft_ms: number | null;
  estimated_queue_ms: number | null;
  cache_savings_ms: number | null;
  candidates: Candidate[];
  actual_ttft_ms: number | null;
}

export interface Metrics {
  queue_seconds: number;
  requests_active: number;
  requests_queued: number;
  active_prefills: number;
  active_decodes: number;
  output_tokens_total: number;
  interactive_slo_violations: number;
  placement_rejections: number;
  requests_by_outcome: Record<string, number>;
  cache_affinity_bypassed: Record<string, number>;
  ttft_p50_s: number | null;
  ttft_p95_s: number | null;
  itl_p50_ms: number | null;
}

export interface Snapshot {
  ts: number;
  gateway_url: string;
  workers: Worker[];
  nodes: NodeState[];
  metrics: Metrics;
  decisions: Decision[];
  scenario: { name: string | null; state: string; started_at: number | null };
}

export interface FeedEntry {
  ts: number;
  kind: "worker" | "node" | "system";
  text: string;
  state: string;
}

const [snapshot, setSnapshot] = createSignal<Snapshot | null>(null);
const [feed, setFeed] = createSignal<FeedEntry[]>([]);
const [scenarioLog, setScenarioLog] = createSignal<string[]>([]);
const [connected, setConnected] = createSignal(false);
const [flash, setFlash] = createSignal(0);

export {
  snapshot,
  setSnapshot,
  feed,
  setFeed,
  scenarioLog,
  setScenarioLog,
  connected,
  setConnected,
  flash,
  setFlash,
};

export function pushFeed(entry: FeedEntry) {
  setFeed((f) => [entry, ...f].slice(0, 80));
}

export function fmtS(ms: number | null | undefined, digits = 1): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(digits)} s`;
}

/** State text always accompanies color — never color alone. */
export const STATE_CLASS: Record<string, string> = {
  ready: "ok",
  degraded: "warn",
  cordoned: "warn",
  draining: "warn",
  warming: "info",
  recovering: "info",
  provisioning: "idle",
  maintenance: "idle",
  stopped: "idle",
  unhealthy: "bad",
  failed: "bad",
};
