//! Fleet controller (phase 6D/6E) — nodes above workers, and the
//! telemetry that moves them.
//!
//! Two responsibilities, one reconcile loop:
//!
//! 1. **Node state reconciliation.** The [`NodeTable`] owns a
//!    [`NodeLifecycle`] per simulated node (workers carry
//!    `inference-lab.io/node` labels). Kubernetes reality arrives as node
//!    events: a READY worker means the node validated (`WorkersOnline`);
//!    an empty registry entry means the node is lost and re-provisioning
//!    (`WorkersLost`); a fully-drained node enters maintenance. After
//!    every pass, each worker's *placement gate* is rewritten from its
//!    node's state — the bridge between fleet state and the router's
//!    placement filter.
//!
//! 2. **Simulated GPU telemetry.** When `FLEET_DCGM_URL` is configured the
//!    loop scrapes the mock-DCGM exporter (DCGM-shaped metrics labeled
//!    `simulation="true"`, keyed by GPU UUID) and responds the way a real
//!    controller would:
//!
//!    | signal                       | response                                   |
//!    |------------------------------|--------------------------------------------|
//!    | XID 48/74/79/95              | node FAILED → drain workers, stop scheduling |
//!    | temp ≥ FLEET_THERMAL_C       | node CORDONED + worker ThermalThrottling (DEGRADED, traffic continues) |
//!    | temp < clear threshold       | ThermalCleared → node back to READY        |
//!    | FB_USED ≥ VRAM (OOM)         | worker MemoryPressure → immediate UNHEALTHY |
//!
//!    Every number here is `simulation="true"` — see the README scope note.

use crate::AppState;
use inference_lab_scheduler::{WorkerHandle, NODE_GATE_CLOSED, NODE_GATE_OPEN, NODE_GATE_RESIDENT_ONLY};
use inference_lab_worker_lifecycle::{
    Dimension, FailureReason, NodeEvent, NodeEventKind, NodeLifecycle, NodeState, WorkerState,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FleetConfig {
    /// Mock-DCGM exporter metrics URL (`None` = no telemetry, admin-only).
    pub dcgm_url: Option<String>,
    pub poll_s: f64,
    /// Throttle thresholds: at `thermal_c` the node cordons (soft), below
    /// `thermal_clear_c` the episode ends.
    pub thermal_c: f64,
    pub thermal_clear_c: f64,
}

impl FleetConfig {
    pub fn from_env() -> Self {
        Self {
            dcgm_url: std::env::var("FLEET_DCGM_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            poll_s: crate::env_f64("FLEET_POLL_S", 2.0).clamp(0.5, 60.0),
            thermal_c: crate::env_f64("FLEET_THERMAL_C", 87.0),
            thermal_clear_c: crate::env_f64("FLEET_THERMAL_CLEAR_C", 80.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Node table
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct NodeTable {
    nodes: Mutex<HashMap<String, NodeLifecycle>>,
}

impl NodeTable {
    pub fn observe(&self, id: &str, event: NodeEvent, kind: NodeEventKind) -> Vec<inference_lab_worker_lifecycle::NodeTransition> {
        let mut g = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let node = g.entry(id.to_string()).or_insert_with(NodeLifecycle::new);
        let t = node.observe(event, kind);
        if !t.is_empty() {
            tracing::info!(
                node = %id,
                from = t[0].from.as_str(),
                to = t[0].to.as_str(),
                kind = t[0].kind.as_str(),
                "node transition"
            );
        }
        t
    }

    pub fn state_of(&self, id: &str) -> NodeState {
        self.nodes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .map(|n| n.state())
            .unwrap_or(NodeState::Provisioning)
    }

    pub fn snapshot(&self) -> Vec<(String, NodeState, u64)> {
        self.nodes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(id, n)| (id.clone(), n.state(), n.total_transitions()))
            .collect()
    }
}

/// Observe a node event AND record its transitions as metrics. Every
/// node-event call site goes through here.
fn observe_node(st: &Arc<AppState>, node: &str, event: NodeEvent, kind: NodeEventKind) {
    for t in st.nodes.observe(node, event, kind) {
        st.metrics
            .fleet_node_transitions_total
            .with_label_values(&[node, t.from.as_str(), t.to.as_str(), t.kind.as_str()])
            .inc();
    }
}

// ---------------------------------------------------------------------------
// Reconcile + telemetry loop
// ---------------------------------------------------------------------------

pub async fn fleet_loop(st: Arc<AppState>, cfg: FleetConfig) {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(5))
        .build()
        .expect("fleet http client");
    let mut tick = tokio::time::interval(Duration::from_secs_f64(cfg.poll_s));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Per-GPU episode state: an OOM fires once per spike, not every scrape.
    let mut oom_active: HashMap<String, bool> = HashMap::new();
    let mut thermal_active: HashMap<String, bool> = HashMap::new();
    let mut xid_seen: HashMap<String, f64> = HashMap::new();
    loop {
        tick.tick().await;
        reconcile(&st);
        if let Some(url) = &cfg.dcgm_url {
            if let Err(e) = apply_telemetry(&st, url, &cfg, &http, &mut oom_active, &mut thermal_active, &mut xid_seen).await {
                tracing::debug!(error = %e, "fleet telemetry scrape failed");
            }
            // Faults raised by telemetry change placement gates — apply
            // them in the SAME tick, not the next one (a one-tick window
            // is enough to route onto a failing node).
            reconcile(&st);
        }
    }
}

/// Sync the NodeTable + placement gates with registry reality. Cheap,
/// idempotent, runs every tick.
fn reconcile(st: &Arc<AppState>) {
    let workers = st.router.workers();
    // membership: node -> workers
    let mut membership: HashMap<&str, Vec<&Arc<WorkerHandle>>> = HashMap::new();
    for w in &workers {
        if let Some(node) = &w.node_id {
            membership.entry(node.as_str()).or_default().push(w);
        }
    }

    for (node, members) in &membership {
        let before = st.nodes.state_of(node);
        // Any READY worker validates the node.
        if before == NodeState::Provisioning && members.iter().any(|w| w.state() == WorkerState::Ready)
        {
            observe_node(st, node, NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        }
        // A fully-drained node completes its drain into maintenance.
        if before == NodeState::Draining
            && members
                .iter()
                .all(|w| matches!(w.state(), WorkerState::Draining | WorkerState::Stopped))
        {
            observe_node(st, node, NodeEvent::WorkersDrained, NodeEventKind::Fleet);
        }
        // Recovery validation: a RECOVERING node passes when a worker is
        // READY again.
        if before == NodeState::Recovering
            && members.iter().any(|w| w.state() == WorkerState::Ready)
        {
            observe_node(st, node, NodeEvent::ValidationPassed, NodeEventKind::Fleet);
        }
    }

    // A DRAINING node whose workers were all removed (pod deletion rather
    // than graceful finish) still completes its drain.
    for (id, _, _) in st.nodes.snapshot() {
        if st.nodes.state_of(&id) == NodeState::Draining && !membership.contains_key(id.as_str())
        {
            observe_node(st, &id, NodeEvent::WorkersDrained, NodeEventKind::Fleet);
        }
    }

    // Nodes whose workers all vanished: lost (only meaningful while they
    // were serving).
    let known_nodes: Vec<String> = st.nodes.snapshot().into_iter().map(|(id, _, _)| id).collect();
    for node in known_nodes {
        let has_workers = membership.keys().any(|k| k == &node);
        if !has_workers {
            let state = st.nodes.state_of(&node);
            if matches!(state, NodeState::Ready | NodeState::Cordoned | NodeState::Recovering) {
                observe_node(st, &node, NodeEvent::WorkersLost, NodeEventKind::Fleet);
            }
        }
    }

    // Rewrite every worker's placement gate from its node's state.
    for w in &workers {
        let Some(node) = &w.node_id else { continue };
        let state = st.nodes.state_of(node);
        let (gate, label) = match state {
            NodeState::Ready => (NODE_GATE_OPEN, "ready"),
            NodeState::Cordoned => (NODE_GATE_RESIDENT_ONLY, "cordoned"),
            _ => (NODE_GATE_CLOSED, state.as_str()),
        };
        if w.node_gate() != gate {
            w.set_node_gate(gate, label);
        }
    }

    // Metrics: one-hot node states.
    for (id, _, _) in st.nodes.snapshot() {
        let state = st.nodes.state_of(&id);
        for s in NodeState::all() {
            st.metrics
                .fleet_node_state
                .with_label_values(&[&id, s.as_str()])
                .set((state == s) as i64);
        }
    }
}

// ---------------------------------------------------------------------------
// Telemetry (mock DCGM)
// ---------------------------------------------------------------------------

/// Critical XID codes: DOUBLE_BIT_ECC-style errors that fail a node.
const CRITICAL_XIDS: [f64; 4] = [48.0, 74.0, 79.0, 95.0];

/// Parse the DCGM exposition text into per-GPU (xid, temp_c, fb_used_mib).
///
/// The value is the token after the LAST space: label values may contain
/// spaces (real dcgm-exporter emits `modelName="NVIDIA H100 80GB HBM3"`),
/// so a first-space split truncates the series and drops every line.
pub fn parse_dcgm_series(text: &str) -> HashMap<String, (f64, f64, f64)> {
    let mut per_gpu: HashMap<String, (f64, f64, f64)> = HashMap::new();
    for line in text.lines() {
        let Some((series, value)) = line.rsplit_once(' ') else { continue };
        if !series.starts_with("DCGM_FI_DEV_") {
            continue;
        }
        // dcgm-exporter label semantics: gpu=<index>, UUID=<device uuid>.
        // Correlate on the UUID — the worker's simulated GPU identity.
        let Some(uuid) = series.split("UUID=\"").nth(1).and_then(|r| r.split('"').next()) else {
            continue;
        };
        let Ok(v) = value.trim().parse::<f64>() else { continue };
        let e = per_gpu.entry(uuid.to_string()).or_insert((0.0, 0.0, 0.0));
        if series.starts_with("DCGM_FI_DEV_XID_ERRORS{") {
            e.0 = v;
        } else if series.starts_with("DCGM_FI_DEV_GPU_TEMP{") {
            e.1 = v;
        } else if series.starts_with("DCGM_FI_DEV_FB_USED{") {
            e.2 = v;
        }
    }
    per_gpu
}

#[allow(clippy::too_many_arguments)]
async fn apply_telemetry(
    st: &Arc<AppState>,
    url: &str,
    cfg: &FleetConfig,
    http: &reqwest::Client,
    oom_active: &mut HashMap<String, bool>,
    thermal_active: &mut HashMap<String, bool>,
    xid_seen: &mut HashMap<String, f64>,
) -> anyhow::Result<()> {
    let text = http.get(url).send().await?.error_for_status()?.text().await?;
    let per_gpu = parse_dcgm_series(&text);

    let workers = st.router.workers();
    for w in &workers {
        let Some(uuid) = &w.gpu_uuid else { continue };
        let Some((xid, temp, fb_used_mib)) = per_gpu.get(uuid.as_str()).copied() else {
            continue;
        };
        // DCGM reports framebuffer in MiB; capabilities are in GB. Normalize
        // BEFORE comparing (1 GiB = 1024 MiB) — comparing raw MiB against a
        // GB number quarantines every GPU on the first scrape.
        let fb_used = fb_used_mib / 1024.0;
        let node = w.node_id.clone().unwrap_or_default();

        // XID errors: DCGM reports the LATEST error code on the GPU.
        let prev = xid_seen.insert(uuid.clone(), xid).unwrap_or(0.0);
        if xid != prev && CRITICAL_XIDS.contains(&xid) {
            tracing::error!(node = %node, gpu = %uuid, xid = xid as u64,
                "critical XID error — failing node");
            observe_node(st, &node, NodeEvent::CriticalFault, NodeEventKind::Telemetry);
        } else if xid != prev && xid > 0.0 {
            tracing::warn!(node = %node, gpu = %uuid, xid = xid as u64, "non-critical XID error");
        }

        // Thermal: soft response — cordon, degrade the worker, keep serving.
        let therm = thermal_active.entry(uuid.clone()).or_insert(false);
        if temp >= cfg.thermal_c && !*therm {
            *therm = true;
            observe_node(st, &node, NodeEvent::ThermalEvent, NodeEventKind::Telemetry);
            let transitions = w.record_failure(Dimension::Functional, FailureReason::ThermalThrottling);
            st.metrics.record_transitions(w, &transitions);
            tracing::warn!(node = %node, gpu = %uuid, temp, "thermal event — node cordoned, traffic continues");
        } else if temp < cfg.thermal_clear_c && *therm {
            *therm = false;
            observe_node(st, &node, NodeEvent::ThermalCleared, NodeEventKind::Telemetry);
            tracing::info!(node = %node, gpu = %uuid, temp, "thermal episode cleared");
        }

        // OOM: FB_USED at/over VRAM is an immediate worker quarantine.
        let Some(vram) = w.vram_gb else { continue };
        let oom = oom_active.entry(uuid.clone()).or_insert(false);
        if fb_used >= vram * 0.98 && !*oom {
            *oom = true;
            let transitions = w.record_failure(Dimension::Process, FailureReason::MemoryPressure);
            st.metrics.record_transitions(w, &transitions);
            tracing::error!(node = %node, gpu = %uuid,
                fb_used_gib = fb_used, vram_gb = vram,
                "simulated GPU OOM — worker quarantined (memory_pressure)");
        } else if fb_used < vram * 0.90 && *oom {
            *oom = false; // probe recovery takes it from here
        }
    }
    Ok(())
}

/// Admin surface used by `POST /admin/nodes/{id}/{action}`.
pub fn admin_action(st: &AppState, node: &str, action: &str) -> Result<(NodeState, String), String> {
    use NodeEvent as E;
    use NodeEventKind as K;
    let (event, kind) = match action {
        "cordon" => (E::Cordon, K::Admin),
        "uncordon" => (E::Uncordon, K::Admin),
        "drain" => (E::Drain, K::Admin),
        "maintenance" => (E::Maintenance, K::Admin),
        "return" => (E::ReturnToService, K::Admin),
        "fail" => (E::CriticalFault, K::Admin),
        "recover" => (E::FaultCleared, K::Admin),
        _ => return Err(format!("unknown node action {action} (cordon|uncordon|drain|maintenance|return|fail|recover)")),
    };
    // `drain` also drains every worker on the node (existing phase-4D path).
    if matches!(event, E::Drain) {
        for w in st.router.workers() {
            if w.node_id.as_deref() == Some(node) {
                let transitions = w.drain();
                st.metrics.record_transitions(&w, &transitions);
            }
        }
    }
    let t = st.nodes.observe(node, event, kind);
    if t.is_empty() {
        return Err(format!(
            "node {node}: action {action} is a no-op in state {}",
            st.nodes.state_of(node).as_str()
        ));
    }
    let state = st.nodes.state_of(node);
    Ok((state, format!("{} -> {}", t[0].from.as_str(), t[0].to.as_str())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dcgm_series_with_spaces_in_label_values() {
        // Real dcgm-exporter emits label values with spaces
        // (modelName="NVIDIA H100 80GB HBM3"); the value is the token
        // after the LAST space. A first-space split turns the value into
        // `H100`, drops every line, and the controller goes blind — the
        // failure mode this test pins.
        let text = "\
# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (in C) [simulation=true]
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu=\"0\",UUID=\"gpu-a\",device=\"nvidia0\",modelName=\"NVIDIA H100 80GB HBM3\",Hostname=\"node-gpu-01\",simulation=\"true\"} 94.0
DCGM_FI_DEV_FB_USED{gpu=\"0\",UUID=\"gpu-a\",device=\"nvidia0\",modelName=\"NVIDIA H100 80GB HBM3\",Hostname=\"node-gpu-01\",simulation=\"true\"} 4915.2
DCGM_FI_DEV_XID_ERRORS{gpu=\"0\",UUID=\"gpu-a\",device=\"nvidia0\",modelName=\"NVIDIA H100 80GB HBM3\",Hostname=\"node-gpu-01\",simulation=\"true\"} 48
DCGM_FI_DEV_GPU_TEMP{gpu=\"1\",UUID=\"gpu-b\",device=\"nvidia1\",modelName=\"NVIDIA L40S\",Hostname=\"node-gpu-02\",simulation=\"true\"} 42.5
";
        let parsed = parse_dcgm_series(text);
        assert_eq!(parsed.len(), 2, "two GPUs parsed");
        let (xid, temp, fb_mib) = parsed["gpu-a"];
        assert_eq!(xid, 48.0);
        assert_eq!(temp, 94.0);
        assert_eq!(fb_mib, 4915.2);
        let (_, temp_b, _) = parsed["gpu-b"];
        assert_eq!(temp_b, 42.5);
    }

    #[test]
    fn ignores_unparseable_and_non_device_lines() {
        let text = "\
DCGM_FI_PROF_PCIE_TX_BYTES{UUID=\"gpu-a\",simulation=\"true\"} 1.2e6
DCGM_FI_DEV_GPU_TEMP{UUID=\"gpu-b\"} not-a-number
garbage line without braces 5
";
        let parsed = parse_dcgm_series(text);
        // Only DCGM_FI_DEV_* families are interpreted; PROF_* lines and
        // unparseable values never create entries.
        assert!(parsed.is_empty());
    }
}
