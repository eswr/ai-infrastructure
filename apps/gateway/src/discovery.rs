//! Kubernetes worker discovery (phase 5).
//!
//! In `WORKER_DISCOVERY=kubernetes` mode the gateway does not get a static
//! `WORKERS` list. Instead it watches, in-cluster:
//!
//! * **Pods** carrying the worker label selector — existence, IP, and the
//!   capability labels a worker registers with:
//!   `inference-lab.io/runtime`, `inference-lab.io/model`. The pod spec IS
//!   the registration; there is no side-channel to hard-code.
//! * **Endpoints** of the matching Services — the "in service" view. This
//!   is OBSERVED, never obeyed: when a worker fails its functional probe,
//!   the gateway's lifecycle controller marks it UNHEALTHY, the pod's
//!   readiness (which this same gateway answers — see
//!   `/admin/workers/{id}/readiness`) goes false, and the pod drops out of
//!   Endpoints. If discovery removed the registry entry on that alone, the
//!   gateway would lose the IP it needs to keep probing — and a worker
//!   could never walk UNHEALTHY → RECOVERING → WARMING → READY and re-enter
//!   service. So a NotReady pod stays registered; only a pod whose object
//!   is gone (or terminating: `deletionTimestamp` set) is drained and
//!   removed. The lifecycle controller stays authoritative for routing.
//!
//! Implementation is a poll loop (list, diff, apply) rather than a watch
//! stream: at lab scale the list is a few pods, a 1–2 s poll is faster than
//! the functional-probe cadence that actually gates traffic, and there is
//! no resourceVersion/410 machinery to get wrong. The API surface used
//! (pods, endpoints) is the same one a watch would use.

use crate::AppState;
use inference_lab_scheduler::calibration::RuntimeCalibration;
use inference_lab_scheduler::WorkerHandle;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Pod labels a worker registers its capabilities with.
pub const LABEL_WORKER: &str = "inference-lab.io/worker";
pub const LABEL_RUNTIME: &str = "inference-lab.io/runtime";
pub const LABEL_MODEL: &str = "inference-lab.io/model";
// Phase 6A: heterogeneous-fleet capability labels. The label set IS the
// worker's registration — the gateway infers nothing it was not told.
pub const LABEL_NODE: &str = "inference-lab.io/node";
pub const LABEL_ACCELERATOR: &str = "inference-lab.io/accelerator";
pub const LABEL_VRAM_GB: &str = "inference-lab.io/vram-gb";
pub const LABEL_GPU_UUID: &str = "inference-lab.io/gpu-uuid";
pub const LABEL_TP_GROUP: &str = "inference-lab.io/tp-group";

const SA_DIR: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub namespace: String,
    pub selector: String,
    pub poll_s: f64,
    /// Fallback worker port when an Endpoints subset carries no named port.
    pub port: u16,
    pub port_name: String,
}

impl DiscoveryConfig {
    /// `WORKER_DISCOVERY=kubernetes` turns the mode on; everything else has
    /// a lab-sane default.
    pub fn from_env() -> anyhow::Result<Self> {
        let namespace = match std::env::var("K8S_NAMESPACE") {
            Ok(ns) if !ns.trim().is_empty() => ns,
            _ => std::fs::read_to_string(format!("{SA_DIR}/namespace"))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "default".to_string()),
        };
        Ok(Self {
            namespace,
            selector: std::env::var("WORKER_SELECTOR")
                .unwrap_or_else(|_| format!("{LABEL_WORKER}=true")),
            poll_s: crate::env_f64("WORKER_DISCOVERY_POLL_S", 2.0).clamp(0.5, 60.0),
            port: std::env::var("WORKER_PORT")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(8080),
            port_name: std::env::var("WORKER_PORT_NAME").unwrap_or_else(|_| "http".into()),
        })
    }

    pub fn enabled() -> bool {
        std::env::var("WORKER_DISCOVERY").ok().as_deref() == Some("kubernetes")
    }
}

// ---------------------------------------------------------------------------
// Minimal in-cluster API client (list-only; the scaler does the writes)
// ---------------------------------------------------------------------------

pub struct K8sClient {
    http: reqwest::Client,
    base: String,
}

impl K8sClient {
    /// ServiceAccount bearer token + cluster CA, exactly how every
    /// in-cluster client bootstraps — hand-rolled here because one GET
    /// endpoint does not justify the `kube` crate.
    pub fn in_cluster() -> anyhow::Result<Self> {
        let token = std::fs::read_to_string(format!("{SA_DIR}/token"))?
            .trim()
            .to_string();
        let ca = std::fs::read(format!("{SA_DIR}/ca.crt"))?;
        let base = match (
            std::env::var("KUBERNETES_SERVICE_HOST"),
            std::env::var("KUBERNETES_SERVICE_PORT"),
        ) {
            (Ok(h), Ok(p)) if !h.is_empty() && !p.is_empty() => format!("https://{h}:{p}"),
            _ => "https://kubernetes.default.svc".to_string(),
        };
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))?,
        );
        let http = reqwest::Client::builder()
            .add_root_certificate(reqwest::Certificate::from_pem(&ca)?)
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .build()?;
        Ok(Self { http, base })
    }

    async fn get_json(&self, path: &str) -> anyhow::Result<Value> {
        let resp = self.http.get(format!("{}{path}", self.base)).send().await?;
        let resp = resp.error_for_status()?;
        Ok(resp.json().await?)
    }

    /// All pods in `ns` whose labels match `selector` (URL-encoded).
    pub async fn list_pods(&self, ns: &str, selector: &str) -> anyhow::Result<Value> {
        self.get_json(&format!(
            "/api/v1/namespaces/{ns}/pods?labelSelector={}",
            encode_query(selector)
        ))
        .await
    }

    /// Endpoints of every Service in `ns` matching `selector`.
    pub async fn list_endpoints(&self, ns: &str, selector: &str) -> anyhow::Result<Value> {
        self.get_json(&format!(
            "/api/v1/namespaces/{ns}/endpoints?labelSelector={}",
            encode_query(selector)
        ))
        .await
    }
}

/// Percent-encode a query value without pulling a url crate: everything
/// outside the RFC 3986 unreserved set.
fn encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Shared snapshot for gauges and /status
// ---------------------------------------------------------------------------

/// Which registry entries their Service currently lists as ready endpoints,
/// and which workers discovery OWNS. `managed` matters: EXTERNAL_WORKERS
/// are registered statically and share the registry — the sync loop must
/// never drain-and-remove a worker it did not discover (pods come and go;
/// the native Metal worker does not).
#[derive(Default)]
pub struct DiscoveryState {
    in_service: Mutex<HashSet<String>>,
    managed: Mutex<HashSet<String>>,
}

impl DiscoveryState {
    pub fn set_in_service(&self, ids: &HashSet<String>) {
        *self.in_service.lock().unwrap_or_else(|e| e.into_inner()) = ids.clone();
    }

    pub fn in_service(&self, id: &str) -> bool {
        self.in_service
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(id)
    }

    fn own(&self, id: &str) {
        self.managed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.to_string());
    }

    fn managed(&self) -> HashSet<String> {
        self.managed.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

// ---------------------------------------------------------------------------
// The sync loop
// ---------------------------------------------------------------------------

/// One desired registry entry, derived from a pod.
struct Discovered {
    endpoint: String,
    runtime: String,
    model: String,
    node: Option<String>,
    accelerator: Option<String>,
    vram_gb: Option<f64>,
    gpu_uuid: Option<String>,
    tp_group: Option<String>,
}

pub async fn discovery_loop(st: Arc<AppState>, cfg: DiscoveryConfig) {
    let client = match K8sClient::in_cluster() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                error = %e,
                "kubernetes discovery: no in-cluster credentials — no workers \
                 will ever register (static mode via WORKERS is the alternative)"
            );
            return;
        }
    };
    tracing::info!(
        namespace = %cfg.namespace,
        selector = %cfg.selector,
        poll_s = cfg.poll_s,
        "kubernetes discovery watching pods + endpoints"
    );
    let mut tick = tokio::time::interval(Duration::from_secs_f64(cfg.poll_s));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        match sync(&st, &client, &cfg).await {
            Ok((added, removed, updated)) if added + removed + updated > 0 => {
                tracing::info!(added, removed, updated, "discovery sync applied");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "discovery sync failed; retrying next poll")
            }
        }
    }
}

/// List, diff, apply. Idempotent by construction: every step is a pure
/// function of (cluster state, current registry).
async fn sync(
    st: &Arc<AppState>,
    client: &K8sClient,
    cfg: &DiscoveryConfig,
) -> anyhow::Result<(u64, u64, u64)> {
    let pods = client.list_pods(&cfg.namespace, &cfg.selector).await?;
    let endpoints = client.list_endpoints(&cfg.namespace, &cfg.selector).await?;

    // Ready endpoint targets of the matching Services. The endpoints API
    // splits addresses into ready `addresses` vs `notReadyAddresses` — we
    // count only the former, and take the worker port from the Service's
    // named port (WORKER_PORT_NAME), falling back to WORKER_PORT.
    let mut in_service: HashSet<String> = HashSet::new();
    let mut service_ports: HashMap<String, u16> = HashMap::new();
    for ep in items(&endpoints) {
        for subset in arr(&ep["subsets"]) {
            let port = arr(&subset["ports"])
                .iter()
                .find(|p| p["name"].as_str() == Some(cfg.port_name.as_str()))
                .and_then(|p| p["port"].as_u64())
                .map(|p| p as u16);
            for addr in arr(&subset["addresses"]) {
                if let Some(name) = addr["targetRef"]["name"].as_str() {
                    in_service.insert(name.to_string());
                    if let Some(p) = port {
                        service_ports.insert(name.to_string(), p);
                    }
                }
            }
        }
    }
    st.discovery.set_in_service(&in_service);

    // Desired registry: Running pods with an IP, not terminating.
    let mut desired: HashMap<String, Discovered> = HashMap::new();
    for pod in items(&pods) {
        let Some(name) = pod["metadata"]["name"].as_str() else { continue };
        let terminating = pod["metadata"]["deletionTimestamp"].is_string();
        let running = pod["status"]["phase"].as_str() == Some("Running");
        let ip = pod["status"]["podIP"].as_str().unwrap_or("");
        if terminating || !running || ip.is_empty() {
            continue;
        }
        let labels = &pod["metadata"]["labels"];
        let runtime = labels[LABEL_RUNTIME]
            .as_str()
            .or_else(|| name.split('-').next())
            .unwrap_or("unknown")
            .to_string();
        let model = labels[LABEL_MODEL]
            .as_str()
            .unwrap_or("default")
            .to_string();
        let gpu_uuid = labels[LABEL_GPU_UUID]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| pod["metadata"]["uid"].as_str().map(|s| s.to_string()));
        let d = Discovered {
            endpoint: format!(
                "http://{ip}:{port}",
                port = service_ports.get(name).copied().unwrap_or(cfg.port)
            ),
            runtime,
            model,
            node: labels[LABEL_NODE].as_str().map(|s| s.to_string()),
            accelerator: labels[LABEL_ACCELERATOR].as_str().map(|s| s.to_string()),
            vram_gb: labels[LABEL_VRAM_GB].as_str().and_then(|v| v.parse().ok()),
            gpu_uuid,
            tp_group: labels[LABEL_TP_GROUP].as_str().map(|s| s.to_string()),
        };
        desired.insert(name.to_string(), d);
    }

    let current = st.router.workers();
    let managed = st.discovery.managed();
    let (mut added, mut removed, mut updated) = (0u64, 0u64, 0u64);

    for (id, d) in &desired {
        if let Some(w) = current.iter().find(|w| &w.id == id) {
            if w.endpoint != d.endpoint {
                // Same pod name, new IP (container restart): replace the
                // handle so probes and dispatches follow the address.
                st.router.remove_worker(id);
                if let Some(h) = build_handle(st, id, d) {
                    if st.router.add_worker(h) {
                        updated += 1;
                        tracing::info!(worker = %id, endpoint = %d.endpoint, "worker endpoint updated");
                    }
                }
            }
        } else if let Some(h) = build_handle(st, id, d) {
            let (runtime, endpoint) = (h.runtime.clone(), h.endpoint.clone());
            if st.router.add_worker(h) {
                st.discovery.own(id);
                added += 1;
                tracing::info!(worker = %id, %runtime, %endpoint, "worker registered from kubernetes");
            }
        }
    }

    for w in &current {
        // Only workers discovery owns follow the pods. EXTERNAL_WORKERS
        // (static, e.g. the native Metal runtime on the host) stay.
        if !managed.contains(&w.id) {
            continue;
        }
        if !desired.contains_key(&w.id) {
            // Pod gone or terminating: drain, then out of the registry.
            // In-flight streams hold their Arc and run to completion.
            let transitions = w.drain();
            st.metrics.record_transitions(w, &transitions);
            st.router.remove_worker(&w.id);
            removed += 1;
            tracing::info!(worker = %w.id, "worker removed: pod gone from kubernetes");
        }
    }

    if added > 0 {
        st.metrics.discovery_events_total.with_label_values(&["added"]).inc_by(added);
    }
    if removed > 0 {
        st.metrics.discovery_events_total.with_label_values(&["removed"]).inc_by(removed);
    }
    if updated > 0 {
        st.metrics.discovery_events_total.with_label_values(&["updated"]).inc_by(updated);
    }
    st.metrics.discovery_workers.set(st.router.workers().len() as i64);
    Ok((added, removed, updated))
}

fn items(v: &Value) -> Vec<&Value> {
    v["items"].as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

fn arr(v: &Value) -> Vec<&Value> {
    v.as_array().map(|a| a.iter().collect()).unwrap_or_default()
}

/// Calibration resolution for a discovered worker: reuse a matching server
/// calibration when one exists (external/native workers share it), else a
/// per-(runtime, model) seed-based model grown on demand.
#[allow(clippy::too_many_arguments)]
fn build_handle(st: &Arc<AppState>, id: &str, d: &Discovered) -> Option<Arc<WorkerHandle>> {
    let cal = st
        .calibrations
        .iter()
        .find(|c| {
            let p = c.cal.snapshot_profile();
            p.runtime == d.runtime && p.model == d.model
        })
        .map(|c| c.cal.clone())
        .unwrap_or_else(|| {
            let mut map = st.dyn_cals.lock().unwrap_or_else(|e| e.into_inner());
            map.entry(format!("{}\u{0}{}", d.runtime, d.model))
                .or_insert_with(|| {
                    Arc::new(RuntimeCalibration::from_seeds(
                        &st.cal_machine,
                        &d.runtime,
                        &d.model,
                        0,
                        st.service_defaults,
                    ))
                })
                .clone()
        });
    let mut handle = WorkerHandle::with_calibration(
        id,
        d.endpoint.clone(),
        d.model.clone(),
        d.runtime.clone(),
        cal,
    );
    handle.node_id = d.node.clone();
    handle.accelerator = d.accelerator.clone();
    handle.vram_gb = d.vram_gb;
    handle.gpu_uuid = d.gpu_uuid.clone();
    handle.tp_group = d.tp_group.clone();
    Some(Arc::new(handle))
}

#[cfg(test)]
mod tests {
    use super::encode_query;

    #[test]
    fn label_selectors_are_query_safe() {
        assert_eq!(encode_query("inference-lab.io/worker=true"), "inference-lab.io%2Fworker%3Dtrue");
        assert_eq!(encode_query("app=foo,bar baz"), "app%3Dfoo%2Cbar%20baz");
    }
}
