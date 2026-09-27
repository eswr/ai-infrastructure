//! Placement (phase 6A/6C) — *can this request run on this accelerator at
//! all?*, answered BEFORE dispatch, the way a real fleet scheduler does.
//!
//! The pipeline the gateway applies when a request carries
//! `placement` requirements:
//!
//! ```text
//! compatible accelerator?  ->  enough memory?  ->  lifecycle READY?
//!   ->  node accepts placements?  ->  cache locality?  ->  ECT  ->  contention
//! ```
//!
//! The first three gates live here (the rest are the existing router).
//!
//! MEMORY ESTIMATE — planning-grade, not kernel accounting:
//!
//! ```text
//! weights_gb   = params_B × weight_bits / 8 × overhead (1.02)
//! kv_gb        = kv_gb_per_1k_tokens × context_tokens / 1000
//! per_gpu      = (weights/tp + kv/tp + runtime_overhead + 0.25×(tp−1))
//!                × safety_margin (1.05)
//! ```
//!
//! with TP (tensor parallel) splitting weights and KV across N ranks and
//! paying per-rank context overhead. A model already RESIDENT on a GPU
//! skips the weights term (it is loaded; only KV is new) — which is also
//! what makes a CORDONED node able to keep serving what it hosts.
//!
//! The model registry is PLANNING DATA, not measurement: plausible
//! architectures and quantizations for the simulated fleet, keyed by
//! lowercase name fragments with a size/quant heuristic fallback. An
//! explicit `model_memory_gb` requirement always overrides it. The honest
//! scope statement: no CUDA kernel, HBM bandwidth, or NCCL number produced
//! here is a measurement — see the README's scope note.
//!
//! Backward compatibility is a hard rule: requests WITHOUT `placement`
//! requirements skip this module entirely.

use crate::WorkerHandle;
use inference_lab_protocol::PlacementRequirements;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Model registry (planning data)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ModelEntry {
    /// Lowercase fragment the entry keys on (e.g. "llama-3.1-70b").
    pub fragment: &'static str,
    /// Billions of parameters.
    pub params_b: f64,
    pub weight_bits: f64,
    /// GB of KV cache per 1000 context tokens (architecture-derived).
    pub kv_gb_per_1k_tokens: f64,
    /// Runtime fixed overhead: CUDA context, activations, framework buffers.
    pub overhead_gb: f64,
}

pub const WEIGHT_OVERHEAD: f64 = 1.02;
pub const SAFETY_MARGIN: f64 = 1.05;

/// Suffix → weight bits. mlx `-4bit` style and GGUF `q4_k_m` style both
/// fold to the same planning number.
fn weight_bits_from_name(name: &str) -> Option<f64> {
    for (pat, bits) in [
        ("4bit", 4.0),
        ("q4", 4.0),
        ("int4", 4.0),
        ("8bit", 8.0),
        ("q8", 8.0),
        ("int8", 8.0),
        ("fp8", 8.0),
        ("fp16", 16.0),
        ("f16", 16.0),
        ("bf16", 16.0),
        ("fp32", 32.0),
        ("f32", 32.0),
    ] {
        if name.contains(pat) {
            return Some(bits);
        }
    }
    None
}

/// KV-cache planning constant by parameter bucket (GQA-era architectures):
/// smaller models either have few KV heads or short typical contexts; this
/// is deliberately coarse — the registry entries above it carry real values.
fn kv_gb_per_1k_for_params(params_b: f64) -> f64 {
    if params_b <= 2.0 {
        0.03
    } else if params_b <= 9.0 {
        0.26
    } else if params_b <= 16.0 {
        0.35
    } else if params_b <= 40.0 {
        0.45
    } else {
        0.55
    }
}

/// The lab's simulated fleet models — explicit entries so the placement
/// matrix in `run_fleet_check` is stable and reviewable.
pub const REGISTRY: &[ModelEntry] = &[
    ModelEntry {
        fragment: "mock-gpu",
        params_b: 0.05,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.001,
        overhead_gb: 0.05,
    },
    ModelEntry {
        fragment: "qwen3.5-9b",
        params_b: 9.0,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.15,
        overhead_gb: 0.8,
    },
    ModelEntry {
        fragment: "llama-3.1-8b",
        params_b: 8.0,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.26,
        overhead_gb: 1.5,
    },
    ModelEntry {
        fragment: "qwen2.5-7b",
        params_b: 7.0,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.06,
        overhead_gb: 1.5,
    },
    ModelEntry {
        fragment: "qwen2.5-32b",
        params_b: 32.0,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.42,
        overhead_gb: 1.8,
    },
    ModelEntry {
        fragment: "llama-3.1-70b",
        params_b: 70.0,
        weight_bits: 4.0,
        kv_gb_per_1k_tokens: 0.52,
        overhead_gb: 2.0,
    },
];

/// Resolve a model name to planning numbers. Explicit registry fragments
/// win; otherwise parse size ("70b", "0.6b") + quant ("q4"/"fp16"/"-4bit")
/// out of the name; else `None` (the caller must supply model_memory_gb).
pub fn lookup_model(model: &str) -> Option<ModelEntry> {
    let name = model.to_lowercase();
    for e in REGISTRY {
        if name.contains(e.fragment) {
            let mut entry = *e;
            // The entry pins the family's QUANTIZED default; a suffix from
            // the concrete name overrides (…-70b-fp16 is a 16-bit model).
            if let Some(bits) = weight_bits_from_name(&name) {
                if (bits - entry.weight_bits).abs() > f64::EPSILON {
                    entry.weight_bits = bits;
                }
            }
            return Some(entry);
        }
    }
    // Heuristic fallback: scan for a params token ("70b", "0.6b").
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut j = i;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'b') && (j + 1 == bytes.len() || bytes[j + 1] == b'-')
            {
                let params: f64 = name[start..j].parse().ok()?;
                return Some(ModelEntry {
                    fragment: "(parsed)",
                    params_b: params,
                    weight_bits: weight_bits_from_name(&name).unwrap_or(16.0),
                    kv_gb_per_1k_tokens: kv_gb_per_1k_for_params(params),
                    overhead_gb: 1.5,
                });
            }
            i = j;
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub weights_gb: f64,
    pub kv_gb: f64,
    pub overhead_gb: f64,
    /// Per-GPU requirement INCLUDING the safety margin — the number the
    /// placement gate compares against one GPU's VRAM.
    pub required_per_gpu_gb: f64,
}

impl MemoryEstimate {
    pub fn total_gb(&self) -> f64 {
        (self.weights_gb + self.kv_gb + self.overhead_gb) * SAFETY_MARGIN
    }
}

/// Estimate per-GPU memory for `model` at `context_tokens` with
/// `tensor_parallel_size` ranks. `resident` skips the weights term (the
/// model is already loaded on this GPU; only KV is new). `explicit_gb` (a
/// caller-supplied total model-memory figure) overrides the registry
/// entirely.
pub fn estimate_memory(
    model: &str,
    context_tokens: u64,
    tensor_parallel_size: u32,
    resident: bool,
    explicit_gb: Option<f64>,
) -> Option<MemoryEstimate> {
    let tp = tensor_parallel_size.max(1) as f64;
    let context = context_tokens.max(1) as f64 / 1000.0;
    if let Some(gb) = explicit_gb {
        // Caller knows better: treat the figure as the total weights+KV
        // budget and still split across ranks, overhead per rank on top.
        let est = MemoryEstimate {
            weights_gb: if resident { 0.0 } else { gb },
            kv_gb: 0.0,
            overhead_gb: 1.5 + 0.25 * (tp - 1.0),
            required_per_gpu_gb: 0.0,
        };
        let per_gpu = MemoryEstimate {
            required_per_gpu_gb: (est.weights_gb / tp + est.kv_gb + est.overhead_gb)
                * SAFETY_MARGIN,
            ..est
        };
        return Some(per_gpu);
    }
    let entry = lookup_model(model)?;
    let weights_total = if resident {
        0.0
    } else {
        entry.params_b * entry.weight_bits / 8.0 * WEIGHT_OVERHEAD
    };
    let kv_total = entry.kv_gb_per_1k_tokens * context;
    let overhead = entry.overhead_gb + 0.25 * (tp - 1.0);
    Some(MemoryEstimate {
        weights_gb: weights_total,
        kv_gb: kv_total,
        overhead_gb: overhead,
        required_per_gpu_gb: (weights_total / tp + kv_total / tp + overhead) * SAFETY_MARGIN,
    })
}

// ---------------------------------------------------------------------------
// The placement gate
// ---------------------------------------------------------------------------

/// Why a candidate was refused, with stable metric-label strings.
#[derive(Debug, Clone)]
pub enum PlacementReject {
    /// The worker reports no accelerator/VRAM capability, so the gate
    /// cannot verify the request fits — it must refuse.
    NoCapability,
    /// Unknown model, no explicit memory — refusing is the safe default.
    NoEstimate,
    AcceleratorMismatch { required: String, have: String },
    InsufficientMemory { required_gb: f64, available_gb: f64 },
    /// tp > 1 requested but no group of matching healthy GPUs exists.
    NoTensorParallelGroup { tp: u32 },
    /// The worker's node is draining/maintenance/failed/provisioning.
    NodeNotPlacable { node: String, state: &'static str },
    /// Node is CORDONED and this model is not resident on the worker's GPU
    /// — new model loads are refused; resident traffic continues.
    CordonedNewPlacement { node: String },
}

impl PlacementReject {
    /// Prometheus label for `inference_placement_rejections_total`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NoCapability => "no_capability",
            Self::NoEstimate => "no_estimate",
            Self::AcceleratorMismatch { .. } => "accelerator_mismatch",
            Self::InsufficientMemory { .. } => "insufficient_memory",
            Self::NoTensorParallelGroup { .. } => "no_tensor_parallel_group",
            Self::NodeNotPlacable { .. } => "node_not_placable",
            Self::CordonedNewPlacement { .. } => "cordoned_new_placement",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Self::NoCapability => {
                "worker reports no accelerator/VRAM capability".into()
            }
            Self::NoEstimate => "unknown model and no explicit model_memory_gb".into(),
            Self::AcceleratorMismatch { required, have } => {
                format!("needs {required}, worker has {have}")
            }
            Self::InsufficientMemory { required_gb, available_gb } => {
                format!("needs {required_gb:.1} GB, GPU has {available_gb:.1} GB")
            }
            Self::NoTensorParallelGroup { tp } => format!("no TP-{tp} group available"),
            Self::NodeNotPlacable { node, state } => format!("node {node} is {state}"),
            Self::CordonedNewPlacement { node } => {
                format!("node {node} cordoned: model not resident on this GPU")
            }
        }
    }
}

/// Outcome of applying the placement gate to the healthy candidate set.
pub struct PlacementOutcome {
    /// Workers that may receive the request (for TP requests: the group
    /// LEADERS — dispatch goes to one rank; the group is recorded on
    /// `tp_group`).
    pub candidates: Vec<Arc<WorkerHandle>>,
    /// (worker-id, reject) pairs for everything refused — telemetry.
    pub rejections: Vec<(String, PlacementReject)>,
    /// The tensor-parallel group a TP request was admitted to, if any.
    pub tp_group: Option<String>,
}

/// Node-gate states the gateway's fleet controller keeps on each handle
/// (mirrors `inference_lab_worker_lifecycle::NodeState`'s placement rule).
pub const NODE_GATE_OPEN: u8 = 0; // node READY
pub const NODE_GATE_RESIDENT_ONLY: u8 = 1; // node CORDONED
pub const NODE_GATE_CLOSED: u8 = 2; // DRAINING / MAINTENANCE / FAILED / ...

/// Apply the placement pipeline. `reqs` comes straight from the request;
/// when it carries no requirements, use [`PlacementOutcome::pass_all`]
/// semantics instead (the gateway skips calling this entirely).
pub fn filter(
    workers: &[Arc<WorkerHandle>],
    model: &str,
    reqs: &PlacementRequirements,
) -> PlacementOutcome {
    let tp = reqs.tensor_parallel_size.unwrap_or(1);
    let mut rejections = Vec::new();

    if tp > 1 {
        return filter_tp_group(workers, model, reqs, tp, &mut rejections);
    }

    let mut candidates = Vec::new();
    for w in workers {
        match check_single(w, model, reqs) {
            Ok(_) => candidates.push(w.clone()),
            Err(r) => rejections.push((w.id.clone(), r)),
        }
    }
    PlacementOutcome { candidates, rejections, tp_group: None }
}

/// Everything passes — the no-requirements path.
pub fn pass_all(workers: &[Arc<WorkerHandle>]) -> PlacementOutcome {
    PlacementOutcome {
        candidates: workers.to_vec(),
        rejections: Vec::new(),
        tp_group: None,
    }
}

fn check_accelerator(w: &WorkerHandle, reqs: &PlacementRequirements) -> Result<(), PlacementReject> {
    if let Some(want) = &reqs.accelerator {
        let have = w.accelerator.as_deref().unwrap_or("none");
        if have != want {
            return Err(PlacementReject::AcceleratorMismatch {
                required: want.clone(),
                have: have.to_string(),
            });
        }
    }
    Ok(())
}

/// Node gate: closed gates refuse; RESIDENT_ONLY gates refuse models the
/// GPU does not already host.
fn check_node_gate(w: &WorkerHandle, model: &str) -> Result<bool, PlacementReject> {
    match w.node_gate() {
        NODE_GATE_CLOSED => Err(PlacementReject::NodeNotPlacable {
            node: w.node_id.clone().unwrap_or_else(|| "?".into()),
            state: w.node_state_label(),
        }),
        NODE_GATE_RESIDENT_ONLY => {
            if w.model_loaded(model) {
                Ok(true) // resident: may serve
            } else {
                Err(PlacementReject::CordonedNewPlacement {
                    node: w.node_id.clone().unwrap_or_else(|| "?".into()),
                })
            }
        }
        _ => Ok(false),
    }
}

/// One-GPU placement check. Returns `Ok(resident)` where `resident` says
/// the model is already loaded on this GPU (the caller may care).
fn check_single(
    w: &WorkerHandle,
    model: &str,
    reqs: &PlacementRequirements,
) -> Result<bool, PlacementReject> {
    if w.vram_gb.is_none() {
        // No capability report -> no way to verify the placement -> refuse.
        return Err(PlacementReject::NoCapability);
    }
    check_accelerator(w, reqs)?;
    let resident = w.model_loaded(model);
    let cordoned_only = check_node_gate(w, model);
    if let Some(vram) = w.vram_gb {
        let est = estimate_memory(
            model,
            reqs.context_length.unwrap_or(4096),
            1,
            resident,
            reqs.model_memory_gb,
        )
        .ok_or(PlacementReject::NoEstimate)?;
        if est.required_per_gpu_gb > vram {
            return Err(PlacementReject::InsufficientMemory {
                required_gb: est.required_per_gpu_gb,
                available_gb: vram,
            });
        }
    }
    // Node gate AFTER the memory check so cordoned rejections name the real
    // reason (resident vs not) and closed gates still win.
    cordoned_only?;
    Ok(resident)
}

fn filter_tp_group(
    workers: &[Arc<WorkerHandle>],
    model: &str,
    reqs: &PlacementRequirements,
    tp: u32,
    rejections: &mut Vec<(String, PlacementReject)>,
) -> PlacementOutcome {
    // Group healthy candidates by their tp-group label.
    let mut groups: HashMap<&str, Vec<&Arc<WorkerHandle>>> = HashMap::new();
    for w in workers {
        match &w.tp_group {
            Some(g) => groups.entry(g.as_str()).or_default().push(w),
            None => rejections.push((
                w.id.clone(),
                PlacementReject::NoTensorParallelGroup { tp },
            )),
        }
    }

    let mut eligible: Vec<(&str, Vec<&Arc<WorkerHandle>>)> = Vec::new();
    for (group, members) in groups {
        if (members.len() as u32) < tp {
            continue;
        }
        // Every rank must pass accelerator + node gates + per-GPU memory
        // (weights and KV split across ranks; overhead is per-rank).
        let mut ok = true;
        for w in members.iter().take(tp as usize) {
            if let Err(r) = check_accelerator(w, reqs) {
                rejections.push((w.id.clone(), r));
                ok = false;
                break;
            }
            if let Err(r) = check_node_gate(w, model) {
                rejections.push((w.id.clone(), r));
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let resident = members[0].model_loaded(model);
        let Some(est) = estimate_memory(
            model,
            reqs.context_length.unwrap_or(4096),
            tp,
            resident,
            reqs.model_memory_gb,
        ) else {
            continue;
        };
        if members
            .iter()
            .take(tp as usize)
            .any(|w| w.vram_gb.is_some_and(|v| est.required_per_gpu_gb > v))
        {
            for w in members.iter().take(tp as usize) {
                rejections.push((
                    w.id.clone(),
                    PlacementReject::InsufficientMemory {
                        required_gb: est.required_per_gpu_gb,
                        available_gb: w.vram_gb.unwrap_or(0.0),
                    },
                ));
            }
            continue;
        }
        eligible.push((group, members));
    }

    if eligible.is_empty() {
        return PlacementOutcome {
            candidates: Vec::new(),
            rejections: std::mem::take(rejections),
            tp_group: None,
        };
    }

    // Admission, not optimization: deterministic group choice (smallest
    // name) keeps the simulation reviewable. Dispatch lands on the first
    // rank; the group travels on the outcome for telemetry.
    eligible.sort_by(|a, b| a.0.cmp(b.0));
    let (group, members) = eligible.remove(0);
    PlacementOutcome {
        candidates: vec![members[0].clone()],
        rejections: std::mem::take(rejections),
        tp_group: Some(group.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_lab_protocol::PlacementRequirements;

    fn gpu(id: &str, accelerator: &str, vram: f64) -> WorkerHandle {
        let mut w = WorkerHandle::new(id, format!("http://{id}"), "mock", "mock");
        w.accelerator = Some(accelerator.to_string());
        w.vram_gb = Some(vram);
        w
    }

    fn reqs(context: u64) -> PlacementRequirements {
        PlacementRequirements {
            model_memory_gb: None,
            context_length: Some(context),
            tensor_parallel_size: None,
            accelerator: None,
        }
    }

    #[test]
    fn size_classes_route_to_the_right_gpus() {
        let fleet = vec![
            Arc::new(gpu("a10-01", "nvidia-a10", 24.0)),
            Arc::new(gpu("l40s-01", "nvidia-l40s", 48.0)),
            Arc::new(gpu("h100-01", "nvidia-h100", 80.0)),
        ];
        // 8B q4 (~6 GB): everywhere.
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs(4096));
        assert_eq!(out.candidates.len(), 3, "{:?}", out.rejections);
        // 70B q4 (~40 GB): L40S + H100, not the A10.
        let out = filter(&fleet, "llama-3.1-70b-instruct-q4", &reqs(4096));
        assert_eq!(out.candidates.len(), 2, "{:?}", out.rejections);
        assert!(out.rejections.iter().any(|(id, r)| id == "a10-01"
            && matches!(r, PlacementReject::InsufficientMemory { .. })));
        // 70B fp16 (~148 GB): nowhere single-GPU.
        let out = filter(&fleet, "llama-3.1-70b-instruct-fp16", &reqs(4096));
        assert!(out.candidates.is_empty());
        assert_eq!(out.rejections.len(), 3);
    }

    #[test]
    fn kv_estimate_rejects_absurd_contexts() {
        let fleet = vec![Arc::new(gpu("h100-01", "nvidia-h100", 80.0))];
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs(400_000));
        assert!(out.candidates.is_empty(), "260 GB of KV cannot fit 80 GB");
    }

    #[test]
    fn explicit_memory_overrides_the_registry() {
        let fleet = vec![Arc::new(gpu("a10-01", "nvidia-a10", 24.0))];
        let reqs = PlacementRequirements {
            model_memory_gb: Some(20.0),
            context_length: Some(4096),
            tensor_parallel_size: None,
            accelerator: None,
        };
        let out = filter(&fleet, "totally-unknown-model", &reqs);
        assert_eq!(out.candidates.len(), 1);
        let reqs = PlacementRequirements { model_memory_gb: Some(30.0), ..reqs };
        let out = filter(&fleet, "totally-unknown-model", &reqs);
        assert!(out.candidates.is_empty());
    }

    #[test]
    fn unknown_model_without_estimate_is_refused() {
        let fleet = vec![Arc::new(gpu("h100-01", "nvidia-h100", 80.0))];
        let out = filter(&fleet, "mystery-model", &reqs(4096));
        assert!(out.candidates.is_empty());
        assert_eq!(out.rejections[0].1.label(), "no_estimate");
    }

    #[test]
    fn accelerator_mismatch_is_named() {
        let fleet = vec![Arc::new(gpu("a10-01", "nvidia-a10", 24.0))];
        let reqs_h100 = PlacementRequirements {
            accelerator: Some("nvidia-h100".into()),
            ..reqs(4096)
        };
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs_h100);
        assert!(out.candidates.is_empty());
        assert_eq!(out.rejections[0].1.label(), "accelerator_mismatch");
    }

    #[test]
    fn capability_less_workers_refuse_placement_requests() {
        // The generic phase-5 mock pool reports no accelerator/VRAM: it
        // cannot PROVE a placement fits, so it must refuse.
        let fleet = vec![
            Arc::new(WorkerHandle::new("plain-mock", "http://x", "mock", "mock")),
            Arc::new(gpu("a10-01", "nvidia-a10", 24.0)),
        ];
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs(4096));
        assert_eq!(out.candidates.len(), 1);
        assert_eq!(out.candidates[0].id, "a10-01");
        assert_eq!(out.rejections[0].1.label(), "no_capability");
    }

    #[test]
    fn cordoned_node_serves_resident_models_only() {
        let mut w = gpu("l40s-01", "nvidia-l40s", 48.0);
        w.node_id = Some("node-gpu-02".into());
        w.set_node_gate(NODE_GATE_RESIDENT_ONLY, "cordoned");
        let fleet = vec![Arc::new(w)];
        // Resident model passes (weights skipped anyway).
        fleet[0].note_model_loaded("llama-3.1-8b-instruct-q4");
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs(4096));
        assert_eq!(out.candidates.len(), 1);
        // A different model = a NEW placement = refused on the cordoned node.
        let out = filter(&fleet, "qwen2.5-32b-instruct-q4", &reqs(4096));
        assert!(out.candidates.is_empty());
        assert_eq!(out.rejections[0].1.label(), "cordoned_new_placement");
    }

    #[test]
    fn closed_node_gate_refuses_everything() {
        let mut w = gpu("h100-01", "nvidia-h100", 80.0);
        w.node_id = Some("node-gpu-01".into());
        w.set_node_gate(NODE_GATE_CLOSED, "failed");
        let fleet = vec![Arc::new(w)];
        let out = filter(&fleet, "llama-3.1-8b-instruct-q4", &reqs(4096));
        assert_eq!(out.rejections[0].1.label(), "node_not_placable");
    }

    #[test]
    fn tensor_parallel_finds_one_group() {
        let mut h1 = gpu("h100-01", "nvidia-h100", 80.0);
        let mut h2 = gpu("h100-02", "nvidia-h100", 80.0);
        h1.tp_group = Some("tg-h100-01".into());
        h2.tp_group = Some("tg-h100-01".into());
        let fleet = vec![Arc::new(h1), Arc::new(h2), Arc::new(gpu("a10", "nvidia-a10", 24.0))];
        let reqs2 = PlacementRequirements {
            tensor_parallel_size: Some(2),
            ..reqs(4096)
        };
        let out = filter(&fleet, "llama-3.1-70b-instruct-fp16", &reqs2);
        assert_eq!(out.tp_group.as_deref(), Some("tg-h100-01"));
        assert_eq!(out.candidates.len(), 1, "dispatch lands on the first rank");
        // TP-4: the pair is not enough.
        let reqs4 = PlacementRequirements {
            tensor_parallel_size: Some(4),
            ..reqs(4096)
        };
        let out = filter(&fleet, "llama-3.1-70b-instruct-fp16", &reqs4);
        assert!(out.candidates.is_empty());
    }

    #[test]
    fn resident_model_skips_the_weights_term() {
        let mut w = gpu("a10-01", "nvidia-a10", 24.0);
        w.note_model_loaded("qwen2.5-32b-instruct-q4");
        let fleet = vec![Arc::new(w)];
        // Cold: 32B q4 (~17 GB + overhead) fits a 24 GB A10 barely; with a
        // longer context it does not.
        let out = filter(&fleet, "qwen2.5-32b-instruct-q4", &reqs(200_000));
        assert!(out.candidates.is_empty(), "KV alone must blow the budget");
        // But a NON-resident longer-context request was rejected BEFORE the
        // gate — residency would not have saved it either way here; assert
        // the reason is memory, not the node gate.
        assert_eq!(out.rejections[0].1.label(), "insufficient_memory");
    }
}
