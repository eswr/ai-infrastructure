//! Cap-bounded ring of recent routing decisions for observability
//! consumers (the lab console's `GET /decisions`).
//!
//! **Diagnostic only.** Best-effort in-memory history: entries are
//! ephemeral, bounded, and not part of scheduler correctness or
//! persistence. Nothing in the control plane reads this back; ids are
//! monotonically increasing per process so stream consumers can dedup
//! and resynchronize after reconnect.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const CAP: usize = 64;

#[derive(Default)]
pub struct DecisionRing {
    next_id: AtomicU64,
    entries: Mutex<VecDeque<serde_json::Value>>,
}

impl DecisionRing {
    /// Insert a decision, stamp it with its id, return the id so the
    /// request path can attach the actual outcome later.
    pub fn push(&self, mut entry: serde_json::Value) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Some(obj) = entry.as_object_mut() {
            obj.insert("id".into(), serde_json::json!(id));
        }
        let mut q = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        q.push_back(entry);
        while q.len() > CAP {
            q.pop_front();
        }
        id
    }

    /// Backfill the measured first-token latency once the request's
    /// timeline closes. No-op if the entry already aged out of the ring.
    pub fn observe_outcome(&self, id: u64, actual_ttft_ms: f64) {
        let mut q = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        for e in q.iter_mut().rev() {
            if e.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(obj) = e.as_object_mut() {
                    obj.insert("actual_ttft_ms".into(), serde_json::json!(actual_ttft_ms));
                }
                break;
            }
        }
    }

    /// Chronological snapshot (oldest first).
    pub fn snapshot(&self) -> Vec<serde_json::Value> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_assigns_ids_and_caps() {
        let ring = DecisionRing::default();
        for i in 0..(CAP + 10) {
            let id = ring.push(serde_json::json!({ "i": i }));
            let snap = ring.snapshot();
            assert_eq!(snap.last().unwrap()["id"], serde_json::json!(id));
        }
        assert_eq!(ring.snapshot().len(), CAP);
        // Oldest entries aged out.
        assert_eq!(ring.snapshot()[0]["i"], serde_json::json!(10));
    }

    #[test]
    fn outcome_backfills_by_id() {
        let ring = DecisionRing::default();
        let id = ring.push(serde_json::json!({ "actual_ttft_ms": null }));
        ring.observe_outcome(id, 47800.0);
        assert_eq!(ring.snapshot()[0]["actual_ttft_ms"], serde_json::json!(47800.0));
        // Unknown/aged-out ids are a no-op.
        ring.observe_outcome(9999, 1.0);
        assert_eq!(ring.snapshot().len(), 1);
    }
}
