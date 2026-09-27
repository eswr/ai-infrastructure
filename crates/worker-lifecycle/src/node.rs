//! Node lifecycle (phase 6D) — the state machine ONE level above
//! [`crate::WorkerLifecycle`]: a node hosts many workers, and fleet
//! controllers think in node terms (cordon the node, drain its workers,
//! take it to maintenance, validate it back).
//!
//! ```text
//!                ┌──────────────┐  validated   ┌────────┐
//!                │ PROVISIONING │ ───────────► │ READY  │
//!                └──────────────┘              └───┬────┘
//!                       ▲    ▲        ┌──────────┼───────────┐
//!                       │    │        │          │           │
//!   all workers gone ◄──┘    │   cordon   drain    critical fault
//!   (lost node /             │        │      │        (XID, ECC,
//!    replacement)            │        ▼      ▼         lost contact)
//!                       ┌────┴─────┐  ┌────────┐ ┌────────┐
//!                       │ RECOVER- │  │CORDONED│ │DRAINING│
//!                       │   ING    │  └───┬────┘ └───┬────┘
//!                       └────┬─────┘      │          │ all workers
//!                            │            │          ▼ drained
//!              validation    │            │    ┌─────────────┐
//!              passes ───────┤            │    │ MAINTENANCE │
//!              fails → FAILED│            │    └─────────────┘
//!                            ▼            │          │
//!                       ┌─────────┐       │  return to service
//!                       │ FAILED  │ ◄─────┼──── (via RECOVERING)
//!                       └─────────┘ fault │
//!                            │  escalation│
//!                            └──► RECOVERING (fault cleared)
//! ```
//!
//! The policy semantics that matter to the scheduler:
//!
//! | state        | new placements | existing traffic       |
//! |--------------|----------------|------------------------|
//! | READY        | yes            | yes                    |
//! | CORDONED     | no (resident-only) | yes (that is the point) |
//! | DRAINING     | no             | finishing              |
//! | MAINTENANCE  | no             | none left              |
//! | FAILED       | no             | drained                |
//! | RECOVERING   | no             | only after validation  |
//! | PROVISIONING | no             | no workers to serve    |
//!
//! CORDONED deserves its asymmetry: a hot node keeps serving the models
//! already resident on its GPUs while refusing *new* model loads — the
//! "high temperature" response is degrade, not evacuate.
//!
//! Pure logic, no I/O, no clocks beyond `Instant` — same contract as the
//! worker machine. The gateway's fleet controller translates DCGM-shaped
//! telemetry and admin commands into [`NodeEvent`]s; kubernetes state
//! (pods appearing/vanishing) arrives as `WorkersOnline` / `WorkersLost`.

use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeState {
    /// Registered but not validated (no serving workers yet, or the node
    /// just came back and is being checked).
    Provisioning,
    /// Hosting READY workers; takes placements.
    Ready,
    /// Soft gate: no NEW model placements; resident models keep serving.
    Cordoned,
    /// Workers draining; active generations finish.
    Draining,
    /// Emptied and held; nothing schedules here.
    Maintenance,
    /// Critical fault (XID class). Workers drained, scheduling stopped.
    Failed,
    /// Returned/cleared; waiting for functional validation.
    Recovering,
}

impl NodeState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Provisioning => "provisioning",
            Self::Ready => "ready",
            Self::Cordoned => "cordoned",
            Self::Draining => "draining",
            Self::Maintenance => "maintenance",
            Self::Failed => "failed",
            Self::Recovering => "recovering",
        }
    }

    pub fn all() -> [NodeState; 7] {
        [
            Self::Provisioning,
            Self::Ready,
            Self::Cordoned,
            Self::Draining,
            Self::Maintenance,
            Self::Failed,
            Self::Recovering,
        ]
    }

    /// Does this node accept NEW model placements?
    pub fn allows_new_placements(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Do workers on this node currently serve traffic? (CORDONED keeps
    /// serving — that is why it exists.)
    pub fn serves_traffic(&self) -> bool {
        matches!(self, Self::Ready | Self::Cordoned)
    }
}

/// Everything that can happen to a node. The fleet controller owns the
/// mapping from the world (admin commands, telemetry, kubernetes) to these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeEvent {
    /// First worker of the node reached READY (the node proved itself).
    WorkersOnline,
    /// Every worker of the node is gone from the registry (lost node /
    /// replacement pending).
    WorkersLost,
    /// Admin: soft gate — no new placements, traffic continues.
    Cordon,
    /// Admin: lift the soft gate.
    Uncordon,
    /// Begin draining (for maintenance, or as the response to a soft fault
    /// that escalated).
    Drain,
    /// The last worker finished draining.
    WorkersDrained,
    /// Enter maintenance (after drain).
    Maintenance,
    /// Operator returns the node from maintenance.
    ReturnToService,
    /// Critical fault (XID 48/74/79 class): fail immediately.
    CriticalFault,
    /// Soft fault (thermal): cordon, keep serving.
    ThermalEvent,
    /// The soft fault cleared (temperature back in range).
    ThermalCleared,
    /// A failed node's fault was cleared — validate before Ready.
    FaultCleared,
    /// Post-recovery validation passed (workers READY again).
    ValidationPassed,
    /// Post-recovery validation failed — back to FAILED.
    ValidationFailed,
}

/// Why a transition happened — stable metric-label strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeEventKind {
    Admin,
    Telemetry,
    Fleet,
}

impl NodeEventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Admin => "admin",
            Self::Telemetry => "telemetry",
            Self::Fleet => "fleet",
        }
    }
}

/// One node transition, mirroring [`crate::Transition`].
#[derive(Debug, Clone)]
pub struct NodeTransition {
    pub from: NodeState,
    pub to: NodeState,
    pub kind: NodeEventKind,
    pub at: Instant,
}

#[derive(Debug)]
pub struct NodeLifecycle {
    state: NodeState,
    /// Total transitions ever emitted (history itself is not kept).
    total_transitions: u64,
}

impl NodeLifecycle {
    pub fn new() -> Self {
        Self { state: NodeState::Provisioning, total_transitions: 0 }
    }

    pub fn state(&self) -> NodeState {
        self.state
    }

    pub fn total_transitions(&self) -> u64 {
        self.total_transitions
    }

    /// Feed one event; returns the transitions it caused (0 if the event
    /// was a no-op for the current state — e.g. cordoning a failed node).
    pub fn observe(&mut self, event: NodeEvent, kind: NodeEventKind) -> Vec<NodeTransition> {
        let next = match (self.state, event) {
            (NodeState::Provisioning, NodeEvent::WorkersOnline) => Some(NodeState::Ready),
            (NodeState::Provisioning, NodeEvent::CriticalFault) => Some(NodeState::Failed),

            (NodeState::Ready, NodeEvent::WorkersLost) => Some(NodeState::Provisioning),
            (NodeState::Ready, NodeEvent::Cordon) => Some(NodeState::Cordoned),
            (NodeState::Ready, NodeEvent::Drain) => Some(NodeState::Draining),
            (NodeState::Ready, NodeEvent::CriticalFault) => Some(NodeState::Failed),
            (NodeState::Ready, NodeEvent::ThermalEvent) => Some(NodeState::Cordoned),

            (NodeState::Cordoned, NodeEvent::Uncordon) => Some(NodeState::Ready),
            (NodeState::Cordoned, NodeEvent::ThermalCleared) => Some(NodeState::Ready),
            (NodeState::Cordoned, NodeEvent::Drain) => Some(NodeState::Draining),
            (NodeState::Cordoned, NodeEvent::CriticalFault) => Some(NodeState::Failed),
            (NodeState::Cordoned, NodeEvent::ThermalEvent) => None, // already degraded

            (NodeState::Draining, NodeEvent::WorkersDrained) => Some(NodeState::Maintenance),
            (NodeState::Draining, NodeEvent::CriticalFault) => Some(NodeState::Failed),
            // An operator can cancel a drain/maintenance episode.
            (NodeState::Draining, NodeEvent::FaultCleared) => Some(NodeState::Recovering),

            (NodeState::Maintenance, NodeEvent::ReturnToService) => Some(NodeState::Recovering),
            (NodeState::Maintenance, NodeEvent::FaultCleared) => Some(NodeState::Recovering),
            (NodeState::Maintenance, NodeEvent::CriticalFault) => Some(NodeState::Failed),

            (NodeState::Failed, NodeEvent::FaultCleared) => Some(NodeState::Recovering),

            (NodeState::Recovering, NodeEvent::ValidationPassed) => Some(NodeState::Ready),
            (NodeState::Recovering, NodeEvent::ValidationFailed) => Some(NodeState::Failed),
            (NodeState::Recovering, NodeEvent::WorkersLost) => Some(NodeState::Provisioning),

            _ => None,
        };
        match next {
            Some(to) if to != self.state => {
                let t = NodeTransition {
                    from: self.state,
                    to,
                    kind,
                    at: Instant::now(),
                };
                self.state = to;
                self.total_transitions += 1;
                vec![t]
            }
            _ => vec![],
        }
    }
}

impl Default for NodeLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(node: &mut NodeLifecycle, events: &[(NodeEvent, NodeEventKind)]) -> Vec<NodeTransition> {
        events.iter().flat_map(|(e, k)| node.observe(*e, *k)).collect()
    }

    #[test]
    fn provisioning_validates_into_ready() {
        let mut n = NodeLifecycle::new();
        let t = drive(&mut n, &[
            (NodeEvent::Cordon, NodeEventKind::Admin), // no-op before validation
            (NodeEvent::WorkersOnline, NodeEventKind::Fleet),
        ]);
        assert_eq!(t.len(), 1);
        assert_eq!(n.state(), NodeState::Ready);
        assert!(n.state().allows_new_placements());
        assert!(n.state().serves_traffic());
    }

    #[test]
    fn cordon_is_a_soft_gate() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        let t = drive(&mut n, &[
            (NodeEvent::ThermalEvent, NodeEventKind::Telemetry),
            (NodeEvent::ThermalEvent, NodeEventKind::Telemetry), // idempotent
            (NodeEvent::ThermalCleared, NodeEventKind::Telemetry),
        ]);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].to, NodeState::Cordoned);
        assert_eq!(t[1].to, NodeState::Ready);
        assert!(NodeState::Cordoned.serves_traffic());
        assert!(!NodeState::Cordoned.allows_new_placements());
    }

    #[test]
    fn maintenance_round_trip_validates() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        let t = drive(&mut n, &[
            (NodeEvent::Drain, NodeEventKind::Admin),
            (NodeEvent::WorkersDrained, NodeEventKind::Fleet),
            (NodeEvent::ReturnToService, NodeEventKind::Admin),
            (NodeEvent::ValidationPassed, NodeEventKind::Fleet),
        ]);
        let path: Vec<NodeState> = t.iter().map(|x| x.to).collect();
        assert_eq!(path, vec![
            NodeState::Draining,
            NodeState::Maintenance,
            NodeState::Recovering,
            NodeState::Ready,
        ]);
    }

    #[test]
    fn critical_fault_fails_and_recovery_validates() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        let t = drive(&mut n, &[
            (NodeEvent::CriticalFault, NodeEventKind::Telemetry),
            (NodeEvent::Cordon, NodeEventKind::Admin), // no-op on FAILED
            (NodeEvent::FaultCleared, NodeEventKind::Admin),
            (NodeEvent::ValidationFailed, NodeEventKind::Fleet),
            (NodeEvent::FaultCleared, NodeEventKind::Admin),
            (NodeEvent::ValidationPassed, NodeEventKind::Fleet),
        ]);
        let path: Vec<NodeState> = t.iter().map(|x| x.to).collect();
        assert_eq!(path, vec![
            NodeState::Failed,
            NodeState::Recovering,
            NodeState::Failed,
            NodeState::Recovering,
            NodeState::Ready,
        ]);
    }

    #[test]
    fn losing_all_workers_sends_a_ready_node_back_to_provisioning() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        let t = drive(&mut n, &[
            (NodeEvent::WorkersLost, NodeEventKind::Fleet),
            (NodeEvent::WorkersOnline, NodeEventKind::Fleet),
        ]);
        let path: Vec<NodeState> = t.iter().map(|x| x.to).collect();
        assert_eq!(path, vec![NodeState::Provisioning, NodeState::Ready]);
    }

    #[test]
    fn operator_can_cancel_drain_and_maintenance() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        n.observe(NodeEvent::Drain, NodeEventKind::Admin);
        n.observe(NodeEvent::WorkersDrained, NodeEventKind::Fleet);
        assert_eq!(n.state(), NodeState::Maintenance);
        // recover (FaultCleared) cancels maintenance instead of failing.
        n.observe(NodeEvent::FaultCleared, NodeEventKind::Admin);
        assert_eq!(n.state(), NodeState::Recovering);
        n.observe(NodeEvent::ValidationPassed, NodeEventKind::Fleet);
        assert_eq!(n.state(), NodeState::Ready);
    }

    #[test]
    fn failed_node_ignores_thermal_and_cordon() {
        let mut n = NodeLifecycle::new();
        n.observe(NodeEvent::WorkersOnline, NodeEventKind::Fleet);
        n.observe(NodeEvent::CriticalFault, NodeEventKind::Telemetry);
        assert!(drive(&mut n, &[
            (NodeEvent::ThermalEvent, NodeEventKind::Telemetry),
            (NodeEvent::Uncordon, NodeEventKind::Admin),
        ])
        .is_empty());
        assert_eq!(n.state(), NodeState::Failed);
    }
}
