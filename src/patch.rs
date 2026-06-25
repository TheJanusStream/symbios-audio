//! Patch schema — the serializable description of a DSP graph.
//!
//! An [`AudioPatch`] is the top-level document: a `seed` for the
//! deterministic RNG plus a [`NodeGraph`].  The graph is a DAG of
//! [`GraphNode`]s; each node has an identifier, a configuration
//! ([`crate::node::NodeKind`]), and a map of named input ports to lists of
//! [`Connection`]s.  A connection sources its value either from a literal
//! constant or from another node's output.  Multiple connections may target
//! the same port — their resolved values are **summed** before delivery, so
//! signal mixing and modulation stacking are expressible without a dedicated
//! node (though [`crate::node::NodeKind::Mix`] exists for clarity).
//!
//! Sample rate and duration are *not* part of the patch — they are passed
//! to [`crate::bake::bake`] at bake time, so the same patch can render at
//! 44.1 kHz for playback and 48 kHz for capture without editing the JSON.
//!
//! The schema is intentionally Bevy-free and `serde`-only — DAG-CBOR or any
//! other transport encoding is the caller's concern (Overlands handles
//! ATProto/CBOR at the PDS boundary).
//!
//! Topological sorting lives here as a free function ([`topo_sort`]); the
//! sample-loop evaluator in [`mod@crate::bake`] computes the order once
//! and reuses it across the bake.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::node::NodeKind;

/// Stable identifier for a node within a [`NodeGraph`].
///
/// `Default` returns `NodeId(0)` — the first id in a freshly-built
/// graph.  Useful for Sovereign* mirror shims (e.g. Overlands' PDS
/// types) that need to construct an empty node by default.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct NodeId(pub u32);

/// Top-level patch — a DSP graph plus the seed that drives stochastic nodes.
///
/// Patches are sample-rate-agnostic: the consumer chooses the rate at
/// [`crate::bake::bake`] time.  This matches how the rest of the schema
/// works (durations are in beats/seconds, not samples) and lets the same
/// patch render at 44.1 kHz for playback and 48 kHz for capture without
/// editing the JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioPatch {
    /// Seed for the deterministic RNG that drives any stochastic node
    /// (noise generators, randomised LFOs, etc.).  Two bakes of the same
    /// patch at the same sample rate and duration with the same seed
    /// produce a bit-identical buffer.
    #[serde(default)]
    pub seed: u32,
    /// The DAG of nodes that produces the output sample.
    pub graph: NodeGraph,
}

/// Directed acyclic graph of [`GraphNode`]s.
///
/// `nodes` is the unordered set of placed nodes.  `output` names the node
/// whose final sample value is the patch's output.  Evaluation order is
/// derived from [`topo_sort`] — call it once before the sample loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeGraph {
    /// Unordered set of placed nodes.  Evaluation order is derived
    /// from [`topo_sort`], not from this list's order, so callers may
    /// store nodes in any sequence they like.
    pub nodes: Vec<GraphNode>,
    /// Identifier of the node whose per-sample output is the patch's
    /// final mono sample.  Must refer to a node present in `nodes` or
    /// [`topo_sort`] returns [`GraphError::MissingOutput`].
    pub output: NodeId,
}

impl Default for NodeGraph {
    fn default() -> Self {
        // The empty graph would have no output; default to a single silent
        // node so a fresh patch is a valid (silent) one.
        Self {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind: NodeKind::Silence,
                inputs: BTreeMap::new(),
            }],
            output: NodeId(0),
        }
    }
}

/// A node placed in a [`NodeGraph`].
///
/// `Default` is `{ id: NodeId(0), kind: NodeKind::Silence, inputs:
/// empty }` — a silent zero-id node, useful as a placeholder during
/// graph construction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Stable id used by other nodes' [`Connection::Node`] entries and
    /// by [`NodeGraph::output`].  Must be unique within the parent
    /// graph or [`topo_sort`] returns [`GraphError::DuplicateId`].
    pub id: NodeId,
    /// Concrete node configuration (oscillator settings, filter
    /// cutoff, etc.).  Tagged on the serde wire by `"kind"`.
    pub kind: NodeKind,
    /// Wired inputs, keyed by port name.  Each port holds a *list* of
    /// connections whose resolved values are **summed** at evaluation
    /// time, so several sources can feed one port (signal mixing,
    /// modulation stacking).  A missing port — or an empty list — reads
    /// `0.0`, so partially-wired nodes behave sensibly without filling
    /// in every slot.
    #[serde(default)]
    pub inputs: BTreeMap<String, Vec<Connection>>,
}

impl GraphNode {
    /// Builder helper: append `conn` to the named port's connection list,
    /// creating the list if absent.  Returns `self` for chaining.
    ///
    /// Because ports sum their connections, calling this twice for the
    /// same port wires *both* sources into it.
    pub fn with_input(mut self, port: impl Into<String>, conn: Connection) -> Self {
        self.inputs.entry(port.into()).or_default().push(conn);
        self
    }
}

/// Source of a single input contribution — either a literal value or
/// another node's output.
///
/// `Node` connections carry an `amount` multiplier (default `1.0`).  At
/// bake time the value contributed to the downstream port is
/// `upstream_output * amount`.  This is how modulation routing is
/// expressed: an LFO wired to a filter's `"cutoff_hz"` port with `amount =
/// 900.0` sweeps the cutoff by ±900 Hz around the filter's base setting.
/// Pass-through signal connections leave `amount` at the default `1.0`.
///
/// Each port holds a *list* of connections (see [`GraphNode::inputs`]);
/// when more than one targets a port, their contributions are summed.
///
/// `Default` is `Constant { value: 0.0 }` — a silent fixed connection.
/// Useful as a placeholder for Sovereign* mirror shims and for
/// programmatically built graphs that want to fill an input slot before
/// settling its final source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum Connection {
    /// A fixed DC value.
    Constant { value: f32 },
    /// Reads from another node's output, scaled by `amount` (default
    /// `1.0` if absent in JSON).
    Node {
        id: NodeId,
        #[serde(default = "default_amount")]
        amount: f32,
    },
}

impl Default for Connection {
    /// `Constant { value: 0.0 }` — a silent fixed connection.
    /// `#[derive(Default)]` doesn't accept `#[default]` on struct
    /// variants, so this is written by hand.
    fn default() -> Self {
        Self::Constant { value: 0.0 }
    }
}

fn default_amount() -> f32 {
    1.0
}

impl Connection {
    /// Convenience constructor for a unity-amount `Node` connection.
    pub fn from_node(id: NodeId) -> Self {
        Self::Node {
            id,
            amount: default_amount(),
        }
    }

    /// Modulation connection from `id`'s output, scaled by `amount` before
    /// delivery.  Use this for LFO → cutoff, envelope → amplitude,
    /// oscillator → oscillator (FM) wiring.
    pub fn modulation(id: NodeId, amount: f32) -> Self {
        Self::Node { id, amount }
    }

    /// Convenience constructor for a constant DC connection.
    pub fn constant(value: f32) -> Self {
        Self::Constant { value }
    }
}

/// Reasons a [`NodeGraph`] is structurally invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A connection references a node id that isn't present in the graph.
    UnknownNode(NodeId),
    /// The graph's declared output node isn't present.
    MissingOutput(NodeId),
    /// The graph contains a cycle.
    Cycle,
    /// Two or more nodes share the same id.
    DuplicateId(NodeId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownNode(id) => {
                write!(f, "connection references unknown node {}", id.0)
            }
            Self::MissingOutput(id) => {
                write!(f, "graph output {} not present in nodes", id.0)
            }
            Self::Cycle => f.write_str("graph contains a cycle"),
            Self::DuplicateId(id) => write!(f, "duplicate node id {}", id.0),
        }
    }
}

impl std::error::Error for GraphError {}

/// Topologically sort the graph so dependencies come before dependents.
///
/// Uses Kahn's algorithm with a deterministic queue ordering: among nodes
/// with equal in-degree, the one with the lower [`NodeId`] is emitted first.
/// This makes the evaluation order reproducible for a given patch.
///
/// Returns the node ids in evaluation order, or [`GraphError`] on structural
/// problems (cycle, unknown reference, duplicate id, or missing output).
pub fn topo_sort(graph: &NodeGraph) -> Result<Vec<NodeId>, GraphError> {
    // In-degree count per node, and adjacency from upstream → downstream.
    let mut indegree: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut adjacency: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();

    for node in &graph.nodes {
        if indegree.insert(node.id, 0).is_some() {
            return Err(GraphError::DuplicateId(node.id));
        }
        adjacency.entry(node.id).or_default();
    }

    if !indegree.contains_key(&graph.output) {
        return Err(GraphError::MissingOutput(graph.output));
    }

    for node in &graph.nodes {
        for conns in node.inputs.values() {
            for conn in conns {
                if let Connection::Node { id, .. } = conn {
                    if !indegree.contains_key(id) {
                        return Err(GraphError::UnknownNode(*id));
                    }
                    adjacency.get_mut(id).expect("inserted above").push(node.id);
                    *indegree.get_mut(&node.id).expect("inserted above") += 1;
                }
            }
        }
    }

    // Seed the queue with every zero-indegree node in id order (BTreeMap
    // gives us deterministic iteration).
    let mut queue: VecDeque<NodeId> = indegree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(id, _)| *id)
        .collect();

    let mut order = Vec::with_capacity(graph.nodes.len());
    while let Some(id) = queue.pop_front() {
        order.push(id);
        if let Some(downstream) = adjacency.get(&id) {
            // Sort the downstream list each step to keep tie-breaking
            // deterministic — small graphs, cheap.
            let mut to_decrement: Vec<NodeId> = downstream.clone();
            to_decrement.sort();
            for d in to_decrement {
                let deg = indegree.get_mut(&d).expect("known node");
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(d);
                }
            }
        }
    }

    if order.len() != graph.nodes.len() {
        return Err(GraphError::Cycle);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(id: u32) -> GraphNode {
        GraphNode {
            id: NodeId(id),
            kind: NodeKind::Silence,
            inputs: BTreeMap::new(),
        }
    }

    fn silence_with(id: u32, inputs: &[(&str, Connection)]) -> GraphNode {
        let mut m: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
        for (k, v) in inputs {
            m.entry((*k).to_string()).or_default().push(v.clone());
        }
        GraphNode {
            id: NodeId(id),
            kind: NodeKind::Silence,
            inputs: m,
        }
    }

    #[test]
    fn topo_sort_single_node() {
        let g = NodeGraph {
            nodes: vec![silence(0)],
            output: NodeId(0),
        };
        let order = topo_sort(&g).unwrap();
        assert_eq!(order, vec![NodeId(0)]);
    }

    #[test]
    fn topo_sort_linear_chain() {
        // 0 -> 1 -> 2
        let g = NodeGraph {
            nodes: vec![
                silence(0),
                silence_with(1, &[("in", Connection::from_node(NodeId(0)))]),
                silence_with(2, &[("in", Connection::from_node(NodeId(1)))]),
            ],
            output: NodeId(2),
        };
        let order = topo_sort(&g).unwrap();
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn topo_sort_diamond() {
        // 0 -> 1, 0 -> 2, 1+2 -> 3
        let g = NodeGraph {
            nodes: vec![
                silence(0),
                silence_with(1, &[("in", Connection::from_node(NodeId(0)))]),
                silence_with(2, &[("in", Connection::from_node(NodeId(0)))]),
                silence_with(
                    3,
                    &[
                        ("a", Connection::from_node(NodeId(1))),
                        ("b", Connection::from_node(NodeId(2))),
                    ],
                ),
            ],
            output: NodeId(3),
        };
        let order = topo_sort(&g).unwrap();
        // 0 must come first; 3 must come last; 1 and 2 between (in id order
        // because of deterministic tie-breaking).
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]);
    }

    #[test]
    fn topo_sort_detects_cycle() {
        // 0 <-> 1
        let g = NodeGraph {
            nodes: vec![
                silence_with(0, &[("in", Connection::from_node(NodeId(1)))]),
                silence_with(1, &[("in", Connection::from_node(NodeId(0)))]),
            ],
            output: NodeId(0),
        };
        assert_eq!(topo_sort(&g), Err(GraphError::Cycle));
    }

    #[test]
    fn topo_sort_detects_unknown_node() {
        let g = NodeGraph {
            nodes: vec![silence_with(
                0,
                &[("in", Connection::from_node(NodeId(99)))],
            )],
            output: NodeId(0),
        };
        assert_eq!(topo_sort(&g), Err(GraphError::UnknownNode(NodeId(99))));
    }

    #[test]
    fn topo_sort_detects_missing_output() {
        let g = NodeGraph {
            nodes: vec![silence(0)],
            output: NodeId(5),
        };
        assert_eq!(topo_sort(&g), Err(GraphError::MissingOutput(NodeId(5))));
    }

    #[test]
    fn topo_sort_detects_duplicate_id() {
        let g = NodeGraph {
            nodes: vec![silence(0), silence(0)],
            output: NodeId(0),
        };
        assert_eq!(topo_sort(&g), Err(GraphError::DuplicateId(NodeId(0))));
    }

    #[test]
    fn connection_default_amount_deserialises() {
        let json = r#"{"source":"node","id":7}"#;
        let conn: Connection = serde_json::from_str(json).unwrap();
        match conn {
            Connection::Node { id, amount } => {
                assert_eq!(id, NodeId(7));
                assert_eq!(amount, 1.0);
            }
            _ => panic!("expected Node connection"),
        }
    }

    #[test]
    fn connection_node_with_amount_round_trips() {
        let conn = Connection::modulation(NodeId(2), 500.0);
        let json = serde_json::to_string(&conn).unwrap();
        let back: Connection = serde_json::from_str(&json).unwrap();
        assert_eq!(conn, back);
        // And explicit JSON with both fields parses back correctly.
        let explicit = r#"{"source":"node","id":2,"amount":500.0}"#;
        let parsed: Connection = serde_json::from_str(explicit).unwrap();
        assert_eq!(parsed, conn);
    }

    #[test]
    fn multiple_connections_per_port_are_retained() {
        // Fan-in: two sources on one port survive a round-trip and are
        // both present (the baker sums them).
        let node = GraphNode::default()
            .with_input("in", Connection::from_node(NodeId(0)))
            .with_input("in", Connection::constant(0.5));
        assert_eq!(node.inputs["in"].len(), 2);
        let json = serde_json::to_string(&node).unwrap();
        let back: GraphNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.inputs["in"].len(), 2);
    }
}
