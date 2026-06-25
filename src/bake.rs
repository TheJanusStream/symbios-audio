//! Sample-loop evaluator — turns an [`AudioPatch`] into a mono `Vec<f32>`.
//!
//! Phase 1 ticket #3.  Single-threaded, single-buffer, deterministic.  The
//! graph is compiled once into a flat plan in topological order (dense node
//! indices, pre-resolved input sources); the inner loop then iterates from
//! sample 0 to `duration_samples`, evaluates every node once per sample,
//! and pushes the output node's value into the buffer.  Outputs live in a
//! `Vec<f32>` indexed by plan position and the per-node input values are
//! filled into reusable scratch, so the hot loop allocates nothing.
//!
//! # Determinism
//!
//! Two bakes of the same patch with the same sample rate, duration, and
//! seed produce a bit-identical buffer.  This only requires two things, and
//! the evaluator guarantees both:
//! - A deterministic **node evaluation order** — [`crate::patch::topo_sort`]
//!   uses Kahn's algorithm with sorted tie-breaking (no `HashMap`
//!   iteration), and that order is frozen into the plan.
//! - A deterministic **RNG draw order** — a single [`ChaCha8Rng`] seeded
//!   from `AudioPatch::seed`, advanced only by node draws in evaluation
//!   order, never reset or reseeded mid-bake.
//!
//! Input resolution and output storage draw no RNG, so their containers
//! (a flat `Vec`, a fixed-order scratch list) need only be deterministic,
//! not sorted — which is why this layer no longer leans on `BTreeMap` in
//! the inner loop.
//!
//! See the `tests` module at the bottom for the regression hash test.
//!
//! # Scope and out-of-scope
//!
//! Correctness over speed.  The rayon-backed parallel bake pool lives in
//! the `bevy_symbios_audio` wrapper crate's async-bake pool — the inner
//! loop here stays single-threaded per patch, which lets the pool just
//! dispatch one bake per pending request.
//! Soft-clipping / master gain belongs to the mixdown baker
//! ([`crate::mixdown::bake_sequence`]).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::node::{BakeContext, Node, NodeKind};
use crate::patch::{AudioPatch, Connection, GraphError, GraphNode, NodeId, topo_sort};
use crate::wav::MAX_WAV_SAMPLES;

/// How often [`bake_inner`] polls the cancellation flag inside the sample
/// loop.  Coarse enough that the relaxed atomic load is free in the hot path,
/// fine enough that a dropped pending audio patch (in the wrapper crate)
/// aborts an in-flight bake within a fraction of a millisecond at audio rates.
const CANCEL_CHECK_INTERVAL: u64 = 4096;

/// Bake `patch` into a mono `Vec<f32>` at `sample_rate` Hz for
/// `duration_secs` seconds.
///
/// Returns an empty buffer if `duration_secs <= 0.0`.  **Panics** if the
/// underlying [`crate::patch::NodeGraph`] is structurally invalid (cycle,
/// dangling reference, duplicate id, or missing output).  Callers that
/// can't trust their patch should use [`try_bake`], which surfaces the
/// [`GraphError`] instead.
///
/// # Determinism
///
/// `bake(p, sr, d) == bake(p, sr, d)` bit-for-bit for any well-formed
/// patch, sample rate, and duration.  Seed lives on [`AudioPatch::seed`].
pub fn bake(patch: &AudioPatch, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    try_bake(patch, sample_rate, duration_secs)
        .expect("bake: AudioPatch.graph is structurally invalid")
}

/// Fallible sibling of [`bake`] — bake `patch` into a mono `Vec<f32>`, or
/// return the [`GraphError`] that made the graph un-bakeable.
///
/// Returns `Ok(empty)` for a non-positive duration.  Use this on the async
/// / sequencer paths where a malformed patch should be reported or skipped
/// rather than panicking a worker thread.
pub fn try_bake(
    patch: &AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
) -> Result<Vec<f32>, GraphError> {
    bake_inner(
        patch,
        sample_rate,
        duration_samples(sample_rate, duration_secs),
        None,
        None,
    )
}

/// [`try_bake`] that aborts early if `cancelled` flips to `true` while the
/// bake is in flight.  Used by the async pool so dropping a pending audio
/// patch (in the wrapper crate) stops an already-running bake
/// instead of letting it occupy a worker until completion.  The returned
/// buffer is partial (and discarded) when cancellation fires mid-bake.
///
/// Public (rather than `pub(crate)`) so the `bevy_symbios_audio` wrapper
/// crate's async-bake pool — which lives in a separate crate after the
/// Bevy-free / Bevy split — can drive cancellable bakes.
pub fn try_bake_cancellable(
    patch: &AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, GraphError> {
    bake_inner(
        patch,
        sample_rate,
        duration_samples(sample_rate, duration_secs),
        None,
        Some(cancelled),
    )
}

/// One resolved contribution to an input port.
enum InputSource {
    /// A fixed DC value.
    Const(f32),
    /// `outputs[src] * amount`, where `src` is the upstream node's dense
    /// plan index (resolved once, so the hot loop indexes a `Vec`).
    Node { src: usize, amount: f32 },
}

/// A single input port of a planned node: its name plus the (summed) list
/// of contributions feeding it.
struct PortPlan {
    name: String,
    sources: Vec<InputSource>,
}

/// One node, compiled for the sample loop: its config and its resolved
/// input ports.  Borrows the [`NodeKind`] out of the patch for the bake.
struct NodePlan<'a> {
    kind: &'a NodeKind,
    ports: Vec<PortPlan>,
}

/// Core evaluator shared by [`bake`] / [`try_bake`] and the sequencer.
///
/// `gate_samples` threads a per-event gate window into every
/// [`BakeContext`] so [`NodeKind::Gate`] can open the gate for the first
/// `gate_samples` samples and close it (triggering envelope release) for
/// the rest of the bake.  `None` leaves the gate always open.
pub(crate) fn bake_inner(
    patch: &AudioPatch,
    sample_rate: u32,
    duration_samples: u64,
    gate_samples: Option<u64>,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<f32>, GraphError> {
    if duration_samples == 0 {
        return Ok(Vec::new());
    }

    // Deterministic evaluation order — the one thing the bake's output
    // identity actually hinges on (alongside RNG draw order).
    let order = topo_sort(&patch.graph)?;

    let node_by_id: BTreeMap<NodeId, &GraphNode> =
        patch.graph.nodes.iter().map(|n| (n.id, n)).collect();
    // NodeId → dense plan index, so node-sourced inputs can read a Vec.
    let index_of: BTreeMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    // Compile the graph into a flat, index-addressed plan once.
    let plan: Vec<NodePlan> = order
        .iter()
        .map(|id| {
            let node = node_by_id[id];
            let ports = node
                .inputs
                .iter()
                .map(|(name, conns)| {
                    let sources = conns
                        .iter()
                        .map(|c| match c {
                            Connection::Constant { value } => InputSource::Const(*value),
                            Connection::Node { id, amount } => InputSource::Node {
                                src: index_of[id],
                                amount: *amount,
                            },
                        })
                        .collect();
                    PortPlan {
                        name: name.clone(),
                        sources,
                    }
                })
                .collect();
            NodePlan {
                kind: &node.kind,
                ports,
            }
        })
        .collect();

    let output_index = index_of[&patch.graph.output];

    let mut rng = ChaCha8Rng::seed_from_u64(u64::from(patch.seed));

    // Per-node persistent state, indexed by plan position.  Type-erased so
    // each node kind owns its own state shape; impls downcast via
    // BakeContext::state_mut::<S>().
    let mut states: Vec<Option<Box<dyn Any + Send>>> =
        plan.iter().map(|np| np.kind.init_state()).collect();

    // Reusable scratch: dense outputs and a per-node (port, value) list.
    let mut outputs = vec![0.0_f32; plan.len()];
    let mut input_scratch: Vec<(&str, f32)> = Vec::new();

    let mut buffer = Vec::with_capacity(duration_samples as usize);

    for sample_index in 0..duration_samples {
        // Abort an in-flight bake whose owning PendingAudioPatch was dropped,
        // so a despawned entity's work stops occupying a pool worker rather
        // than running to completion as zombie computation.  Checked on a
        // coarse stride so the relaxed load stays out of the per-sample path.
        if sample_index % CANCEL_CHECK_INTERVAL == 0
            && cancelled.is_some_and(|c| c.load(Ordering::Relaxed))
        {
            return Ok(buffer);
        }
        for i in 0..plan.len() {
            let np = &plan[i];
            // Resolve this node's ports by summing each port's sources.
            input_scratch.clear();
            for port in &np.ports {
                let mut value = 0.0_f32;
                for src in &port.sources {
                    value += match *src {
                        InputSource::Const(c) => c,
                        InputSource::Node { src, amount } => outputs[src] * amount,
                    };
                }
                input_scratch.push((port.name.as_str(), value));
            }
            let state_ref = states[i].as_deref_mut();
            let mut ctx = BakeContext::new(
                sample_rate,
                sample_index,
                duration_samples,
                &mut rng,
                &input_scratch,
                state_ref,
            )
            .with_gate(gate_samples);
            outputs[i] = np.kind.sample(&mut ctx);
        }
        buffer.push(outputs[output_index]);
    }

    Ok(buffer)
}

/// Convert a `f32` seconds duration into a sample count for the given rate.
///
/// Computes in `f64` to keep precision sane for long bakes (a 32-bit float
/// loses ~1 sample of accuracy by ~30 seconds at 48 kHz).  Rounds to the
/// nearest sample so user-friendly values like `0.01` (which isn't exactly
/// representable in `f32`) yield the expected 441 samples at 44.1 kHz
/// rather than 440.  Clamps negative values to zero.
#[inline]
fn duration_samples(sample_rate: u32, duration_secs: f32) -> u64 {
    if duration_secs <= 0.0 {
        return 0;
    }
    let samples = (f64::from(duration_secs) * f64::from(sample_rate)).round();
    // Clamp to the WAV ceiling so an untrusted `duration_secs` can't saturate
    // the cast to u64::MAX and make bake_inner's `Vec::with_capacity` attempt
    // a usize::MAX allocation.
    if samples >= MAX_WAV_SAMPLES as f64 {
        MAX_WAV_SAMPLES as u64
    } else {
        samples as u64
    }
}

#[cfg(test)]
mod tests {
    use crate::node::NodeKind;
    use crate::patch::{AudioPatch, Connection, GraphNode, NodeGraph, NodeId};

    use super::*;

    /// Stable, version-portable hash of an `f32` buffer.  FNV-1a over each
    /// sample's IEEE-754 little-endian bit pattern.  Not cryptographic;
    /// good enough to detect any change to the bake output.
    fn fnv1a_64(samples: &[f32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        for s in samples {
            for byte in s.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    fn silence_patch(seed: u32) -> AudioPatch {
        AudioPatch {
            seed,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        }
    }

    #[test]
    fn zero_duration_yields_empty_buffer() {
        let p = silence_patch(0);
        assert!(bake(&p, 44_100, 0.0).is_empty());
    }

    #[test]
    fn negative_duration_yields_empty_buffer() {
        let p = silence_patch(0);
        assert!(bake(&p, 44_100, -1.0).is_empty());
    }

    #[test]
    fn buffer_length_matches_rate_times_duration() {
        let p = silence_patch(0);
        // 0.5 s @ 48 kHz = 24 000 samples.
        let buf = bake(&p, 48_000, 0.5);
        assert_eq!(buf.len(), 24_000);
        // 1 s @ 44.1 kHz = 44 100 samples.
        let buf2 = bake(&p, 44_100, 1.0);
        assert_eq!(buf2.len(), 44_100);
    }

    #[test]
    fn silence_only_patch_produces_all_zeros() {
        let p = silence_patch(0);
        let buf = bake(&p, 44_100, 0.01); // 441 samples
        assert_eq!(buf.len(), 441);
        for (i, s) in buf.iter().enumerate() {
            assert_eq!(*s, 0.0_f32, "sample {i} not zero: {s}");
        }
    }

    #[test]
    fn bake_is_deterministic_across_repeated_calls() {
        // A graph with constant-wired and node-wired inputs, plus a
        // multi-node DAG, exercises the per-sample input resolution path
        // even though Silence ignores its inputs.
        let mut n1_inputs: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
        n1_inputs.insert("a".to_string(), vec![Connection::from_node(NodeId(0))]);
        n1_inputs.insert("b".to_string(), vec![Connection::constant(0.25)]);
        let patch = AudioPatch {
            seed: 0xDEAD_BEEF,
            graph: NodeGraph {
                nodes: vec![
                    GraphNode {
                        id: NodeId(0),
                        kind: NodeKind::Silence,
                        inputs: BTreeMap::new(),
                    },
                    GraphNode {
                        id: NodeId(1),
                        kind: NodeKind::Silence,
                        inputs: n1_inputs,
                    },
                ],
                output: NodeId(1),
            },
        };
        let a = bake(&patch, 44_100, 0.1);
        let b = bake(&patch, 44_100, 0.1);
        assert_eq!(a, b);
    }

    #[test]
    fn bake_hash_pinned_for_silent_quarter_second() {
        // Regression pin: every f32 in the buffer is +0.0, and the FNV-1a
        // hash of 11_025 zero bytes-worth-of-zero-f32s is a constant.  Any
        // change to bake's output shape (length, samples, alignment) flips
        // this hash and the test fails loudly.
        let p = silence_patch(0);
        let buf = bake(&p, 44_100, 0.25);
        assert_eq!(buf.len(), 11_025);
        let h = fnv1a_64(&buf);
        // Computed once and pinned.  This is the hash of 11_025 f32 zeros.
        assert_eq!(h, 0xC7D0_6137_6364_38F5);
    }

    #[test]
    fn fnv1a_self_check_against_known_input() {
        // Lock down the hash function itself — if the FNV constants drift,
        // this fails before the bake-hash test does, giving a cleaner
        // diagnosis.
        let h = fnv1a_64(&[0.0_f32]);
        // FNV-1a over four zero bytes (the bit pattern of +0.0_f32).
        let expected: u64 = {
            const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
            const FNV_PRIME: u64 = 0x100000001b3;
            let mut h = FNV_OFFSET;
            for _ in 0..4 {
                h = h.wrapping_mul(FNV_PRIME);
            }
            h
        };
        assert_eq!(h, expected);
    }

    #[test]
    #[should_panic(expected = "structurally invalid")]
    fn invalid_graph_panics_loudly() {
        // Output points at a node id that doesn't exist.
        let p = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(99),
            },
        };
        let _ = bake(&p, 44_100, 0.01);
    }

    #[test]
    fn try_bake_returns_err_on_invalid_graph() {
        // The fallible sibling surfaces the GraphError instead of panicking.
        let p = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(99),
            },
        };
        assert_eq!(
            try_bake(&p, 44_100, 0.01),
            Err(crate::patch::GraphError::MissingOutput(NodeId(99)))
        );
    }

    #[test]
    fn try_bake_ok_matches_bake_for_valid_patch() {
        let p = silence_patch(0);
        let via_try = try_bake(&p, 44_100, 0.05).expect("valid patch bakes");
        let via_bake = bake(&p, 44_100, 0.05);
        assert_eq!(via_try, via_bake);
    }

    #[test]
    fn bake_inner_aborts_when_cancelled() {
        let p = silence_patch(0);
        let cancelled = AtomicBool::new(true);
        // Pre-cancelled: the loop checks at sample_index 0 and returns before
        // producing the full buffer, so even a multi-million-sample request
        // stops immediately instead of running to completion.
        let out =
            bake_inner(&p, 44_100, 5_000_000, None, Some(&cancelled)).expect("valid patch bakes");
        assert!(
            out.len() < 5_000_000,
            "cancelled bake must not run to completion: got {} samples",
            out.len()
        );
    }

    #[test]
    fn bake_inner_completes_when_not_cancelled() {
        let p = silence_patch(0);
        let cancelled = AtomicBool::new(false);
        let out = bake_inner(&p, 44_100, 1_000, None, Some(&cancelled)).expect("valid patch bakes");
        assert_eq!(out.len(), 1_000);
    }

    #[test]
    fn duration_samples_clamps_absurd_duration() {
        // An astronomical duration clamps to the WAV ceiling instead of
        // saturating to u64::MAX (which would OOM bake_inner's allocation).
        assert_eq!(
            duration_samples(44_100, f32::MAX),
            crate::wav::MAX_WAV_SAMPLES as u64
        );
        // Sane durations are untouched: 0.5 s × 48 kHz.
        assert_eq!(duration_samples(48_000, 0.5), 24_000);
    }
}
