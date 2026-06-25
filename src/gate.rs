//! [`Gate`] — turns the note's gate window into a control signal.
//!
//! A [`crate::adsr::AdsrEnvelope`] is edge-triggered off its `"gate"` input,
//! but nothing in a standalone patch produces a *falling* edge at a known
//! time — so wiring the gate to a constant `1.0` leaves the envelope stuck
//! at sustain and the note is cut off, never released.
//!
//! `Gate` closes that loop.  It reads the per-evaluation gate window the
//! baker carries on [`BakeContext`] ([`BakeContext::gate_open`]) and emits
//! `1.0` while the gate is open, `0.0` once it closes.  The sequencer
//! ([`crate::mixdown::bake_sequence`]) opens the gate for an event's
//! `gate_beats` and then keeps baking through `release_beats` of tail, so
//! the wiring `Gate → AdsrEnvelope.gate → amplitude` gives a note that
//! attacks, sustains for the gate, then releases and rings out.
//!
//! For a standalone [`crate::bake::bake`] (no event gate) the window is
//! "always open", so `Gate` holds `1.0` for the whole bake — a held note.
//!
//! # `invert`
//!
//! With `invert = true` the polarity flips: `0.0` while open, `1.0` after
//! the gate closes.  Handy as a release-triggered envelope gate or to fade
//! a layer *in* on note-off.

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

/// Emits a `1.0`/`0.0` gate signal driven by the baker's gate window.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    /// Flip the polarity — `0.0` while the gate is open, `1.0` after it
    /// closes.  Defaults to `false` (open ⇒ `1.0`).
    #[serde(default)]
    pub invert: bool,
}

impl Node for Gate {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        // `open != invert`: open&&!invert -> 1, open&&invert -> 0, etc.
        if ctx.gate_open() != self.invert {
            1.0
        } else {
            0.0
        }
    }
}

crate::impl_genotype!(Gate { invert: bool });

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn gate_value(gate: &Gate, sample_index: u64, gate_samples: Option<u64>) -> f32 {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs: &[(&str, f32)] = &[];
        let mut ctx = BakeContext::new(44_100, sample_index, 44_100, &mut rng, inputs, None)
            .with_gate(gate_samples);
        gate.sample(&mut ctx)
    }

    #[test]
    fn open_while_inside_window_then_closes() {
        let gate = Gate::default();
        // Window of 100 samples: open for [0, 100), closed at >= 100.
        assert_eq!(gate_value(&gate, 0, Some(100)), 1.0);
        assert_eq!(gate_value(&gate, 99, Some(100)), 1.0);
        assert_eq!(gate_value(&gate, 100, Some(100)), 0.0);
        assert_eq!(gate_value(&gate, 5_000, Some(100)), 0.0);
    }

    #[test]
    fn ungated_bake_holds_gate_open() {
        let gate = Gate::default();
        // No window => always open (held note).
        assert_eq!(gate_value(&gate, 0, None), 1.0);
        assert_eq!(gate_value(&gate, 1_000_000, None), 1.0);
    }

    #[test]
    fn invert_flips_polarity() {
        let gate = Gate { invert: true };
        assert_eq!(gate_value(&gate, 10, Some(100)), 0.0);
        assert_eq!(gate_value(&gate, 100, Some(100)), 1.0);
    }

    #[test]
    fn serde_round_trips_with_kind_tag() {
        // Gate must serialise cleanly inside the internally-tagged NodeKind.
        let json = serde_json::to_string(&Gate::default()).unwrap();
        let back: Gate = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Gate::default());
        // The default-invert field may be omitted on the wire.
        let parsed: Gate = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, Gate::default());
    }
}
