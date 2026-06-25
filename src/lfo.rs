//! Low-frequency oscillator.
//!
//! Emits a slow control signal — one sample per bake step — intended to
//! drive other nodes' parameters via the `Connection` `amount` field
//! (see [`crate::patch::Connection`]).  Not for the audio output buffer,
//! though there's nothing actually stopping you from listening to a 0.3 Hz
//! square wave as a click train if that's the aesthetic.
//!
//! # Output shape
//!
//! For all shapes, the raw waveform value lives in roughly `[-1, 1]`.
//! `depth` scales it; `offset` shifts the result.  Final output:
//!
//! ```text
//! output = waveform(phase) * depth + offset
//! ```
//!
//! So a `Lfo { rate_hz: 0.3, shape: Sine, depth: 900.0, offset: 1100.0 }`
//! sweeps between 200 and 2000 — exactly the cutoff range the ticket's
//! "LFO sweeping a filter sounds like wind" acceptance test asks for.
//!
//! # Random shape
//!
//! `LfoShape::Random` is sample-and-hold: a fresh uniform `[-1, 1]` value
//! is drawn from `BakeContext::rng` once per cycle and held flat until the
//! phase wraps.  This gives the stepped, irregular flavour beloved of
//! 70s synth bleeps without needing a separate sequencer.

use std::any::Any;
use std::f32::consts::PI;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

/// Output waveform shape of an [`Lfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LfoShape {
    /// `sin(2π · phase)`.
    #[default]
    Sine,
    /// Symmetric triangle: peaks at +1, troughs at −1.
    Triangle,
    /// Bipolar square: +1 first half of cycle, −1 second half.
    Square,
    /// Rising sawtooth, −1 at phase 0 to +1 at phase 1.
    Saw,
    /// Sample-and-hold: fresh uniform `[-1, 1]` drawn at each cycle wrap,
    /// held flat across the cycle.
    Random,
}

/// Low-frequency oscillator config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lfo {
    pub rate_hz: f32,
    pub shape: LfoShape,
    pub depth: f32,
    pub offset: f32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            rate_hz: 1.0,
            shape: LfoShape::Sine,
            depth: 1.0,
            offset: 0.0,
        }
    }
}

/// Persistent state for an [`Lfo`].
///
/// `phase` is the running phase accumulator in `[0, 1)`.  `held_value` is
/// the most recent sample drawn for the `Random` shape; `cycle_started` is
/// `false` only on the very first sample so the random draw fires on
/// sample 0 rather than waiting a full cycle.
#[derive(Debug, Clone, Copy, Default)]
pub struct LfoState {
    pub(crate) phase: f32,
    pub(crate) held_value: f32,
    pub(crate) cycle_started: bool,
}

impl Node for Lfo {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let rate = self.rate_hz.max(0.0);
        let is_random = matches!(self.shape, LfoShape::Random);

        // Copy the state out so the (mutable) state borrow is released
        // before we touch the rng — the two share `ctx` and can't be
        // borrowed at once.
        let mut st = match ctx.state_mut::<LfoState>() {
            Some(s) => *s,
            // No state container: emit the offset (raw waveform = 0).
            None => return self.offset,
        };

        // Sample-and-hold: draw the very first held value on sample 0 so
        // the first cycle isn't silent.  (Subsequent draws happen on wrap,
        // below.)
        if is_random && !st.cycle_started {
            st.held_value = ctx.rng().random_range(-1.0_f32..1.0_f32);
        }
        st.cycle_started = true;

        let phase = st.phase;
        let raw = match self.shape {
            LfoShape::Sine => (2.0 * PI * phase).sin(),
            LfoShape::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
            LfoShape::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoShape::Saw => 2.0 * phase - 1.0,
            LfoShape::Random => st.held_value,
        };

        // Advance the phase.  rem_euclid keeps it in [0, 1) even if a
        // negative rate slips through.
        let next_phase = (phase + rate / sr).rem_euclid(1.0);
        let wrapped = next_phase < phase;
        st.phase = next_phase;

        // On a cycle wrap, draw the next cycle's held value (Random only).
        // Drawing exactly once per cycle — rather than every sample —
        // keeps the step-and-hold honest and the rng stream lean.
        if is_random && wrapped {
            st.held_value = ctx.rng().random_range(-1.0_f32..1.0_f32);
        }

        if let Some(s) = ctx.state_mut::<LfoState>() {
            *s = st;
        }

        raw * self.depth + self.offset
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(LfoState::default()))
    }
}

crate::impl_genotype!(Lfo {
    rate_hz: f32_log(0.5, 0.01, 30.0),
    shape: enum_cycle([
        LfoShape::Sine,
        LfoShape::Triangle,
        LfoShape::Square,
        LfoShape::Saw,
        LfoShape::Random
    ]),
    depth: f32(0.2, 0.0, 10_000.0),
    // Negative literals must be parenthesised here so the macro's
    // tt-matched parameter list keeps `-10_000.0` as one token tree.
    offset: f32(0.2, (-10_000.0), 10_000.0),
});

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn drive(lfo: &Lfo, sample_rate: u32, n: usize) -> Vec<f32> {
        let mut state = lfo.init_state();
        let inputs: &[(&str, f32)] = &[];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
            let mut ctx =
                BakeContext::new(sample_rate, i as u64, n as u64, &mut rng, inputs, state_ref);
            out.push(lfo.sample(&mut ctx));
        }
        out
    }

    #[test]
    fn sine_lfo_at_one_hz_completes_full_cycle_in_one_second() {
        // 1 Hz, depth=1, offset=0 → value at t=0.25s is sin(π/2) ≈ 1.
        let lfo = Lfo {
            rate_hz: 1.0,
            shape: LfoShape::Sine,
            depth: 1.0,
            offset: 0.0,
        };
        let buf = drive(&lfo, 44_100, 44_100);
        // ~quarter cycle.
        let quarter = buf[11_025];
        assert!(
            (quarter - 1.0).abs() < 0.01,
            "quarter cycle value: {quarter}"
        );
        // ~half cycle should be near 0.
        assert!(buf[22_050].abs() < 0.05);
    }

    #[test]
    fn lfo_output_obeys_depth_and_offset() {
        let lfo = Lfo {
            rate_hz: 0.5,
            shape: LfoShape::Sine,
            depth: 900.0,
            offset: 1_100.0,
        };
        // 0.5 Hz over 2 s = one full cycle; min and max reach the rails.
        let buf = drive(&lfo, 44_100, 88_200);
        let max = buf.iter().cloned().fold(f32::MIN, f32::max);
        let min = buf.iter().cloned().fold(f32::MAX, f32::min);
        // Should sweep roughly 200..2000 — the wind-LFO acceptance range.
        assert!(min > 100.0 && min < 300.0, "min: {min}");
        assert!(max > 1900.0 && max < 2100.0, "max: {max}");
    }

    #[test]
    fn square_lfo_takes_only_two_distinct_values() {
        let lfo = Lfo {
            rate_hz: 4.0,
            shape: LfoShape::Square,
            depth: 1.0,
            offset: 0.0,
        };
        let buf = drive(&lfo, 44_100, 44_100);
        for s in &buf {
            assert!(
                (*s - 1.0).abs() < 1e-6 || (*s + 1.0).abs() < 1e-6,
                "square value off-rail: {s}"
            );
        }
    }

    #[test]
    fn random_lfo_outputs_change_over_time() {
        let lfo = Lfo {
            rate_hz: 5.0,
            shape: LfoShape::Random,
            depth: 1.0,
            offset: 0.0,
        };
        let buf = drive(&lfo, 44_100, 44_100);
        // Across a 1-second buffer of a 5 Hz S&H, we expect at least
        // five distinct held values.
        let mut distinct: Vec<f32> = Vec::new();
        for v in &buf {
            if !distinct.iter().any(|x| (x - v).abs() < 1e-6) {
                distinct.push(*v);
            }
        }
        assert!(
            distinct.len() >= 5,
            "S&H gave only {} distinct values",
            distinct.len()
        );
    }

    #[test]
    fn random_lfo_first_cycle_is_a_nonzero_held_value() {
        // Regression for the first-cycle-silence bug: sample 0 must draw a
        // real value and hold it flat for the whole first cycle (instead of
        // being stuck at 0 until the first phase wrap).
        let lfo = Lfo {
            rate_hz: 2.0, // first cycle = sr / 2 = 22_050 samples
            shape: LfoShape::Random,
            depth: 1.0,
            offset: 0.0,
        };
        // Drive a sub-range comfortably inside the ~22_050-sample first
        // cycle (the exact wrap drifts a sample or two from f32 phase
        // accumulation, so we stay clear of the boundary).
        let buf = drive(&lfo, 44_100, 20_000);
        let first = buf[0];
        assert!(first != 0.0, "first S&H value should not be silent");
        for (i, s) in buf.iter().enumerate() {
            assert!(
                (*s - first).abs() < 1e-9,
                "first cycle not held flat at sample {i}: {s} vs {first}"
            );
        }
    }

    #[test]
    fn lfo_falls_back_to_offset_when_state_missing() {
        let lfo = Lfo {
            rate_hz: 1.0,
            shape: LfoShape::Sine,
            depth: 1.0,
            offset: 0.5,
        };
        let inputs: &[(&str, f32)] = &[];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, inputs, None);
        let s = lfo.sample(&mut ctx);
        // With no state, raw waveform is 0, so output is just offset.
        assert!((s - 0.5).abs() < 1e-6);
    }
}
