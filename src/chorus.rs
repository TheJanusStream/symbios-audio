//! Chorus — an internally-modulated fractional delay line.
//!
//! A chorus thickens a signal by mixing it with a copy of itself read back
//! from a short delay whose length is swept by a low-frequency oscillator.
//! The moving delay detunes the wet copy a few cents either side of the dry
//! signal; summing the two produces the shimmering, ensemble-like widening
//! the effect is named for.
//!
//! Unlike the crate's filters — which read their modulation off a wired port
//! so an external [`crate::lfo::Lfo`] can sweep them — a chorus carries its
//! *own* sine LFO ([`Chorus::rate_hz`] / [`Chorus::depth_ms`]).  The internal
//! modulator is what makes a chorus a chorus rather than a static delay, so
//! the node is self-contained: drop it after any signal source and it works.
//!
//! # Input ports
//!
//! - `"in"` — signal to process.  Defaults to zero (silence) when unwired,
//!   so an isolated chorus node bakes silence rather than crashing — the same
//!   convention the filters use.
//!
//! # Parameters
//!
//! - [`Chorus::base_delay_ms`] — centre delay the LFO sweeps around.
//! - [`Chorus::depth_ms`] — peak deviation of the sweep, added to and
//!   subtracted from the base delay across one LFO cycle.
//! - [`Chorus::rate_hz`] — LFO frequency.
//! - [`Chorus::feedback`] — fraction of the delayed signal fed back into the
//!   line (0 is a plain chorus; higher values push toward flanger territory).
//!   Clamped below 1.0 so the line can't run away.
//! - [`Chorus::mix`] — dry/wet blend.  `0.0` is the untouched input (an exact
//!   pass-through); `1.0` is the wet delayed signal only.
//!
//! # State machinery
//!
//! [`ChorusState`] holds the ring buffer, its write cursor, and the LFO
//! phase.  The buffer's length depends on the sample rate, which
//! [`Node::init_state`] does not receive — so the buffer is sized lazily on
//! the first [`Node::sample`] call (where `ctx.sample_rate` is known and
//! constant for the whole bake).  Sizing draws no RNG, so two bakes at the
//! same rate still match bit-for-bit.

use std::any::Any;
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

/// Hard ceiling on feedback, keeping the delay line strictly contractive so
/// it can never blow up under sustained input.
const MAX_FEEDBACK: f32 = 0.95;

/// Internally-modulated chorus effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chorus {
    /// LFO rate in Hz — how fast the delay is swept.
    pub rate_hz: f32,
    /// Peak sweep deviation in milliseconds, added to / subtracted from
    /// [`Self::base_delay_ms`] over one LFO cycle.
    pub depth_ms: f32,
    /// Centre delay in milliseconds the LFO sweeps around.
    pub base_delay_ms: f32,
    /// Feedback fraction (clamped to `[0, 0.95]`).  Zero is a clean chorus.
    pub feedback: f32,
    /// Dry/wet blend in `[0, 1]`.  `0.0` passes the input through unchanged.
    pub mix: f32,
}

impl Default for Chorus {
    fn default() -> Self {
        Self {
            rate_hz: 0.8,
            depth_ms: 2.0,
            base_delay_ms: 8.0,
            feedback: 0.0,
            mix: 0.5,
        }
    }
}

/// Persistent state for a [`Chorus`].
///
/// `buf` is the delay ring (sized on the first sample); `write` is the next
/// write index; `phase` is the LFO phase accumulator in `[0, 1)`.
#[derive(Debug, Clone, Default)]
pub struct ChorusState {
    buf: Vec<f32>,
    write: usize,
    phase: f32,
}

impl Chorus {
    /// Number of delay samples the buffer must hold for the current sample
    /// rate: the longest delay the LFO can reach (`base + depth`) plus a few
    /// samples of headroom for the fractional read's interpolation tap.
    fn buffer_len(&self, sample_rate: f32) -> usize {
        let max_ms = (self.base_delay_ms + self.depth_ms).max(0.0);
        let samples = (max_ms * 0.001 * sample_rate).ceil() as usize;
        samples.max(1) + 4
    }
}

impl Node for Chorus {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let x = ctx.input("in");
        let sr = ctx.sample_rate as f32;

        let st = match ctx.state_mut::<ChorusState>() {
            Some(s) => s,
            // No state container — behave as a dry pass-through, matching the
            // filters' "missing state ⇒ identity" convention.
            None => return x,
        };

        // Lazy buffer sizing: the length depends on the sample rate, which
        // isn't available at `init_state` time.  Allocated once on sample 0.
        let needed = self.buffer_len(sr);
        if st.buf.len() < needed {
            st.buf = vec![0.0; needed];
            st.write = 0;
        }
        let len = st.buf.len();

        // Modulated delay: a sine LFO sweeps the read position ±depth around
        // the base delay.  Clamp into the buffer so a misconfigured depth
        // can't read out of bounds.
        let lfo = (2.0 * PI * st.phase).sin();
        let delay_ms = self.base_delay_ms + self.depth_ms * lfo;
        let delay_samples = (delay_ms.max(0.0) * 0.001 * sr).clamp(0.0, (len - 1) as f32);

        // Fractional read, `delay_samples` behind the write cursor, with
        // linear interpolation between the two straddling taps.
        let read_pos = (st.write as f32 - delay_samples).rem_euclid(len as f32);
        let i0 = read_pos.floor() as usize % len;
        let i1 = (i0 + 1) % len;
        let frac = read_pos - read_pos.floor();
        let delayed = st.buf[i0] * (1.0 - frac) + st.buf[i1] * frac;

        // Write input plus feedback, then advance the cursor and LFO phase.
        let fb = self.feedback.clamp(0.0, MAX_FEEDBACK);
        st.buf[st.write] = x + delayed * fb;
        st.write = (st.write + 1) % len;
        let rate = self.rate_hz.max(0.0);
        st.phase = (st.phase + rate / sr).rem_euclid(1.0);

        let mix = self.mix.clamp(0.0, 1.0);
        x * (1.0 - mix) + delayed * mix
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(ChorusState::default()))
    }
}

crate::impl_genotype!(Chorus {
    rate_hz: f32_log(0.5, 0.05, 12.0),
    depth_ms: f32(2.0, 0.0, 20.0),
    base_delay_ms: f32(2.0, 1.0, 40.0),
    feedback: f32(0.1, 0.0, 0.95),
    mix: f32(0.1, 0.0, 1.0),
});

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Drive a signal through one chorus via `Node::sample`, returning the
    /// output buffer.  Mirrors the filter module's `drive` harness.
    fn drive(ch: &Chorus, sample_rate: u32, signal: &[f32]) -> Vec<f32> {
        let mut state = ch.init_state().expect("chorus installs state");
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let total = signal.len() as u64;
        let mut out = Vec::with_capacity(signal.len());
        for (i, x) in signal.iter().copied().enumerate() {
            let inputs = [("in", x)];
            let mut ctx = BakeContext::new(
                sample_rate,
                i as u64,
                total,
                &mut rng,
                &inputs,
                Some(&mut *state),
            );
            out.push(ch.sample(&mut ctx));
        }
        out
    }

    fn sine_buffer(sample_rate: u32, freq: f32, secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stays_finite_under_random_input() {
        let ch = Chorus {
            feedback: 0.9,
            ..Chorus::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let signal: Vec<f32> = (0..20_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&ch, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite(), "sample {i} not finite: {y}");
        }
    }

    #[test]
    fn mix_zero_is_exact_pass_through() {
        let ch = Chorus {
            mix: 0.0,
            ..Chorus::default()
        };
        let signal = sine_buffer(44_100, 440.0, 0.05);
        let out = drive(&ch, 44_100, &signal);
        for (i, (y, x)) in out.iter().zip(signal.iter()).enumerate() {
            assert_eq!(*y, *x, "mix=0 should pass input through at sample {i}");
        }
    }

    #[test]
    fn wet_signal_differs_from_dry() {
        // With a non-zero mix and depth, the swept delay must actually colour
        // the signal — the output can't equal the dry input everywhere.
        let ch = Chorus {
            mix: 1.0,
            depth_ms: 3.0,
            ..Chorus::default()
        };
        let signal = sine_buffer(44_100, 440.0, 0.2);
        let out = drive(&ch, 44_100, &signal);
        // Compare past the initial buffer fill so the delay line has content.
        let differs = out
            .iter()
            .zip(signal.iter())
            .skip(2_000)
            .any(|(y, x)| (y - x).abs() > 1e-3);
        assert!(differs, "wet chorus output should differ from dry input");
    }

    #[test]
    fn falls_back_to_pass_through_when_state_missing() {
        let ch = Chorus::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("in", 0.37_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        assert_eq!(ch.sample(&mut ctx), 0.37);
    }

    #[test]
    fn high_feedback_does_not_explode() {
        // Feedback above the cap is clamped, so even a constant DC drive
        // stays bounded rather than integrating to infinity.
        let ch = Chorus {
            feedback: 5.0,
            mix: 1.0,
            ..Chorus::default()
        };
        let signal = vec![1.0_f32; 44_100];
        let out = drive(&ch, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite() && y.abs() < 1_000.0, "runaway at {i}: {y}");
        }
    }

    #[test]
    fn genotype_clamps_fields_to_ranges() {
        use symbios_genetics::Genotype;
        let mut ch = Chorus::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..200 {
            ch.mutate(&mut rng, 1.0);
            assert!((0.05..=12.0).contains(&ch.rate_hz));
            assert!((0.0..=20.0).contains(&ch.depth_ms));
            assert!((1.0..=40.0).contains(&ch.base_delay_ms));
            assert!((0.0..=0.95).contains(&ch.feedback));
            assert!((0.0..=1.0).contains(&ch.mix));
        }
    }

    #[test]
    fn serde_round_trips_through_node_kind() {
        use crate::node::NodeKind;
        let kind = NodeKind::Chorus(Chorus::default());
        let json = serde_json::to_string(&kind).unwrap();
        let back: NodeKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }
}
