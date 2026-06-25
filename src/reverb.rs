//! Reverb — a mono Freeverb (Schroeder-Moorer) reverberator.
//!
//! Jezar at Dreampoint's "Freeverb" is the canonical small, good-sounding
//! algorithmic reverb: a bank of parallel damped feedback-comb filters whose
//! summed output is smeared through a chain of allpass filters.  The combs
//! build the dense decaying tail; the allpasses diffuse it so individual echo
//! repeats blur into a smooth wash.  This is the mono reduction — one channel,
//! the classic comb/allpass tunings, no stereo spread.
//!
//! # Input ports
//!
//! - `"in"` — signal to reverberate.  Defaults to zero (silence) when
//!   unwired, so an isolated reverb node bakes silence — the filters' "missing
//!   connection reads zero" convention.
//!
//! # Parameters
//!
//! - [`Reverb::room_size`] — `[0, 1]`, mapped to comb feedback in
//!   `[0.70, 0.98]`.  Larger rooms ring longer.
//! - [`Reverb::damping`] — `[0, 1]`, how quickly high frequencies are absorbed
//!   in the comb feedback path.  Higher is darker / more felt-lined.
//! - [`Reverb::mix`] — dry/wet blend.  `0.0` is an exact pass-through; `1.0`
//!   is the wet tail only (still attenuated by Freeverb's fixed input gain, so
//!   the reverb sits politely under the dry signal at moderate mixes).
//!
//! # State machinery
//!
//! [`ReverbState`] owns the eight comb and four allpass delay lines.  Their
//! lengths scale with the sample rate (the tunings below are quoted at
//! 44.1 kHz), which [`Node::init_state`] does not receive — so the lines are
//! built lazily on the first [`Node::sample`] call.  No RNG is drawn, so two
//! bakes at the same rate remain bit-identical.

use std::any::Any;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

// --- Freeverb tuning constants ----------------------------------------------
//
// Quoted for 44.1 kHz; scaled to the bake's actual rate at state-build time.

/// Comb delay lengths in samples (parallel bank).
const COMB_TUNINGS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
/// Allpass delay lengths in samples (series chain).
const ALLPASS_TUNINGS: [usize; 4] = [556, 441, 341, 225];
/// Sample rate the tunings are quoted at.
const TUNING_SAMPLE_RATE: f32 = 44_100.0;
/// Fixed allpass feedback coefficient (Freeverb constant).
const ALLPASS_FEEDBACK: f32 = 0.5;
/// Input attenuation so a wet-only mix sits under the dry signal.
const FIXED_GAIN: f32 = 0.015;
/// `room_size` → comb-feedback mapping: `feedback = size * SCALE + OFFSET`.
const ROOM_SCALE: f32 = 0.28;
const ROOM_OFFSET: f32 = 0.7;
/// `damping` → comb damping coefficient: `damp = damping * DAMP_SCALE`.
const DAMP_SCALE: f32 = 0.4;

/// One damped feedback-comb filter — the tail-builder.
#[derive(Debug, Clone, Default)]
struct Comb {
    buf: Vec<f32>,
    idx: usize,
    /// One-pole lowpass state in the feedback path (the "damping").
    filter_store: f32,
}

impl Comb {
    fn with_len(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            idx: 0,
            filter_store: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32, feedback: f32, damp: f32) -> f32 {
        let output = self.buf[self.idx];
        // Lowpass the feedback so each pass loses highs (felt-lined room).
        self.filter_store = output * (1.0 - damp) + self.filter_store * damp;
        self.buf[self.idx] = input + self.filter_store * feedback;
        self.idx = (self.idx + 1) % self.buf.len();
        output
    }
}

/// One Schroeder allpass filter — the diffuser.
#[derive(Debug, Clone, Default)]
struct Allpass {
    buf: Vec<f32>,
    idx: usize,
}

impl Allpass {
    fn with_len(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            idx: 0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buf[self.idx];
        let output = -input + buffered;
        self.buf[self.idx] = input + buffered * ALLPASS_FEEDBACK;
        self.idx = (self.idx + 1) % self.buf.len();
        output
    }
}

/// Mono Freeverb reverberator config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reverb {
    /// Room size in `[0, 1]` — maps to comb feedback (longer tail when larger).
    pub room_size: f32,
    /// High-frequency damping in `[0, 1]` — higher is darker.
    pub damping: f32,
    /// Dry/wet blend in `[0, 1]`.  `0.0` passes the input through unchanged.
    pub mix: f32,
}

impl Default for Reverb {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            mix: 0.3,
        }
    }
}

/// Persistent state for a [`Reverb`] — the comb and allpass delay lines,
/// built lazily on the first sample (their lengths depend on the rate).
#[derive(Debug, Clone, Default)]
pub struct ReverbState {
    combs: Vec<Comb>,
    allpasses: Vec<Allpass>,
}

/// Scale a 44.1 kHz tuning to `sample_rate`, never shorter than one sample.
#[inline]
fn scaled_len(tuning: usize, sample_rate: f32) -> usize {
    ((tuning as f32 * sample_rate / TUNING_SAMPLE_RATE).round() as usize).max(1)
}

impl Node for Reverb {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let x = ctx.input("in");
        let sr = ctx.sample_rate as f32;

        let st = match ctx.state_mut::<ReverbState>() {
            Some(s) => s,
            // No state container — dry pass-through, matching the filters.
            None => return x,
        };

        // Lazy line construction: lengths depend on the sample rate, unknown
        // at `init_state` time.  Built once on sample 0.
        if st.combs.is_empty() {
            st.combs = COMB_TUNINGS
                .iter()
                .map(|&t| Comb::with_len(scaled_len(t, sr)))
                .collect();
            st.allpasses = ALLPASS_TUNINGS
                .iter()
                .map(|&t| Allpass::with_len(scaled_len(t, sr)))
                .collect();
        }

        let feedback = self.room_size.clamp(0.0, 1.0) * ROOM_SCALE + ROOM_OFFSET;
        let damp = self.damping.clamp(0.0, 1.0) * DAMP_SCALE;
        let input = x * FIXED_GAIN;

        // Parallel combs build the tail; series allpasses diffuse it.
        let mut wet = 0.0;
        for comb in &mut st.combs {
            wet += comb.process(input, feedback, damp);
        }
        for ap in &mut st.allpasses {
            wet = ap.process(wet);
        }

        let mix = self.mix.clamp(0.0, 1.0);
        x * (1.0 - mix) + wet * mix
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(ReverbState::default()))
    }
}

crate::impl_genotype!(Reverb {
    room_size: f32(0.1, 0.0, 1.0),
    damping: f32(0.1, 0.0, 1.0),
    mix: f32(0.1, 0.0, 1.0),
});

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Drive a signal through one reverb via `Node::sample`.
    fn drive(rv: &Reverb, sample_rate: u32, signal: &[f32]) -> Vec<f32> {
        let mut state = rv.init_state().expect("reverb installs state");
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
            out.push(rv.sample(&mut ctx));
        }
        out
    }

    fn energy(buf: &[f32]) -> f64 {
        buf.iter().map(|s| (*s as f64) * (*s as f64)).sum()
    }

    #[test]
    fn stays_finite_under_random_input() {
        let rv = Reverb {
            room_size: 1.0,
            damping: 0.0,
            mix: 1.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let signal: Vec<f32> = (0..20_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&rv, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite(), "sample {i} not finite: {y}");
        }
    }

    #[test]
    fn mix_zero_is_exact_pass_through() {
        let rv = Reverb {
            mix: 0.0,
            ..Reverb::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let signal: Vec<f32> = (0..5_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&rv, 44_100, &signal);
        for (i, (y, x)) in out.iter().zip(signal.iter()).enumerate() {
            assert_eq!(*y, *x, "mix=0 should pass input through at sample {i}");
        }
    }

    #[test]
    fn impulse_produces_a_decaying_tail() {
        // A single-sample impulse, wet-only.  After the input stops there must
        // be a non-trivial tail, and a later window must carry less energy
        // than an earlier one (the reverb decays rather than sustains).
        let rv = Reverb {
            room_size: 0.7,
            damping: 0.3,
            mix: 1.0,
        };
        let mut signal = vec![0.0_f32; 44_100];
        signal[0] = 1.0;
        let out = drive(&rv, 44_100, &signal);

        // Tail past the impulse must be audible.
        let tail = energy(&out[100..]);
        assert!(tail > 1e-6, "reverb tail should be non-trivial, got {tail}");

        // Energy in an early window exceeds a later one → decay.
        let early = energy(&out[2_000..6_000]);
        let late = energy(&out[30_000..34_000]);
        assert!(
            early > late,
            "reverb should decay: early {early} should exceed late {late}"
        );
    }

    #[test]
    fn falls_back_to_pass_through_when_state_missing() {
        let rv = Reverb::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("in", 0.42_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        assert_eq!(rv.sample(&mut ctx), 0.42);
    }

    #[test]
    fn line_lengths_scale_with_sample_rate() {
        // At 88.2 kHz the comb/allpass lines should be ~twice their 44.1 kHz
        // length, so the algorithm holds its tuning across rates.
        assert_eq!(scaled_len(COMB_TUNINGS[0], 88_200.0), COMB_TUNINGS[0] * 2);
        assert_eq!(scaled_len(ALLPASS_TUNINGS[0], 44_100.0), ALLPASS_TUNINGS[0]);
        assert_eq!(scaled_len(10, 1.0), 1, "never shorter than one sample");
    }

    #[test]
    fn genotype_clamps_fields_to_ranges() {
        use symbios_genetics::Genotype;
        let mut rv = Reverb::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..200 {
            rv.mutate(&mut rng, 1.0);
            assert!((0.0..=1.0).contains(&rv.room_size));
            assert!((0.0..=1.0).contains(&rv.damping));
            assert!((0.0..=1.0).contains(&rv.mix));
        }
    }

    #[test]
    fn serde_round_trips_through_node_kind() {
        use crate::node::NodeKind;
        let kind = NodeKind::Reverb(Reverb::default());
        let json = serde_json::to_string(&kind).unwrap();
        let back: NodeKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }
}
