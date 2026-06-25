//! Biquad filters — lowpass, highpass, bandpass.
//!
//! Single Robert Bristow-Johnson cookbook implementation, three variants
//! distinguished by their coefficient calculation.  Direct Form I —
//! straightforward to read, four floats of state per voice, and the
//! numerical behaviour is good enough for the audible signal levels this
//! crate produces.
//!
//! Coefficients are recomputed every sample.  That is the *whole point*
//! of doing it this way rather than caching: because cross-node modulation
//! is wired up, an LFO driving the cutoff produces audio-rate frequency
//! sweeps for free — the filter already reads the current cutoff value, no
//! extra plumbing needed.  If profiling later shows it matters, a
//! static-cutoff fast path can be added behind a detection check.
//!
//! # Input ports
//!
//! - `"in"` — signal to filter.  Defaults to zero (silent input) if
//!   unwired, so an isolated filter node bakes silence rather than
//!   crashing.
//! - `"cutoff_hz"` (lowpass / highpass) or `"center_hz"` (bandpass) —
//!   *added* to the configured cutoff/centre frequency each sample, so an
//!   LFO here sweeps the corner (this is how the wind drone gets its
//!   motion).  Note the deliberate name difference: a lowpass/highpass has
//!   a *cutoff*, a bandpass has a *centre*.
//! - `"q"` — added to the configured `q` each sample.
//!
//! All modulation ports default to zero when unwired, leaving the
//! configured value untouched.  Cutoff and Q are clamped to a stable range
//! at sample time, so out-of-range modulation can't blow the filter up.
//!
//! # State machinery
//!
//! [`BiquadState`] is installed via [`Node::init_state`] using the
//! type-erased state path from Phase 1 #5.  All three filter types share
//! the same state shape and use the same Direct-Form-I update.

use std::any::Any;
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

/// Direct-Form-I biquad history: two input samples and two output samples.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiquadState {
    pub(crate) x1: f32,
    pub(crate) x2: f32,
    pub(crate) y1: f32,
    pub(crate) y2: f32,
}

/// Internal coefficient bundle for one biquad evaluation.  Already
/// normalised by `a0` so the difference equation reads cleanly.
#[derive(Debug, Clone, Copy)]
struct BiquadCoefs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

/// Which biquad variant a coefficient calculator should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BiquadKind {
    Lowpass,
    Highpass,
    Bandpass,
}

const MIN_Q: f32 = 0.001;
const MIN_CUTOFF_HZ: f32 = 1.0;
/// Keep the warped digital cutoff strictly inside `(0, π)` to stay
/// numerically stable.  `0.499` rather than `0.5` to leave room for
/// rounding without crossing Nyquist.
const NYQUIST_FRACTION: f32 = 0.499;

#[inline]
fn clamp_cutoff(cutoff_hz: f32, sample_rate: f32) -> f32 {
    let max = sample_rate * NYQUIST_FRACTION;
    cutoff_hz.clamp(MIN_CUTOFF_HZ, max.max(MIN_CUTOFF_HZ + 1.0))
}

#[inline]
fn clamp_q(q: f32) -> f32 {
    if q.is_finite() { q.max(MIN_Q) } else { MIN_Q }
}

/// Robert Bristow-Johnson cookbook coefficients for the chosen filter
/// kind at the requested cutoff and Q.  See the audio-eq-cookbook PDF
/// (Bristow-Johnson, 2005) for the derivations.
fn compute_coefs(kind: BiquadKind, cutoff_hz: f32, q: f32, sample_rate: f32) -> BiquadCoefs {
    let cutoff = clamp_cutoff(cutoff_hz, sample_rate);
    let q = clamp_q(q);
    let omega = 2.0 * PI * cutoff / sample_rate;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let alpha = sin_w / (2.0 * q);

    let (b0_raw, b1_raw, b2_raw) = match kind {
        BiquadKind::Lowpass => ((1.0 - cos_w) * 0.5, 1.0 - cos_w, (1.0 - cos_w) * 0.5),
        BiquadKind::Highpass => ((1.0 + cos_w) * 0.5, -(1.0 + cos_w), (1.0 + cos_w) * 0.5),
        BiquadKind::Bandpass => (alpha, 0.0, -alpha),
    };
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha;

    BiquadCoefs {
        b0: b0_raw / a0,
        b1: b1_raw / a0,
        b2: b2_raw / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// One step of the Direct-Form-I difference equation.  Returns the new
/// output sample and updates the four state values in place.
#[inline]
fn process(input: f32, state: &mut BiquadState, c: &BiquadCoefs) -> f32 {
    let y = c.b0 * input + c.b1 * state.x1 + c.b2 * state.x2 - c.a1 * state.y1 - c.a2 * state.y2;
    state.x2 = state.x1;
    state.x1 = input;
    state.y2 = state.y1;
    state.y1 = y;
    y
}

// --- Lowpass ----------------------------------------------------------------

/// Second-order biquad lowpass.  Passes frequencies below `cutoff_hz` and
/// attenuates everything above at ~12 dB/octave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiquadLowpass {
    pub cutoff_hz: f32,
    pub q: f32,
}

impl Default for BiquadLowpass {
    fn default() -> Self {
        Self {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2, // ~0.707, Butterworth
        }
    }
}

impl Node for BiquadLowpass {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let input = ctx.input("in");
        let sr = ctx.sample_rate as f32;
        let cutoff = self.cutoff_hz + ctx.input("cutoff_hz");
        let q = self.q + ctx.input("q");
        let coefs = compute_coefs(BiquadKind::Lowpass, cutoff, q, sr);
        match ctx.state_mut::<BiquadState>() {
            Some(state) => process(input, state, &coefs),
            None => input,
        }
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(BiquadState::default()))
    }
}

// --- Highpass ---------------------------------------------------------------

/// Second-order biquad highpass.  Passes frequencies above `cutoff_hz`
/// and attenuates everything below at ~12 dB/octave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiquadHighpass {
    pub cutoff_hz: f32,
    pub q: f32,
}

impl Default for BiquadHighpass {
    fn default() -> Self {
        Self {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        }
    }
}

impl Node for BiquadHighpass {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let input = ctx.input("in");
        let sr = ctx.sample_rate as f32;
        let cutoff = self.cutoff_hz + ctx.input("cutoff_hz");
        let q = self.q + ctx.input("q");
        let coefs = compute_coefs(BiquadKind::Highpass, cutoff, q, sr);
        match ctx.state_mut::<BiquadState>() {
            Some(state) => process(input, state, &coefs),
            None => input,
        }
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(BiquadState::default()))
    }
}

// --- Bandpass (constant peak-gain variant) ----------------------------------

/// Second-order biquad bandpass centered at `center_hz`.  Constant-peak-
/// gain variant (cookbook), so peak amplitude at the centre is roughly 1.0
/// regardless of `Q` — high-Q just narrows the band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiquadBandpass {
    pub center_hz: f32,
    pub q: f32,
}

impl Default for BiquadBandpass {
    fn default() -> Self {
        Self {
            center_hz: 1_000.0,
            q: 1.0,
        }
    }
}

impl Node for BiquadBandpass {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let input = ctx.input("in");
        let sr = ctx.sample_rate as f32;
        let center = self.center_hz + ctx.input("center_hz");
        let q = self.q + ctx.input("q");
        let coefs = compute_coefs(BiquadKind::Bandpass, center, q, sr);
        match ctx.state_mut::<BiquadState>() {
            Some(state) => process(input, state, &coefs),
            None => input,
        }
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(BiquadState::default()))
    }
}

// --- Genotype impls ---------------------------------------------------------

crate::impl_genotype!(BiquadLowpass {
    cutoff_hz: f32_log(0.5, 20.0, 20_000.0),
    q: f32(0.5, 0.1, 20.0),
});

crate::impl_genotype!(BiquadHighpass {
    cutoff_hz: f32_log(0.5, 20.0, 20_000.0),
    q: f32(0.5, 0.1, 20.0),
});

crate::impl_genotype!(BiquadBandpass {
    center_hz: f32_log(0.5, 20.0, 20_000.0),
    q: f32(0.5, 0.1, 20.0),
});

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Drive `n` raw inputs through a single filter's `Node::sample`,
    /// returning the output buffer.  Each step builds a fresh `inputs`
    /// list so the wired sample value can change per step.
    fn drive<F: Node>(filt: &F, sample_rate: u32, signal: &[f32]) -> Vec<f32> {
        let mut state_box = filt.init_state().expect("filter must install state");
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut out = Vec::with_capacity(signal.len());
        let total = signal.len() as u64;
        for (i, x) in signal.iter().copied().enumerate() {
            let inputs = [("in", x)];
            let mut ctx = BakeContext::new(
                sample_rate,
                i as u64,
                total,
                &mut rng,
                &inputs,
                Some(&mut *state_box),
            );
            out.push(filt.sample(&mut ctx));
        }
        out
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f64 = buf.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum_sq / buf.len() as f64).sqrt() as f32
    }

    fn sine_buffer(sample_rate: u32, freq: f32, secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    // --- stability / numerical hygiene -------------------------------------

    #[test]
    fn lowpass_stays_finite_under_random_input() {
        let filt = BiquadLowpass {
            cutoff_hz: 5_000.0,
            q: 2.5,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let signal: Vec<f32> = (0..10_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&filt, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite(), "sample {i} not finite: {y}");
        }
    }

    #[test]
    fn highpass_stays_finite_under_random_input() {
        let filt = BiquadHighpass {
            cutoff_hz: 200.0,
            q: 5.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let signal: Vec<f32> = (0..10_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&filt, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite(), "sample {i} not finite: {y}");
        }
    }

    #[test]
    fn bandpass_stays_finite_under_random_input() {
        let filt = BiquadBandpass {
            center_hz: 1_000.0,
            q: 8.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let signal: Vec<f32> = (0..10_000)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        let out = drive(&filt, 44_100, &signal);
        for (i, y) in out.iter().enumerate() {
            assert!(y.is_finite(), "sample {i} not finite: {y}");
        }
    }

    // --- frequency response sanity -----------------------------------------

    #[test]
    fn lowpass_passes_low_blocks_high() {
        let filt = BiquadLowpass {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        };
        // Skip the first ~1 ms of output to let the filter settle.
        let low = sine_buffer(44_100, 200.0, 0.25);
        let high = sine_buffer(44_100, 8_000.0, 0.25);
        let low_out = drive(&filt, 44_100, &low);
        let high_out = drive(&filt, 44_100, &high);
        let r_low = rms(&low_out[1_000..]);
        let r_high = rms(&high_out[1_000..]);
        let atten_db = 20.0 * (r_low / r_high).log10();
        assert!(
            atten_db > 24.0,
            "LP attenuation insufficient: {atten_db} dB (need >24 dB)"
        );
    }

    #[test]
    fn highpass_blocks_low_passes_high() {
        let filt = BiquadHighpass {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        };
        let low = sine_buffer(44_100, 200.0, 0.25);
        let high = sine_buffer(44_100, 8_000.0, 0.25);
        let low_out = drive(&filt, 44_100, &low);
        let high_out = drive(&filt, 44_100, &high);
        let r_low = rms(&low_out[1_000..]);
        let r_high = rms(&high_out[1_000..]);
        let atten_db = 20.0 * (r_high / r_low).log10();
        assert!(
            atten_db > 24.0,
            "HP attenuation insufficient: {atten_db} dB (need >24 dB)"
        );
    }

    #[test]
    fn bandpass_peaks_at_center() {
        let filt = BiquadBandpass {
            center_hz: 1_000.0,
            q: 4.0,
        };
        let center = sine_buffer(44_100, 1_000.0, 0.25);
        let off = sine_buffer(44_100, 100.0, 0.25);
        let c_out = drive(&filt, 44_100, &center);
        let o_out = drive(&filt, 44_100, &off);
        let r_c = rms(&c_out[1_000..]);
        let r_o = rms(&o_out[1_000..]);
        assert!(r_c > r_o, "BP center {r_c} should exceed off-band {r_o}");
    }

    // --- defensive / config edge cases -------------------------------------

    #[test]
    fn cutoff_above_nyquist_does_not_explode() {
        let filt = BiquadLowpass {
            cutoff_hz: 1_000_000.0,
            q: 1.0,
        };
        let signal = sine_buffer(44_100, 1_000.0, 0.05);
        let out = drive(&filt, 44_100, &signal);
        for y in out {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn zero_or_negative_q_clamps_safely() {
        let filt = BiquadLowpass {
            cutoff_hz: 1_000.0,
            q: 0.0,
        };
        let signal = sine_buffer(44_100, 1_000.0, 0.05);
        let out = drive(&filt, 44_100, &signal);
        for y in out {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn falls_back_to_pass_through_when_state_missing() {
        let filt = BiquadLowpass::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("in", 0.42_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        let y = filt.sample(&mut ctx);
        assert_eq!(y, 0.42);
    }

    // --- genotype ----------------------------------------------------------

    #[test]
    fn genotype_clamps_cutoff_and_q_to_ranges() {
        use symbios_genetics::Genotype;
        let mut filt = BiquadLowpass {
            cutoff_hz: 19_500.0,
            q: 19.0,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..200 {
            filt.mutate(&mut rng, 1.0);
            assert!((20.0..=20_000.0).contains(&filt.cutoff_hz));
            assert!((0.1..=20.0).contains(&filt.q));
        }
    }
}
