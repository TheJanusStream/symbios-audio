//! White, pink, and brown noise generators.
//!
//! All three draw from [`crate::node::BakeContext::rng`], so every bake of
//! a noise-bearing patch with the same seed produces an identical buffer —
//! this is the contract the DID-seeded ambient layer in the Overlands
//! integration relies on.
//!
//! # Pink-noise filter choice
//!
//! Phase 1 picks **Paul Kellet's 3-band approximation** over the Voss-
//! McCartney algorithm or Kellet's more accurate 7-band variant.  Reasons:
//!
//! - Only three filter coefficients of state per voice (b0, b1, b2),
//!   which keeps the per-node state container compact.
//! - The −3 dB/octave slope is correct across the audible range to within
//!   ~0.4 dB, well below the perceptual threshold for sustained noise.
//! - Voss-McCartney requires a running pseudo-octave update schedule
//!   keyed off the sample index, which is awkward when sample_index is a
//!   `u64` — small bit-counting tricks work but obscure the intent.
//!
//! If the 0.4 dB ripple ever matters, the more accurate 7-band Kellet
//! filter is a drop-in replacement (extend [`PinkState`] and update the
//! coefficient table).
//!
//! # Brown-noise design
//!
//! A leaky integrator after Larry Trammel:
//! `last = (last + 0.02 * white) / 1.02`, output `= last * 3.5`.  The
//! `1 / 1.02` leak factor is the DC block — over long horizons the
//! integrator drains back toward zero, so output stays bounded around
//! ±1 without a separate high-pass filter.

use std::any::Any;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

// --- White ------------------------------------------------------------------

/// Uniform white noise, drawn fresh from the seeded RNG every sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhiteNoise {
    /// Output scale.  The raw RNG draw is in [−1, 1]; the sample is
    /// `draw * amplitude`.
    pub amplitude: f32,
}

impl Default for WhiteNoise {
    fn default() -> Self {
        Self { amplitude: 0.5 }
    }
}

impl Node for WhiteNoise {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let r = ctx.rng().random_range(-1.0_f32..1.0_f32);
        r * self.amplitude
    }
}

// --- Pink (Paul Kellet 3-band) ----------------------------------------------

/// Pink noise — Paul Kellet's 3-band filter, ~−3 dB/octave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinkNoise {
    pub amplitude: f32,
}

impl Default for PinkNoise {
    fn default() -> Self {
        Self { amplitude: 0.5 }
    }
}

/// Filter state for [`PinkNoise`].  Three first-order LP feedback values.
#[derive(Debug, Clone, Copy, Default)]
pub struct PinkState {
    b0: f32,
    b1: f32,
    b2: f32,
}

/// Output normalisation: the raw sum `b0 + b1 + b2 + 0.1848·white` runs
/// to roughly ±5 when `white ∈ [−1, 1]`, so scale to bring the typical
/// envelope inside ±1 before the user's amplitude multiplier.
const PINK_NORMALISE: f32 = 0.18;

impl Node for PinkNoise {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let white = ctx.rng().random_range(-1.0_f32..1.0_f32);
        // state_mut should never return None here because init_state
        // installs a PinkState — but if it ever does (baker bug or a node
        // wired up outside the standard path), gracefully fall back to
        // raw white scaled by amplitude rather than panicking mid-bake.
        let state = match ctx.state_mut::<PinkState>() {
            Some(s) => s,
            None => return white * self.amplitude,
        };
        state.b0 = 0.99765 * state.b0 + white * 0.0990460;
        state.b1 = 0.96300 * state.b1 + white * 0.2965164;
        state.b2 = 0.57000 * state.b2 + white * 1.0526913;
        let pink = (state.b0 + state.b1 + state.b2 + white * 0.1848) * PINK_NORMALISE;
        pink * self.amplitude
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(PinkState::default()))
    }
}

// --- Brown (Trammel leaky integrator) ---------------------------------------

/// Brown noise — leaky integrator of white, ~−6 dB/octave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrownNoise {
    pub amplitude: f32,
}

impl Default for BrownNoise {
    fn default() -> Self {
        Self { amplitude: 0.5 }
    }
}

/// Filter state for [`BrownNoise`].  Single first-order accumulator.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrownState {
    last: f32,
}

/// Trammel scale factor that maps the leaky-integrator output back into
/// approximately ±1 before the user's amplitude multiplier.
const BROWN_NORMALISE: f32 = 3.5;
const BROWN_STEP: f32 = 0.02;
const BROWN_LEAK_DIVISOR: f32 = 1.02;

impl Node for BrownNoise {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let white = ctx.rng().random_range(-1.0_f32..1.0_f32);
        let state = match ctx.state_mut::<BrownState>() {
            Some(s) => s,
            None => return white * self.amplitude,
        };
        state.last = (state.last + white * BROWN_STEP) / BROWN_LEAK_DIVISOR;
        // Rare excursions can momentarily exceed ±1; clamping protects
        // downstream stages from clipping without distorting the typical
        // signal because the integrator naturally lives well inside the
        // clamp range.
        let brown = (state.last * BROWN_NORMALISE).clamp(-1.0, 1.0);
        brown * self.amplitude
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(BrownState::default()))
    }
}

// --- Genotype impls ---------------------------------------------------------

crate::impl_genotype!(WhiteNoise {
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(PinkNoise {
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(BrownNoise {
    amplitude: f32(0.1, 0.0, 1.0),
});

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn pink_ctx<'a>(
        rng: &'a mut ChaCha8Rng,
        state: &'a mut PinkState,
        inputs: &'a [(&'a str, f32)],
    ) -> BakeContext<'a> {
        BakeContext::new(44_100, 0, 44_100, rng, inputs, Some(state))
    }

    fn brown_ctx<'a>(
        rng: &'a mut ChaCha8Rng,
        state: &'a mut BrownState,
        inputs: &'a [(&'a str, f32)],
    ) -> BakeContext<'a> {
        BakeContext::new(44_100, 0, 44_100, rng, inputs, Some(state))
    }

    #[test]
    fn white_is_bounded_by_amplitude() {
        let osc = WhiteNoise { amplitude: 0.4 };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs: &[(&str, f32)] = &[];
        for _ in 0..10_000 {
            let mut ctx = BakeContext::new(44_100, 0, 44_100, &mut rng, inputs, None);
            let s = osc.sample(&mut ctx);
            assert!(s.abs() < 0.4, "white sample {s} out of |s|<0.4");
        }
    }

    #[test]
    fn white_is_not_silent() {
        let osc = WhiteNoise::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs: &[(&str, f32)] = &[];
        let mut nonzero = 0;
        for _ in 0..1000 {
            let mut ctx = BakeContext::new(44_100, 0, 44_100, &mut rng, inputs, None);
            if osc.sample(&mut ctx).abs() > 1e-6 {
                nonzero += 1;
            }
        }
        // Practically all samples should be non-zero.
        assert!(
            nonzero > 990,
            "white noise far too quiet: {nonzero}/1000 non-zero"
        );
    }

    #[test]
    fn pink_state_evolves_with_samples() {
        let osc = PinkNoise::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut state = PinkState::default();
        let inputs: &[(&str, f32)] = &[];
        for _ in 0..200 {
            let mut ctx = pink_ctx(&mut rng, &mut state, inputs);
            osc.sample(&mut ctx);
        }
        // After 200 samples driven by uniform white, the LP-filter values
        // should have moved well off zero.
        assert!(state.b0.abs() > 0.01 || state.b1.abs() > 0.01 || state.b2.abs() > 0.01);
    }

    #[test]
    fn pink_falls_back_when_state_missing() {
        // Documenting the recovery path: without a PinkState in
        // BakeContext, the impl returns scaled white instead of panicking.
        let osc = PinkNoise { amplitude: 1.0 };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs: &[(&str, f32)] = &[];
        let mut ctx = BakeContext::new(44_100, 0, 44_100, &mut rng, inputs, None);
        let s = osc.sample(&mut ctx);
        assert!(s.abs() <= 1.0);
    }

    #[test]
    fn brown_stays_bounded_over_long_run() {
        let osc = BrownNoise { amplitude: 1.0 };
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD);
        let mut state = BrownState::default();
        let inputs: &[(&str, f32)] = &[];
        let mut max_abs = 0.0_f32;
        for _ in 0..100_000 {
            let mut ctx = brown_ctx(&mut rng, &mut state, inputs);
            let s = osc.sample(&mut ctx);
            max_abs = max_abs.max(s.abs());
        }
        // The integrator + clamp must keep output inside ±1.
        assert!(max_abs <= 1.0, "brown leaked past 1.0: {max_abs}");
    }

    #[test]
    fn state_downcast_returns_none_for_wrong_type() {
        // BrownState container with a PinkNoise impl asking for PinkState
        // should return None — the runtime downcast is the safety net.
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut wrong = BrownState::default();
        let inputs: &[(&str, f32)] = &[];
        let mut ctx = BakeContext::new(44_100, 0, 44_100, &mut rng, inputs, Some(&mut wrong));
        assert!(ctx.state_mut::<PinkState>().is_none());
    }
}
