//! Signal-combining nodes — [`Mix`] and [`Gain`].
//!
//! The oscillators, noise sources, and filters in this crate each carry a
//! single signal through one downstream edge.  These two nodes are the glue
//! for *combining* and *scaling* signals inside one [`crate::patch::AudioPatch`]
//! so a single voice can be more than a linear chain.
//!
//! # [`Mix`] — additive bus
//!
//! Sums **every** wired input port (see [`BakeContext::input_sum`]) and
//! scales the result by `gain`.  Because a port already sums its own
//! fan-in connections (see [`crate::patch::GraphNode::inputs`]), a `Mix`
//! is only needed when the sources sit on *differently named* ports — but
//! it reads cleaner than relying on fan-in and gives one obvious place to
//! attach a master `gain`.  Wire a sine and a saw into, say, `"a"` and
//! `"b"` and the `Mix` output is their sum.
//!
//! # [`Gain`] — voltage-controlled amplifier
//!
//! Multiplies the `"in"` port by `gain + input("gain")`.  Unlike an
//! oscillator's *additive* `"amplitude"` port, this is a true multiply, so
//! wiring an envelope or LFO into `"gain"` gives a clean VCA / ring-mod /
//! tremolo.  With the default `gain = 1.0` and nothing on `"gain"`, it is a
//! unity pass-through.

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

fn default_gain() -> f32 {
    1.0
}

// --- Mix --------------------------------------------------------------------

/// Additive mixer — sums all wired input ports, scaled by `gain`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mix {
    /// Master multiplier applied to the summed inputs.
    #[serde(default = "default_gain")]
    pub gain: f32,
}

impl Default for Mix {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Node for Mix {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        ctx.input_sum() * self.gain
    }
}

// --- Gain (VCA) -------------------------------------------------------------

/// Voltage-controlled amplifier — `in * (gain + input("gain"))`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gain {
    /// Base multiplier.  Wired modulation on the `"gain"` input is added to
    /// this before the multiply, so an envelope on `"gain"` with `gain =
    /// 0.0` is a textbook VCA.
    #[serde(default = "default_gain")]
    pub gain: f32,
}

impl Default for Gain {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Node for Gain {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let input = ctx.input("in");
        let gain = self.gain + ctx.input("gain");
        input * gain
    }
}

// --- Genotype ---------------------------------------------------------------

crate::impl_genotype!(Mix {
    gain: f32(0.1, 0.0, 4.0),
});

crate::impl_genotype!(Gain {
    gain: f32(0.1, 0.0, 4.0),
});

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn mix_sums_all_ports_scaled_by_gain() {
        let mix = Mix { gain: 0.5 };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("a", 1.0_f32), ("b", 0.5), ("c", -0.25)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        // (1.0 + 0.5 - 0.25) * 0.5 = 0.625
        assert!((mix.sample(&mut ctx) - 0.625).abs() < 1e-6);
    }

    #[test]
    fn mix_of_no_inputs_is_silent() {
        let mix = Mix::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs: &[(&str, f32)] = &[];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, inputs, None);
        assert_eq!(mix.sample(&mut ctx), 0.0);
    }

    #[test]
    fn gain_multiplies_input_by_base_plus_cv() {
        let g = Gain { gain: 0.25 };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // gain = 0.25 + 0.5 = 0.75; in = 0.8 → 0.6
        let inputs = [("in", 0.8_f32), ("gain", 0.5)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        assert!((g.sample(&mut ctx) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn gain_default_is_unity_pass_through() {
        let g = Gain::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("in", 0.42_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        assert!((g.sample(&mut ctx) - 0.42).abs() < 1e-6);
    }

    #[test]
    fn gain_acts_as_vca_when_base_is_zero() {
        // Base gain 0 + envelope CV on "gain" => output is purely the
        // CV-scaled input, the classic VCA wiring.
        let g = Gain { gain: 0.0 };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let silent = [("in", 0.9_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &silent, None);
        assert_eq!(g.sample(&mut ctx), 0.0);
    }

    #[test]
    fn genotype_clamps_gain_to_range() {
        use symbios_genetics::Genotype;
        let mut mix = Mix { gain: 3.9 };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..200 {
            mix.mutate(&mut rng, 1.0);
            assert!((0.0..=4.0).contains(&mix.gain));
        }
    }
}
