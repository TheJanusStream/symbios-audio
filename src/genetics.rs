//! [`symbios_genetics::Genotype`] integration — declarative macro and
//! shared mutation helpers.
//!
//! Lifted from `bevy_symbios_texture::genetics` and trimmed to the field
//! kinds audio node configs need.  Every concrete node configuration
//! struct in this crate (oscillators, noise generators, ADSR, filters,
//! LFO, sequencer recipe) invokes [`crate::impl_genotype!`] so the type
//! plugs into `SimpleGA`, `Nsga2`, and `MapElites` from the
//! `symbios-genetics` crate for free.
//!
//! # Field kinds
//!
//! | Kind | Mutation | Crossover |
//! |------|----------|-----------|
//! | `seed` | Replace with random `u32` | 50/50 pick |
//! | `f64(half, min, max)` | Uniform perturbation, clamp | 50/50 pick |
//! | `f64_round(half, min, max)` | Perturbation + `.round()` | 50/50 pick |
//! | `f32(half, min, max)` | Uniform perturbation, clamp | 50/50 pick |
//! | `f32_log(half_octaves, min, max)` | Multiply by `2^uniform(±half)`, clamp | 50/50 pick |
//! | `usize(min, max)` | ±1 step, clamp | 50/50 pick |
//! | `bool` | Flip with probability `rate` | 50/50 pick |
//! | `enum_cycle([V1, V2, …])` | Cycle to next variant | 50/50 pick (clone) |
//! | `genotype` | Delegate to nested `Genotype` | Delegate to nested crossover |
//!
//! Optional `post_mutate` / `post_crossover` blocks run custom fixup logic
//! after the per-field operations.
//!
//! # Coverage note
//!
//! The macro is deliberately general.  The audio node configs in this crate
//! only exercise the `f32`, `f32_log`, `bool`, and `enum_cycle` kinds; the
//! `seed`, `f64`, `f64_round`, and `usize` kinds (and the matching mutation
//! helpers below, each `#[allow(dead_code)]`) are kept for parity with the
//! sibling `symbios-texture` configs and are covered only by this module's
//! `TestConfig`, not by any real node.

use rand::Rng;

/// Perturb a `f64` by a uniform step in `(-half_range, +half_range)` with
/// probability `rate`, clamped to `[min, max]`.
#[allow(dead_code)]
#[inline]
pub(crate) fn mutate_f64<R: Rng>(
    val: f64,
    rng: &mut R,
    rate: f32,
    half_range: f64,
    min: f64,
    max: f64,
) -> f64 {
    if rng.random::<f32>() < rate {
        (val + (rng.random::<f64>() - 0.5) * 2.0 * half_range).clamp(min, max)
    } else {
        val
    }
}

/// Perturb a `f32` by a uniform step in `(-half_range, +half_range)` with
/// probability `rate`, clamped to `[min, max]`.
#[allow(dead_code)]
#[inline]
pub(crate) fn mutate_f32<R: Rng>(
    val: f32,
    rng: &mut R,
    rate: f32,
    half_range: f32,
    min: f32,
    max: f32,
) -> f32 {
    if rng.random::<f32>() < rate {
        (val + (rng.random::<f32>() - 0.5) * 2.0 * half_range).clamp(min, max)
    } else {
        val
    }
}

/// Multiplicative `f32` perturbation: scale by `2^uniform(-half_octaves, +half_octaves)`
/// with probability `rate`, clamped to `[min, max]`.
///
/// Frequency fields in audio configs are log-perceptual — a doubling is one
/// octave regardless of whether you start at 100 Hz or 1 kHz.  Mutating
/// additively biases search toward small absolute moves, which at high
/// frequencies barely shifts pitch and at low frequencies cracks the
/// signal in half.  Mutating multiplicatively keeps the genetic walk
/// musically reasonable everywhere on the spectrum.
#[allow(dead_code)]
#[inline]
pub(crate) fn mutate_f32_log<R: Rng>(
    val: f32,
    rng: &mut R,
    rate: f32,
    half_octaves: f32,
    min: f32,
    max: f32,
) -> f32 {
    if rng.random::<f32>() < rate {
        let octaves = (rng.random::<f32>() - 0.5) * 2.0 * half_octaves;
        (val * 2.0_f32.powf(octaves)).clamp(min, max)
    } else {
        val
    }
}

/// Perturb a `usize` by ±1 with probability `rate`, clamped to `[min, max]`.
#[allow(dead_code)]
#[inline]
pub(crate) fn mutate_usize<R: Rng>(
    val: usize,
    rng: &mut R,
    rate: f32,
    min: usize,
    max: usize,
) -> usize {
    if rng.random::<f32>() < rate {
        if rng.random::<bool>() {
            val.saturating_add(1).min(max)
        } else {
            val.saturating_sub(1).max(min)
        }
    } else {
        val
    }
}

/// Replace a `u32` seed entirely with probability `rate`.
#[allow(dead_code)]
#[inline]
pub(crate) fn mutate_seed<R: Rng>(val: u32, rng: &mut R, rate: f32) -> u32 {
    if rng.random::<f32>() < rate {
        rng.random::<u32>()
    } else {
        val
    }
}

/// Generates a `Genotype` (mutate + crossover) impl for a config struct.
///
/// See module docs for the supported field kinds.  Invoked once per
/// node-config struct in this crate.
#[macro_export]
macro_rules! impl_genotype {
    // Entry point
    (
        $Config:ty {
            $( $field:ident : $kind:tt $( ( $($param:tt),* ) )? ),+ $(,)?
        }
        $( post_mutate: |$ms:ident| $mutate_fix:block )?
        $( post_crossover: |$cs:ident| $crossover_fix:block )?
    ) => {
        impl ::symbios_genetics::Genotype for $Config {
            fn mutate<R: ::rand::Rng>(&mut self, rng: &mut R, rate: f32) {
                $(
                    $crate::impl_genotype!(
                        @mutate self, rng, rate, $field, $kind
                        $( ( $($param),* ) )?
                    );
                )+
                $( let $ms = self; $mutate_fix )?
            }

            fn crossover<R: ::rand::Rng>(&self, other: &Self, rng: &mut R) -> Self {
                #[allow(unused_mut)]
                let mut child = Self {
                    $(
                        $field: $crate::impl_genotype!(
                            @crossover self, other, rng, $field, $kind
                            $( ( $($param),* ) )?
                        ),
                    )+
                };
                $( let $cs = &mut child; $crossover_fix )?
                child
            }
        }
    };

    // --- mutate arms --------------------------------------------------------

    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, seed) => {
        $s.$f = $crate::genetics::mutate_seed($s.$f, $rng, $rate);
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, f64 ($hr:expr, $min:expr, $max:expr)) => {
        $s.$f = $crate::genetics::mutate_f64($s.$f, $rng, $rate, $hr, $min, $max);
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, f64_round ($hr:expr, $min:expr, $max:expr)) => {
        $s.$f = $crate::genetics::mutate_f64($s.$f, $rng, $rate, $hr, $min, $max).round();
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, f32 ($hr:expr, $min:expr, $max:expr)) => {
        $s.$f = $crate::genetics::mutate_f32($s.$f, $rng, $rate, $hr, $min, $max);
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, f32_log ($hr:expr, $min:expr, $max:expr)) => {
        $s.$f = $crate::genetics::mutate_f32_log($s.$f, $rng, $rate, $hr, $min, $max);
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, usize ($min:expr, $max:expr)) => {
        $s.$f = $crate::genetics::mutate_usize($s.$f, $rng, $rate, $min, $max);
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, bool) => {
        if ::rand::Rng::random::<f32>($rng) < $rate { $s.$f = !$s.$f; }
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, enum_cycle ([ $($variant:expr),+ ])) => {
        if ::rand::Rng::random::<f32>($rng) < $rate {
            let variants = [ $($variant),+ ];
            let cur = variants.iter().position(|v| *v == $s.$f).unwrap_or(0);
            $s.$f = variants[(cur + 1) % variants.len()].clone();
        }
    };
    (@mutate $s:ident, $rng:ident, $rate:ident, $f:ident, genotype) => {
        ::symbios_genetics::Genotype::mutate(&mut $s.$f, $rng, $rate);
    };

    // --- crossover arms -----------------------------------------------------

    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, seed) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, f64 ($($p:tt)*)) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, f64_round ($($p:tt)*)) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, f32 ($($p:tt)*)) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, f32_log ($($p:tt)*)) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, usize ($($p:tt)*)) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, bool) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f } else { $b.$f }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, enum_cycle ([ $($variant:expr),+ ])) => {
        if ::rand::Rng::random::<bool>($rng) { $a.$f.clone() } else { $b.$f.clone() }
    };
    (@crossover $a:ident, $b:ident, $rng:ident, $f:ident, genotype) => {
        ::symbios_genetics::Genotype::crossover(&$a.$f, &$b.$f, $rng)
    };
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use serde::{Deserialize, Serialize};
    use symbios_genetics::Genotype;

    // Exercise every field kind so the macro and helpers are covered by
    // regression tests before any real node config is added.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    enum Flavour {
        Sine,
        Saw,
        Square,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestConfig {
        seed: u32,
        frequency: f64,
        partials: f64,
        gain: f32,
        cycles: usize,
        inverted: bool,
        flavour: Flavour,
    }

    impl_genotype!(TestConfig {
        seed: seed,
        frequency: f64(50.0, 20.0, 20_000.0),
        partials: f64_round(1.0, 1.0, 64.0),
        gain: f32(0.1, 0.0, 1.0),
        cycles: usize(1, 32),
        inverted: bool,
        flavour: enum_cycle([Flavour::Sine, Flavour::Saw, Flavour::Square]),
    });

    fn seeded() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0xA1B2C3)
    }

    fn base() -> TestConfig {
        TestConfig {
            seed: 7,
            frequency: 440.0,
            partials: 4.0,
            gain: 0.5,
            cycles: 8,
            inverted: false,
            flavour: Flavour::Sine,
        }
    }

    #[test]
    fn mutate_rate_zero_is_identity() {
        let mut c = base();
        let snapshot = c.clone();
        c.mutate(&mut seeded(), 0.0);
        assert_eq!(c.seed, snapshot.seed);
        assert_eq!(c.cycles, snapshot.cycles);
        assert_eq!(c.inverted, snapshot.inverted);
        assert_eq!(c.flavour, snapshot.flavour);
        assert!((c.frequency - snapshot.frequency).abs() < f64::EPSILON);
        assert!((c.partials - snapshot.partials).abs() < f64::EPSILON);
        assert!((c.gain - snapshot.gain).abs() < f32::EPSILON);
    }

    #[test]
    fn mutate_clamps_within_field_bounds() {
        let mut c = TestConfig {
            frequency: 19_999.0,
            gain: 0.99,
            ..base()
        };
        let mut rng = seeded();
        for _ in 0..200 {
            c.mutate(&mut rng, 1.0);
            assert!(c.frequency >= 20.0 && c.frequency <= 20_000.0);
            assert!((0.0..=1.0).contains(&c.gain));
            assert!((1.0..=64.0).contains(&c.partials));
            assert!((1..=32).contains(&c.cycles));
        }
    }

    #[test]
    fn f64_round_kind_produces_integer_values() {
        let mut c = base();
        c.mutate(&mut seeded(), 1.0);
        assert!((c.partials - c.partials.round()).abs() < f64::EPSILON);
    }

    #[test]
    fn enum_cycle_cycles_variants() {
        let mut c = TestConfig {
            flavour: Flavour::Sine,
            ..base()
        };
        // With rate = 1.0 a cycle step happens deterministically.
        c.mutate(&mut seeded(), 1.0);
        assert_eq!(c.flavour, Flavour::Saw);
        c.mutate(&mut seeded(), 1.0);
        assert_eq!(c.flavour, Flavour::Square);
        c.mutate(&mut seeded(), 1.0);
        assert_eq!(c.flavour, Flavour::Sine);
    }

    #[test]
    fn crossover_takes_each_field_from_one_parent() {
        let a = base();
        let b = TestConfig {
            seed: 99,
            frequency: 880.0,
            partials: 16.0,
            gain: 0.1,
            cycles: 24,
            inverted: true,
            flavour: Flavour::Square,
        };
        let child = a.crossover(&b, &mut seeded());
        assert!(child.seed == a.seed || child.seed == b.seed);
        assert!(child.frequency == a.frequency || child.frequency == b.frequency);
        assert!(child.partials == a.partials || child.partials == b.partials);
        assert!(child.gain == a.gain || child.gain == b.gain);
        assert!(child.cycles == a.cycles || child.cycles == b.cycles);
        assert!(child.inverted == a.inverted || child.inverted == b.inverted);
        assert!(child.flavour == a.flavour || child.flavour == b.flavour);
    }
}
