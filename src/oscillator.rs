//! Four classic waveform generators — sine, square, sawtooth, triangle.
//!
//! All four share a single phase-accumulator state shape ([`OscPhase`])
//! and integrate the instantaneous frequency sample by sample, so wiring
//! another node into the `"freq"` input port gives proper per-sample
//! frequency modulation (an LFO sweep produces a vibrato, an audio-rate
//! oscillator produces FM sidebands).  Amplitude is similarly modulatable
//! through the `"amplitude"` input port.
//!
//! Aliasing is opt-out, not mandatory.  Square/saw/triangle default to the
//! naïve generators ([`AntiAlias::Naive`]) — The Janus Stream targets
//! *texture* over purity, and the audible grit of a 440 Hz square at 32 kHz
//! is part of the aesthetic.  Set [`SquareOsc::anti_alias`] (and the saw /
//! triangle equivalents) to [`AntiAlias::PolyBlep`] to band-limit the
//! discontinuities with PolyBLEP (value-step correction for square/saw) and
//! polyBLAMP (slope-corner correction for triangle).  Both are pure
//! arithmetic — no transcendentals — so the band-limited path keeps the
//! crate's bit-identical-across-machines determinism contract, and both work
//! under per-sample `"freq"` modulation since the correction reads the
//! instantaneous phase increment `dt = freq / sr`.
//!
//! # Input ports
//!
//! - `"freq"` — added to `freq_hz` per sample.  Wire an LFO here for
//!   vibrato; wire an audio-rate oscillator for FM.
//! - `"amplitude"` — added to `amplitude` per sample.  Wire an envelope
//!   here for AM/tremolo or volume shaping.
//!
//! Unwired ports read zero, leaving the configured value untouched.

use std::any::Any;
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

const TWO_PI: f32 = 2.0 * PI;

/// Shared phase accumulator for every oscillator in this module (and the
/// LFO, conceptually — though that one carries extra fields for the
/// sample-and-hold shape).
///
/// One running phase value in `[0, 1)` that advances by `freq / sr` per
/// sample.  Lifted to a struct rather than a bare `f32` so the
/// `#[non_exhaustive]` extension story stays open.
#[derive(Debug, Clone, Copy, Default)]
pub struct OscPhase {
    pub(crate) phase: f32,
}

/// Direction of a sawtooth ramp.  `Up` rises from −1 to +1 over the period
/// (the classic bright sawtooth ramp); `Down` falls from +1 to −1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SawPolarity {
    #[default]
    Up,
    Down,
}

/// Band-limiting mode for the discontinuous oscillators (square / saw /
/// triangle).
///
/// `Naive` (the default) is the raw phase-accumulator generator: its hard
/// edges and corners inject energy above Nyquist that folds back as
/// aliasing — intentional grit, see the module header.  `PolyBlep` applies
/// a polynomial correction around each discontinuity (PolyBLEP for the
/// value steps in square / saw, polyBLAMP for the slope corners in
/// triangle), which removes most of that aliasing for a tiny per-sample
/// cost and no extra state.
///
/// The default is `Naive` so every pre-existing patch (which has no
/// `anti_alias` field on the wire) deserializes to the exact same bytes it
/// baked before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AntiAlias {
    /// Raw generator — aliased, deterministic, the historical behaviour.
    #[default]
    Naive,
    /// PolyBLEP / polyBLAMP band-limited generator.
    PolyBlep,
}

fn default_amplitude() -> f32 {
    1.0
}

/// Advance the phase accumulator and return the *previous* phase so the
/// sample produced at index N reflects N's contribution rather than
/// N+1's.  Returns the phase to use for this sample.
#[inline]
fn step_phase(state: &mut OscPhase, effective_freq: f32, sample_rate: f32) -> f32 {
    let p = state.phase;
    state.phase = (p + effective_freq / sample_rate).rem_euclid(1.0);
    p
}

/// Stateless fallback: when no [`OscPhase`] state is installed (typical
/// for direct unit tests that bypass the baker), reconstruct phase from
/// `t * freq`.  Callers pass the *effective* (post-modulation) frequency so
/// a wired `"freq"` input still shifts the pitch — equivalent to the
/// accumulator path when that frequency is constant, and only divergent
/// under per-sample-varying modulation, which genuinely needs the
/// integrating accumulator.
#[inline]
fn stateless_phase(time_secs: f64, freq_hz: f32) -> f32 {
    (time_secs as f32 * freq_hz).rem_euclid(1.0)
}

/// Largest normalized phase increment `dt = freq / sr` for which the
/// PolyBLEP / polyBLAMP correction windows (`t < dt` near an edge and
/// `t > 1 - dt` before the next one) stay disjoint.  Above this the two
/// windows overlap and the correction is meaningless, so we fall back to
/// the naïve sample — which is already saturated with aliasing at that
/// pitch anyway (a freq past a quarter of the sample rate).
const MAX_BLEP_DT: f32 = 0.5;

/// PolyBLEP residual for a band-limited **value step**, matched to a unit
/// upward step at the phase wrap.
///
/// `t` is the phase in `[0, 1)`; `dt` is the per-sample phase increment
/// `freq / sr`.  Returns 0 outside the two-sample neighbourhood of a
/// discontinuity, so callers add/subtract it unconditionally.  Subtracting
/// it from a naïve upward saw (`2·phase − 1`) band-limits the reset; the
/// square uses one copy per edge.  Pure `+ − ×` arithmetic — deterministic
/// across machines.
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if !(dt > 0.0 && dt < MAX_BLEP_DT) {
        return 0.0;
    }
    if t < dt {
        let x = t / dt;
        2.0 * x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + 2.0 * x + 1.0
    } else {
        0.0
    }
}

/// polyBLAMP residual for a band-limited **slope corner** — the integral of
/// [`poly_blep`], used to round the triangle's corners.
///
/// Same conventions as [`poly_blep`]: `t` is the phase in `[0, 1)`, `dt` the
/// per-sample increment, result 0 outside the corner neighbourhood.  The
/// caller scales it by the per-sample slope change at the corner (`4·dt` for
/// a unit triangle) and adds it.
#[inline]
fn poly_blamp(t: f32, dt: f32) -> f32 {
    if !(dt > 0.0 && dt < MAX_BLEP_DT) {
        return 0.0;
    }
    if t < dt {
        let x = t / dt - 1.0;
        -1.0 / 3.0 * x * x * x
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt + 1.0;
        1.0 / 3.0 * x * x * x
    } else {
        0.0
    }
}

// --- Sine -------------------------------------------------------------------

/// Pure-tone sine oscillator with a constant phase offset.
///
/// `phase_offset` is in units of one cycle (0.0 to 1.0 covers a full
/// rotation), so two sines at the same frequency with `phase_offset` 0.0
/// and 0.25 are 90° apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SineOsc {
    pub freq_hz: f32,
    pub phase_offset: f32,
    /// Output gain, multiplied with the waveform.  Wired modulation on
    /// the `"amplitude"` input is added to this value per sample.
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

impl Default for SineOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            phase_offset: 0.0,
            amplitude: 1.0,
        }
    }
}

impl Node for SineOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), freq),
        };
        let total = (phase + self.phase_offset).rem_euclid(1.0);
        (TWO_PI * total).sin() * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Square -----------------------------------------------------------------

/// Naïve pulse-width square at `freq_hz`.
///
/// `duty` is the fraction of the period spent at +1, clamped to `(0, 1)`
/// at sample time so the canonical 0.5 case yields a symmetric square
/// wave.  Duty values near 0 or 1 produce a thin pulse — useful as a
/// click train and as a target for PWM modulation later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquareOsc {
    pub freq_hz: f32,
    pub duty: f32,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
    /// Band-limiting mode.  Defaults to [`AntiAlias::Naive`] so existing
    /// patches (no `anti_alias` on the wire) deserialize unchanged.
    #[serde(default)]
    pub anti_alias: AntiAlias,
}

impl Default for SquareOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            duty: 0.5,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        }
    }
}

impl Node for SquareOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), freq),
        };
        let duty = self.duty.clamp(f32::EPSILON, 1.0 - f32::EPSILON);
        let mut raw = if phase < duty { 1.0 } else { -1.0 };
        if self.anti_alias == AntiAlias::PolyBlep {
            let dt = freq / sr;
            // Rising edge (upward step) at phase 0, falling edge (downward
            // step) at phase `duty` — add at the former, subtract at the
            // latter, each evaluated at the phase relative to that edge.
            raw += poly_blep(phase, dt);
            raw -= poly_blep((phase - duty).rem_euclid(1.0), dt);
        }
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Sawtooth ---------------------------------------------------------------

/// Naïve sawtooth.  Polarity flips the ramp direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SawtoothOsc {
    pub freq_hz: f32,
    pub polarity: SawPolarity,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
    /// Band-limiting mode.  Defaults to [`AntiAlias::Naive`] so existing
    /// patches (no `anti_alias` on the wire) deserialize unchanged.
    #[serde(default)]
    pub anti_alias: AntiAlias,
}

impl Default for SawtoothOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        }
    }
}

impl Node for SawtoothOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), freq),
        };
        let mut raw = match self.polarity {
            SawPolarity::Up => 2.0 * phase - 1.0,
            SawPolarity::Down => 1.0 - 2.0 * phase,
        };
        if self.anti_alias == AntiAlias::PolyBlep {
            let dt = freq / sr;
            // The reset is a downward step for an `Up` ramp (subtract the
            // BLEP) and an upward step for a `Down` ramp (add it).
            let blep = poly_blep(phase, dt);
            raw += match self.polarity {
                SawPolarity::Up => -blep,
                SawPolarity::Down => blep,
            };
        }
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Triangle ---------------------------------------------------------------

/// Naïve triangle.  Symmetric — peaks at +1 at the half-period mark and
/// bottoms out at −1 at the phase boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriangleOsc {
    pub freq_hz: f32,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
    /// Band-limiting mode.  Defaults to [`AntiAlias::Naive`] so existing
    /// patches (no `anti_alias` on the wire) deserialize unchanged.
    #[serde(default)]
    pub anti_alias: AntiAlias,
}

impl Default for TriangleOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        }
    }
}

impl Node for TriangleOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), freq),
        };
        // Naïve triangle: min (−1) at phase 0, max (+1) at phase 0.5, with a
        // per-phase slope of ±4.
        let mut raw = 1.0 - 4.0 * (phase - 0.5).abs();
        if self.anti_alias == AntiAlias::PolyBlep {
            let dt = freq / sr;
            // Slope corners at phase 0 (trough, slope +8) and phase 0.5
            // (peak, slope −8).  The polyBLAMP residual scaled by the
            // per-sample slope change `4·dt` rounds each corner — added at
            // the trough, subtracted at the peak.
            raw += 4.0 * dt * poly_blamp(phase, dt);
            raw -= 4.0 * dt * poly_blamp((phase - 0.5).rem_euclid(1.0), dt);
        }
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Genotype ---------------------------------------------------------------

crate::impl_genotype!(SineOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    phase_offset: f32(0.05, 0.0, 1.0),
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(SquareOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    duty: f32(0.05, 0.05, 0.95),
    amplitude: f32(0.1, 0.0, 1.0),
    anti_alias: enum_cycle([AntiAlias::Naive, AntiAlias::PolyBlep]),
});

crate::impl_genotype!(SawtoothOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    polarity: enum_cycle([SawPolarity::Up, SawPolarity::Down]),
    amplitude: f32(0.1, 0.0, 1.0),
    anti_alias: enum_cycle([AntiAlias::Naive, AntiAlias::PolyBlep]),
});

crate::impl_genotype!(TriangleOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    amplitude: f32(0.1, 0.0, 1.0),
    anti_alias: enum_cycle([AntiAlias::Naive, AntiAlias::PolyBlep]),
});

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Drive an oscillator through N samples with installed state and no
    /// modulation, returning the per-sample buffer.  Matches what the
    /// production bake() path does for a one-node patch.
    fn drive<N: Node>(node: &N, sample_rate: u32, n: usize) -> Vec<f32> {
        let mut state = node.init_state();
        let inputs: &[(&str, f32)] = &[];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
            let mut ctx =
                BakeContext::new(sample_rate, i as u64, n as u64, &mut rng, inputs, state_ref);
            out.push(node.sample(&mut ctx));
        }
        out
    }

    #[test]
    fn sine_zero_phase_at_t_zero() {
        let osc = SineOsc::default();
        let buf = drive(&osc, 44_100, 1);
        assert!(buf[0].abs() < 1e-6, "sample[0] = {}", buf[0]);
    }

    #[test]
    fn sine_quarter_period_is_one() {
        // sr / (4 * 440) ≈ 25 samples per quarter period at 440 Hz.
        let osc = SineOsc::default();
        let sr: u32 = 44_100;
        let quarter = (sr as f32 / (4.0 * 440.0)).round() as usize;
        let buf = drive(&osc, sr, quarter + 1);
        assert!(
            buf[quarter] > 0.99,
            "quarter-period sample: {}",
            buf[quarter]
        );
    }

    #[test]
    fn square_50pct_duty_is_bipolar() {
        let osc = SquareOsc::default();
        let sr: u32 = 44_100;
        // Sample 10 is in the first half-cycle (positive); 10 + half period is
        // in the second (negative).
        let half_period = (sr as f32 / (2.0 * 440.0)).round() as usize;
        let buf = drive(&osc, sr, 10 + half_period + 1);
        assert_eq!(buf[10], 1.0);
        assert_eq!(buf[10 + half_period], -1.0);
    }

    #[test]
    fn sawtooth_up_rises_linearly_within_period() {
        let osc = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        };
        let buf = drive(&osc, 1_000, 501);
        // Phase 0 → -1.  Phase 0.5 (sample 500 at 1 Hz sr=1000) → 0.
        assert!((buf[0] - -1.0).abs() < 1e-3);
        assert!(buf[500].abs() < 1e-2);
    }

    #[test]
    fn sawtooth_down_is_negated_up() {
        let up = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        };
        let down = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Down,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        };
        let up_buf = drive(&up, 1_000, 1_000);
        let down_buf = drive(&down, 1_000, 1_000);
        for i in (0..1_000).step_by(73) {
            let sum = up_buf[i] + down_buf[i];
            assert!(sum.abs() < 1e-3, "up+down ≠ 0 at i={i}: {sum}");
        }
    }

    #[test]
    fn triangle_is_symmetric_around_half_period() {
        let osc = TriangleOsc {
            freq_hz: 1.0,
            amplitude: 1.0,
            anti_alias: AntiAlias::Naive,
        };
        let buf = drive(&osc, 1_000, 1_000);
        // f(0) = -1; f(0.25) = 0; f(0.5) = +1; f(0.75) = 0.
        assert!(buf[250].abs() < 1e-2);
        assert!((buf[500] - 1.0).abs() < 1e-2);
        assert!(buf[750].abs() < 1e-2);
    }

    #[test]
    fn freq_input_modulates_pitch() {
        // Phase-accumulator FM: drive a sine with a positive constant
        // freq input.  Total frequency should be config + mod_value, so
        // the output cycles faster than the config alone.
        let osc = SineOsc {
            freq_hz: 440.0,
            phase_offset: 0.0,
            amplitude: 1.0,
        };
        let sr: u32 = 44_100;
        let mut state = osc.init_state();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // Mod input is +440 Hz, so total freq = 880 Hz.
        let inputs = [("freq", 440.0_f32)];
        let half_period_at_880 = (sr as f32 / (4.0 * 880.0)).round() as u64;
        let mut samples = Vec::new();
        for i in 0..=half_period_at_880 {
            let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
            let mut ctx = BakeContext::new(sr, i, 100, &mut rng, &inputs, state_ref);
            samples.push(osc.sample(&mut ctx));
        }
        // At sample sr/(4*880), a sine at 880 Hz should be at peak.
        let last = *samples.last().unwrap();
        assert!(last > 0.99, "modulated quarter-period: {last}");
    }

    #[test]
    fn stateless_path_reflects_freq_modulation() {
        // With no OscPhase state installed (the stateless fallback), a wired
        // "freq" input must still shift the pitch — previously it was
        // silently dropped in favour of the configured freq_hz.
        let osc = SineOsc {
            freq_hz: 0.0,
            phase_offset: 0.0,
            amplitude: 1.0,
        };
        let sr: u32 = 44_100;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // Effective freq = 0 + 100 = 100 Hz.  At t = sample_index / sr ≈
        // 1/400 s, phase = 100 × t ≈ 0.25 → sin(π/2) ≈ 1.0.  Under the old
        // behaviour (freq_hz = 0) phase stays 0 and the output is 0.
        let inputs = [("freq", 100.0_f32)];
        let sample_index = u64::from(sr / 400);
        let state_ref: Option<&mut (dyn Any + Send)> = None;
        let mut ctx = BakeContext::new(
            sr,
            sample_index,
            u64::from(sr),
            &mut rng,
            &inputs,
            state_ref,
        );
        let s = osc.sample(&mut ctx);
        assert!((s - 1.0).abs() < 1e-2, "stateless freq-modulated peak: {s}");
    }

    #[test]
    fn amplitude_input_scales_output() {
        let osc = SineOsc {
            freq_hz: 440.0,
            phase_offset: 0.25, // start at peak
            amplitude: 0.0,
        };
        let sr: u32 = 44_100;
        let mut state = osc.init_state();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("amplitude", 0.5_f32)];
        let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
        let mut ctx = BakeContext::new(sr, 0, 1, &mut rng, &inputs, state_ref);
        // amplitude is 0.0 + 0.5 = 0.5; phase 0.25 means sin(π/2) = 1.0.
        // Output should be 0.5.
        let s = osc.sample(&mut ctx);
        assert!((s - 0.5).abs() < 1e-3, "amplitude-modulated peak: {s}");
    }

    #[test]
    fn genotype_clamps_frequencies_to_audible_range() {
        use symbios_genetics::Genotype;
        let mut osc = SineOsc {
            freq_hz: 19_500.0,
            phase_offset: 0.0,
            amplitude: 0.9,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..200 {
            osc.mutate(&mut rng, 1.0);
            assert!((20.0..=20_000.0).contains(&osc.freq_hz));
            assert!((0.0..=1.0).contains(&osc.phase_offset));
            assert!((0.0..=1.0).contains(&osc.amplitude));
        }
    }

    // --- Anti-aliasing (PolyBLEP / polyBLAMP) ------------------------------

    #[test]
    fn anti_alias_defaults_to_naive() {
        // The default keeps every pre-existing patch baking byte-for-byte
        // identical, so it must stay Naive on all three discontinuous oscs.
        assert_eq!(SquareOsc::default().anti_alias, AntiAlias::Naive);
        assert_eq!(SawtoothOsc::default().anti_alias, AntiAlias::Naive);
        assert_eq!(TriangleOsc::default().anti_alias, AntiAlias::Naive);
    }

    #[test]
    fn missing_anti_alias_field_deserializes_to_naive() {
        // A patch authored before this field existed has no `anti_alias`
        // key; serde's #[serde(default)] must fill in Naive (not error,
        // not PolyBlep) so old JSON round-trips to the old behaviour.
        let json = r#"{"freq_hz":440.0,"duty":0.5,"amplitude":1.0}"#;
        let osc: SquareOsc = serde_json::from_str(json).unwrap();
        assert_eq!(osc.anti_alias, AntiAlias::Naive);
    }

    #[test]
    fn poly_blep_is_zero_outside_edge_window_and_for_extreme_dt() {
        let dt = 0.05;
        // Interior phases (away from both edges) get no correction.
        for &t in &[0.2_f32, 0.5, 0.8] {
            assert_eq!(poly_blep(t, dt), 0.0);
            assert_eq!(poly_blamp(t, dt), 0.0);
        }
        // dt out of the usable (0, 0.5) range disables the correction
        // entirely rather than producing garbage from overlapping windows.
        assert_eq!(poly_blep(0.0, 0.0), 0.0);
        assert_eq!(poly_blep(0.0, 0.9), 0.0);
        assert_eq!(poly_blamp(0.0, 0.9), 0.0);
    }

    #[test]
    fn polyblep_saw_matches_naive_away_from_the_wrap() {
        // PolyBLEP only touches the two-sample neighbourhood of the reset.
        // Interior samples must be bit-for-bit the naïve ramp.
        let sr = 44_100;
        let naive = drive(
            &SawtoothOsc {
                freq_hz: 2205.0, // dt = 0.05 → 20 samples/period
                polarity: SawPolarity::Up,
                amplitude: 1.0,
                anti_alias: AntiAlias::Naive,
            },
            sr,
            60,
        );
        let blep = drive(
            &SawtoothOsc {
                freq_hz: 2205.0,
                polarity: SawPolarity::Up,
                amplitude: 1.0,
                anti_alias: AntiAlias::PolyBlep,
            },
            sr,
            60,
        );
        // Samples whose phase sits in [0.05, 0.95] (indices 1..=19 mod 20,
        // excluding the wrap-straddling 0/19) are untouched.
        for i in 0..60 {
            let phase_idx = i % 20;
            if (2..=18).contains(&phase_idx) {
                assert!(
                    (naive[i] - blep[i]).abs() < 1e-6,
                    "interior sample {i} diverged: naive={} blep={}",
                    naive[i],
                    blep[i]
                );
            }
        }
    }

    #[test]
    fn polyblep_saw_softens_the_reset_discontinuity() {
        // The largest sample-to-sample jump (at the ramp reset) must shrink
        // markedly once band-limited — that jump is exactly the aliasing
        // source.
        let sr = 44_100;
        let cfg = |aa| SawtoothOsc {
            freq_hz: 2205.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
            anti_alias: aa,
        };
        let naive = drive(&cfg(AntiAlias::Naive), sr, 200);
        let blep = drive(&cfg(AntiAlias::PolyBlep), sr, 200);
        let max_jump = |buf: &[f32]| {
            buf.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0_f32, f32::max)
        };
        let naive_jump = max_jump(&naive);
        let blep_jump = max_jump(&blep);
        assert!(
            naive_jump > 1.8,
            "naive reset jump unexpectedly small: {naive_jump}"
        );
        assert!(
            blep_jump < naive_jump * 0.8,
            "PolyBLEP reset jump {blep_jump} not < 0.8 × naive {naive_jump}"
        );
    }

    #[test]
    fn polyblep_square_stays_bipolar_and_bounded() {
        let blep = drive(
            &SquareOsc {
                freq_hz: 2205.0,
                duty: 0.5,
                amplitude: 1.0,
                anti_alias: AntiAlias::PolyBlep,
            },
            44_100,
            200,
        );
        // PolyBLEP can overshoot a touch past ±1 near the edges, but never
        // wildly, and both rails are still reached.
        assert!(
            blep.iter().all(|s| s.abs() <= 1.3),
            "square overshoot too large"
        );
        assert!(blep.iter().any(|&s| s > 0.5), "square never goes high");
        assert!(blep.iter().any(|&s| s < -0.5), "square never goes low");
    }

    #[test]
    fn polyblamp_triangle_stays_centered_and_bounded() {
        // Drive an exact integer number of periods (dt = 0.05 → 20
        // samples/period × 10 = 200 samples) so the mean of a centred
        // triangle is ~0; polyBLAMP must not introduce DC or blow up.
        let blep = drive(
            &TriangleOsc {
                freq_hz: 2205.0,
                amplitude: 1.0,
                anti_alias: AntiAlias::PolyBlep,
            },
            44_100,
            200,
        );
        let mean = blep.iter().sum::<f32>() / blep.len() as f32;
        assert!(mean.abs() < 0.05, "triangle DC offset too large: {mean}");
        assert!(
            blep.iter().all(|s| s.abs() <= 1.1),
            "triangle overshoot too large"
        );
    }
}
