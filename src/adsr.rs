//! Attack / Decay / Sustain / Release envelope.
//!
//! Turns a `gate` input signal (>0.5 = on, ≤0.5 = off) into a control-voltage
//! envelope in `[0, 1]`.  Typical wiring:
//! `gate → AdsrEnvelope → amplitude input of an oscillator or filter`.
//!
//! # Gate semantics
//!
//! Edge-triggered, not level-triggered:
//! - **Rising edge** (gate ≤0.5 → >0.5) restarts the attack stage from
//!   0.0.  Re-triggering during a held note always returns to zero —
//!   this is the classic synth-keyboard behaviour and avoids the
//!   "stuck loud" failure mode that level-trigger designs suffer from.
//! - **Falling edge** (gate >0.5 → ≤0.5) captures whatever value the
//!   envelope is currently producing and ramps it down to zero over
//!   `release_s`.  So a key released mid-attack releases from the partial
//!   value, not the full peak.
//!
//! # Curve shapes
//!
//! Linear: straight ramps.  Exponential: `1 − (1 − α)²` applied to every
//! ramp.  The Exponential shape is "ease-out" — fast initial change, slow
//! approach to the target — which mimics a capacitor charging through a
//! resistor and is what most analogue synth ADSRs feel like.  All three
//! ramps (attack, decay, release) use the same curve function, just with
//! different start/end points.
//!
//! # State machinery
//!
//! Builds on the per-node state plumbing from Phase 1 #5: [`AdsrState`] is
//! installed via [`Node::init_state`] and reached via
//! [`BakeContext::state_mut`].  See the module-level comment in
//! [`crate::noise`] for the design rationale.

use std::any::Any;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

/// Curve shape used for all three ramps of an [`AdsrEnvelope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AdsrCurve {
    /// Straight-line ramps.  Useful for synthetic clicks and mathematical
    /// fades where the maths-perfect linearity matters.
    #[default]
    Linear,
    /// Ease-out curve `1 − (1 − α)²`.  Capacitor-charge shape — fast
    /// initial change, asymptotic approach.  Reach for this if you don't
    /// know which to pick (note the type's `Default` is `Linear`).
    Exponential,
}

/// Attack / Decay / Sustain / Release envelope.  Drives an amplitude (or
/// any other CV target) in `[0, 1]` over the course of a note.
///
/// Field units:
/// - `attack_s`, `decay_s`, `release_s`: seconds (clamped to ≥ 1 sample at
///   sample time so a zero stays a one-sample transition rather than a
///   division-by-zero).
/// - `sustain_level`: dimensionless `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsrEnvelope {
    pub attack_s: f32,
    pub decay_s: f32,
    pub sustain_level: f32,
    pub release_s: f32,
    pub curve: AdsrCurve,
}

impl Default for AdsrEnvelope {
    fn default() -> Self {
        Self {
            attack_s: 0.01,
            decay_s: 0.1,
            sustain_level: 0.7,
            release_s: 0.2,
            curve: AdsrCurve::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum AdsrPhase {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Per-node persistent state for an [`AdsrEnvelope`].  Tracks the current
/// phase of the envelope, how many samples it has spent there, the last
/// observed gate level (for edge detection), and the value at the moment
/// the release phase started (for partial-release ramps).
///
/// `pub` so external code that pokes at envelope state in tests (or in a
/// future per-graph state inspector) can construct one; the field set is
/// `pub(crate)` to leave room to refactor without breaking downstream.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdsrState {
    pub(crate) phase: AdsrPhase,
    pub(crate) elapsed_samples: u64,
    pub(crate) last_gate_high: bool,
    pub(crate) release_start_level: f32,
}

const GATE_THRESHOLD: f32 = 0.5;
const MIN_PHASE_SAMPLES: f32 = 1.0;

#[inline]
fn curve_eval(curve: AdsrCurve, alpha: f32) -> f32 {
    let a = alpha.clamp(0.0, 1.0);
    match curve {
        AdsrCurve::Linear => a,
        // Ease-out: fast initial change, slow approach.
        AdsrCurve::Exponential => 1.0 - (1.0 - a) * (1.0 - a),
    }
}

impl AdsrEnvelope {
    /// Lerp from `start` to `end` along the configured curve.  All three
    /// ramps (attack, decay, release) share this shape — only the
    /// endpoints differ.
    #[inline]
    fn ramp(&self, start: f32, end: f32, alpha: f32) -> f32 {
        start + (end - start) * curve_eval(self.curve, alpha)
    }

    /// Compute the envelope value for the current phase + elapsed time.
    fn value(&self, state: &AdsrState, sample_rate: f32) -> f32 {
        match state.phase {
            AdsrPhase::Idle => 0.0,
            AdsrPhase::Attack => {
                let dur = (self.attack_s * sample_rate).max(MIN_PHASE_SAMPLES);
                self.ramp(0.0, 1.0, state.elapsed_samples as f32 / dur)
            }
            AdsrPhase::Decay => {
                let dur = (self.decay_s * sample_rate).max(MIN_PHASE_SAMPLES);
                self.ramp(1.0, self.sustain_level, state.elapsed_samples as f32 / dur)
            }
            AdsrPhase::Sustain => self.sustain_level,
            AdsrPhase::Release => {
                let dur = (self.release_s * sample_rate).max(MIN_PHASE_SAMPLES);
                self.ramp(
                    state.release_start_level,
                    0.0,
                    state.elapsed_samples as f32 / dur,
                )
            }
        }
    }

    /// Duration in samples of the phase, or `u64::MAX` for Idle/Sustain
    /// which never time out by themselves.
    fn phase_duration_samples(&self, phase: AdsrPhase, sample_rate: f32) -> u64 {
        match phase {
            AdsrPhase::Idle | AdsrPhase::Sustain => u64::MAX,
            AdsrPhase::Attack => (self.attack_s * sample_rate).max(MIN_PHASE_SAMPLES) as u64,
            AdsrPhase::Decay => (self.decay_s * sample_rate).max(MIN_PHASE_SAMPLES) as u64,
            AdsrPhase::Release => (self.release_s * sample_rate).max(MIN_PHASE_SAMPLES) as u64,
        }
    }
}

impl Node for AdsrEnvelope {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let gate_high = ctx.input("gate") > GATE_THRESHOLD;
        let state = match ctx.state_mut::<AdsrState>() {
            Some(s) => s,
            // Without a state container we still return a usable signal:
            // 0 when the gate is low, sustain when it's high.  Avoids
            // mid-bake panic if the baker ever fails to install state.
            None => return if gate_high { self.sustain_level } else { 0.0 },
        };

        // --- Edge detection --------------------------------------------
        if gate_high && !state.last_gate_high {
            // Rising edge: restart attack from zero.
            state.phase = AdsrPhase::Attack;
            state.elapsed_samples = 0;
        } else if !gate_high
            && state.last_gate_high
            && !matches!(state.phase, AdsrPhase::Idle | AdsrPhase::Release)
        {
            // Falling edge mid-note: capture the current value and start
            // releasing from there so a key released during attack/decay
            // doesn't snap to the full peak before fading.
            state.release_start_level = self.value(state, sr);
            state.phase = AdsrPhase::Release;
            state.elapsed_samples = 0;
        }
        state.last_gate_high = gate_high;

        // --- Output ----------------------------------------------------
        let out = self.value(state, sr);

        // --- Advance phase --------------------------------------------
        state.elapsed_samples += 1;
        if state.elapsed_samples >= self.phase_duration_samples(state.phase, sr) {
            match state.phase {
                AdsrPhase::Attack => {
                    state.phase = AdsrPhase::Decay;
                    state.elapsed_samples = 0;
                }
                AdsrPhase::Decay => {
                    state.phase = AdsrPhase::Sustain;
                    state.elapsed_samples = 0;
                }
                AdsrPhase::Release => {
                    state.phase = AdsrPhase::Idle;
                    state.elapsed_samples = 0;
                    state.release_start_level = 0.0;
                }
                _ => {}
            }
        }

        out
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(AdsrState::default()))
    }
}

// --- Genotype ---------------------------------------------------------------

crate::impl_genotype!(AdsrEnvelope {
    attack_s: f32_log(0.5, 0.001, 10.0),
    decay_s: f32_log(0.5, 0.001, 10.0),
    sustain_level: f32(0.1, 0.0, 1.0),
    release_s: f32_log(0.5, 0.001, 10.0),
    curve: enum_cycle([AdsrCurve::Linear, AdsrCurve::Exponential]),
}
post_mutate: |s| {
    // The ticket flagged this: without a floor on the total envelope time,
    // multiplicative mutation keeps making A/D/R smaller, collapsing every
    // envelope to an instant click and homogenising the population.  Scale
    // all three up proportionally if the sum falls below the floor.
    let min_total = 0.05_f32;
    let total = s.attack_s + s.decay_s + s.release_s;
    if total > 0.0 && total < min_total {
        let scale = min_total / total;
        s.attack_s *= scale;
        s.decay_s *= scale;
        s.release_s *= scale;
    }
}
post_crossover: |c| {
    let min_total = 0.05_f32;
    let total = c.attack_s + c.decay_s + c.release_s;
    if total > 0.0 && total < min_total {
        let scale = min_total / total;
        c.attack_s *= scale;
        c.decay_s *= scale;
        c.release_s *= scale;
    }
});

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn linear_test_envelope() -> AdsrEnvelope {
        AdsrEnvelope {
            attack_s: 0.1,
            decay_s: 0.1,
            sustain_level: 0.5,
            release_s: 0.1,
            curve: AdsrCurve::Linear,
        }
    }

    /// Drive an envelope through `gate_pattern` and return per-sample
    /// values.  `gate_pattern` is a slice of `(sample_count, gate_high)`
    /// segments — `[(4410, true), (100, false)]` means 4410 samples gated
    /// on, then 100 samples gated off.
    fn run(env: &AdsrEnvelope, sample_rate: u32, pattern: &[(u64, bool)]) -> (Vec<f32>, AdsrState) {
        let mut state = AdsrState::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut samples = Vec::new();
        let mut sample_index: u64 = 0;
        let total: u64 = pattern.iter().map(|(n, _)| *n).sum();
        for &(count, high) in pattern {
            let inputs = [("gate", if high { 1.0_f32 } else { 0.0 })];
            for _ in 0..count {
                let mut ctx = BakeContext::new(
                    sample_rate,
                    sample_index,
                    total,
                    &mut rng,
                    &inputs,
                    Some(&mut state),
                );
                samples.push(env.sample(&mut ctx));
                sample_index += 1;
            }
        }
        (samples, state)
    }

    #[test]
    fn idle_with_no_gate_stays_at_zero() {
        let env = linear_test_envelope();
        let (samples, state) = run(&env, 44_100, &[(1000, false)]);
        assert!(samples.iter().all(|s| *s == 0.0));
        assert_eq!(state.phase, AdsrPhase::Idle);
    }

    #[test]
    fn linear_attack_reaches_one_at_end() {
        // A = 0.1s @ 44.1kHz = 4410 samples.  After 4410 samples the
        // envelope must be at 1.0 (end of attack, start of decay).  We
        // include one extra sample so the phase transition + post-decay
        // index lands clearly.
        let env = linear_test_envelope();
        let (samples, _) = run(&env, 44_100, &[(4411, true)]);
        // The instant the attack phase hits its duration, the next
        // sample lands at the attack peak as decay begins from 1.0.
        assert!(
            (samples[4410] - 1.0).abs() < 1e-3,
            "end-of-attack ≠ 1.0: {}",
            samples[4410]
        );
    }

    #[test]
    fn linear_decay_reaches_sustain() {
        // A=0.1, D=0.1 → end of decay at sample 8820.
        let env = linear_test_envelope();
        let (samples, state) = run(&env, 44_100, &[(8821, true)]);
        assert!(
            (samples[8820] - 0.5).abs() < 1e-3,
            "end-of-decay ≠ 0.5: {}",
            samples[8820]
        );
        assert_eq!(state.phase, AdsrPhase::Sustain);
    }

    #[test]
    fn sustain_holds_value_until_released() {
        let env = linear_test_envelope();
        // Gate on for 1 s (44 100 samples) — should be in sustain at the
        // end with value = sustain_level.
        let (samples, state) = run(&env, 44_100, &[(44_100, true)]);
        assert!((samples[44_099] - 0.5).abs() < 1e-3);
        assert_eq!(state.phase, AdsrPhase::Sustain);
    }

    #[test]
    fn release_after_one_second_gate_finishes_at_zero() {
        // A=0.1, D=0.1, R=0.1, gate high for 1 s.
        let env = linear_test_envelope();
        let (samples, state) = run(&env, 44_100, &[(44_100, true), (4_411, false)]);
        // Sample index 44_100 is the first sample after the gate goes
        // low — the release starts immediately, beginning at the sustain
        // level (0.5).
        assert!(
            (samples[44_100] - 0.5).abs() < 1e-3,
            "start of release ≠ 0.5: {}",
            samples[44_100]
        );
        // Sample index 44_100 + 4410 = 48_510: end of release, value 0.
        assert!(
            samples[48_510].abs() < 1e-3,
            "end of release ≠ 0.0: {}",
            samples[48_510]
        );
        assert_eq!(state.phase, AdsrPhase::Idle);
    }

    #[test]
    fn release_from_mid_attack_starts_from_partial_value() {
        // Gate on for half the attack, then off — release starts from
        // ~0.5, not from 1.0.
        let env = linear_test_envelope();
        let (samples, _) = run(&env, 44_100, &[(2_205, true), (4_410, false)]);
        // At sample 2205 (start of release), value should be ~0.5
        // (half-way through linear attack).
        assert!(
            (samples[2_205] - 0.5).abs() < 1e-2,
            "mid-attack release start: {}",
            samples[2_205]
        );
        // At end of release the envelope is zero.
        assert!(samples[2_205 + 4_409].abs() < 1e-3);
    }

    #[test]
    fn rising_edge_during_release_retriggers_attack_from_zero() {
        // After release fully completes, a new gate-on should restart
        // from zero (not jump back to the last release value).
        let env = linear_test_envelope();
        let (samples, state) = run(&env, 44_100, &[(44_100, true), (4_411, false), (10, true)]);
        // Last 10 samples are the freshly-retriggered attack — first
        // sample is zero, then ramps up.
        let retrig_first = samples[44_100 + 4_411];
        assert!(
            retrig_first.abs() < 1e-3,
            "retrigger first sample ≠ 0: {retrig_first}"
        );
        assert_eq!(state.phase, AdsrPhase::Attack);
    }

    #[test]
    fn exponential_curve_reaches_endpoints_exactly() {
        // The exponential ease-out hits 0 at α=0 and 1 at α=1 just like
        // the linear curve does — so the test's exact endpoint checks
        // still pass.  The midpoint differs (linear=0.5, exp=0.75).
        let env = AdsrEnvelope {
            curve: AdsrCurve::Exponential,
            ..linear_test_envelope()
        };
        let (samples, _) = run(&env, 44_100, &[(8_821, true)]);
        assert!((samples[4_410] - 1.0).abs() < 1e-3);
        assert!((samples[8_820] - 0.5).abs() < 1e-3);
        // And the midpoint of attack is at ~0.75, not 0.5.
        assert!(
            samples[2_205] > 0.7 && samples[2_205] < 0.8,
            "exp midpoint: {}",
            samples[2_205]
        );
    }

    #[test]
    fn genotype_keeps_total_envelope_time_above_floor() {
        use symbios_genetics::Genotype;
        let mut env = AdsrEnvelope {
            attack_s: 0.001,
            decay_s: 0.001,
            release_s: 0.001,
            ..linear_test_envelope()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..50 {
            env.mutate(&mut rng, 1.0);
            let total = env.attack_s + env.decay_s + env.release_s;
            assert!(total >= 0.05, "envelope collapsed: total={total}");
        }
    }

    #[test]
    fn falls_back_to_sustain_when_state_missing() {
        // Defensive path: if init_state isn't installed, gate-high should
        // still produce non-zero output (sustain_level) rather than
        // crashing.
        let env = linear_test_envelope();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("gate", 1.0_f32)];
        let mut ctx = BakeContext::new(44_100, 0, 1, &mut rng, &inputs, None);
        let s = env.sample(&mut ctx);
        assert!((s - env.sustain_level).abs() < 1e-6);
    }
}
