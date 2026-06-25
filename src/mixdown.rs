//! Mixdown baker — turn a [`SequenceRecipe`] into a single mono
//! master timeline buffer.
//!
//! Sits one layer above [`crate::bake::bake`]: takes a sequencer
//! recipe with named instruments and timed events, bakes each
//! instrument once per unique gate/release length, resamples by the
//! event's `pitch_multiplier`, sums all events into a master buffer at
//! the right sample offsets, applies a smooth `tanh` soft-clip so peaks
//! don't punch through `[-1, 1]`, and — when looping is enabled —
//! pre-mixes the tail crossfade into the loop region so a hard
//! `Source::loop_..()` is click-free at the seam.
//!
//! # Gate and release
//!
//! Each event bakes its instrument with a **gate window** open for
//! `gate_beats` and then keeps baking through `release_beats` of tail
//! (via the crate-internal `bake_inner`, the gated core of
//! [`crate::bake::try_bake`]).  An instrument that wires a
//! [`crate::node::NodeKind::Gate`] into an [`crate::adsr::AdsrEnvelope`]
//! therefore attacks and sustains for the gate, then releases and rings
//! out across the tail — so `gate_beats` is a real note length, not just
//! a buffer trim.  `release_beats = 0` reproduces the old hard one-shot.
//!
//! # Algorithm
//!
//! 1. Bake each `(instrument_id, gate_beats, release_beats)` triple
//!    exactly once, with the gate open for `gate_beats`.  Identical
//!    events anywhere on the timeline share the bake.
//! 2. Allocate the master buffer.  If `loop_start_beats` is set, the
//!    buffer is provisioned with an extra `loop_crossfade_beats` of
//!    tail so late event releases that spill past `duration_beats`
//!    can be folded back into the loop start.
//! 3. For each event: apply its `pitch_multiplier` per its
//!    [`PitchMode`] (Varispeed → linear-interpolation resample of the
//!    native bake; TimePreserving → use the already-retuned bake as-is),
//!    scale by `volume`, sum into the master starting at
//!    `time_beats × beat_secs × sample_rate`.
//! 4. `master[i] = master[i].tanh()` — saturates roughly linearly up
//!    to ±0.5 then smoothly compresses, which sounds nicer than a
//!    hard `clamp` and avoids the harmonic spray a true clipping
//!    introduces.
//! 5. If `loop_start_beats` is set: crossfade the tail samples down
//!    over the loop region starting at `loop_start_beats`, then
//!    truncate the buffer to exactly `main_samples` so the returned
//!    `Vec<f32>` is `duration_beats × beat_secs × sample_rate`
//!    samples long.  Otherwise the buffer is just the main timeline.
//!
//! # Pitch and time
//!
//! Each event's [`crate::sequence::PitchMode`] selects how its
//! `pitch_multiplier` is applied:
//!
//! - [`PitchMode::Varispeed`] (default) resamples the native bake, so a
//!   pitch-up event finishes sooner than its gate length and a pitch-down
//!   event hangs past it — tape-style, pitch and time coupled.  Every
//!   pitch of an instrument shares one native bake.
//! - [`PitchMode::TimePreserving`] retunes the instrument's oscillators
//!   (via [`crate::node::NodeKind::scale_pitch`]) and bakes the event at
//!   its true `gate + release` length, so no resampling is needed: pitch
//!   and time are independent and there are no resampling artifacts.  This
//!   is a synthesizer retuning its oscillators, not a sampler stretching a
//!   recording — strictly higher quality than PSOLA / phase-vocoder
//!   stretching of the wrong-length bake would be.  Each distinct pitch is
//!   its own bake (the bake cache keys on it).
//!
//! # Dangling references and bad graphs
//!
//! Events whose `instrument_id` doesn't appear in `recipe.instruments`
//! are skipped with a warn log rather than panicking — a typo shouldn't
//! crash the bake.  Likewise an instrument whose patch graph is
//! structurally invalid is skipped (the gated baker returns a `Result`,
//! like [`crate::bake::try_bake`]) instead of aborting the whole mixdown.

use std::collections::HashMap;

use crate::wav::MAX_WAV_SAMPLES;
use crate::bake::bake_inner;
use crate::patch::AudioPatch;
use crate::sequence::{Event, PitchMode, SequenceRecipe};

/// Bake a [`SequenceRecipe`] into a mono `Vec<f32>` mixdown buffer.
///
/// Buffer length:
/// - When `loop_start_beats` is `None`: exactly
///   `(duration_beats + loop_crossfade_beats) × (60 / bpm) ×
///   sample_rate` samples (the crossfade tail is left in place since
///   there is no loop region to fold it into).
/// - When `loop_start_beats` is `Some(b)`: exactly
///   `duration_beats × (60 / bpm) × sample_rate` samples — the tail
///   is pre-mixed back into the loop region starting at beat `b`
///   and then dropped, so a hard `Source::loop_..()` over the
///   returned buffer is click-free at the seam.
///
/// See the module docs for the algorithm and limitations.
pub fn bake_sequence(recipe: &SequenceRecipe) -> Vec<f32> {
    let beat_secs = beat_seconds(recipe.bpm);
    let sr = recipe.sample_rate;
    let main_samples = duration_to_samples(recipe.duration_beats, beat_secs, sr);
    let tail_samples = duration_to_samples(recipe.loop_crossfade_beats, beat_secs, sr);
    // saturating_add + clamp: `duration_to_samples` already caps each term at
    // MAX_WAV_SAMPLES, but guard the sum too so an untrusted recipe can never
    // overflow usize here or provision a buffer past the WAV-encodable ceiling
    // (which `samples_to_wav_bytes` would later reject anyway).
    let master_len = main_samples
        .saturating_add(tail_samples)
        .min(MAX_WAV_SAMPLES);

    let mut master = vec![0.0_f32; master_len];
    if master_len == 0 {
        return master;
    }

    // Per-instrument lookup so events resolve in O(1) instead of O(N)
    // scans of recipe.instruments per event.
    let instrument_lookup: HashMap<&str, &AudioPatch> = recipe
        .instruments
        .iter()
        .map(|i| (i.id.as_str(), &i.patch))
        .collect();

    // Bake cache keyed by (instrument_id, gate, release, pitch) bit
    // patterns so two events with identical timing reuse the source buffer.
    // The pitch slot is a fixed `1.0` sentinel for Varispeed events (they
    // bake once at native pitch and resample per event, sharing one bake
    // across every pitch) and the actual multiplier for TimePreserving ones
    // (each pitch is a distinct retuned bake).
    let mut bake_cache: HashMap<(String, u32, u32, u32), Vec<f32>> = HashMap::new();

    for track in &recipe.tracks {
        for event in &track.events {
            let key = bake_key(event);
            if bake_cache.contains_key(&key) {
                continue;
            }
            let Some(patch) = instrument_lookup.get(event.instrument_id.as_str()) else {
                log::warn!(
                    "mixdown: event references unknown instrument '{}'; skipping",
                    event.instrument_id
                );
                continue;
            };
            // The gate is open for `gate_beats`, then the bake continues
            // through `release_beats` of tail so a Gate→ADSR release rings
            // out instead of being cut off at the gate edge.
            let gate_samples = duration_to_samples(event.gate_beats, beat_secs, sr) as u64;
            let total_beats = event.gate_beats.max(0.0) + event.release_beats.max(0.0);
            let total_samples = duration_to_samples(total_beats, beat_secs, sr) as u64;
            // TimePreserving retunes the oscillators and bakes at the true
            // gate+release length (no later resample); Varispeed bakes the
            // instrument at native pitch and resamples in the sum pass.  A
            // non-positive / non-finite TimePreserving pitch is nonsense —
            // skip the bake (the sum pass then finds no cache entry and
            // skips the event too), mirroring resample_linear's guard.
            let result = match event.pitch_mode {
                PitchMode::Varispeed => {
                    bake_inner(patch, sr, total_samples, Some(gate_samples), None)
                }
                PitchMode::TimePreserving => {
                    if !event.pitch_multiplier.is_finite() || event.pitch_multiplier <= 0.0 {
                        log::warn!(
                            "mixdown: time-preserving event on '{}' has non-positive pitch {}; \
                             skipping",
                            event.instrument_id,
                            event.pitch_multiplier
                        );
                        continue;
                    }
                    let retuned = retuned_patch(patch, event.pitch_multiplier);
                    bake_inner(&retuned, sr, total_samples, Some(gate_samples), None)
                }
            };
            // A malformed instrument graph shouldn't take down the whole
            // mixdown — warn and skip it, like an unknown instrument ref.
            let buf = match result {
                Ok(buf) => buf,
                Err(err) => {
                    log::warn!(
                        "mixdown: instrument '{}' has an invalid graph ({err}); skipping",
                        event.instrument_id
                    );
                    continue;
                }
            };
            bake_cache.insert(key, buf);
        }
    }

    // Sum events into the master.
    for track in &recipe.tracks {
        for event in &track.events {
            let key = bake_key(event);
            let Some(source) = bake_cache.get(&key) else {
                continue;
            };
            if source.is_empty() {
                continue;
            }
            let start = (f64::from(event.time_beats) * f64::from(beat_secs) * f64::from(sr)).round()
                as usize;
            let max_out = master.len().saturating_sub(start);
            if max_out == 0 {
                continue;
            }
            let volume = event.volume.clamp(0.0, 1.0);
            match event.pitch_mode {
                PitchMode::Varispeed => {
                    // Resample only as many samples as can land in the master
                    // from `start`: anything past the end is discarded by
                    // write_into anyway, and bounding here stops a subnormal
                    // pitch_multiplier from inflating the output length to
                    // usize::MAX (capacity-overflow panic / OOM) before a
                    // single sample is written.
                    let resampled = resample_linear(source, event.pitch_multiplier, max_out);
                    if resampled.is_empty() {
                        continue;
                    }
                    write_into(&mut master, start, &resampled, volume);
                }
                PitchMode::TimePreserving => {
                    // The source was baked at the target pitch and the event's
                    // full gate+release length — no resampling.  write_into
                    // clips whatever runs past the master end.
                    write_into(&mut master, start, source, volume);
                }
            }
        }
    }

    // Soft-clip with tanh.  Smooth saturation: leaves small signals
    // essentially untouched, compresses peaks gracefully.
    for s in &mut master {
        *s = s.tanh();
    }

    // Tail-crossfade for seamless looping.  When loop_start_beats is
    // set, the bake has been kept running past duration_beats into
    // the crossfade tail (events whose release extends past the
    // timeline end land in this region).  We fade the tail down
    // linearly and overlay it onto the loop region starting at
    // loop_start_beats, faded up symmetrically.  After the overlay
    // the tail samples are dropped and the buffer is truncated to
    // exactly main_samples — playing it on a hard rodio
    // Source::loop_..() loop is seamless because the tail's
    // late-event release has been pre-mixed into the loop start.
    if let Some(loop_start_beats) = recipe.loop_start_beats {
        apply_loop_crossfade(
            &mut master,
            loop_start_beats,
            beat_secs,
            sr,
            main_samples,
            tail_samples,
        );
        master.truncate(main_samples);
    }

    master
}

/// Apply the tail-crossfade described in the module docs.  No-op if
/// any of the inputs make the operation nonsensical (loop_start past
/// the end, no tail samples, etc.) — bake_sequence is `-> Vec<f32>`
/// with no error channel, so this stays quiet.
fn apply_loop_crossfade(
    master: &mut [f32],
    loop_start_beats: f32,
    beat_secs: f32,
    sample_rate: u32,
    main_samples: usize,
    tail_samples: usize,
) {
    if tail_samples == 0 || main_samples == 0 {
        return;
    }
    let loop_start = (f64::from(loop_start_beats) * f64::from(beat_secs) * f64::from(sample_rate))
        .round() as usize;
    if loop_start >= main_samples {
        log::warn!(
            "mixdown: loop_start_beats ({loop_start_beats}) is past duration_beats; \
             skipping crossfade"
        );
        return;
    }
    // Don't run the crossfade past the truncation point — if the loop
    // region is shorter than the configured tail, clip the window.
    let crossfade = tail_samples
        .min(main_samples.saturating_sub(loop_start))
        .min(master.len().saturating_sub(main_samples));
    if crossfade == 0 {
        return;
    }
    // Linear sum-to-one crossfade: at i=0 the loop region takes the
    // full tail value (the "next sample after duration_beats" — the
    // seam connects perfectly).  At i=crossfade-1 the loop region is
    // back to its own pre-crossfade content.
    let denom = crossfade as f32;
    for i in 0..crossfade {
        let alpha = i as f32 / denom;
        let tail = master[main_samples + i];
        let loop_content = master[loop_start + i];
        master[loop_start + i] = (1.0 - alpha) * tail + alpha * loop_content;
    }
}

#[inline]
fn beat_seconds(bpm: f32) -> f32 {
    if bpm <= 0.0 { 0.0 } else { 60.0 / bpm }
}

#[inline]
fn duration_to_samples(beats: f32, beat_secs: f32, sample_rate: u32) -> usize {
    if beats <= 0.0 || beat_secs <= 0.0 {
        return 0;
    }
    let samples = (f64::from(beats) * f64::from(beat_secs) * f64::from(sample_rate)).round();
    // Clamp at the WAV ceiling so an astronomical duration / gate / release
    // from an untrusted recipe can't saturate the float→usize cast to
    // usize::MAX and blow up a downstream Vec allocation (the master buffer
    // here, or bake_inner's per-event buffer when this feeds gate/total).
    if samples >= MAX_WAV_SAMPLES as f64 {
        MAX_WAV_SAMPLES
    } else {
        samples as usize
    }
}

#[inline]
fn bake_key(event: &Event) -> (String, u32, u32, u32) {
    // Varispeed bakes once at native pitch (all its pitches share one bake),
    // so it uses a fixed `1.0` sentinel; TimePreserving keys on the actual
    // multiplier so each retuned pitch is cached separately.  A
    // TimePreserving event at exactly 1.0 collides with the sentinel, which
    // is harmless: retuning by 1.0 is the native bake either way.
    let pitch_bits = match event.pitch_mode {
        PitchMode::Varispeed => 1.0_f32.to_bits(),
        PitchMode::TimePreserving => event.pitch_multiplier.to_bits(),
    };
    (
        event.instrument_id.clone(),
        event.gate_beats.to_bits(),
        event.release_beats.to_bits(),
        pitch_bits,
    )
}

/// Clone `patch` with every oscillator's pitch scaled by `mult` — the
/// synthesis-time transposition behind [`PitchMode::TimePreserving`].  See
/// [`crate::node::NodeKind::scale_pitch`] for which nodes are (and are not)
/// pitched.
fn retuned_patch(patch: &AudioPatch, mult: f32) -> AudioPatch {
    let mut retuned = patch.clone();
    for node in &mut retuned.graph.nodes {
        node.kind.scale_pitch(mult);
    }
    retuned
}

/// Linear-interpolation resampler.  Output length is approximately
/// `source.len() / pitch_multiplier`, capped at `max_out` — the number of
/// samples the caller can actually consume — so a near-zero pitch can't drive
/// an unbounded allocation.  Pitch-up shortens the buffer; pitch-down
/// lengthens it.  An empty source or non-positive pitch produces an empty
/// output (defensive — bake() returns empty for non-positive duration, and a
/// zero or negative pitch is nonsense).
fn resample_linear(source: &[f32], pitch_multiplier: f32, max_out: usize) -> Vec<f32> {
    if source.is_empty() || !pitch_multiplier.is_finite() || pitch_multiplier <= 0.0 {
        return Vec::new();
    }
    // Compute the length in f64 and cap it at `max_out`.  A subnormal
    // pitch_multiplier (e.g. 1e-30) sends source.len()/pitch enormous — in
    // f32 it overflows to +Inf and the `as usize` cast saturates to
    // usize::MAX, which panics Vec::with_capacity.  The cap also avoids
    // materialising more samples than the caller can use.
    let raw_len = (source.len() as f64 / f64::from(pitch_multiplier)).floor();
    let out_len = if raw_len >= max_out as f64 {
        max_out
    } else {
        raw_len as usize
    };
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f32 * pitch_multiplier;
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_idx as f32;
        let next = src_idx + 1;
        if next < source.len() {
            out.push(source[src_idx] * (1.0 - frac) + source[next] * frac);
        } else if src_idx < source.len() {
            // Last sample — extrapolate by holding the final value.
            out.push(source[src_idx]);
        } else {
            break;
        }
    }
    out
}

#[inline]
fn write_into(master: &mut [f32], start: usize, src: &[f32], volume: f32) {
    let len = src.len().min(master.len().saturating_sub(start));
    for (i, s) in src.iter().take(len).enumerate() {
        master[start + i] += s * volume;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_at_unit_pitch_is_identity() {
        let src = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let out = resample_linear(&src, 1.0, usize::MAX);
        assert_eq!(out.len(), src.len());
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_at_pitch_two_halves_length() {
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&src, 2.0, usize::MAX);
        // 100 / 2 = 50 output samples; out[i] ≈ src[2i] = 2i.
        assert_eq!(out.len(), 50);
        for (i, s) in out.iter().enumerate() {
            assert!((s - (2 * i) as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn resample_at_pitch_half_doubles_length_with_interp() {
        let src = vec![0.0_f32, 10.0, 20.0, 30.0];
        let out = resample_linear(&src, 0.5, usize::MAX);
        // src.len() / 0.5 = 8; out[i] = src[i/2] interp.
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
        assert!((out[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn resample_rejects_non_positive_pitch() {
        assert!(resample_linear(&[1.0, 2.0], 0.0, usize::MAX).is_empty());
        assert!(resample_linear(&[1.0, 2.0], -1.0, usize::MAX).is_empty());
        assert!(resample_linear(&[1.0, 2.0], f32::NAN, usize::MAX).is_empty());
    }

    #[test]
    fn resample_empty_source_is_empty() {
        assert!(resample_linear(&[], 1.0, usize::MAX).is_empty());
    }

    #[test]
    fn resample_caps_output_at_max_out() {
        // Pitch-down would double the length to 200, but max_out clips it.
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample_linear(&src, 0.5, 10);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn resample_subnormal_pitch_is_bounded_not_oom() {
        // f32::MIN_POSITIVE sends source.len()/pitch to ~1e38; the old f32
        // path saturated the usize cast to usize::MAX and panicked
        // Vec::with_capacity.  With the cap the output is bounded at max_out.
        let src = vec![0.5_f32; 8];
        let out = resample_linear(&src, f32::MIN_POSITIVE, 32);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|s| (*s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn duration_to_samples_clamps_absurd_durations() {
        // An astronomical beat count clamps to the WAV ceiling instead of
        // saturating the float→usize cast to usize::MAX.
        assert_eq!(
            duration_to_samples(f32::MAX, 0.5, 44_100),
            MAX_WAV_SAMPLES,
            "huge duration must clamp to the WAV ceiling"
        );
        // Sane values are unaffected: 2 beats × 0.5 s/beat × 44.1 kHz.
        assert_eq!(duration_to_samples(2.0, 0.5, 44_100), 44_100);
    }

    #[test]
    fn write_into_clips_at_master_end() {
        let mut master = vec![0.0_f32; 5];
        let src = [1.0_f32, 1.0, 1.0, 1.0];
        write_into(&mut master, 3, &src, 1.0);
        // Only master[3] and master[4] get written.
        assert_eq!(master, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn write_into_applies_volume_scaler() {
        let mut master = vec![0.0_f32; 4];
        let src = [1.0_f32, 1.0, 1.0, 1.0];
        write_into(&mut master, 0, &src, 0.25);
        for s in master {
            assert!((s - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn write_into_at_or_past_master_end_is_noop() {
        let mut master = vec![0.0_f32; 4];
        let src = [1.0_f32, 1.0];
        write_into(&mut master, 10, &src, 1.0);
        assert!(master.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn empty_recipe_returns_empty_master() {
        let recipe = SequenceRecipe {
            duration_beats: 0.0,
            ..SequenceRecipe::default()
        };
        assert!(bake_sequence(&recipe).is_empty());
    }

    #[test]
    fn beat_seconds_handles_invalid_bpm() {
        assert_eq!(beat_seconds(120.0), 0.5);
        assert_eq!(beat_seconds(0.0), 0.0);
        assert_eq!(beat_seconds(-50.0), 0.0);
    }

    // --- loop crossfade ----------------------------------------------------

    #[test]
    fn apply_loop_crossfade_blends_tail_into_loop_start() {
        // 8-sample buffer = 4 main + 4 tail.  loop_start = 0, so the
        // 4 tail samples crossfade across the whole main region.
        let mut master = vec![
            // main region (loop content, all 0.0 — so crossfade reveals
            // pure tail at i=0 and pure (loop=0) at i=crossfade-1).
            0.0_f32, 0.0, 0.0, 0.0, // tail region: all 1.0
            1.0, 1.0, 1.0, 1.0,
        ];
        apply_loop_crossfade(&mut master, 0.0, 1.0, 4, 4, 4);
        // alpha = i/4. main[i] = (1-alpha)*1.0 + alpha*0.0 = 1 - i/4.
        assert!((master[0] - 1.0).abs() < 1e-6);
        assert!((master[1] - 0.75).abs() < 1e-6);
        assert!((master[2] - 0.5).abs() < 1e-6);
        assert!((master[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn apply_loop_crossfade_is_noop_when_no_tail() {
        let mut master = vec![0.5_f32; 4];
        let before = master.clone();
        apply_loop_crossfade(&mut master, 0.0, 1.0, 4, 4, 0);
        assert_eq!(master, before);
    }

    #[test]
    fn apply_loop_crossfade_skips_when_loop_start_past_main() {
        // loop_start_beats = 5 with main = 4 samples (and 1 sec/beat,
        // 4 sample rate) → loop_start sample index 5, past main_samples
        // of 4.  Function must skip without panicking.
        let mut master = vec![0.5_f32; 8];
        apply_loop_crossfade(&mut master, 5.0, 1.0, 4, 4, 4);
        // No change to the buffer.
        assert!(master.iter().all(|s| (*s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn apply_loop_crossfade_clips_window_when_loop_too_close_to_end() {
        // 8-sample buffer.  main_samples = 6, tail_samples = 2 (so the
        // last 2 indices [6, 7] are the tail).  loop_start_beats * sr =
        // 1.0 * 4 = sample 4, leaving only main_samples - loop_start =
        // 2 samples for the crossfade.  Even though tail_samples is
        // configured here as 4 (an over-request), the function clips
        // the effective window to min(tail, main_left, buffer_left) =
        // min(4, 2, 2) = 2.
        let mut master = vec![0.0_f32; 8];
        master[6] = 0.8;
        master[7] = 0.6;
        apply_loop_crossfade(&mut master, 1.0, 1.0, 4, 6, 4);
        // crossfade=2 → alpha = i/2.
        // master[4] = (1 - 0) * tail[0] + 0 * 0 = 0.8.
        // master[5] = (1 - 0.5) * tail[1] + 0.5 * 0 = 0.3.
        assert!((master[4] - 0.8).abs() < 1e-6, "got {}", master[4]);
        assert!((master[5] - 0.3).abs() < 1e-6, "got {}", master[5]);
    }
}
