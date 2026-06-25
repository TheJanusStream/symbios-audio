//! `SequenceRecipe` — the "MIDI-ish" timeline layer that sits on top
//! of [`AudioPatch`].
//!
//! Where [`AudioPatch`] describes a single synth voice, a
//! [`SequenceRecipe`] orchestrates multiple voices into ambient tracks:
//! instruments are named, embedded patches; tracks are sequences of
//! [`Event`]s scheduled in beats; the recipe carries its own BPM and
//! optional loop window so the same JSON describes both the events *and*
//! the temporal grid they sit on.
//!
//! This module is pure schema + [`Genotype`] integration.  The recipe is
//! turned into a mono `Vec<f32>` master buffer by
//! [`crate::mixdown::bake_sequence`]; the same function also bakes the
//! seamless-loop tail crossfade when `loop_start_beats` is set.
//!
//! # Continuous pitch
//!
//! `Event::pitch_multiplier` is an `f32`, not a semitone integer.  Not
//! everything is music — wind gusts, ambient drones, and environmental
//! layers want microtonal / continuous pitch sweeps.  A multiplier of
//! 1.0 plays the instrument at its native pitch; 2.0 is an octave up;
//! 0.97 is a 3% detune.  Quantised semitone playback is a
//! caller-side convention: `2.0_f32.powf(semitones / 12.0)`.
//!
//! # Pitch vs. time ([`PitchMode`])
//!
//! `Event::pitch_mode` chooses *how* the multiplier is realised:
//!
//! - [`PitchMode::Varispeed`] (the default) resamples the native bake —
//!   tape-style, so pitch and duration are coupled: a pitch-up finishes
//!   before its gate window and a pitch-down hangs past it.  Cheap, shares
//!   one bake across every pitch of an instrument, and the right call for
//!   SFX where the speed-up *is* the effect.
//! - [`PitchMode::TimePreserving`] retunes the instrument's oscillators at
//!   synthesis time and bakes the event at its true `gate + release`
//!   length, so pitch and duration are independent — the note fills its
//!   slot regardless of transposition, with no resampling artifacts.  Each
//!   distinct pitch is its own bake (see [`crate::mixdown::bake_sequence`]).

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::Genotype;

use crate::genetics::{mutate_f32, mutate_f32_log};
use crate::patch::AudioPatch;

/// Top-level recipe — the JSON document a sequencer authors and
/// [`crate::mixdown::bake_sequence`] consumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRecipe {
    /// Tempo for beat-relative timestamps (BPM).
    pub bpm: f32,
    /// Target sample rate when the recipe is baked.  Default `44_100`.
    pub sample_rate: u32,
    /// Total length of the timeline *before* the loop tail.  The
    /// mixdown baker renders this many beats; when looping is enabled
    /// it then extends the bake by `loop_crossfade_beats` of tail so
    /// release tails that spill past `duration_beats` can be folded
    /// back into the loop start by the crossfade.
    pub duration_beats: f32,
    /// Beat at which the seamless loop should restart.  `None` means
    /// "play once, don't loop"; `Some(b)` means "after the timeline
    /// completes, hop back here and continue".  When set, the mixdown
    /// baker pre-mixes the tail crossfade into the buffer so a hard
    /// `Source::loop_..()` is click-free at the seam.
    pub loop_start_beats: Option<f32>,
    /// Tail crossfade window in beats — how much of the end of the
    /// timeline overlaps with the start of the loop region when
    /// looping.  Zero disables the crossfade (hard cut at the loop
    /// point).
    pub loop_crossfade_beats: f32,
    /// Instruments referenced by `tracks[].events[].instrument_id`.
    /// Each carries its full `AudioPatch` so the recipe is
    /// self-contained — no external patch library required.
    pub instruments: Vec<Instrument>,
    /// Parallel tracks of events.  Tracks have no inherent semantics
    /// (mute/solo, panning, etc. live on the instrument or the event);
    /// they're a grouping hint for editors, summed flat by the
    /// mixdown baker.
    pub tracks: Vec<Track>,
}

impl Default for SequenceRecipe {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            sample_rate: 44_100,
            duration_beats: 4.0,
            loop_start_beats: None,
            loop_crossfade_beats: 0.0,
            instruments: Vec::new(),
            tracks: Vec::new(),
        }
    }
}

/// One named, sequenced instrument — an `AudioPatch` plus the string
/// id that events use to reference it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Instrument {
    /// Stable identifier used by `Event::instrument_id`.  String rather
    /// than an integer index so a recipe survives a reorder of the
    /// `instruments` list without invalidating events.
    pub id: String,
    /// The full DSP graph that produces this instrument's voice.
    pub patch: AudioPatch,
}

/// A parallel track of events.  All events on one track share an
/// implicit position in the mix; the mixdown baker sums all tracks
/// into a single master buffer.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Track {
    pub events: Vec<Event>,
}

/// How an [`Event`]'s `pitch_multiplier` is realised at mixdown time.
///
/// See the module-level "Pitch vs. time" section for the trade-off.  The
/// default is [`PitchMode::Varispeed`] so recipes authored before this
/// field existed (no `pitch_mode` on the wire) bake byte-for-byte as
/// before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PitchMode {
    /// Resample the native bake — pitch and duration are coupled
    /// (tape varispeed).  The historical behaviour.
    #[default]
    Varispeed,
    /// Retune the oscillators at synthesis time and bake at the event's
    /// true `gate + release` length — pitch and duration are independent.
    TimePreserving,
}

/// One scheduled note / sound / gust on a track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Start time in beats from the recipe's `t = 0`.
    pub time_beats: f32,
    /// Which [`Instrument::id`] to play.
    pub instrument_id: String,
    /// Continuous pitch scaler.  `1.0` is the patch's native pitch;
    /// `2.0` is one octave up; `0.5` is one down; any positive value
    /// is valid for microtonal / drone sweeps.
    pub pitch_multiplier: f32,
    /// Volume scaler applied at mixdown time.  Bounded `[0, 1]` by
    /// convention; the baker clamps before summation.
    pub volume: f32,
    /// How long the note's gate is held **open**, in beats.  The mixdown
    /// baker opens the instrument's gate window for this long (see
    /// [`crate::node::NodeKind::Gate`]), so an [`crate::adsr::AdsrEnvelope`]
    /// wired `Gate → gate` attacks, decays, and sustains for the gate.  At
    /// the end of the window the gate closes and the envelope enters its
    /// release stage.  Longer than the patch's attack+decay → it holds at
    /// sustain for the extra time; shorter → it releases early, from a
    /// partial value.
    pub gate_beats: f32,
    /// Extra tail baked *after* the gate closes, in beats — enough for the
    /// envelope's release to ring out (and for resonant filters / delays
    /// to decay).  `0.0` (the default) cuts the note the instant the gate
    /// closes, reproducing a hard one-shot.  An instrument with no
    /// gate-driven release can leave this at `0.0`.
    #[serde(default)]
    pub release_beats: f32,
    /// How `pitch_multiplier` is realised — resample (default) or
    /// synthesis-time retune.  See [`PitchMode`].  `#[serde(default)]`
    /// keeps pre-existing recipes (no `pitch_mode` key) on the historical
    /// resample path.
    #[serde(default)]
    pub pitch_mode: PitchMode,
}

impl Default for Event {
    fn default() -> Self {
        Self {
            time_beats: 0.0,
            instrument_id: String::new(),
            pitch_multiplier: 1.0,
            volume: 1.0,
            gate_beats: 1.0,
            release_beats: 0.0,
            pitch_mode: PitchMode::Varispeed,
        }
    }
}

// --- Genotype ---------------------------------------------------------------

const BPM_MIN: f32 = 20.0;
const BPM_MAX: f32 = 300.0;
/// How much the per-mutate volume jitter can move a value, in absolute
/// terms.  Smaller than the oscillator-amplitude jitter because
/// over-pumping a single event's volume across mutations would make
/// every track gradually compress toward the rail.
const VOLUME_HALF_RANGE: f32 = 0.1;

impl Genotype for SequenceRecipe {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // Primary axes per the ticket: BPM and event volumes.
        self.bpm = mutate_f32_log(self.bpm, rng, rate, 0.2, BPM_MIN, BPM_MAX);
        for track in &mut self.tracks {
            for event in &mut track.events {
                event.volume = mutate_f32(event.volume, rng, rate, VOLUME_HALF_RANGE, 0.0, 1.0);
            }
        }
        // Structural mutation (add/remove events, swap instruments,
        // change pitches, slide times) is deliberately deferred — it
        // requires schema-aware moves that risk silent recipes if done
        // naively, and the genetic search is just as well served by
        // mutating the underlying AudioPatches (which already implement
        // Genotype through impl_genotype!).
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // Field-uniform crossover: each top-level field is drawn from
        // one parent.  Tracks and instruments are taken whole rather
        // than spliced so event references stay coherent.
        Self {
            bpm: if rng.random::<bool>() {
                self.bpm
            } else {
                other.bpm
            },
            sample_rate: if rng.random::<bool>() {
                self.sample_rate
            } else {
                other.sample_rate
            },
            duration_beats: if rng.random::<bool>() {
                self.duration_beats
            } else {
                other.duration_beats
            },
            loop_start_beats: if rng.random::<bool>() {
                self.loop_start_beats
            } else {
                other.loop_start_beats
            },
            loop_crossfade_beats: if rng.random::<bool>() {
                self.loop_crossfade_beats
            } else {
                other.loop_crossfade_beats
            },
            instruments: if rng.random::<bool>() {
                self.instruments.clone()
            } else {
                other.instruments.clone()
            },
            tracks: if rng.random::<bool>() {
                self.tracks.clone()
            } else {
                other.tracks.clone()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn small_recipe() -> SequenceRecipe {
        SequenceRecipe {
            bpm: 110.0,
            sample_rate: 44_100,
            duration_beats: 8.0,
            loop_start_beats: Some(4.0),
            loop_crossfade_beats: 1.0,
            instruments: vec![
                Instrument {
                    id: "wind".into(),
                    patch: AudioPatch::default(),
                },
                Instrument {
                    id: "kick".into(),
                    patch: AudioPatch::default(),
                },
            ],
            tracks: vec![
                Track {
                    events: vec![
                        Event {
                            time_beats: 0.0,
                            instrument_id: "wind".into(),
                            pitch_multiplier: 1.0,
                            volume: 0.6,
                            gate_beats: 8.0,
                            release_beats: 0.0,
                            ..Default::default()
                        },
                        Event {
                            time_beats: 4.0,
                            instrument_id: "wind".into(),
                            pitch_multiplier: 0.97,
                            volume: 0.4,
                            gate_beats: 4.0,
                            release_beats: 0.0,
                            ..Default::default()
                        },
                    ],
                },
                Track {
                    events: vec![Event {
                        time_beats: 0.5,
                        instrument_id: "kick".into(),
                        pitch_multiplier: 1.0,
                        volume: 0.8,
                        gate_beats: 0.25,
                        release_beats: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        }
    }

    // --- defaults ----------------------------------------------------------

    #[test]
    fn default_recipe_has_sensible_shape() {
        let r = SequenceRecipe::default();
        assert_eq!(r.sample_rate, 44_100);
        assert_eq!(r.bpm, 120.0);
        assert!(r.instruments.is_empty());
        assert!(r.tracks.is_empty());
        assert!(r.loop_start_beats.is_none());
    }

    #[test]
    fn default_event_uses_unit_pitch_and_volume() {
        let e = Event::default();
        assert_eq!(e.pitch_multiplier, 1.0);
        assert_eq!(e.volume, 1.0);
        assert_eq!(e.time_beats, 0.0);
        // Default pitch mode is the historical resample path so existing
        // recipes bake unchanged.
        assert_eq!(e.pitch_mode, PitchMode::Varispeed);
    }

    #[test]
    fn event_without_pitch_mode_field_deserializes_to_varispeed() {
        // A recipe authored before pitch_mode existed has no such key;
        // #[serde(default)] must fill Varispeed rather than erroring.
        let json = r#"{
            "time_beats": 0.0,
            "instrument_id": "x",
            "pitch_multiplier": 1.0,
            "volume": 1.0,
            "gate_beats": 1.0
        }"#;
        let e: Event = serde_json::from_str(json).unwrap();
        assert_eq!(e.pitch_mode, PitchMode::Varispeed);
    }

    // --- Genotype ----------------------------------------------------------

    #[test]
    fn mutate_keeps_bpm_inside_audible_tempo_range() {
        let mut recipe = SequenceRecipe {
            bpm: 25.0,
            ..small_recipe()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..500 {
            recipe.mutate(&mut rng, 1.0);
            assert!(
                (BPM_MIN..=BPM_MAX).contains(&recipe.bpm),
                "BPM {} escaped [{BPM_MIN}, {BPM_MAX}]",
                recipe.bpm
            );
        }
    }

    #[test]
    fn mutate_keeps_event_volumes_inside_unit_interval() {
        let mut recipe = small_recipe();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..200 {
            recipe.mutate(&mut rng, 1.0);
            for t in &recipe.tracks {
                for e in &t.events {
                    assert!(
                        (0.0..=1.0).contains(&e.volume),
                        "event volume {} escaped [0,1]",
                        e.volume
                    );
                }
            }
        }
    }

    #[test]
    fn mutate_preserves_structure_lengths() {
        // Per ticket: structural mutation is deferred.  Track and
        // instrument counts must stay fixed across mutations.
        let mut recipe = small_recipe();
        let inst_count = recipe.instruments.len();
        let track_count = recipe.tracks.len();
        let event_counts: Vec<usize> = recipe.tracks.iter().map(|t| t.events.len()).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..50 {
            recipe.mutate(&mut rng, 1.0);
        }
        assert_eq!(recipe.instruments.len(), inst_count);
        assert_eq!(recipe.tracks.len(), track_count);
        for (t, expected) in recipe.tracks.iter().zip(event_counts) {
            assert_eq!(t.events.len(), expected);
        }
    }

    #[test]
    fn mutate_at_rate_zero_is_identity() {
        let original = small_recipe();
        let mut mutated = original.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        mutated.mutate(&mut rng, 0.0);
        assert_eq!(mutated, original);
    }

    #[test]
    fn crossover_takes_each_field_from_one_parent() {
        let a = small_recipe();
        let b = SequenceRecipe {
            bpm: 200.0,
            sample_rate: 48_000,
            duration_beats: 16.0,
            loop_start_beats: None,
            loop_crossfade_beats: 0.0,
            ..a.clone()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let child = a.crossover(&b, &mut rng);
        assert!(child.bpm == a.bpm || child.bpm == b.bpm);
        assert!(child.sample_rate == a.sample_rate || child.sample_rate == b.sample_rate);
        assert!(
            child.duration_beats == a.duration_beats || child.duration_beats == b.duration_beats
        );
    }
}
