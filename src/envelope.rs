//! Record-boundary clamps — the bounds outside which an [`AudioPatch`] or
//! [`SequenceRecipe`] is not a sound design choice but an attack.
//!
//! # Why this lives here
//!
//! [`crate::bake::bake`] is unbounded by construction: node count drives
//! per-sample work, event count drives the mixdown's bake count, and several
//! configs turn a float straight into an allocation ([`crate::chorus::Chorus`]
//! delay times size a ring buffer).  A consumer that bakes a patch it did not
//! author — one that arrived over a network, or out of a file — therefore has
//! to bound it first, and until 0.2 every such consumer had to write those
//! bounds itself.  Overlands did, for the mirror it publishes to a PDS; its
//! own bake worker, holding the native types, did not.
//!
//! # What an envelope is, and is not
//!
//! An envelope is a **validity bound**, not a taste bound and not a mutation
//! range.  Everything any legitimate producer makes has to fit inside it, so
//! clamping an in-envelope patch is a no-op **bit for bit** — asserted by
//! `clamping_an_in_envelope_patch_changes_nothing`, over every kind in
//! [`NodeKind::defaults`].  Narrowing one of these numbers to what today's
//! content happens to use would silently rewrite a value somebody authored.
//!
//! # NaN and infinity
//!
//! A non-finite field resolves to **that field's default**, not to the nearest
//! bound.  This is deliberately the opposite of `symbios-texture`'s envelope,
//! where NaN belongs to no range and `f32::clamp` propagates it, so it settles
//! at the minimum.  Here the fields are audio parameters whose minimum is
//! usually silence or a degenerate filter: a NaN cutoff clamped to 1 Hz is a
//! dead voice, while a NaN cutoff reset to 1 kHz is the patch the author
//! most likely meant.  The rule is stated once here and applied by this
//! module's `clamp_finite`.

use crate::node::NodeKind;
use crate::patch::{AudioPatch, Connection, NodeGraph};
use crate::sequence::{Event, SequenceRecipe, Track};

/// Collection-size caps for one patch or recipe.
///
/// Every field is a *count*; the per-field numeric ranges are fixed
/// properties of each node config and are not configurable — a caller who
/// wants a 30 kHz oscillator wants a different synth, not a wider envelope.
///
/// [`Default`] is the set Overlands has enforced at its record boundary
/// since #695, so a consumer that adopts this type keeps the bounds that
/// were already load-bearing there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Envelope {
    /// Maximum nodes in a single [`NodeGraph`].  A graph this size already
    /// bakes for tens of seconds at audio rate; past it, an attack is far
    /// likelier than a sound design choice.
    pub max_nodes: usize,
    /// Maximum connections wired into one input port.  Ports sum their
    /// connections, so a realistic port holds one signal plus a handful of
    /// modulators.
    pub max_connections_per_port: usize,
    /// Maximum [`Event`]s on one [`Track`].  Events compound in the mixdown
    /// baker — one bake per unique `(instrument, gate)` — so an unbounded
    /// list amplifies bake cost quadratically.
    pub max_track_events: usize,
    /// Maximum [`crate::sequence::Instrument`]s in a recipe.  Each carries a
    /// whole patch, itself bounded by `max_nodes`.
    pub max_instruments: usize,
    /// Maximum [`Track`]s in a recipe.
    pub max_tracks: usize,
    /// Maximum length in **bytes** of an instrument id.  Truncation walks
    /// back to a UTF-8 boundary, so a multi-byte character straddling the cap
    /// is dropped whole rather than split.
    pub max_instrument_id_bytes: usize,
}

impl Default for Envelope {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_connections_per_port: 64,
            max_track_events: 4096,
            max_instruments: 64,
            max_tracks: 64,
            max_instrument_id_bytes: 128,
        }
    }
}

/// Bring a value inside the envelope.
///
/// Implemented for the two documents a consumer receives — [`AudioPatch`] and
/// [`SequenceRecipe`] — and for the pieces they nest, so a consumer holding
/// only a [`NodeGraph`] can clamp that.
pub trait ClampToEnvelope {
    /// Clamp every field and collection to `limits`, in place.
    ///
    /// A no-op, bit for bit, on a value already inside the envelope.
    fn clamp_to_envelope(&mut self, limits: &Envelope);
}

/// Clamp `v` to `[lo, hi]`, resolving NaN and infinity to `default`.
///
/// See the module docs for why a non-finite value takes the field's default
/// rather than the nearest bound.
fn clamp_finite(v: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        default
    }
}

/// Trim `s` to at most `max_bytes`, walking back to the previous UTF-8
/// boundary so `String::truncate`'s boundary panic cannot be triggered by a
/// multi-byte character straddling the cap.
fn truncate_on_char_boundary(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

impl ClampToEnvelope for AudioPatch {
    fn clamp_to_envelope(&mut self, limits: &Envelope) {
        self.graph.clamp_to_envelope(limits);
    }
}

impl ClampToEnvelope for NodeGraph {
    fn clamp_to_envelope(&mut self, limits: &Envelope) {
        // Cap node count first so the per-node walk below cannot be made to
        // traverse a hostile list.  Truncated from the tail, because the head
        // usually carries the wired output node.
        self.nodes.truncate(limits.max_nodes);
        for node in &mut self.nodes {
            node.kind.clamp_to_envelope(limits);
            for connections in node.inputs.values_mut() {
                connections.truncate(limits.max_connections_per_port);
                for connection in connections.iter_mut() {
                    connection.clamp_to_envelope(limits);
                }
            }
        }
    }
}

impl ClampToEnvelope for NodeKind {
    fn clamp_to_envelope(&mut self, _limits: &Envelope) {
        // The upper bounds here are conservative by design: this runs
        // *before* a patch reaches the baker, so it defuses a hostile value
        // rather than repairing it at synthesis time the way
        // `filter::clamp_cutoff` does.  22_050 Hz is Nyquist at the 44.1 kHz
        // default rate; a patch baked at a higher rate is clamped no further
        // than that, which is a bound on the record, not on the DSP.
        match self {
            // Nothing numeric to clamp.  `Unknown` in particular is left
            // alone: this build knows nothing about the config it stood for,
            // and there is nothing here to bound.
            Self::Silence | Self::Gate(_) | Self::Unknown => {}
            Self::Sine(c) => {
                c.freq_hz = clamp_finite(c.freq_hz, 0.0, 22_050.0, 440.0);
                c.phase_offset = clamp_finite(c.phase_offset, -1.0, 1.0, 0.0);
                c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 1.0);
            }
            Self::Square(c) => {
                c.freq_hz = clamp_finite(c.freq_hz, 0.0, 22_050.0, 440.0);
                c.duty = clamp_finite(c.duty, 0.0, 1.0, 0.5);
                c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 1.0);
            }
            Self::Sawtooth(c) => {
                c.freq_hz = clamp_finite(c.freq_hz, 0.0, 22_050.0, 440.0);
                c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 1.0);
            }
            Self::Triangle(c) => {
                c.freq_hz = clamp_finite(c.freq_hz, 0.0, 22_050.0, 440.0);
                c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 1.0);
            }
            Self::WhiteNoise(c) => c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 0.5),
            Self::PinkNoise(c) => c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 0.5),
            Self::BrownNoise(c) => c.amplitude = clamp_finite(c.amplitude, -8.0, 8.0, 0.5),
            Self::Adsr(c) => {
                c.attack_s = clamp_finite(c.attack_s, 0.0, 60.0, 0.01);
                c.decay_s = clamp_finite(c.decay_s, 0.0, 60.0, 0.1);
                c.sustain_level = clamp_finite(c.sustain_level, 0.0, 1.0, 0.7);
                c.release_s = clamp_finite(c.release_s, 0.0, 60.0, 0.2);
            }
            Self::BiquadLowpass(c) => {
                c.cutoff_hz = clamp_finite(c.cutoff_hz, 1.0, 22_050.0, 1_000.0);
                c.q = clamp_finite(c.q, 0.001, 64.0, 0.707);
            }
            Self::BiquadHighpass(c) => {
                c.cutoff_hz = clamp_finite(c.cutoff_hz, 1.0, 22_050.0, 1_000.0);
                c.q = clamp_finite(c.q, 0.001, 64.0, 0.707);
            }
            Self::BiquadBandpass(c) => {
                c.center_hz = clamp_finite(c.center_hz, 1.0, 22_050.0, 1_000.0);
                c.q = clamp_finite(c.q, 0.001, 64.0, 1.0);
            }
            Self::Lfo(c) => {
                c.rate_hz = clamp_finite(c.rate_hz, 0.0, 1_000.0, 1.0);
                c.depth = clamp_finite(c.depth, -10_000.0, 10_000.0, 1.0);
                c.offset = clamp_finite(c.offset, -10_000.0, 10_000.0, 0.0);
            }
            // Combiners — gain is a plain multiplier.  Bounded well past
            // unity but away from the float rails, so a hostile value cannot
            // drive the summed bus to infinity.
            Self::Mix(c) => c.gain = clamp_finite(c.gain, -64.0, 64.0, 1.0),
            Self::Gain(c) => c.gain = clamp_finite(c.gain, -64.0, 64.0, 1.0),
            Self::Chorus(c) => {
                c.rate_hz = clamp_finite(c.rate_hz, 0.0, 100.0, 0.8);
                // Delay times size the ring buffer, so a giant value here is
                // an allocation, not a sound.
                c.depth_ms = clamp_finite(c.depth_ms, 0.0, 100.0, 2.0);
                c.base_delay_ms = clamp_finite(c.base_delay_ms, 0.0, 100.0, 8.0);
                // The chorus line clamps feedback below 1.0 internally;
                // mirroring that ceiling keeps the line contractive.
                c.feedback = clamp_finite(c.feedback, 0.0, 0.95, 0.0);
                c.mix = clamp_finite(c.mix, 0.0, 1.0, 0.5);
            }
            Self::Reverb(c) => {
                c.room_size = clamp_finite(c.room_size, 0.0, 1.0, 0.5);
                c.damping = clamp_finite(c.damping, 0.0, 1.0, 0.5);
                c.mix = clamp_finite(c.mix, 0.0, 1.0, 0.3);
            }
        }
    }
}

/// Bound on a connection's DC value and on a node connection's `amount`.
///
/// Two reasons, and either would do on its own.
///
/// It is the only bound in this module with no relation to the others: the
/// combiner gains are `±64`, the widest thing here is an LFO depth at
/// `±10_000`, and a DC constant or modulation amount of a *million* is not a
/// value anybody reaches for — it is four orders past anything a patch does.
/// It was picked as "away from the float rails", which bounds the arithmetic
/// but says nothing about the sound.
///
/// And it has to survive a mirror. A schema with no floating-point type —
/// AT Protocol is the one driving this, but it is the general case — carries
/// these as scaled integers, and at the customary 1/10000 resolution `±1e6`
/// needs 10<sup>10</sup> ticks, past `i32`, where the cast saturates and the
/// value silently changes. `±1e5` needs 10<sup>9</sup>, which fits with room
/// to spare. Every other bound in this module already did.
const MAX_CONNECTION_MAGNITUDE: f32 = 100_000.0;

impl ClampToEnvelope for Connection {
    fn clamp_to_envelope(&mut self, _limits: &Envelope) {
        match self {
            Self::Constant { value } => {
                *value = clamp_finite(
                    *value,
                    -MAX_CONNECTION_MAGNITUDE,
                    MAX_CONNECTION_MAGNITUDE,
                    0.0,
                );
            }
            Self::Node { amount, .. } => {
                *amount = clamp_finite(
                    *amount,
                    -MAX_CONNECTION_MAGNITUDE,
                    MAX_CONNECTION_MAGNITUDE,
                    1.0,
                );
            }
        }
    }
}

impl ClampToEnvelope for SequenceRecipe {
    fn clamp_to_envelope(&mut self, limits: &Envelope) {
        self.bpm = clamp_finite(self.bpm, 1.0, 1_000.0, 120.0);
        // `sample_rate` is a u32, so finiteness is free; bounded to a
        // plausible audio range regardless.
        self.sample_rate = self.sample_rate.clamp(8_000, 192_000);
        self.duration_beats = clamp_finite(self.duration_beats, 0.0, 100_000.0, 4.0);
        // Both loop bounds are clamped against the *already clamped*
        // duration, so a hostile duration cannot smuggle a huge loop window
        // through behind it.
        if let Some(loop_start) = self.loop_start_beats.as_mut() {
            *loop_start = clamp_finite(*loop_start, 0.0, self.duration_beats.max(0.0), 0.0);
        }
        self.loop_crossfade_beats = clamp_finite(
            self.loop_crossfade_beats,
            0.0,
            self.duration_beats.max(0.0),
            0.0,
        );
        self.instruments.truncate(limits.max_instruments);
        for instrument in &mut self.instruments {
            truncate_on_char_boundary(&mut instrument.id, limits.max_instrument_id_bytes);
            instrument.patch.clamp_to_envelope(limits);
        }
        self.tracks.truncate(limits.max_tracks);
        for track in &mut self.tracks {
            track.clamp_to_envelope(limits);
        }
    }
}

impl ClampToEnvelope for Track {
    fn clamp_to_envelope(&mut self, limits: &Envelope) {
        self.events.truncate(limits.max_track_events);
        for event in &mut self.events {
            event.clamp_to_envelope(limits);
        }
    }
}

impl ClampToEnvelope for Event {
    fn clamp_to_envelope(&mut self, limits: &Envelope) {
        self.time_beats = clamp_finite(self.time_beats, 0.0, 100_000.0, 0.0);
        truncate_on_char_boundary(&mut self.instrument_id, limits.max_instrument_id_bytes);
        // Pitch is continuous (see the `sequence` module docs), so this is
        // not quantised to semitones — only bounded away from zero, below
        // which playback speed degenerates.
        self.pitch_multiplier = clamp_finite(self.pitch_multiplier, 0.001, 64.0, 1.0);
        self.volume = clamp_finite(self.volume, 0.0, 1.0, 1.0);
        self.gate_beats = clamp_finite(self.gate_beats, 0.0, 100_000.0, 1.0);
        // The release tail bakes extra samples after the gate closes, so it
        // is bounded like `gate_beats`.
        self.release_beats = clamp_finite(self.release_beats, 0.0, 100_000.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chorus::Chorus;
    use crate::mix::{Gain, Mix};
    use crate::oscillator::SineOsc;
    use crate::patch::{GraphNode, NodeId};
    use crate::reverb::Reverb;
    use crate::sequence::Instrument;

    /// The property the whole envelope rests on: it bounds validity, not
    /// taste. Every default-configured node, and a recipe built from them,
    /// must come through untouched — otherwise clamping a record silently
    /// rewrites something its author chose.
    #[test]
    fn clamping_an_in_envelope_patch_changes_nothing() {
        let limits = Envelope::default();
        for kind in NodeKind::defaults() {
            let mut patch = AudioPatch {
                seed: 7,
                graph: NodeGraph {
                    nodes: vec![GraphNode {
                        id: NodeId(0),
                        kind: kind.clone(),
                        inputs: Default::default(),
                    }],
                    output: NodeId(0),
                },
            };
            let before = patch.clone();
            patch.clamp_to_envelope(&limits);
            assert_eq!(patch, before, "{} was rewritten in envelope", kind.label());
        }

        let mut recipe = SequenceRecipe {
            instruments: vec![Instrument {
                id: String::from("bell"),
                patch: AudioPatch::default(),
            }],
            tracks: vec![Track {
                events: vec![Event::default()],
            }],
            ..SequenceRecipe::default()
        };
        let before = recipe.clone();
        recipe.clamp_to_envelope(&limits);
        assert_eq!(recipe, before);
    }

    /// And it is idempotent: clamping a clamped value is a second no-op, so
    /// a consumer may clamp on load and again before baking.
    #[test]
    fn clamping_is_idempotent() {
        let limits = Envelope::default();
        let mut patch = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: f32::INFINITY,
                        phase_offset: 99.0,
                        amplitude: f32::NAN,
                    }),
                    inputs: Default::default(),
                }],
                output: NodeId(0),
            },
        };
        patch.clamp_to_envelope(&limits);
        let once = patch.clone();
        patch.clamp_to_envelope(&limits);
        assert_eq!(patch, once);
    }

    #[test]
    fn chorus_clamps_hostile_values() {
        // Feedback past the contractive ceiling, a NaN delay, an
        // out-of-range mix — all must land back in safe bounds.
        let mut kind = NodeKind::Chorus(Chorus {
            rate_hz: 1e9,
            depth_ms: f32::NAN,
            base_delay_ms: 1e9,
            feedback: 5.0,
            mix: 50.0,
        });
        kind.clamp_to_envelope(&Envelope::default());
        let NodeKind::Chorus(c) = kind else {
            panic!("variant changed");
        };
        assert_eq!(c.rate_hz, 100.0);
        assert_eq!(c.depth_ms, 2.0, "NaN must fall back to the default");
        assert_eq!(c.base_delay_ms, 100.0);
        assert_eq!(c.feedback, 0.95, "feedback must stay contractive");
        assert_eq!(c.mix, 1.0);
    }

    #[test]
    fn reverb_clamps_to_unit_ranges() {
        let mut kind = NodeKind::Reverb(Reverb {
            room_size: 9.0,
            damping: -9.0,
            mix: f32::INFINITY,
        });
        kind.clamp_to_envelope(&Envelope::default());
        let NodeKind::Reverb(r) = kind else {
            panic!("variant changed");
        };
        assert_eq!(r.room_size, 1.0);
        assert_eq!(r.damping, 0.0);
        assert_eq!(r.mix, 0.3, "Inf must fall back to the default");
    }

    #[test]
    fn mix_and_gain_clamp_gain() {
        let mut m = NodeKind::Mix(Mix { gain: 1e6 });
        m.clamp_to_envelope(&Envelope::default());
        let NodeKind::Mix(m) = m else {
            panic!("variant changed");
        };
        assert_eq!(m.gain, 64.0);

        let mut g = NodeKind::Gain(Gain { gain: f32::NAN });
        g.clamp_to_envelope(&Envelope::default());
        let NodeKind::Gain(g) = g else {
            panic!("variant changed");
        };
        assert_eq!(g.gain, 1.0, "NaN gain must fall back to unity");
    }

    /// An unknown node has no config this build can bound, and clamping
    /// must not quietly turn it into something else — the bake warns and
    /// plays silence, which is a different thing from rewriting the record.
    #[test]
    fn an_unknown_kind_survives_clamping_unchanged() {
        let mut kind = NodeKind::Unknown;
        kind.clamp_to_envelope(&Envelope::default());
        assert_eq!(kind, NodeKind::Unknown);
    }

    #[test]
    fn collections_truncate_to_their_caps() {
        let limits = Envelope::default();
        let mut patch = AudioPatch::default();
        patch.graph.nodes = vec![GraphNode::default(); limits.max_nodes + 50];
        patch.graph.nodes[0]
            .inputs
            .insert(String::from("in"), vec![Connection::default(); 500]);
        patch.clamp_to_envelope(&limits);
        assert_eq!(patch.graph.nodes.len(), limits.max_nodes);
        assert_eq!(
            patch.graph.nodes[0].inputs["in"].len(),
            limits.max_connections_per_port
        );

        let mut recipe = SequenceRecipe {
            instruments: vec![Instrument::default(); limits.max_instruments + 10],
            tracks: vec![
                Track {
                    events: vec![Event::default(); limits.max_track_events + 10],
                };
                limits.max_tracks + 10
            ],
            ..SequenceRecipe::default()
        };
        recipe.clamp_to_envelope(&limits);
        assert_eq!(recipe.instruments.len(), limits.max_instruments);
        assert_eq!(recipe.tracks.len(), limits.max_tracks);
        assert_eq!(recipe.tracks[0].events.len(), limits.max_track_events);
    }

    /// A caller may hand its own caps; nothing above reads a constant.
    #[test]
    fn a_caller_supplied_envelope_is_the_one_enforced() {
        let limits = Envelope {
            max_nodes: 2,
            ..Envelope::default()
        };
        let mut patch = AudioPatch::default();
        patch.graph.nodes = vec![GraphNode::default(); 10];
        patch.clamp_to_envelope(&limits);
        assert_eq!(patch.graph.nodes.len(), 2);
    }

    /// The cap is in bytes, and the trim walks back to a character
    /// boundary — `String::truncate` would panic mid-character otherwise.
    #[test]
    fn an_instrument_id_is_trimmed_on_a_char_boundary() {
        let limits = Envelope {
            max_instrument_id_bytes: 4,
            ..Envelope::default()
        };
        // Three bytes each, so the cap falls inside the second character.
        let mut event = Event {
            instrument_id: "☃☃☃".into(),
            ..Event::default()
        };
        event.clamp_to_envelope(&limits);
        assert_eq!(event.instrument_id, "☃");

        let mut recipe = SequenceRecipe {
            instruments: vec![Instrument {
                id: "☃☃☃".into(),
                patch: AudioPatch::default(),
            }],
            ..SequenceRecipe::default()
        };
        recipe.clamp_to_envelope(&limits);
        assert_eq!(recipe.instruments[0].id, "☃");
    }

    /// A hostile duration must not smuggle a huge loop window in behind it:
    /// both loop bounds clamp against the duration *after* it is clamped.
    #[test]
    fn loop_bounds_follow_the_clamped_duration() {
        let mut recipe = SequenceRecipe {
            duration_beats: f32::INFINITY,
            loop_start_beats: Some(1e9),
            loop_crossfade_beats: 1e9,
            ..SequenceRecipe::default()
        };
        recipe.clamp_to_envelope(&Envelope::default());
        assert_eq!(recipe.duration_beats, 4.0, "Inf takes the field default");
        assert_eq!(recipe.loop_start_beats, Some(4.0));
        assert_eq!(recipe.loop_crossfade_beats, 4.0);
    }

    /// Connections are the one place an arbitrary float reaches the summed
    /// bus directly.
    #[test]
    fn connections_clamp_their_values() {
        let limits = Envelope::default();
        let mut constant = Connection::Constant { value: f32::NAN };
        constant.clamp_to_envelope(&limits);
        assert_eq!(constant, Connection::Constant { value: 0.0 });

        let mut node = Connection::Node {
            id: NodeId(3),
            amount: f32::NEG_INFINITY,
        };
        node.clamp_to_envelope(&limits);
        assert_eq!(
            node,
            Connection::Node {
                id: NodeId(3),
                amount: 1.0
            }
        );

        // Finite but absurd lands on the bound, both signs, both forms.
        let mut big = Connection::Constant { value: 1e9 };
        big.clamp_to_envelope(&limits);
        assert_eq!(
            big,
            Connection::Constant {
                value: MAX_CONNECTION_MAGNITUDE
            }
        );
        let mut small = Connection::Node {
            id: NodeId(1),
            amount: -1e9,
        };
        small.clamp_to_envelope(&limits);
        assert_eq!(
            small,
            Connection::Node {
                id: NodeId(1),
                amount: -MAX_CONNECTION_MAGNITUDE
            }
        );
    }

    /// Every bound in this module survives a 1/10000 fixed-point mirror.
    ///
    /// A consumer putting these on a schema with no float type carries them
    /// as scaled integers, and a bound past `i32` at that resolution is a
    /// bound the consumer cannot store: the cast saturates and the value
    /// silently changes. `MAX_CONNECTION_MAGNITUDE` was the one that did not
    /// fit — it needed 10^10 ticks against a ceiling of 2.1 x 10^9.
    ///
    /// Stated as a test rather than a comment because the next bound anyone
    /// adds is the one at risk, and this is cheaper to read than the
    /// arithmetic.
    #[test]
    fn every_bound_survives_a_fixed_point_mirror() {
        const SCALE: f64 = 10_000.0;
        const CEILING: f64 = i32::MAX as f64;

        let widest = [
            ("connection", MAX_CONNECTION_MAGNITUDE),
            // The other extremes of the table, so a future widening of any
            // of them trips this too.
            ("lfo depth", 10_000.0),
            ("beats", 100_000.0),
            ("frequency", 22_050.0),
            ("gain", 64.0),
        ];
        for (what, bound) in widest {
            assert!(
                f64::from(bound) * SCALE <= CEILING,
                "the {what} bound of {bound} needs {} ticks at 1/{SCALE:.0} \
                 resolution, past the {CEILING} a 32-bit mirror can carry",
                f64::from(bound) * SCALE
            );
        }
    }
}
