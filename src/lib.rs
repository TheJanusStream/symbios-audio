//! `symbios-audio` — the Bevy-free core of `bevy_symbios_audio`.
//!
//! A DAG-of-nodes synth (sine / square / sawtooth / triangle oscillators,
//! white / pink / brown noise, ADSR envelopes, biquad LP/HP/BP filters,
//! LFOs, [`Mix`] / [`Gain`] (VCA) combiners, a sequencer-driven [`Gate`],
//! [`Chorus`] and [`Reverb`] delay-line effects, and cross-node modulation
//! routing) producing deterministic `Vec<f32>` buffers, plus a pure
//! RIFF/WAVE encoder.
//!
//! This crate holds the pure DSP language with no dependency on Bevy,
//! `bevy_egui`, `wgpu`, or `winit`.  The Bevy integration — the
//! `AudioSource` asset bridge, the async-bake rayon pool + ECS handover,
//! the `Resource` cache wrapper, the egui editor, and the CLI — lives in
//! the `bevy_symbios_audio` wrapper crate, which re-exports everything
//! here.
//!
//! # Layers
//!
//! - [`patch`] — schema + topology.
//! - [`node`] — `Node` trait + `BakeContext` + the closed `NodeKind`
//!   enum that all built-in nodes plug into.
//! - [`oscillator`], [`noise`], [`adsr`], [`filter`], [`lfo`], [`mix`]
//!   (Mix/Gain), [`gate`], [`chorus`], [`reverb`] — the built-in node
//!   implementations.
//! - [`mod@bake`] — turns one [`AudioPatch`] into `Vec<f32>`.
//! - [`sequence`] + [`mixdown`] — the timeline-of-events layer and the
//!   seamless-loop-aware [`bake_sequence`].
//! - [`wav`] — pure RIFF/WAVE encoder for baked buffers (32-bit IEEE float and
//!   half-size 16-bit PCM).
//! - [`genetics`] — declarative [`impl_genotype!`] macro and shared
//!   mutation helpers that wire every config struct into
//!   `symbios-genetics`.
//!
//! Every config type implements [`symbios_genetics::Genotype`] via the
//! [`impl_genotype!`] declarative macro, so the entire DSP language plugs
//! into the evolutionary search algorithms in the `symbios-genetics`
//! crate.

pub mod adsr;
pub mod bake;
pub mod chorus;
pub mod filter;
pub mod gate;
pub mod genetics;
pub mod lfo;
pub mod mix;
pub mod mixdown;
pub mod node;
pub mod noise;
pub mod oscillator;
pub mod patch;
pub mod reverb;
pub mod sequence;
pub mod wav;

pub use adsr::{AdsrCurve, AdsrEnvelope};
pub use bake::{bake, try_bake};
pub use chorus::Chorus;
pub use filter::{BiquadBandpass, BiquadHighpass, BiquadLowpass, BiquadState};
pub use gate::Gate;
pub use lfo::{Lfo, LfoShape};
pub use mix::{Gain, Mix};
pub use mixdown::bake_sequence;
pub use node::{BakeContext, Node, NodeKind};
pub use noise::{BrownNoise, PinkNoise, WhiteNoise};
pub use oscillator::{
    AntiAlias, OscPhase, SawPolarity, SawtoothOsc, SineOsc, SquareOsc, TriangleOsc,
};
pub use patch::{AudioPatch, Connection, GraphError, GraphNode, NodeGraph, NodeId, topo_sort};
pub use reverb::Reverb;
pub use sequence::{Event, Instrument, PitchMode, SequenceRecipe, Track};
pub use wav::{
    MAX_WAV_SAMPLES, MAX_WAV_SAMPLES_PCM16, samples_to_wav_bytes, samples_to_wav_bytes_pcm16,
};
