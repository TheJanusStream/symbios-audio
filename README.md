# symbios-audio

Deterministic algorithmic audio synthesis, engine-free. This crate is the
core of `bevy_symbios_audio`: a DAG-of-nodes synth and a sequencer that
render to plain `Vec<f32>` buffers with no dependency on Bevy, `wgpu`, or an
audio device, so they can be driven from any Rust project — a game, a
headless bake worker, a command-line tool.

The Bevy-coupled layer — the `AudioSource` asset bridge, the async bake pool
and ECS handover, the resource cache, the egui editor, and the CLI — lives in
the `bevy_symbios_audio` wrapper crate, which re-exports this crate wholesale.

## What you get

- **18 node kinds**: sine / square / sawtooth / triangle oscillators (with
  optional PolyBLEP anti-aliasing), white / pink / brown noise, an ADSR
  envelope, biquad lowpass / highpass / bandpass filters, an LFO, `Mix` and
  `Gain` combiners, a sequencer-driven `Gate`, and `Chorus` and `Reverb`
  delay-line effects.
- **Cross-node modulation**: every input port is a *list* of connections
  whose values are summed, so an LFO into a filter's `cutoff_hz`, an envelope
  into an amplitude, or oscillator-into-oscillator FM all fall out of the
  same wiring rule. Connections carry an `amount` multiplier.
- **A sequencer layer**: a `SequenceRecipe` names instruments (each a whole
  patch), schedules `Event`s in beats, and mixes down to one master buffer —
  with an optional loop point whose crossfade is baked in, so the buffer
  loops without a click from its last sample back to the loop start
  (`loop_start_sample`). The beats before the loop start are a one-shot
  run-up: loop from the loop start, not from the first sample. Pitch is
  continuous, and each event chooses whether to resample (`Varispeed`) or
  retune the oscillators at synthesis time (`TimePreserving`).
- **A record-boundary envelope**: `clamp_to_envelope` bounds node counts,
  event counts and every numeric field, so a patch that arrived over a
  network can be baked safely. Clamping a patch that is already in envelope
  is a no-op, bit for bit.
- **Evolvable configs**: every config struct implements
  `symbios_genetics::Genotype` (mutation + crossover) through the
  `impl_genotype!` macro, so sounds can be bred rather than dialled in.
- **An enumerable node roster**: `for_each_node_kind!` and
  `NodeKind::defaults()` expose the built-in kinds as data, so a mirror,
  editor picker or wire test in another crate tracks this one instead of
  keeping a hand-written copy that goes stale.
- **Deterministic**: the same patch, seed, sample rate and duration always
  produce a bit-identical buffer. Evaluation order comes from a topological
  sort with a stable tie-break, not from list order.
- **A pure WAV encoder**: 32-bit IEEE float and half-size 16-bit PCM, no
  external crate.

## Quick start

```rust
use symbios_audio::{AudioPatch, Connection, GraphNode, NodeGraph, NodeId, NodeKind, bake};
use symbios_audio::oscillator::SineOsc;
use symbios_audio::mix::Gain;

// A 440 Hz sine through a VCA at half gain.
let patch = AudioPatch {
    seed: 0,
    graph: NodeGraph {
        nodes: vec![
            GraphNode {
                id: NodeId(0),
                kind: NodeKind::Sine(SineOsc { freq_hz: 440.0, ..SineOsc::default() }),
                inputs: Default::default(),
            },
            GraphNode {
                id: NodeId(1),
                kind: NodeKind::Gain(Gain { gain: 0.5 }),
                inputs: Default::default(),
            }
            .with_input("in", Connection::from_node(NodeId(0))),
        ],
        output: NodeId(1),
    },
};

let samples = bake(&patch, 44_100, 0.25);
assert_eq!(samples.len(), 11_025);
```

`bake` panics on a structurally invalid graph (a cycle, a dangling
reference, a duplicate id, a missing output). On any path where the patch is
not yours — a worker thread, a network payload — use `try_bake`, which
returns the `GraphError` instead, and clamp first:

```rust
use symbios_audio::{AudioPatch, ClampToEnvelope, Envelope, try_bake};

fn bake_untrusted(mut patch: AudioPatch) -> Vec<f32> {
    patch.clamp_to_envelope(&Envelope::default());
    try_bake(&patch, 44_100, 1.0).unwrap_or_default()
}
```

## Forward compatibility

`NodeKind` is `#[non_exhaustive]` and an unrecognised `"kind"` tag decodes to
`NodeKind::Unknown`, so a document written by a newer version still loads —
the rest of the graph survives and the unknown node bakes as silence, with
one warning per bake.

`Unknown` deliberately **cannot be serialized**: writing a graph that holds
one returns an error rather than replacing a node this build merely failed to
recognise with a husk. A consumer that round-trips documents should surface
that error to whoever is saving.

## Crate layout

- `patch` — the schema (`AudioPatch`, `NodeGraph`, `GraphNode`,
  `Connection`) and `topo_sort`.
- `node` — the `Node` trait, `BakeContext`, the `NodeKind` enum, and the
  `for_each_node_kind!` roster.
- `oscillator`, `noise`, `adsr`, `filter`, `lfo`, `mix`, `gate`, `chorus`,
  `reverb` — the built-in node implementations.
- `bake` — one `AudioPatch` into a `Vec<f32>`.
- `sequence` + `mixdown` — the timeline-of-events layer, `bake_sequence`,
  and `loop_start_sample`, the sample a looping buffer goes back to.
- `envelope` — `Envelope` and `ClampToEnvelope`, the record-boundary clamp.
- `wav` — RIFF/WAVE encoding for baked buffers.
- `genetics` — the `impl_genotype!` macro and shared mutation helpers.

## License

MIT.
