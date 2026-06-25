//! Pure RIFF/WAVE IEEE-float encoder for baked `Vec<f32>` buffers.
//!
//! This is the Bevy-free half of the old `audio_source` module: it turns a
//! mono `f32` buffer into a minimal in-memory RIFF/WAVE blob with the
//! IEEE-float (format code `0x0003`) flavour.  The Bevy `AudioSource`
//! bridge ([`samples_to_audio_source`]) lives in the wrapper crate and
//! wraps the bytes this module produces.
//!
//! [`samples_to_audio_source`]: https://docs.rs/bevy_symbios_audio
//!
//! # Format
//!
//! - PCM container: RIFF / WAVE.
//! - Codec: IEEE float, 32-bit little-endian.
//! - Channels: mono only.  Both [`crate::bake::bake`] and
//!   [`crate::mixdown::bake_sequence`] produce mono buffers; stereo
//!   and multichannel routing remain out of scope.
//! - Includes a `fact` chunk so strict decoders (which require it for
//!   non-PCM formats per the WAV spec) accept the blob without warning.
//! - Header size: 12 (RIFF) + 8 + 16 (fmt) + 8 + 4 (fact) + 8 (data) = 56
//!   bytes, followed by `samples.len() * 4` bytes of float data.
//!
//! # Limits
//!
//! `data_size` in the WAV header is a 32-bit field.  At `4 bytes/sample`
//! mono, the upper bound is ~1.07 billion samples — about 6.7 hours at
//! 44.1 kHz or 6.2 hours at 48 kHz.  Bakes longer than that should be
//! split into segments.

const NUM_CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 32;
const FORMAT_IEEE_FLOAT: u16 = 3;
/// Bytes per (mono) sample frame: `channels * bits / 8`.
const BLOCK_ALIGN: u16 = NUM_CHANNELS * BITS_PER_SAMPLE / 8;
/// Fixed header bytes preceding the sample data (everything counted by the
/// RIFF chunk-size field besides `data_size`): `"WAVE"` + fmt + fact + data
/// chunk headers = 4 + 24 + 12 + 8.
const HEADER_BYTES: u32 = 4 + 8 + 16 + 8 + 4 + 8;

/// Largest number of mono `f32` samples that fit in one WAV blob, bounded by
/// the 32-bit RIFF size fields: `(u32::MAX − header) / bytes-per-sample`
/// ≈ 1.07 G samples (~6.7 h at 44.1 kHz).  [`samples_to_wav_bytes`] panics
/// above this rather than emit a silently-wrapped header; callers that build
/// buffers from untrusted sizes (the [`mod@crate::bake`] / [`crate::mixdown`]
/// paths) clamp their allocations to it so an absurd request can't drive a
/// `usize::MAX` allocation.
pub const MAX_WAV_SAMPLES: usize = ((u32::MAX - HEADER_BYTES) / (BLOCK_ALIGN as u32)) as usize;

/// Compute the WAV `data` chunk size for `num_samples` mono float samples,
/// asserting it (plus the header) stays inside the 32-bit RIFF size fields.
///
/// Beyond [`MAX_WAV_SAMPLES`] (≈ 6.7 h at 44.1 kHz) the `u32` size fields
/// would silently wrap and produce a corrupt file; we panic with a clear
/// message instead, since a silently-broken multi-gigabyte WAV is the worse
/// outcome.  Split longer bakes into segments.
fn wav_data_size(num_samples: usize) -> u32 {
    assert!(
        num_samples <= MAX_WAV_SAMPLES,
        "samples_to_wav_bytes: {num_samples} samples exceed the WAV 32-bit \
         size limit (~1.07 G samples / ~6.7 h at 44.1 kHz); split the bake \
         into segments"
    );
    // num_samples ≤ MAX_WAV_SAMPLES, so the cast is exact and the product
    // stays inside u32 — no overflow.
    num_samples as u32 * u32::from(BLOCK_ALIGN)
}

/// Encode `samples` as a complete RIFF/WAVE IEEE-float mono blob at
/// `sample_rate` Hz.  Exposed publicly so callers who already manage
/// their own `Bytes` plumbing (CLI tools writing to disk, the offline
/// baker in ticket #12) can skip the `AudioSource` wrapper.
pub fn samples_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    // Panics on bakes past the 32-bit WAV size limit rather than emitting a
    // silently-wrapped (corrupt) header — see [`wav_data_size`].
    let data_size: u32 = wav_data_size(samples.len());
    let num_samples_per_channel = samples.len() as u32;
    // WAV ByteRate = SampleRate × BlockAlign.  Compute from BLOCK_ALIGN
    // directly (rather than `sample_rate * channels * bits / 8`, whose
    // `* bits` intermediate overflows u32 around a 134 MHz sample_rate) and
    // saturate: a sample_rate large enough to overflow even this is far past
    // any real audio rate and unrepresentable in the u32 header field anyway,
    // so clamp rather than panic in debug / silently wrap in release.
    let byte_rate: u32 = sample_rate.saturating_mul(u32::from(BLOCK_ALIGN));
    let block_align: u16 = BLOCK_ALIGN;

    // RIFF chunk content size (everything after `RIFF<size>`):
    //   "WAVE" (4)
    // + "fmt " chunk header (8) + body (16)
    // + "fact" chunk header (8) + body (4)
    // + "data" chunk header (8) + body (data_size)
    let riff_chunk_size: u32 = HEADER_BYTES + data_size;

    let mut buf = Vec::with_capacity(8 + riff_chunk_size as usize);

    // --- RIFF / WAVE container -------------------------------------------
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // --- fmt chunk -------------------------------------------------------
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // body size
    buf.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
    buf.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // --- fact chunk (required by strict WAV spec for non-PCM formats) ----
    buf.extend_from_slice(b"fact");
    buf.extend_from_slice(&4u32.to_le_bytes()); // body size
    buf.extend_from_slice(&num_samples_per_channel.to_le_bytes());

    // --- data chunk ------------------------------------------------------
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    buf
}

const FORMAT_PCM: u16 = 1;
const PCM16_BITS_PER_SAMPLE: u16 = 16;
/// Bytes per (mono) sample frame for 16-bit PCM: `channels * 16 / 8`.
const PCM16_BLOCK_ALIGN: u16 = NUM_CHANNELS * PCM16_BITS_PER_SAMPLE / 8;
/// Header bytes preceding the data for a canonical PCM WAV (no `fact` chunk,
/// which the spec only requires for non-PCM formats): `"WAVE"` + fmt
/// header+body + data header = 4 + 8 + 16 + 8.
const PCM16_HEADER_BYTES: u32 = 4 + 8 + 16 + 8;

/// Largest number of mono 16-bit samples that fit in one PCM WAV blob, bounded
/// by the 32-bit RIFF size fields. See [`MAX_WAV_SAMPLES`] for the float path.
pub const MAX_WAV_SAMPLES_PCM16: usize =
    ((u32::MAX - PCM16_HEADER_BYTES) / (PCM16_BLOCK_ALIGN as u32)) as usize;

/// Encode `samples` as a complete RIFF/WAVE **16-bit PCM** mono blob at
/// `sample_rate` Hz — half the byte size of the 32-bit float
/// [`samples_to_wav_bytes`] at the cost of quantising to 16 bits, which is
/// inaudible for the procedural pads / hums baked here and the right default
/// when resident memory matters (notably wasm, where the heap never shrinks).
///
/// `f32` samples outside `[-1.0, 1.0]` are clamped before quantising. Panics
/// above [`MAX_WAV_SAMPLES_PCM16`] rather than emit a silently-wrapped header
/// (mirrors [`samples_to_wav_bytes`]); split longer bakes into segments.
pub fn samples_to_wav_bytes_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    assert!(
        samples.len() <= MAX_WAV_SAMPLES_PCM16,
        "samples_to_wav_bytes_pcm16: {} samples exceed the WAV 32-bit size \
         limit; split the bake into segments",
        samples.len()
    );
    let data_size: u32 = samples.len() as u32 * u32::from(PCM16_BLOCK_ALIGN);
    let byte_rate: u32 = sample_rate.saturating_mul(u32::from(PCM16_BLOCK_ALIGN));
    let riff_chunk_size: u32 = PCM16_HEADER_BYTES + data_size;

    let mut buf = Vec::with_capacity(8 + riff_chunk_size as usize);

    // --- RIFF / WAVE container -------------------------------------------
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // --- fmt chunk (canonical 16-byte PCM body) --------------------------
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&FORMAT_PCM.to_le_bytes());
    buf.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&PCM16_BLOCK_ALIGN.to_le_bytes());
    buf.extend_from_slice(&PCM16_BITS_PER_SAMPLE.to_le_bytes());

    // --- data chunk ------------------------------------------------------
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let q = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        buf.extend_from_slice(&q.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-test WAV decoder that walks the chunk list and returns
    /// `(sample_rate, samples)`.  Independent of [`samples_to_wav_bytes`]
    /// so a bug in the encoder doesn't slip past by being symmetrically
    /// reflected in a hand-rolled decoder pulling from the same broken
    /// shape — we read by spec, not by encoder shape.
    fn parse_wav(bytes: &[u8]) -> (u16, u32, Vec<f32>) {
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let mut pos = 12usize;
        let mut format_code: u16 = 0;
        let mut sample_rate: u32 = 0;
        let mut data_start: Option<usize> = None;
        let mut data_size: u32 = 0;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            let body = pos + 8;
            if id == b"fmt " {
                format_code = u16::from_le_bytes([bytes[body], bytes[body + 1]]);
                sample_rate = u32::from_le_bytes([
                    bytes[body + 4],
                    bytes[body + 5],
                    bytes[body + 6],
                    bytes[body + 7],
                ]);
            } else if id == b"data" {
                data_start = Some(body);
                data_size = size as u32;
            }
            pos = body + size;
        }
        let data_start = data_start.expect("data chunk");
        let mut samples = Vec::new();
        let mut i = data_start;
        let end = data_start + data_size as usize;
        while i + 4 <= end {
            let bits = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
            samples.push(f32::from_bits(bits));
            i += 4;
        }
        (format_code, sample_rate, samples)
    }

    #[test]
    fn wav_header_has_riff_wave_fmt_fact_data_markers() {
        let bytes = samples_to_wav_bytes(&[0.0_f32; 4], 44_100);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        // fmt body is 16 bytes → fact starts at 12 + 8 + 16 = 36.
        assert_eq!(&bytes[36..40], b"fact");
        // fact body is 4 bytes → data starts at 36 + 8 + 4 = 48.
        assert_eq!(&bytes[48..52], b"data");
    }

    #[test]
    fn format_code_is_ieee_float() {
        let bytes = samples_to_wav_bytes(&[0.5_f32], 48_000);
        let (format_code, _, _) = parse_wav(&bytes);
        assert_eq!(format_code, 3, "expected IEEE float (3), got {format_code}");
    }

    /// Walk the chunk list of a 16-bit PCM blob, returning
    /// `(format_code, bits_per_sample, sample_rate, i16 samples)`.
    fn parse_wav_pcm16(bytes: &[u8]) -> (u16, u16, u32, Vec<i16>) {
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let mut pos = 12usize;
        let (mut format_code, mut bits, mut sample_rate) = (0u16, 0u16, 0u32);
        let (mut data_start, mut data_size) = (None, 0u32);
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            let body = pos + 8;
            if id == b"fmt " {
                format_code = u16::from_le_bytes([bytes[body], bytes[body + 1]]);
                sample_rate = u32::from_le_bytes([
                    bytes[body + 4],
                    bytes[body + 5],
                    bytes[body + 6],
                    bytes[body + 7],
                ]);
                bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            } else if id == b"data" {
                data_start = Some(body);
                data_size = size as u32;
            }
            pos = body + size;
        }
        let data_start = data_start.expect("data chunk");
        let mut samples = Vec::new();
        let mut i = data_start;
        let end = data_start + data_size as usize;
        while i + 2 <= end {
            samples.push(i16::from_le_bytes([bytes[i], bytes[i + 1]]));
            i += 2;
        }
        (format_code, bits, sample_rate, samples)
    }

    #[test]
    fn pcm16_header_and_quantisation_round_trip() {
        let bytes = samples_to_wav_bytes_pcm16(&[0.0, 1.0, -1.0, 0.5], 22_050);
        let (format_code, bits, sr, samples) = parse_wav_pcm16(&bytes);
        assert_eq!(format_code, 1, "expected PCM (1), got {format_code}");
        assert_eq!(bits, 16);
        assert_eq!(sr, 22_050);
        // Canonical 44-byte PCM header + 4 samples * 2 bytes.
        assert_eq!(bytes.len(), 44 + 8);
        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], i16::MAX);
        assert_eq!(samples[2], -i16::MAX);
        assert_eq!(samples[3], (0.5 * f32::from(i16::MAX)).round() as i16);
    }

    #[test]
    fn pcm16_clamps_out_of_range_samples() {
        let bytes = samples_to_wav_bytes_pcm16(&[2.0, -2.0], 44_100);
        let (_, _, _, samples) = parse_wav_pcm16(&bytes);
        assert_eq!(samples, vec![i16::MAX, -i16::MAX]);
    }

    #[test]
    fn pcm16_is_half_the_bytes_of_float() {
        let s = [0.25_f32; 100];
        let f32_data = samples_to_wav_bytes(&s, 44_100).len();
        let pcm16_data = samples_to_wav_bytes_pcm16(&s, 44_100).len();
        // Data payload halves (200 vs 400 bytes). Headers differ: PCM is a
        // canonical 44-byte header; float carries an extra 12-byte `fact`
        // chunk → 56 bytes.
        assert_eq!(pcm16_data, 44 + 200);
        assert_eq!(f32_data, 56 + 400);
    }

    #[test]
    fn sample_rate_round_trips_through_header() {
        for sr in [22_050_u32, 32_000, 44_100, 48_000, 96_000, 192_000] {
            let bytes = samples_to_wav_bytes(&[0.0_f32; 1], sr);
            let (_, parsed_sr, _) = parse_wav(&bytes);
            assert_eq!(parsed_sr, sr);
        }
    }

    #[test]
    fn samples_round_trip_bit_for_bit() {
        let original = vec![
            0.0_f32,
            1.0,
            -1.0,
            0.5,
            -0.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-9,
            -1e-9,
        ];
        let bytes = samples_to_wav_bytes(&original, 44_100);
        let (_, _, back) = parse_wav(&bytes);
        assert_eq!(original.len(), back.len());
        for (i, (a, b)) in original.iter().zip(back.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "sample {i} bit-pattern mismatch: {a} → {b}"
            );
        }
    }

    #[test]
    fn riff_chunk_size_matches_actual_body_length() {
        let samples = vec![0.0_f32; 100];
        let bytes = samples_to_wav_bytes(&samples, 44_100);
        let declared = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        // `RIFF` + the u32 itself are NOT counted in the declared size.
        assert_eq!(declared, bytes.len() - 8);
    }

    #[test]
    fn data_chunk_size_matches_sample_byte_count() {
        let samples = vec![0.0_f32; 50];
        let bytes = samples_to_wav_bytes(&samples, 44_100);
        // Walk to the data chunk header (offset 48 in our layout).
        let data_size = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]) as usize;
        assert_eq!(data_size, samples.len() * 4);
    }

    #[test]
    fn empty_samples_buffer_yields_valid_header() {
        let bytes = samples_to_wav_bytes(&[], 44_100);
        let (format_code, sr, samples) = parse_wav(&bytes);
        assert_eq!(format_code, 3);
        assert_eq!(sr, 44_100);
        assert!(samples.is_empty());
    }

    #[test]
    fn wav_data_size_is_four_bytes_per_sample() {
        assert_eq!(wav_data_size(0), 0);
        assert_eq!(wav_data_size(1), 4);
        assert_eq!(wav_data_size(1_000), 4_000);
    }

    #[test]
    #[should_panic(expected = "32-bit")]
    fn wav_data_size_panics_past_32bit_limit() {
        // 2 billion samples × 4 bytes overflows the u32 size fields; the
        // helper must panic rather than wrap.  (Just an integer — no
        // multi-gigabyte allocation.)
        let _ = wav_data_size(2_000_000_000);
    }
}
