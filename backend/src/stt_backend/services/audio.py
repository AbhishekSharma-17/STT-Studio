"""Small audio utilities shared by REST and WebSocket paths.

We intentionally keep this module tiny and IO-free:
- Browser sends raw PCM16 @ 16kHz mono via AudioWorklet, so no server-side
  decoding is needed for the WS path.
- For the REST path we pass bytes straight through to vLLM (librosa/soundfile
  handles wav/mp3/flac/ogg/m4a there).

This module provides only the WAV-container wrapper so we can post a VAD-emitted
PCM16 utterance to vLLM's /v1/audio/transcriptions as a proper WAV.
"""

from __future__ import annotations

import io
import struct

SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Wrap raw little-endian mono PCM16 bytes in a minimal WAV header.

    Cheap (~70 bytes of prefix), no dependencies, no resampling. If `pcm`
    isn't 16-bit mono at `sample_rate`, the resulting WAV is wrong — callers
    are responsible for the format.
    """
    if len(pcm) % BYTES_PER_SAMPLE != 0:
        raise ValueError("PCM buffer length must be a multiple of 2 bytes (16-bit)")

    num_samples = len(pcm) // BYTES_PER_SAMPLE
    byte_rate = sample_rate * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    data_size = num_samples * CHANNELS * BYTES_PER_SAMPLE
    riff_size = 36 + data_size

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # PCM fmt chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", CHANNELS))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", BITS_PER_SAMPLE))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


def ms_to_samples(ms: int, sample_rate: int = SAMPLE_RATE_HZ) -> int:
    return (ms * sample_rate) // 1000


def samples_to_ms(samples: int, sample_rate: int = SAMPLE_RATE_HZ) -> int:
    return (samples * 1000) // sample_rate
