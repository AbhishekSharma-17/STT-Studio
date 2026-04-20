from __future__ import annotations

import struct
import wave

import pytest

from stt_backend.services.audio import ms_to_samples, pcm16_to_wav, samples_to_ms


def test_pcm16_to_wav_roundtrip_via_stdlib() -> None:
    # 1 kHz tone, 100 ms, 16 kHz mono
    import math

    sr = 16_000
    duration_s = 0.1
    samples = [
        int(32767 * 0.5 * math.sin(2 * math.pi * 1000 * t / sr))
        for t in range(int(sr * duration_s))
    ]
    pcm = struct.pack(f"<{len(samples)}h", *samples)

    wav = pcm16_to_wav(pcm, sample_rate=sr)

    import io

    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sr
        assert wf.getnframes() == len(samples)
        read = wf.readframes(wf.getnframes())
        assert read == pcm


def test_pcm16_to_wav_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="multiple of 2"):
        pcm16_to_wav(b"\x00\x01\x02")


def test_ms_sample_conversions() -> None:
    assert ms_to_samples(500) == 8000
    assert samples_to_ms(8000) == 500
    assert samples_to_ms(ms_to_samples(1234)) == 1234
