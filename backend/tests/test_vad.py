from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from stt_backend.services.vad import VadChunker, VadConfig


def _sine_pcm16(freq_hz: int, duration_ms: int, amp: float = 0.5, sr: int = 16000) -> bytes:
    n = (sr * duration_ms) // 1000
    samples = [int(32767 * amp * math.sin(2 * math.pi * freq_hz * t / sr)) for t in range(n)]
    return struct.pack(f"<{n}h", *samples)


def _silence_pcm16(duration_ms: int, sr: int = 16000) -> bytes:
    n = (sr * duration_ms) // 1000
    return b"\x00\x00" * n


def test_silence_only_emits_nothing() -> None:
    chunker = VadChunker(VadConfig(min_silence_ms=200, min_speech_ms=100, max_segment_ms=2000))
    out = chunker.feed(_silence_pcm16(1000))
    assert out == []
    flushed = chunker.flush()
    assert flushed is None


def test_max_segment_length_forces_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silero only triggers on speech-like audio; bypass the classifier so we
    # can exercise the max-length cut logic deterministically.
    from stt_backend.services import vad as vad_mod

    monkeypatch.setattr(vad_mod.VadChunker, "_probability", lambda self, w: 1.0)

    chunker = VadChunker(
        VadConfig(min_silence_ms=500, min_speech_ms=100, max_segment_ms=1000)
    )
    audio = _sine_pcm16(440, 1500)  # 1.5s > max_segment_ms
    emitted = []
    for i in range(0, len(audio), 16000 // 5 * 2):  # 200ms steps
        emitted.extend(chunker.feed(audio[i : i + 16000 // 5 * 2]))

    assert emitted, "expected a max_length emission"
    assert any(s.reason == "max_length" for s in emitted)
    # Emitted segment should be close to max_segment_ms (within one Silero window)
    assert all(900 <= s.duration_ms <= 1100 for s in emitted if s.reason == "max_length")


def test_silence_after_speech_emits_on_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    # First 400ms of "speech" then 800ms of silence → one segment, reason=silence.
    from stt_backend.services import vad as vad_mod

    calls = {"n": 0}

    def fake_prob(self: VadChunker, window: bytes) -> float:
        # Return 1.0 for the first N windows (speech), 0.0 afterwards (silence).
        calls["n"] += 1
        # 400ms of speech = 400ms / 32ms per window ≈ 12 windows
        return 1.0 if calls["n"] <= 12 else 0.0

    monkeypatch.setattr(vad_mod.VadChunker, "_probability", fake_prob)

    chunker = VadChunker(
        VadConfig(min_silence_ms=200, min_speech_ms=100, max_segment_ms=5000)
    )
    # 1.2s of any audio — VAD logic is driven by fake_prob
    audio = _silence_pcm16(1200)
    emitted = chunker.feed(audio)

    assert any(s.reason == "silence" for s in emitted), emitted


def test_config_defaults() -> None:
    c = VadConfig()
    assert c.threshold == 0.5
    assert c.min_silence_ms == 500
    assert c.sample_rate == 16000


def test_feed_empty_returns_empty() -> None:
    c = VadChunker()
    assert c.feed(b"") == []


def test_odd_length_bytes_do_not_crash() -> None:
    # Silero expects whole samples; odd-length tails should just be buffered.
    c = VadChunker()
    assert c.feed(_silence_pcm16(10) + b"\x00") == []  # silence + odd byte


# Guard against accidental regression of the Silero window size contract
def test_silero_window_matches_expected() -> None:
    from stt_backend.services.vad import _SILERO_WINDOW_SAMPLES  # type: ignore[attr-defined]

    assert _SILERO_WINDOW_SAMPLES == 512


def test_numpy_import_sanity() -> None:
    # If numpy<>torch mismatch lands, the VAD module import itself fails first.
    assert np.__version__
