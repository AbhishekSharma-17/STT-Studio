"""Silero VAD chunker: stream bytes in, get utterance segments out.

Design:
- Caller feeds raw PCM16 little-endian mono @ 16kHz via `feed(bytes)`.
- Silero operates on 512-sample windows (32ms @ 16kHz); we buffer accordingly.
- We emit a segment when:
    (a) we see `min_silence_ms` of silence after at least `min_speech_ms` of speech, OR
    (b) the current segment reaches `max_segment_ms` (hard cut).
- Emissions are yielded as plain PCM16 bytes; the route wraps them in a WAV
  container before POSTing to vLLM.

The Silero model is loaded once and shared. Inference is CPU-only and small
(~2ms per window), so we don't bother with a worker thread unless a profile
says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import torch
from silero_vad import load_silero_vad

from stt_backend.services.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE_HZ,
    ms_to_samples,
    samples_to_ms,
)

# Silero v5 expects exactly 512 samples per call at 16kHz.
_SILERO_WINDOW_SAMPLES = 512
_SILERO_WINDOW_BYTES = _SILERO_WINDOW_SAMPLES * BYTES_PER_SAMPLE


@dataclass(frozen=True)
class VadConfig:
    threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 250
    max_segment_ms: int = 15_000
    sample_rate: int = SAMPLE_RATE_HZ


@dataclass
class Segment:
    pcm: bytes
    duration_ms: int
    reason: str  # "silence" | "max_length" | "flush"


class _SileroHolder:
    """Lazy single-load of the Silero VAD weights.

    Typed as Any because the returned object has non-standard methods
    (reset_states, __call__(tensor, sample_rate)) that don't fit torch.nn.Module
    stubs cleanly.
    """

    _model: Any = None
    _lock = Lock()

    @classmethod
    def get(cls) -> Any:
        with cls._lock:
            if cls._model is None:
                model = load_silero_vad()
                model.eval()
                cls._model = model
            return cls._model


class VadChunker:
    """Streaming VAD segmenter. Not thread-safe — one instance per session."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self._cfg = config or VadConfig()
        self._model = _SileroHolder.get()

        self._carry_bytes = b""  # sub-window leftover across feed() calls
        self._current_pcm: bytearray = bytearray()  # bytes of the segment being built
        self._in_speech: bool = False
        self._speech_samples: int = 0  # accumulated speech samples in current seg
        self._trailing_silence_samples: int = 0

        # Silero stateful model — reset per new session
        self._model.reset_states()

    # --------------------------------------------------------------
    @property
    def config(self) -> VadConfig:
        return self._cfg

    def feed(self, pcm16_bytes: bytes) -> list[Segment]:
        """Feed an arbitrary chunk of PCM16. Returns zero or more completed segments."""
        if not pcm16_bytes:
            return []

        segments: list[Segment] = []
        buf = self._carry_bytes + pcm16_bytes
        offset = 0
        cfg = self._cfg
        silence_limit = ms_to_samples(cfg.min_silence_ms, cfg.sample_rate)
        min_speech = ms_to_samples(cfg.min_speech_ms, cfg.sample_rate)
        max_samples = ms_to_samples(cfg.max_segment_ms, cfg.sample_rate)

        while len(buf) - offset >= _SILERO_WINDOW_BYTES:
            window = buf[offset : offset + _SILERO_WINDOW_BYTES]
            offset += _SILERO_WINDOW_BYTES

            prob = self._probability(window)
            is_speech = prob >= cfg.threshold

            # Accumulate raw bytes regardless — decision to keep/trim happens on emit.
            self._current_pcm.extend(window)

            if is_speech:
                self._in_speech = True
                self._speech_samples += _SILERO_WINDOW_SAMPLES
                self._trailing_silence_samples = 0
            else:
                if self._in_speech:
                    self._trailing_silence_samples += _SILERO_WINDOW_SAMPLES

            # Emit if we had speech and enough trailing silence
            if (
                self._in_speech
                and self._speech_samples >= min_speech
                and self._trailing_silence_samples >= silence_limit
            ):
                segments.append(self._flush_segment(reason="silence"))
                continue

            # Hard-cut if segment is too long (regardless of speech state)
            current_samples = len(self._current_pcm) // BYTES_PER_SAMPLE
            if current_samples >= max_samples:
                if self._in_speech and self._speech_samples >= min_speech:
                    segments.append(self._flush_segment(reason="max_length"))
                else:
                    # Just drop silence-only overrun to avoid posting empty segments
                    self._reset_segment_state()

        # Preserve leftover bytes smaller than a window for next feed()
        self._carry_bytes = buf[offset:]
        return segments

    def flush(self) -> Segment | None:
        """Force-emit the pending segment if it has any speech."""
        if self._in_speech and self._speech_samples > 0:
            return self._flush_segment(reason="flush")
        self._reset_segment_state()
        return None

    # --------------------------------------------------------------
    def _probability(self, window_bytes: bytes) -> float:
        # Convert 512 little-endian int16 samples to float32 in [-1, 1]
        samples = np.frombuffer(window_bytes, dtype="<i2").astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)
        with torch.no_grad():
            prob: float = self._model(tensor, self._cfg.sample_rate).item()
        return prob

    def _flush_segment(self, *, reason: str) -> Segment:
        pcm = bytes(self._current_pcm)
        duration_ms = samples_to_ms(
            len(pcm) // BYTES_PER_SAMPLE, self._cfg.sample_rate
        )
        self._reset_segment_state()
        return Segment(pcm=pcm, duration_ms=duration_ms, reason=reason)

    def _reset_segment_state(self) -> None:
        self._current_pcm = bytearray()
        self._in_speech = False
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self._model.reset_states()
