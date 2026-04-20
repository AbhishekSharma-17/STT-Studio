"""WebSocket integration test using TestClient + respx.

We stub Silero VAD to force one segment so we exercise the full happy path
without needing real speech audio.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from stt_backend.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Make VAD deterministic: every window is "speech" so the max-segment cut fires.
    from stt_backend.services import vad as vad_mod

    monkeypatch.setattr(vad_mod.VadChunker, "_probability", lambda self, w: 1.0)

    with TestClient(create_app()) as c:
        yield c


def _silence_pcm16(ms: int, sr: int = 16000) -> bytes:
    return b"\x00\x00" * ((sr * ms) // 1000)


def _tone_pcm16(ms: int, sr: int = 16000) -> bytes:
    import math

    n = (sr * ms) // 1000
    samples = [int(10_000 * math.sin(2 * math.pi * 440 * t / sr)) for t in range(n)]
    return struct.pack(f"<{n}h", *samples)


@respx.mock
def test_ws_happy_path_emits_segment(client: TestClient) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello", "language": "en"})
    )

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "session.start",
                    "model": "qwen3-asr",
                    "language": "en",
                    "sample_rate": 16000,
                }
            )
        )
        accepted = json.loads(ws.receive_text())
        assert accepted["type"] == "session.accepted"

        # Send ~1.2s of audio → max-segment cut fires at max_segment_ms default (15s)
        # So instead we rely on commit to force flush.
        audio = _tone_pcm16(600) + _silence_pcm16(600)
        # Split into 200ms chunks
        chunk = 16000 // 5 * 2
        for i in range(0, len(audio), chunk):
            ws.send_bytes(audio[i : i + chunk])

        ws.send_text(json.dumps({"type": "session.commit"}))

        msgs: list[dict[str, object]] = []
        while True:
            m = json.loads(ws.receive_text())
            msgs.append(m)
            if m["type"] == "session.ended":
                break

    segments = [m for m in msgs if m["type"] == "transcription.segment"]
    assert segments, "expected at least one segment"
    seg = segments[0]
    assert seg["text"] == "hello"
    assert seg["model"] == "qwen3-asr"
    assert seg["language"] == "en"


def test_ws_rejects_missing_start(client: TestClient) -> None:
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({"type": "not.start"}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["code"] in {"no_start", "bad_start"}


def test_ws_rejects_wrong_sample_rate(client: TestClient) -> None:
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "session.start",
                    "model": "qwen3-asr",
                    "sample_rate": 44100,
                }
            )
        )
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["code"] == "unsupported_sample_rate"
