from __future__ import annotations

import io
from collections.abc import Iterator

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from stt_backend.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Use `with` so lifespan runs and app.state.vllm_client is created.
    with TestClient(create_app()) as c:
        yield c


@respx.mock
def test_transcribe_rest_qwen_ok(client: TestClient) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "مرحبا", "language": "ar"})
    )

    files = {"file": ("sample.wav", io.BytesIO(b"RIFF....WAVE...."), "audio/wav")}
    data = {"model": "qwen3-asr", "language": "ar"}
    r = client.post("/transcribe", files=files, data=data)

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "مرحبا"
    assert body["language"] == "ar"
    assert body["model"] == "qwen3-asr"


def test_transcribe_rejects_unknown_model(client: TestClient) -> None:
    files = {"file": ("sample.wav", io.BytesIO(b"..."), "audio/wav")}
    data = {"model": "not-a-model"}
    r = client.post("/transcribe", files=files, data=data)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_model"


def test_transcribe_rejects_empty_file(client: TestClient) -> None:
    files = {"file": ("sample.wav", io.BytesIO(b""), "audio/wav")}
    data = {"model": "qwen3-asr"}
    r = client.post("/transcribe", files=files, data=data)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "empty_audio"


@respx.mock
def test_transcribe_502_on_upstream_failure(client: TestClient) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(500, text="boom")
    )

    files = {"file": ("sample.wav", io.BytesIO(b"..."), "audio/wav")}
    data = {"model": "qwen3-asr"}
    r = client.post("/transcribe", files=files, data=data)
    assert r.status_code == 502
    body = r.json()
    assert body["code"] == "vllm_500"
