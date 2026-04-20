from __future__ import annotations

import httpx
import pytest
import respx

from stt_backend.config import Settings
from stt_backend.schemas.transcription import TranscriptionRequest
from stt_backend.services.vllm_client import VllmClient, VllmClientError


@pytest.fixture
def settings() -> Settings:
    # Disable retries so tests that exercise failure don't take forever
    return Settings(vllm_retries=0, vllm_timeout_seconds=5.0)


@respx.mock
async def test_transcribe_qwen_success(settings: Settings) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "مرحبا", "language": "ar"})
    )

    client = VllmClient(settings)
    try:
        result = await client.transcribe(
            audio=b"RIFF....WAVE....",
            req=TranscriptionRequest(model="qwen3-asr", language="ar"),
        )
    finally:
        await client.aclose()

    assert result.text == "مرحبا"
    assert result.language == "ar"
    assert result.model == "qwen3-asr"
    assert result.duration_ms >= 0


@respx.mock
async def test_transcribe_whisper_success(settings: Settings) -> None:
    respx.post("http://whisper:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello world", "language": "en"})
    )

    client = VllmClient(settings)
    try:
        result = await client.transcribe(
            audio=b"RIFF....WAVE....",
            req=TranscriptionRequest(model="whisper"),
        )
    finally:
        await client.aclose()

    assert result.text == "hello world"
    assert result.model == "whisper"


@respx.mock
async def test_transcribe_forwards_language_and_prompt(settings: Settings) -> None:
    route = respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    client = VllmClient(settings)
    try:
        await client.transcribe(
            audio=b"...",
            req=TranscriptionRequest(model="qwen3-asr", language="ar", prompt="context hint"),
        )
    finally:
        await client.aclose()

    sent = route.calls.last.request
    body = sent.content.decode("utf-8", errors="ignore")
    assert "name=\"language\"" in body and "ar" in body
    assert "name=\"prompt\"" in body and "context hint" in body


@respx.mock
async def test_transcribe_raises_on_http_error(settings: Settings) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        return_value=httpx.Response(500, text="internal error")
    )

    client = VllmClient(settings)
    try:
        with pytest.raises(VllmClientError) as exc_info:
            await client.transcribe(
                audio=b"...",
                req=TranscriptionRequest(model="qwen3-asr"),
            )
    finally:
        await client.aclose()

    assert exc_info.value.code == "vllm_500"


@respx.mock
async def test_transcribe_raises_on_connection_error(settings: Settings) -> None:
    respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        side_effect=httpx.ConnectError("boom")
    )

    client = VllmClient(settings)
    try:
        with pytest.raises(VllmClientError) as exc_info:
            await client.transcribe(
                audio=b"...",
                req=TranscriptionRequest(model="qwen3-asr"),
            )
    finally:
        await client.aclose()

    assert exc_info.value.code == "vllm_unreachable"


@respx.mock
async def test_transcribe_retries_transient_errors() -> None:
    s = Settings(vllm_retries=2, vllm_timeout_seconds=5.0)
    route = respx.post("http://qwen:8000/v1/audio/transcriptions").mock(
        side_effect=[
            httpx.ConnectError("first fail"),
            httpx.ConnectError("second fail"),
            httpx.Response(200, json={"text": "ok"}),
        ]
    )
    client = VllmClient(s)
    try:
        result = await client.transcribe(
            audio=b"...",
            req=TranscriptionRequest(model="qwen3-asr"),
        )
    finally:
        await client.aclose()

    assert result.text == "ok"
    assert route.call_count == 3
