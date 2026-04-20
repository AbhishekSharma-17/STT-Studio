"""Async client for vLLM's OpenAI-compatible /v1/audio/transcriptions endpoint.

Handles both Qwen3-ASR and Whisper; selects the upstream URL + model id by name.
Retries transient failures; raises VllmClientError on persistent ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stt_backend.config import Settings
from stt_backend.logging import get_logger
from stt_backend.schemas.transcription import ModelName, TranscriptionRequest, TranscriptionResult

log = get_logger(__name__)


class VllmClientError(RuntimeError):
    """Raised when the upstream vLLM cannot produce a transcription."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _Upstream:
    base_url: str
    model_id: str


class VllmClient:
    """Async, pooled client. Create one per app; reuse across requests."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        # Reuse a single HTTP client — connection pool, keep-alive, HTTP/2 later.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.vllm_timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        self._routes: dict[ModelName, _Upstream] = {
            "qwen3-asr": _Upstream(str(settings.qwen_url).rstrip("/"), settings.qwen_model),
            "whisper": _Upstream(str(settings.whisper_url).rstrip("/"), settings.whisper_model),
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self,
        audio: bytes,
        req: TranscriptionRequest,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> TranscriptionResult:
        """POST audio to the selected vLLM. Returns the parsed transcription."""
        upstream = self._routes[req.model]
        url = f"{upstream.base_url}/v1/audio/transcriptions"

        # OpenAI transcription API is multipart/form-data.
        files = {"file": (filename, audio, content_type)}
        data: dict[str, str] = {
            "model": upstream.model_id,
            "temperature": str(req.temperature),
            "response_format": "json",
        }
        if req.language:
            data["language"] = req.language
        if req.prompt:
            data["prompt"] = req.prompt

        attempts = self._settings.vllm_retries + 1
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=0.25, min=0.25, max=2.0),
                retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
            ):
                with attempt:
                    t0 = time.perf_counter()
                    resp = await self._client.post(url, data=data, files=files)
                    dur_ms = int((time.perf_counter() - t0) * 1000)
        except RetryError as exc:  # pragma: no cover — re-raise is idiomatic w/ reraise=True
            raise VllmClientError("vllm_unreachable", str(exc)) from exc
        except httpx.TransportError as exc:
            raise VllmClientError("vllm_unreachable", str(exc)) from exc

        if resp.status_code >= 400:
            body_preview = resp.text[:200]
            log.warning(
                "vllm.error",
                model=req.model,
                status=resp.status_code,
                body=body_preview,
            )
            raise VllmClientError(
                f"vllm_{resp.status_code}",
                f"upstream returned {resp.status_code}: {body_preview}",
            )

        try:
            body: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise VllmClientError("vllm_bad_json", str(exc)) from exc

        text = body.get("text", "").strip()
        lang = body.get("language") or req.language

        log.info(
            "vllm.ok",
            model=req.model,
            language=lang,
            duration_ms=dur_ms,
            text_chars=len(text),
            text=text if self._settings.log_transcript_content else None,
        )

        return TranscriptionResult(
            text=text,
            language=lang,
            model=req.model,
            duration_ms=dur_ms,
        )
