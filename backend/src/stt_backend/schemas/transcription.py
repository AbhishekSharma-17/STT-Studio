"""Request/response schemas shared by REST + WebSocket paths."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["qwen3-asr", "whisper"]
"""The two model backends we serve. Maps to settings.qwen_model / whisper_model."""


class TranscriptionRequest(BaseModel):
    """Inputs to a transcription call. Audio bytes are passed separately."""

    model: ModelName
    language: str | None = Field(
        default=None,
        description="ISO 639-1 code, e.g. 'ar' or 'en'. None lets the model auto-detect.",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional text prompt to bias the model (both Qwen and Whisper support this).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class TranscriptionResult(BaseModel):
    """Canonical transcription result returned by the backend."""

    text: str
    language: str | None = None
    model: ModelName
    duration_ms: int
    """Wall-clock upstream latency for the transcription call."""


class ErrorDetail(BaseModel):
    """Consistent error body for REST endpoints."""

    code: str
    message: str
