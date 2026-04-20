"""Typed WebSocket message envelopes.

Wire format
-----------
- Control messages are JSON text frames.
- Audio is sent as binary frames: raw PCM16 little-endian mono @ 16 kHz.

Client → server
---------------
- ClientStart
- ClientCommit  (optional — closing the socket has the same effect)

Server → client
---------------
- ServerAccepted
- ServerSegment   (one per VAD-emitted utterance)
- ServerEnded     (session closed cleanly)
- ServerError
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stt_backend.schemas.transcription import ModelName

# ---- Client → Server ----

class ClientStart(BaseModel):
    type: Literal["session.start"]
    model: ModelName
    language: str | None = None
    prompt: str | None = None
    sample_rate: int = 16_000


class ClientCommit(BaseModel):
    type: Literal["session.commit"]


# ---- Server → Client ----

class ServerAccepted(BaseModel):
    type: Literal["session.accepted"] = "session.accepted"
    model: ModelName
    session_id: str


class ServerSegment(BaseModel):
    type: Literal["transcription.segment"] = "transcription.segment"
    segment_id: int
    text: str
    language: str | None = None
    model: ModelName
    audio_duration_ms: int
    upstream_duration_ms: int
    reason: str = Field(description="VAD cut reason: silence | max_length | flush")


class ServerEnded(BaseModel):
    type: Literal["session.ended"] = "session.ended"
    total_segments: int


class ServerError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
