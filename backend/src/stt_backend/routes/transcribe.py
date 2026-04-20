"""REST /transcribe — single-file upload, returns a transcription result.

Used for batch testing and as a simple verification path before the
WebSocket route is wired up.
"""

from __future__ import annotations

from typing import Annotated, get_args

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from stt_backend.logging import get_logger
from stt_backend.schemas.transcription import (
    ErrorDetail,
    ModelName,
    TranscriptionRequest,
    TranscriptionResult,
)
from stt_backend.services.vllm_client import VllmClient, VllmClientError

log = get_logger(__name__)
router = APIRouter(tags=["transcribe"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — enough for ~30 min mono 16k PCM
_VALID_MODELS: frozenset[str] = frozenset(get_args(ModelName))


def _get_client(request: Request) -> VllmClient:
    client: VllmClient | None = getattr(request.app.state, "vllm_client", None)
    if client is None:  # pragma: no cover — set at startup
        raise HTTPException(status_code=500, detail="vLLM client not initialized")
    return client


@router.post(
    "/transcribe",
    response_model=TranscriptionResult,
    responses={
        400: {"model": ErrorDetail},
        413: {"model": ErrorDetail},
        502: {"model": ErrorDetail},
    },
    summary="Transcribe an uploaded audio file",
)
async def transcribe(
    file: Annotated[UploadFile, File(description="Audio file (wav/mp3/flac/ogg/webm/m4a)")],
    model: Annotated[str, Form(description="qwen3-asr | whisper")],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    client: VllmClient = Depends(_get_client),
) -> TranscriptionResult | JSONResponse:
    if model not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_model", "message": f"model must be one of {sorted(_VALID_MODELS)}"},
        )

    audio = await file.read()
    size = len(audio)
    if size == 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_audio", "message": "uploaded file was empty"},
        )
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "audio_too_large",
                "message": f"audio is {size} bytes, max {MAX_UPLOAD_BYTES}",
            },
        )

    req = TranscriptionRequest(
        model=model,  # type: ignore[arg-type]  # validated against _VALID_MODELS above
        language=language,
        prompt=prompt,
        temperature=temperature,
    )

    log.info(
        "transcribe.start",
        model=req.model,
        language=req.language,
        bytes=size,
        filename=file.filename,
        content_type=file.content_type,
    )

    try:
        result = await client.transcribe(
            audio=audio,
            req=req,
            filename=file.filename or "audio",
            content_type=file.content_type or "application/octet-stream",
        )
    except VllmClientError as exc:
        log.warning("transcribe.fail", model=req.model, code=exc.code)
        return JSONResponse(
            status_code=502,
            content={"code": exc.code, "message": exc.message},
        )

    return result
