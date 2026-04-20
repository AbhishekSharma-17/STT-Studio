"""WebSocket /ws/transcribe — live streaming transcription.

Architecture per connection:

   receive_task ──> audio Queue ──> vad_task ──> segments Queue ──> send

- `receive_task`: reads client frames; pushes binary audio into `audio_q`;
  handles control JSON (session.start / session.commit).
- `vad_task`: consumes `audio_q`, feeds VadChunker, and for each emitted
  utterance POSTs to the selected vLLM and pushes a ServerSegment into
  `segments_q`.
- The main coroutine drains `segments_q` and writes frames back to the
  client, then sends `session.ended` on clean close.

All queues are bounded to cap memory under backpressure; if the audio queue
fills we log a warning and drop the oldest bytes (keep latest audio).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from stt_backend.config import Settings
from stt_backend.logging import get_logger
from stt_backend.schemas.transcription import TranscriptionRequest
from stt_backend.schemas.ws_messages import (
    ClientCommit,
    ClientStart,
    ServerAccepted,
    ServerEnded,
    ServerError,
    ServerSegment,
)
from stt_backend.services.audio import pcm16_to_wav
from stt_backend.services.vad import Segment, VadChunker, VadConfig
from stt_backend.services.vllm_client import VllmClient, VllmClientError

router = APIRouter(tags=["ws"])

_EOF = object()  # sentinel: no more audio


@router.websocket("/ws/transcribe")
async def transcribe_ws(ws: WebSocket) -> None:
    settings: Settings = ws.app.state.settings
    client: VllmClient = ws.app.state.vllm_client
    log = get_logger(__name__).bind(session_id=str(uuid.uuid4())[:8])

    await ws.accept()
    session_start_ts = time.monotonic()

    # 1) Wait for session.start
    try:
        start = await _recv_start(ws, log)
    except _ClientMisbehavedError as exc:
        await _send_error(ws, exc.code, exc.message)
        await ws.close(code=1008)
        return

    # VAD config comes from server settings, not client (trust boundary)
    vad = VadChunker(
        VadConfig(
            threshold=settings.vad_threshold,
            min_silence_ms=settings.vad_min_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_segment_ms=settings.vad_max_segment_ms,
        )
    )
    log = log.bind(model=start.model, language=start.language)

    await _send_json(ws, ServerAccepted(model=start.model, session_id=log._context["session_id"]))

    # Bounded queues for backpressure
    audio_q: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=settings.ws_max_audio_queue)
    segments_q: asyncio.Queue[ServerSegment | object] = asyncio.Queue()

    # Shared mutable counter for segment_id
    counter = {"n": 0}

    async def vad_loop() -> None:
        try:
            while True:
                item = await audio_q.get()
                if item is _EOF:
                    # Flush pending VAD buffer one last time
                    flushed = vad.flush()
                    if flushed is not None:
                        await _transcribe_and_enqueue(
                            flushed, start, counter, client, segments_q, log
                        )
                    break
                assert isinstance(item, bytes)
                for seg in vad.feed(item):
                    await _transcribe_and_enqueue(seg, start, counter, client, segments_q, log)
        finally:
            await segments_q.put(_EOF)

    async def receive_loop() -> None:
        try:
            while True:
                if time.monotonic() - session_start_ts > settings.ws_max_session_seconds:
                    log.warning("ws.session_timeout")
                    await _send_error(ws, "session_timeout", "max session length reached")
                    break
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if (b := msg.get("bytes")) is not None:
                    # Drop oldest on pressure: audio freshness > backlog fidelity
                    if audio_q.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            audio_q.get_nowait()
                        log.warning("ws.audio_queue_full_dropping_oldest")
                    await audio_q.put(b)
                elif (t := msg.get("text")) is not None:
                    try:
                        obj = json.loads(t)
                    except json.JSONDecodeError:
                        log.warning("ws.bad_json", text=t[:200])
                        continue
                    if obj.get("type") == "session.commit":
                        try:
                            ClientCommit.model_validate(obj)
                        except ValidationError:
                            continue
                        log.info("ws.commit_received")
                        break  # client asked us to finalise
                    # Ignore other control frames for now
        finally:
            await audio_q.put(_EOF)

    vad_task = asyncio.create_task(vad_loop(), name="vad")
    recv_task = asyncio.create_task(receive_loop(), name="receive")

    try:
        total = 0
        while True:
            seg = await segments_q.get()
            if seg is _EOF:
                break
            assert isinstance(seg, ServerSegment)
            total += 1
            await _send_json(ws, seg)
        if ws.client_state == WebSocketState.CONNECTED:
            await _send_json(ws, ServerEnded(total_segments=total))
    except WebSocketDisconnect:
        log.info("ws.client_disconnected")
    finally:
        # Ensure background tasks tidy up
        for t in (vad_task, recv_task):
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await ws.close()
        log.info(
            "ws.session_closed",
            duration_s=round(time.monotonic() - session_start_ts, 2),
        )


# ------------------------------------------------------------------------------


class _ClientMisbehavedError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def _recv_start(ws: WebSocket, log: structlog.stdlib.BoundLogger) -> ClientStart:
    msg = await ws.receive()
    if msg["type"] != "websocket.receive":
        raise _ClientMisbehavedError("no_start", "expected session.start JSON frame")
    text = msg.get("text")
    if text is None:
        raise _ClientMisbehavedError("no_start", "first frame must be text JSON (session.start)")
    try:
        obj = json.loads(text)
        start = ClientStart.model_validate(obj)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise _ClientMisbehavedError("bad_start", f"invalid session.start: {exc}") from exc
    if start.sample_rate != 16_000:
        raise _ClientMisbehavedError(
            "unsupported_sample_rate",
            f"only 16000 Hz supported, got {start.sample_rate}",
        )
    log.info("ws.session_started", model=start.model, language=start.language)
    return start


async def _transcribe_and_enqueue(
    segment: Segment,
    start: ClientStart,
    counter: dict[str, int],
    client: VllmClient,
    out: asyncio.Queue[ServerSegment | object],
    log: structlog.stdlib.BoundLogger,
) -> None:
    counter["n"] += 1
    seg_id = counter["n"]
    wav = pcm16_to_wav(segment.pcm)
    req = TranscriptionRequest(
        model=start.model,
        language=start.language,
        prompt=start.prompt,
    )
    try:
        res = await client.transcribe(
            audio=wav, req=req, filename=f"seg-{seg_id}.wav", content_type="audio/wav"
        )
    except VllmClientError as exc:
        log.warning("ws.vllm_fail", segment_id=seg_id, code=exc.code)
        # One bad segment shouldn't tear down the session — emit error frame instead
        await out.put(
            ServerSegment(
                segment_id=seg_id,
                text="",
                language=start.language,
                model=start.model,
                audio_duration_ms=segment.duration_ms,
                upstream_duration_ms=0,
                reason=f"error:{exc.code}",
            )
        )
        return

    await out.put(
        ServerSegment(
            segment_id=seg_id,
            text=res.text,
            language=res.language,
            model=res.model,
            audio_duration_ms=segment.duration_ms,
            upstream_duration_ms=res.duration_ms,
            reason=segment.reason,
        )
    )


async def _send_json(ws: WebSocket, payload: Any) -> None:
    if hasattr(payload, "model_dump_json"):
        await ws.send_text(payload.model_dump_json())
    else:  # pragma: no cover — defensive
        await ws.send_text(json.dumps(payload))


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    with contextlib.suppress(Exception):
        await _send_json(ws, ServerError(code=code, message=message))
