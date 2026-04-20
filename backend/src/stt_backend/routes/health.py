"""Health + readiness endpoints.

- /healthz: cheap liveness — 200 if the process is up.
- /readyz: readiness — 200 only if both vLLM upstreams respond to /v1/models.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Response, status

from stt_backend import __version__
from stt_backend.config import Settings, get_settings
from stt_backend.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


async def _probe(client: httpx.AsyncClient, url: str) -> tuple[bool, str]:
    try:
        r = await client.get(f"{url.rstrip('/')}/v1/models", timeout=5.0)
        if r.status_code == 200:
            return True, "ok"
        return False, f"status={r.status_code}"
    except httpx.RequestError as exc:
        return False, f"unreachable: {exc.__class__.__name__}"


@router.get("/readyz", summary="Readiness probe — verifies both vLLM upstreams")
async def readyz(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        qwen_ok, qwen_msg = await _probe(client, str(settings.qwen_url))
        whisper_ok, whisper_msg = await _probe(client, str(settings.whisper_url))

    ready = qwen_ok and whisper_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": ready,
        "upstreams": {
            "qwen": {"ok": qwen_ok, "detail": qwen_msg, "url": str(settings.qwen_url)},
            "whisper": {"ok": whisper_ok, "detail": whisper_msg, "url": str(settings.whisper_url)},
        },
    }
