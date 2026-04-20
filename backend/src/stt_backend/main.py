"""FastAPI app factory + lifespan. Keep this file thin — routes live elsewhere."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from stt_backend import __version__
from stt_backend.config import Settings, get_settings
from stt_backend.logging import configure_logging, get_logger
from stt_backend.routes import health, transcribe, ws
from stt_backend.services.vllm_client import VllmClient

FRONTEND_DIR = Path("/app/frontend")
FRONTEND_DIR_DEV = Path(__file__).resolve().parents[3] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks. Keep IO-free and fast — no model loading here
    (vLLM owns that)."""
    settings: Settings = app.state.settings
    log = get_logger(__name__)

    app.state.vllm_client = VllmClient(settings)

    log.info(
        "backend.start",
        version=__version__,
        qwen_url=str(settings.qwen_url),
        whisper_url=str(settings.whisper_url),
        log_level=settings.log_level,
    )
    try:
        yield
    finally:
        log.info("backend.stop")
        await app.state.vllm_client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="STT Inference Backend",
        version=__version__,
        description="Routes browser audio to Qwen3-ASR or Whisper via vLLM.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Prometheus metrics at /metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # Routes
    app.include_router(health.router)
    app.include_router(transcribe.router)
    app.include_router(ws.router)

    # Static frontend — mounted last so it doesn't shadow API routes.
    static_dir = FRONTEND_DIR if FRONTEND_DIR.is_dir() else FRONTEND_DIR_DEV
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app


app = create_app()
