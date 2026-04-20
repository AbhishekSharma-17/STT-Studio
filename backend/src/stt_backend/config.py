"""Env-driven application settings. Instantiated once via `get_settings()`."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Service identity ----
    backend_host: str = "0.0.0.0"
    backend_port: int = 3000

    # ---- Upstream vLLM URLs (internal compose DNS) ----
    qwen_url: HttpUrl = Field(default=HttpUrl("http://qwen:8000"))
    whisper_url: HttpUrl = Field(default=HttpUrl("http://whisper:8001"))

    # Model identifiers as registered in vLLM (==QWEN_MODEL / WHISPER_MODEL)
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"
    whisper_model: str = "openai/whisper-large-v3"

    # ---- HTTP client tuning ----
    vllm_timeout_seconds: float = 60.0
    vllm_retries: int = 2

    # ---- Logging ----
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    log_transcript_content: bool = False

    # ---- VAD ----
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_silence_ms: int = Field(default=500, ge=50)
    vad_min_speech_ms: int = Field(default=250, ge=50)
    vad_max_segment_ms: int = Field(default=15_000, ge=1_000)

    # ---- WebSocket session limits ----
    ws_max_session_seconds: int = Field(default=1800, ge=60)
    ws_max_audio_queue: int = Field(default=200, ge=10)

    # ---- Security ----
    # NoDecode keeps pydantic-settings from JSON-parsing the env value; our
    # validator below accepts a plain comma-separated string.
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        lvl = v.upper()
        if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log_level: {v}")
        return lvl


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings — cached so imports don't re-read env."""
    return Settings()
