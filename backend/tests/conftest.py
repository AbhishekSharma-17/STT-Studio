"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset relevant env vars per test so Settings pickups are deterministic."""
    for key in list(os.environ):
        if key.startswith(("QWEN_", "WHISPER_", "BACKEND_", "LOG_", "VAD_", "WS_", "VLLM_")):
            monkeypatch.delenv(key, raising=False)
    # Clear the lru_cache so each test gets fresh Settings
    from stt_backend.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
