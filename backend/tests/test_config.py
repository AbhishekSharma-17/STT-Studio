from __future__ import annotations

import pytest

from stt_backend.config import Settings, get_settings


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.backend_port == 3000
    assert s.log_level == "INFO"
    assert s.vad_threshold == 0.5
    assert "http://localhost:3000" in s.allowed_origins


def test_allowed_origins_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    s = Settings()
    assert s.allowed_origins == ["https://a.example", "https://b.example"]


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError, match="invalid log_level"):
        Settings()


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_PORT", "4000")
    s1 = get_settings()
    monkeypatch.setenv("BACKEND_PORT", "5000")
    s2 = get_settings()
    assert s1 is s2, "get_settings() must memoize"


def test_vad_threshold_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAD_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        Settings()
