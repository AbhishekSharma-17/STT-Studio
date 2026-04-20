from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from stt_backend.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Rebuild app so Settings pick up the test env isolated by conftest.
    with TestClient(create_app()) as c:
        yield c


def test_healthz_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


@respx.mock
def test_readyz_ok_when_both_upstreams_up(client: TestClient) -> None:
    respx.get("http://qwen:8000/v1/models").mock(return_value=httpx.Response(200))
    respx.get("http://whisper:8001/v1/models").mock(return_value=httpx.Response(200))

    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["upstreams"]["qwen"]["ok"] is True
    assert body["upstreams"]["whisper"]["ok"] is True


@respx.mock
def test_readyz_503_when_qwen_down(client: TestClient) -> None:
    respx.get("http://qwen:8000/v1/models").mock(side_effect=httpx.ConnectError("nope"))
    respx.get("http://whisper:8001/v1/models").mock(return_value=httpx.Response(200))

    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["upstreams"]["qwen"]["ok"] is False
    assert body["upstreams"]["whisper"]["ok"] is True
