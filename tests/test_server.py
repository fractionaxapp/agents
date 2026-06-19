import pytest
from fastapi.testclient import TestClient

from fractionax_agents.config import get_settings
from fractionax_agents.server import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no API key configured, /chat returns 503 rather than calling Claude.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 503
    finally:
        get_settings.cache_clear()
