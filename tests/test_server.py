import pytest
from fastapi.testclient import TestClient

from fractionax_agents.config import Settings
from fractionax_agents.server import app

client = TestClient(app)


def _no_provider_settings() -> Settings:
    # Ignore any local .env so the test is hermetic regardless of the dev's keys.
    return Settings(_env_file=None, anthropic_api_key=None, minimax_api_key=None)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # With NO provider configured (neither Anthropic nor MiniMax), /chat returns 503
    # rather than calling an LLM.
    monkeypatch.setattr("fractionax_agents.server.get_settings", _no_provider_settings)
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 503
