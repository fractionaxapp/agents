import pytest
from fastapi.testclient import TestClient
from fractionax_core import InvestmentIntent

from fractionax_agents.config import Settings
from fractionax_agents.copilot import intent_to_filter
from fractionax_agents.server import app

client = TestClient(app)


def _no_provider_settings() -> Settings:
    # Ignore any local .env so the test is hermetic regardless of the dev's keys.
    return Settings(_env_file=None, anthropic_api_key=None, minimax_api_key=None)


def test_intent_to_filter_maps_fields() -> None:
    intent = InvestmentIntent(
        action="discover", risk_tier="low", jurisdiction="MY", amount_minor=100_000
    )
    f = intent_to_filter(intent)
    assert f.jurisdiction == "MY"
    assert f.risk_tier == "low"
    assert f.max_min_investment_minor == 100_000


def test_intent_to_filter_passes_through_unset() -> None:
    f = intent_to_filter(InvestmentIntent(action="discover"))
    assert f.jurisdiction is None
    assert f.risk_tier is None
    assert f.max_min_investment_minor is None


def test_copilot_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # With NO provider configured (neither Anthropic nor MiniMax), /copilot returns
    # 503 rather than calling an LLM.
    monkeypatch.setattr("fractionax_agents.server.get_settings", _no_provider_settings)
    resp = client.post("/copilot", json={"message": "invest $1000 in low-risk deals"})
    assert resp.status_code == 503
