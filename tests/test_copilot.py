import pytest
from fastapi.testclient import TestClient
from fractionax_core import InvestmentIntent

from fractionax_agents.config import Settings
from fractionax_agents.copilot import _enrich_intent, intent_to_filter
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


def test_copilot_stream_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fractionax_agents.server.get_settings", _no_provider_settings)
    resp = client.post("/copilot/stream", json={"message": "x"})
    assert resp.status_code == 503


def test_enrich_fills_fields_the_model_missed() -> None:
    # The model returned only action + amount; the backstop fills the rest.
    bare = InvestmentIntent(action="invest", amount_minor=100_000)
    enriched = _enrich_intent(bare, "Invest $1,000 in low-risk Malaysian opportunities")
    assert enriched.currency == "USD"
    assert enriched.jurisdiction == "MY"
    assert enriched.risk_tier == "low"


def test_enrich_parses_amount_and_asset_kind() -> None:
    enriched = _enrich_intent(
        InvestmentIntent(action="discover"), "Show me high-yield revenue-share deals under $2.5k"
    )
    assert enriched.amount_minor == 250_000
    assert enriched.risk_tier == "high"
    assert enriched.asset_kind == "revenue_share"


def test_enrich_never_overrides_the_model() -> None:
    intent = InvestmentIntent(action="discover", risk_tier="high", jurisdiction="SG")
    enriched = _enrich_intent(intent, "low-risk Malaysian")
    assert enriched.risk_tier == "high"
    assert enriched.jurisdiction == "SG"
