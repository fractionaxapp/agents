"""Underwriting Agent: the memo's valuation must be the deterministic NAV-derived
figure, never whatever the LLM happens to return."""

from __future__ import annotations

import pytest
from fractionax_core.domain import Deal, InvestmentMemo, MemoRisk

from fractionax_agents.deals import SEED_ASSETS
from fractionax_agents.memo import generate_memo
from fractionax_agents.oracle import FundamentalNavOracle, illiquidity_adjusted_valuation


def _fake_memo(**_: object) -> InvestmentMemo:
    """Stand in for the LLM call with a memo whose valuation is deliberately bogus."""
    return InvestmentMemo(
        deal_id="from-llm",
        summary="LLM-written narrative.",
        valuation_minor=1,  # wrong on purpose — generate_memo must override this
        projected_yield_pct=7.5,
        risk_tier="low",
        risks=[MemoRisk(title="Concentration", severity="medium", detail="Single obligor.")],
        recommendation="invest",
        generated_at="2026-06-25T00:00:00Z",
    )


def test_memo_valuation_is_nav_derived_not_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fractionax_agents.memo.extract", _fake_memo)

    # Discovery is catalogue-only now, so pair a demo asset with a constructed deal
    # to exercise the NAV-derived valuation path.
    asset = SEED_ASSETS[0]
    deal = Deal(
        id="deal_memo_test",
        asset_id=asset.id,
        asset_class="specialty-finance",
        title=f"Memo test — {asset.name}",
        jurisdiction=asset.jurisdiction,
        currency=asset.currency,
        min_investment_minor=100_000,
        target_raise_minor=4_000_000,
        projected_yield_pct=8.5,
        risk_tier="low",
        status="open",
        sourced_at="2026-06-25T00:00:00Z",
    )
    oracle = FundamentalNavOracle()
    expected = illiquidity_adjusted_valuation(oracle.quote(asset).nav_minor, deal.risk_tier)

    memo = generate_memo(deal, asset, oracle=oracle)

    # The authoritative valuation is the illiquidity-adjusted NAV, not the LLM's figure.
    assert memo.valuation_minor == expected
    assert memo.valuation_minor != 1
    # The narrative and risk assessment still come from the model.
    assert memo.recommendation == "invest"
    assert memo.risks and memo.risks[0].title == "Concentration"


def test_memo_underwrites_a_catalogue_deal_without_an_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery deals carry no typed asset; the memo uses the deal-implied par NAV."""
    monkeypatch.setattr("fractionax_agents.memo.extract", _fake_memo)

    deal = Deal(
        id="deal_rwa_1",
        asset_id="ast_rwa_1",
        title="Catalogue deal without a typed asset",
        jurisdiction="SV",
        currency="USD",
        min_investment_minor=10_000_000,
        target_raise_minor=5_000_000_000,
        projected_yield_pct=0.0,
        risk_tier="low",
        status="open",
        asset_class="stablecoins",
        sourced_at="2026-01-01T00:00:00.000Z",
    )
    expected = illiquidity_adjusted_valuation(deal.target_raise_minor, deal.risk_tier)

    memo = generate_memo(deal)  # no asset

    assert memo.valuation_minor == expected
    assert memo.valuation_minor != 1
    assert memo.recommendation == "invest"
