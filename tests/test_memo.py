"""Underwriting Agent: the memo's valuation must be the deterministic NAV-derived
figure, never whatever the LLM happens to return."""

from __future__ import annotations

import pytest
from fractionax_core.domain import InvestmentMemo, MemoRisk

from fractionax_agents.deals import ASSETS_BY_ID, SEED_DEALS
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

    deal = SEED_DEALS[0]
    asset = ASSETS_BY_ID[deal.asset_id]
    oracle = FundamentalNavOracle()
    expected = illiquidity_adjusted_valuation(oracle.quote(asset).nav_minor, deal.risk_tier)

    memo = generate_memo(deal, asset, oracle=oracle)

    # The authoritative valuation is the illiquidity-adjusted NAV, not the LLM's figure.
    assert memo.valuation_minor == expected
    assert memo.valuation_minor != 1
    # The narrative and risk assessment still come from the model.
    assert memo.recommendation == "invest"
    assert memo.risks and memo.risks[0].title == "Concentration"
