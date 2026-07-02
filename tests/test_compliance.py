"""Compliance Agent: the ALLOW/DENY gate is deterministic. These tests pin the
jurisdiction/accreditation/KYC/sanctions rules and the transfer predicate. All run
with ``with_rationale=False`` so no LLM is involved."""

from __future__ import annotations

from fractionax_core.domain import Deal, Investor, RiskTier

from fractionax_agents.compliance import (
    MockKycProvider,
    check_transfer,
    deal_regime,
    jurisdiction_rules,
    verify_investor,
)


def _deal(jurisdiction: str, risk_tier: RiskTier = "medium") -> Deal:
    return Deal(
        id=f"deal_{jurisdiction.lower()}_{risk_tier}",
        asset_id="ast_x",
        title=f"Test deal {jurisdiction}",
        jurisdiction=jurisdiction,
        currency="USD",
        min_investment_minor=100_000,
        target_raise_minor=5_000_000,
        projected_yield_pct=8.0,
        risk_tier=risk_tier,
        status="open",
        sourced_at="2026-06-25T00:00:00Z",
    )


def _investor(
    jurisdiction: str, accredited: bool = False, id_: str = "inv_1", name: str = "Alex"
) -> Investor:
    return Investor(
        id=id_,
        display_name=name,
        jurisdiction=jurisdiction,
        accredited=accredited,
        risk_appetite="medium",
    )


def _verify(investor: Investor, deal: Deal):
    return verify_investor(investor, deal, with_rationale=False)


def test_regime_inference() -> None:
    assert deal_regime(_deal("US")) == "reg_d"
    assert deal_regime(_deal("MY")) == "reg_s"
    assert deal_regime(_deal("SG")) == "reg_s"


def test_us_accredited_allowed_under_reg_d() -> None:
    d = _verify(_investor("US", accredited=True), _deal("US"))
    assert d.outcome == "allow"
    assert d.regime == "reg_d"
    assert d.reasons == []


def test_us_retail_denied_reg_d_accreditation() -> None:
    d = _verify(_investor("US", accredited=False), _deal("US"))
    assert d.outcome == "deny"
    assert any(r.code == "accreditation_below_minimum" for r in d.reasons)


def test_non_us_retail_allowed_under_reg_s() -> None:
    d = _verify(_investor("MY", accredited=False), _deal("MY"))
    assert d.outcome == "allow"
    assert d.regime == "reg_s"


def test_us_person_blocked_from_reg_s() -> None:
    # Reg S is offshore — a US person is blocked regardless of accreditation.
    d = _verify(_investor("US", accredited=True), _deal("SG"))
    assert d.outcome == "deny"
    assert any(r.code == "jurisdiction_blocked" for r in d.reasons)


def test_sanctioned_jurisdiction_denied() -> None:
    d = _verify(_investor("IR", accredited=True), _deal("MY"))
    assert d.outcome == "deny"
    codes = {r.code for r in d.reasons}
    assert "sanctions_hit" in codes
    assert d.kyc_status == "rejected"


def test_watchlisted_investor_denied() -> None:
    provider = MockKycProvider(watchlist=frozenset({"inv_bad"}))
    inv = _investor("MY", accredited=True, id_="inv_bad")
    d = verify_investor(inv, _deal("MY"), provider=provider, with_rationale=False)
    assert d.outcome == "deny"
    assert any(r.code == "kyc_not_verified" for r in d.reasons)


def test_high_risk_requires_accreditation_even_offshore() -> None:
    d = _verify(_investor("MY", accredited=False), _deal("MY", risk_tier="high"))
    assert d.outcome == "deny"
    assert any(r.code == "risk_tier_requires_accreditation" for r in d.reasons)
    # An accredited investor clears the same high-risk offshore deal.
    ok = _verify(_investor("MY", accredited=True), _deal("MY", risk_tier="high"))
    assert ok.outcome == "allow"


def test_institutional_override_tier() -> None:
    provider = MockKycProvider(institutional_ids=frozenset({"inv_inst"}))
    inv = _investor("US", accredited=False, id_="inv_inst")
    d = verify_investor(inv, _deal("US"), provider=provider, with_rationale=False)
    assert d.accreditation_tier == "institutional"
    assert d.outcome == "allow"


def test_transfer_allowed_to_eligible_buyer() -> None:
    deal = _deal("MY")
    seller = _investor("MY", accredited=True, id_="seller")
    buyer = _investor("SG", accredited=False, id_="buyer")
    check = check_transfer(deal, seller, buyer)
    assert check.allowed
    assert check.reasons == []


def test_transfer_denied_to_us_buyer_under_reg_s() -> None:
    deal = _deal("MY")
    seller = _investor("MY", accredited=True, id_="seller")
    buyer = _investor("US", accredited=True, id_="buyer")
    check = check_transfer(deal, seller, buyer)
    assert not check.allowed
    assert any(r.code == "jurisdiction_blocked" for r in check.reasons)


def test_jurisdiction_rules_exposes_all_regimes() -> None:
    regimes = {r.regime for r in jurisdiction_rules()}
    assert regimes == {"reg_d", "reg_s", "reg_a"}
