"""Underwriting Agent: generate a structured investment memo for a deal.

The memo is produced as structured output (see ``structured.extract``) so the
result is a validated ``InvestmentMemo`` we can store and render directly.
"""

from __future__ import annotations

from fractionax_core import Asset, Deal, InvestmentMemo

from .oracle import (
    NavOracle,
    deal_implied_nav_minor,
    get_nav_oracle,
    illiquidity_adjusted_valuation,
)
from .structured import extract

MEMO_SYSTEM = (
    "You are the FractionAX underwriting agent. Given an alternative-asset deal (and "
    "its underlying asset, when one is available) plus an oracle NAV with an "
    "illiquidity-adjusted valuation, write a concise, decision-useful investment memo. "
    "Use EXACTLY the provided illiquidity-adjusted valuation for valuation_minor — do "
    "not recompute it. Identify the genuine risks — do not pad. Base the projected "
    "yield on the deal's stated figure unless the asset data contradicts it. Never "
    "invent figures that aren't supported by the inputs."
)


def generate_memo(
    deal: Deal, asset: Asset | None = None, *, oracle: NavOracle | None = None
) -> InvestmentMemo:
    """Produce a validated, NAV-grounded investment memo for ``deal``.

    When the deal has a typed underlying asset, the NAV oracle prices it. Otherwise
    (catalogue deals from discovery) the gross NAV is the offering par. Either way the
    memo's valuation is the deterministic illiquidity-adjusted NAV — never an LLM guess
    — and the model writes the narrative and risk assessment around it.
    """
    if asset is not None:
        quote = (oracle or get_nav_oracle()).quote(asset)
        gross_nav_minor, nav_source = quote.nav_minor, quote.source
        asset_block = f"\n\nASSET:\n{asset.model_dump_json(indent=2)}"
    else:
        gross_nav_minor, nav_source = deal_implied_nav_minor(deal), "deal-implied par"
        asset_block = ""
    valuation = illiquidity_adjusted_valuation(gross_nav_minor, deal.risk_tier)
    user = (
        "Write an investment memo for this deal.\n\n"
        f"DEAL:\n{deal.model_dump_json(indent=2)}"
        f"{asset_block}\n\n"
        f"ORACLE NAV ({nav_source}), minor units: {gross_nav_minor}\n"
        f"ILLIQUIDITY-ADJUSTED VALUATION to use (valuation_minor), minor units: {valuation}"
    )
    memo = extract(
        system=MEMO_SYSTEM,
        user=user,
        model_cls=InvestmentMemo,
        tool_name="record_investment_memo",
        tool_description="Record the structured investment memo for the deal.",
    )
    # Authoritative: the valuation is the NAV-derived figure, never an LLM estimate.
    return memo.model_copy(update={"valuation_minor": valuation})
