"""Underwriting Agent: generate a structured investment memo for a deal.

The memo is produced as structured output (see ``structured.extract``) so the
result is a validated ``InvestmentMemo`` we can store and render directly.
"""

from __future__ import annotations

from fractionax_core import Asset, Deal, InvestmentMemo

from .oracle import NavOracle, get_nav_oracle, illiquidity_adjusted_valuation
from .structured import extract

MEMO_SYSTEM = (
    "You are the FractionAX underwriting agent. Given an alternative-asset deal, its "
    "underlying asset, and an oracle NAV with an illiquidity-adjusted valuation, write "
    "a concise, decision-useful investment memo. Use EXACTLY the provided "
    "illiquidity-adjusted valuation for valuation_minor — do not recompute it. "
    "Identify the genuine risks — do not pad. Base the projected yield on the deal's "
    "stated figure unless the asset data contradicts it. Never invent figures that "
    "aren't supported by the inputs."
)


def generate_memo(deal: Deal, asset: Asset, *, oracle: NavOracle | None = None) -> InvestmentMemo:
    """Produce a validated, NAV-grounded investment memo for ``deal``.

    The NAV oracle prices the underlying asset; the memo's valuation is the
    illiquidity-adjusted NAV (deterministic), not an LLM guess. The model writes the
    narrative and risk assessment around that figure.
    """
    nav = (oracle or get_nav_oracle()).quote(asset)
    valuation = illiquidity_adjusted_valuation(nav.nav_minor, deal.risk_tier)
    user = (
        "Write an investment memo for this deal.\n\n"
        f"DEAL:\n{deal.model_dump_json(indent=2)}\n\n"
        f"ASSET:\n{asset.model_dump_json(indent=2)}\n\n"
        f"ORACLE NAV ({nav.source}), minor units: {nav.nav_minor}\n"
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
