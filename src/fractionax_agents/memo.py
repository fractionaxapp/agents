"""Underwriting Agent: generate a structured investment memo for a deal.

The memo is produced as structured output (see ``structured.extract``) so the
result is a validated ``InvestmentMemo`` we can store and render directly.
"""

from __future__ import annotations

from fractionax_core import Asset, Deal, InvestmentMemo

from .structured import extract

MEMO_SYSTEM = (
    "You are the FractionAX underwriting agent. Given an alternative-asset deal and "
    "its underlying asset, write a concise, decision-useful investment memo. Use an "
    "illiquidity-adjusted valuation (discount the asset's gross value for the lock-up "
    "and credit/asset risk). Identify the genuine risks — do not pad. Base the "
    "projected yield on the deal's stated figure unless the asset data contradicts it. "
    "Never invent figures that aren't supported by the inputs."
)


def generate_memo(deal: Deal, asset: Asset) -> InvestmentMemo:
    """Produce a validated investment memo for ``deal`` (backed by ``asset``)."""
    user = (
        "Write an investment memo for this deal.\n\n"
        f"DEAL:\n{deal.model_dump_json(indent=2)}\n\n"
        f"ASSET:\n{asset.model_dump_json(indent=2)}"
    )
    return extract(
        system=MEMO_SYSTEM,
        user=user,
        model_cls=InvestmentMemo,
        tool_name="record_investment_memo",
        tool_description="Record the structured investment memo for the deal.",
    )
