"""User Copilot Agent: natural language -> structured action -> deal discovery
-> investment memo.

This is the orchestration spine of the M1 demo. It parses the user's request
into a structured ``InvestmentIntent``, asks the Deal Sourcing Agent for matching
opportunities, and (for invest/discover intents) generates an investment memo for
the best match via the Underwriting Agent.
"""

from __future__ import annotations

from fractionax_core import Deal, DealFilter, InvestmentIntent, InvestmentMemo
from pydantic import BaseModel

from .deals import ASSETS_BY_ID, source_deals
from .memo import generate_memo
from .structured import extract

INTENT_SYSTEM = (
    "You are the FractionAX copilot's intent parser. Convert the user's natural-language "
    "request into a structured investment intent.\n"
    "- Map dollar amounts to minor units (multiply by 100): $1,000 -> 100000.\n"
    "- Map a named country to its ISO 3166-1 alpha-2 code (Malaysia -> MY, Singapore -> SG, "
    "United States -> US).\n"
    "- Infer risk tier (low/medium/high) and asset kind (ip_royalty/invoice/revenue_share) "
    "only when clearly implied; otherwise leave them unset.\n"
    "- Choose the action: 'invest' to commit money, 'discover' to browse opportunities, "
    "'rebalance' for portfolio changes, 'quote' for a price."
)


class CopilotResult(BaseModel):
    """The Copilot's structured response: the parsed intent, the matching deals,
    and (when applicable) a generated memo for the top match."""

    intent: InvestmentIntent
    deals: list[Deal]
    memo: InvestmentMemo | None = None


def parse_intent(message: str) -> InvestmentIntent:
    """Parse a natural-language request into a structured ``InvestmentIntent``."""
    return extract(
        system=INTENT_SYSTEM,
        user=message,
        model_cls=InvestmentIntent,
        tool_name="record_intent",
        tool_description="Record the structured investment intent parsed from the user's message.",
    )


def intent_to_filter(intent: InvestmentIntent) -> DealFilter:
    """Translate a parsed intent into deal-sourcing criteria."""
    return DealFilter(
        jurisdiction=intent.jurisdiction,
        risk_tier=intent.risk_tier,
        # If the user named an amount, only surface deals they can actually enter.
        max_min_investment_minor=intent.amount_minor,
    )


def run_copilot(message: str, *, with_memo: bool = True) -> CopilotResult:
    """Run the full Copilot flow for a natural-language ``message``."""
    intent = parse_intent(message)
    deals = source_deals(intent_to_filter(intent))

    memo: InvestmentMemo | None = None
    if with_memo and deals and intent.action in ("invest", "discover"):
        top = deals[0]
        asset = ASSETS_BY_ID.get(top.asset_id)
        if asset is not None:
            memo = generate_memo(top, asset)

    return CopilotResult(intent=intent, deals=deals, memo=memo)
