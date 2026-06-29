"""User Copilot Agent: natural language -> structured action -> deal discovery
-> investment memo.

This is the orchestration spine of the M1 demo. It parses the user's request
into a structured ``InvestmentIntent``, asks the Deal Sourcing Agent for matching
opportunities, and (for invest/discover intents) generates an investment memo for
the best match via the Underwriting Agent.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from fractionax_core import Deal, DealFilter, InvestmentIntent, InvestmentMemo
from pydantic import BaseModel

from .deals import ASSETS_BY_ID, source_deals
from .memo import generate_memo
from .structured import extract

INTENT_SYSTEM = (
    "You are the FractionAX copilot's intent parser. Convert the user's "
    "natural-language request into a structured investment intent by calling the "
    "record_intent tool. Extract EVERY field the message implies — do not leave a "
    "field null when the text gives a signal for it.\n"
    "\n"
    "Fields:\n"
    "- action: 'invest' to commit money, 'discover' to browse/find opportunities, "
    "'rebalance' for portfolio changes, 'quote' for a price. Use 'invest' when an "
    "amount to put in is named, otherwise 'discover' when browsing.\n"
    "- amount_minor: monetary amounts in MINOR units (multiply by 100). "
    "'$1,000' -> 100000, '$2.5k' -> 250000.\n"
    "- currency: ISO 4217 code. '$' -> USD. Set it whenever an amount or currency "
    "is mentioned.\n"
    "- jurisdiction: ISO 3166-1 alpha-2 code. Map country names AND adjectives: "
    "Malaysia/Malaysian -> MY, Singapore/Singaporean -> SG, United States/US/American "
    "-> US, United Kingdom/UK/British -> GB, Indonesia/Indonesian -> ID, "
    "Thailand/Thai -> TH. Set it whenever a place is named.\n"
    "- risk_tier: 'low' for low-risk/safe/conservative/stable/defensive; 'medium' for "
    "balanced/moderate; 'high' for high-risk/aggressive/risky/speculative. Risk means "
    "volatility, NOT yield — never infer risk from 'high-yield' or 'high-return' wording "
    "(that is a yield signal, handled below). Set it only when risk appetite is implied.\n"
    "- min_yield_pct: minimum projected annual yield as a number, ONLY when the user "
    "states an explicit floor: 'at least/minimum/over/above N%' -> min_yield_pct=N. A "
    "vague 'high-yield'/'high-return' with no number sets NO floor (results already sort "
    "yield-high-first).\n"
    "- asset_class: one of these exact slugs — stocks, stablecoins, real-estate, "
    "us-treasury-debt, commodities, corporate-credit, asset-backed-credit, private-equity, "
    "venture-capital, specialty-finance. Map: property/real estate -> real-estate; "
    "stablecoin -> stablecoins; treasury/t-bill/government bond -> us-treasury-debt; "
    "stock/equity/shares -> stocks; commodity/gold/metal -> commodities; corporate/private "
    "credit/loan/bond -> corporate-credit; invoice/receivable/factoring/trade finance -> "
    "asset-backed-credit; private equity/buyout -> private-equity; venture/startup -> "
    "venture-capital; revenue-share/royalty/IP/music/patent/licence -> specialty-finance. "
    "Set it whenever an asset type is named.\n"
    "- asset_kind: legacy hint; set when clear (royalties/IP -> ip_royalty, invoices -> "
    "invoice, revenue-share -> revenue_share) but prefer asset_class for filtering.\n"
    "\n"
    "Examples:\n"
    "- 'Invest $1,000 in low-risk Malaysian opportunities' -> action=invest, "
    "amount_minor=100000, currency=USD, jurisdiction=MY, risk_tier=low.\n"
    "- 'Show me high-yield revenue-share deals with minimum 2% yield' -> action=discover, "
    "asset_class=specialty-finance, min_yield_pct=2.\n"
    "- 'Discover invoice deals in Singapore' -> action=discover, "
    "asset_class=asset-backed-credit, jurisdiction=SG.\n"
    "- 'Safe US treasury deals yielding over 4%' -> action=discover, jurisdiction=US, "
    "risk_tier=low, asset_class=us-treasury-debt, min_yield_pct=4.\n"
    "Leave a field null only when the message gives no signal for it."
)


class CopilotResult(BaseModel):
    """The Copilot's structured response: the parsed intent, the matching deals,
    and (when applicable) a generated memo for the top match."""

    intent: InvestmentIntent
    deals: list[Deal]
    memo: InvestmentMemo | None = None


# Deterministic backstop for the obvious mappings. The LLM handles open-ended
# understanding; these rules guarantee high-confidence fields the model may miss
# (important when the fallback provider extracts less reliably than Claude).
_COUNTRY_TO_ISO: dict[str, str] = {
    "malaysian": "MY", "malaysia": "MY",
    "singaporean": "SG", "singapore": "SG",
    "american": "US", "united states": "US", "usa": "US", "us": "US",
    "british": "GB", "united kingdom": "GB", "uk": "GB",
    "indonesian": "ID", "indonesia": "ID",
    "thai": "TH", "thailand": "TH",
}  # fmt: skip
_RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "low": ("low-risk", "low risk", "safe", "conservative", "stable", "defensive"),
    # Yield words (high-yield/high-return) are deliberately NOT here — they signal a
    # yield floor, not risk; they are handled by _YIELD_RE / min_yield_pct below.
    "high": ("high-risk", "high risk", "aggressive", "risky", "speculative"),
    "medium": ("medium-risk", "medium risk", "balanced", "moderate"),
}  # fmt: skip
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "real-estate": ("real estate", "real-estate", "property", "properties"),
    "stablecoins": ("stablecoin", "stable coin"),
    "us-treasury-debt": ("treasury", "treasuries", "t-bill", "government bond", "govt bond"),
    "stocks": ("stock", "stocks", "equity", "equities", "shares"),
    "commodities": ("commodity", "commodities", "gold", "silver", "precious metal"),
    "asset-backed-credit": ("invoice", "receivable", "factoring", "trade finance", "asset-backed"),
    "private-equity": ("private equity", "buyout"),
    "venture-capital": ("venture capital", "venture-capital", "vc deal", "startup"),
    "corporate-credit": ("corporate credit", "private credit", "corporate bond", "corporate loan"),
    "specialty-finance": (
        "revenue share", "revenue-share", "rev share", "rev-share", "franchise revenue",
        "royalty", "royalties", "intellectual property",
        "music catalog", "music catalogue", "patent", "licence", "license",
    ),
}  # fmt: skip
# "minimum 2%", "at least 4%", "over 5%", or "2% yield"/"4% return"/"5% apy".
_YIELD_RE = re.compile(
    r"(?:min(?:imum)?|at\s+least|over|above|>=?)\s*(\d+(?:\.\d+)?)\s*%"
    r"|(\d+(?:\.\d+)?)\s*%\s*(?:\+|yield|return|apy|apr)",
    re.IGNORECASE,
)
_ASSET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue_share": (
        "revenue share", "revenue-share", "rev share", "rev-share", "franchise revenue",
    ),
    "invoice": ("invoice", "receivable", "factoring"),
    "ip_royalty": (
        "royalty", "royalties", "intellectual property",
        "music catalog", "music catalogue", "patent", "licence", "license",
    ),
}  # fmt: skip
_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s?([km])?", re.IGNORECASE)


def _enrich_intent(intent: InvestmentIntent, message: str) -> InvestmentIntent:
    """Fill high-confidence fields the model left unset, from the raw message."""
    text = message.lower()
    updates: dict[str, Any] = {}

    if intent.amount_minor is None:
        m = _AMOUNT_RE.search(message)
        if m:
            amount = float(m.group(1).replace(",", ""))
            mult = {"k": 1_000, "m": 1_000_000}.get((m.group(2) or "").lower(), 1)
            updates["amount_minor"] = int(round(amount * mult * 100))

    if intent.currency is None and "$" in message:
        updates["currency"] = "USD"

    if intent.jurisdiction is None:
        for name, iso in _COUNTRY_TO_ISO.items():
            if re.search(rf"\b{re.escape(name)}\b", text):
                updates["jurisdiction"] = iso
                break

    if intent.risk_tier is None:
        for tier, keywords in _RISK_KEYWORDS.items():
            if any(k in text for k in keywords):
                updates["risk_tier"] = tier
                break

    if intent.asset_kind is None:
        for kind, keywords in _ASSET_KEYWORDS.items():
            if any(k in text for k in keywords):
                updates["asset_kind"] = kind
                break

    if intent.asset_class is None:
        for cls, keywords in _CLASS_KEYWORDS.items():
            if any(k in text for k in keywords):
                updates["asset_class"] = cls
                break

    # Only an explicit floor ("minimum 2%") filters by yield; a vague "high-yield"
    # imposes no hard cutoff — results are already sorted yield-high-first, so a
    # default floor would just hide whole classes (e.g. real estate rarely clears 8%).
    if intent.min_yield_pct is None:
        m = _YIELD_RE.search(message)
        if m:
            updates["min_yield_pct"] = float(m.group(1) or m.group(2))

    return intent.model_copy(update=updates) if updates else intent


def parse_intent(message: str) -> InvestmentIntent:
    """Parse a natural-language request into a structured ``InvestmentIntent``.

    The model extracts the intent; a deterministic backstop then fills any
    high-confidence fields it missed (amount, currency, jurisdiction, risk, asset).
    """
    intent = extract(
        system=INTENT_SYSTEM,
        user=message,
        model_cls=InvestmentIntent,
        tool_name="record_intent",
        tool_description="Record the structured investment intent parsed from the user's message.",
    )
    return _enrich_intent(intent, message)


def intent_to_filter(intent: InvestmentIntent) -> DealFilter:
    """Translate a parsed intent into deal-sourcing criteria."""
    return DealFilter(
        jurisdiction=intent.jurisdiction,
        risk_tier=intent.risk_tier,
        min_yield_pct=intent.min_yield_pct,
        asset_class=intent.asset_class,
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


def stream_copilot(message: str, *, with_memo: bool = True) -> Iterator[tuple[str, Any]]:
    """Yield ``(event, payload)`` as each Copilot stage completes.

    Drives the SSE endpoint so the UI shows the parsed intent immediately, then the
    matching deals, then the memo (the slow LLM step) when it is ready.
    """
    intent = parse_intent(message)
    yield "intent", intent.model_dump()

    deals = source_deals(intent_to_filter(intent))
    yield "deals", [d.model_dump() for d in deals]

    if with_memo and deals and intent.action in ("invest", "discover"):
        asset = ASSETS_BY_ID.get(deals[0].asset_id)
        if asset is not None:
            yield "memo", generate_memo(deals[0], asset).model_dump()

    yield "done", {}
