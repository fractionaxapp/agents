"""Deal Sourcing Agent core: a seed catalogue of alternative-asset opportunities
plus deterministic aggregation/filtering. The seed data stands in for live
sourcing connectors until those land; the filtering logic is what the Copilot
calls to narrow opportunities by the user's intent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fractionax_core import Asset, Deal, DealFilter
from fractionax_core.domain import InvoiceAsset, IpRoyaltyAsset, RevenueShareAsset, RiskTier

# A dated rwa.xyz snapshot used to seed the discovery catalogue for the demo.
# These deals have no backing Asset; the Copilot underwrites them from a
# deal-implied NAV. Replaced by a licensed live connector in Milestone 2.
_SEED_FILE = Path(__file__).with_name("seed_deals.json")

# The seed's risk_tier is a class-level anchor (assigned from the asset class).
# Refine it with per-deal signals so deals within a class aren't uniform, capping
# the move at one tier so the class stays the dominant signal. (A coarse stand-in
# until the licensed feed carries real per-issuer risk.)
_RISK_TIERS: tuple[RiskTier, ...] = ("low", "medium", "high")
_NO_AUM_RAISE_MINOR = 1_000_000  # $10k floor — used when the source disclosed no AUM
_ESTABLISHED_RAISE_MINOR = 10_000_000_000  # >= $100M: larger, more liquid/established
_HIGH_YIELD_PCT = 10.0


def _refine_risk(base: RiskTier, yield_pct: float, target_raise_minor: int) -> RiskTier:
    """Nudge a class-anchored risk tier by per-deal signals (≤ one tier of movement):
    a high projected yield or an undisclosed/opaque size reads riskier; a large,
    more-established offering reads safer."""
    delta = 0
    if yield_pct >= _HIGH_YIELD_PCT:
        delta += 1
    if target_raise_minor <= _NO_AUM_RAISE_MINOR:
        delta += 1
    elif target_raise_minor >= _ESTABLISHED_RAISE_MINOR:
        delta -= 1
    delta = max(-1, min(1, delta))
    return _RISK_TIERS[max(0, min(len(_RISK_TIERS) - 1, _RISK_TIERS.index(base) + delta))]


def _load_catalogue_seed() -> list[Deal]:
    if not _SEED_FILE.exists():
        return []
    try:
        deals = [Deal(**row) for row in json.loads(_SEED_FILE.read_text())]
    except (ValueError, OSError):
        return []
    return [
        d.model_copy(
            update={
                "risk_tier": _refine_risk(d.risk_tier, d.projected_yield_pct, d.target_raise_minor)
            }
        )
        for d in deals
    ]

# --- Seed assets (one per supported alternative-asset class) ---------------

SEED_ASSETS: list[Asset] = [
    IpRoyaltyAsset(
        id="ast_catalog_a",
        name="Indie music catalogue A",
        currency="USD",
        jurisdiction="US",
        licensor="Northwind Records",
        annual_royalty_minor=1_200_000,
        term_months=60,
    ),
    IpRoyaltyAsset(
        id="ast_patent_my",
        name="Agritech patent licence",
        currency="USD",
        jurisdiction="MY",
        licensor="Selangor AgriTech Sdn Bhd",
        annual_royalty_minor=640_000,
        term_months=48,
    ),
    InvoiceAsset(
        id="ast_invoice_sg",
        name="Logistics receivable (90-day)",
        currency="USD",
        jurisdiction="SG",
        debtor="Strait Freight Pte Ltd",
        face_value_minor=5_000_000,
        due_date="2026-09-30",
    ),
    InvoiceAsset(
        id="ast_invoice_us",
        name="SaaS annual receivable",
        currency="USD",
        jurisdiction="US",
        debtor="Cloudfield Inc",
        face_value_minor=8_000_000,
        due_date="2026-12-15",
    ),
    RevenueShareAsset(
        id="ast_revshare_my",
        name="F&B franchise revenue share",
        currency="USD",
        jurisdiction="MY",
        business="Kopi & Co (12 outlets)",
        share_pct=6.0,
        projected_monthly_revenue_minor=2_500_000,
    ),
]

ASSETS_BY_ID: dict[str, Asset] = {a.id: a for a in SEED_ASSETS}

# --- Seed deals -------------------------------------------------------------
# Deal discovery is served entirely from the dated rwa.xyz catalogue snapshot.

SEED_DEALS: list[Deal] = _load_catalogue_seed()


def _title_matches(query: str, title: str) -> bool:
    """True if ``title`` is the deal the user named — exact/substring either way, or
    a strong token overlap (so minor wording differences still resolve)."""
    q, t = query.strip().lower(), title.lower()
    if not q:
        return False
    if q in t or t in q:
        return True
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    t_tokens = set(re.findall(r"[a-z0-9]+", t))
    return bool(q_tokens) and len(q_tokens & t_tokens) / len(q_tokens) >= 0.6


def source_deals(deal_filter: DealFilter | None = None) -> list[Deal]:
    """Aggregate and filter opportunities by yield, risk, and jurisdiction.

    This is the Deal Sourcing Agent's deterministic core — the Copilot translates
    a parsed intent into a ``DealFilter`` and calls this. Results are sorted by
    projected yield (descending) so the best opportunities surface first.
    """
    deals = list(SEED_DEALS)
    if deal_filter is not None:
        f = deal_filter
        # A specifically named deal takes precedence over the broad filters — return
        # just the matches so the Copilot underwrites the deal the user asked about.
        if f.title_query:
            matched = [d for d in deals if _title_matches(f.title_query, d.title)]
            if matched:
                return sorted(matched, key=lambda d: d.projected_yield_pct, reverse=True)
            # No title match — fall through to the broad filters rather than strand.
        deals = [
            d
            for d in deals
            if (f.jurisdiction is None or d.jurisdiction == f.jurisdiction)
            and (f.risk_tier is None or d.risk_tier == f.risk_tier)
            and (f.asset_class is None or d.asset_class == f.asset_class)
            and (f.min_yield_pct is None or d.projected_yield_pct >= f.min_yield_pct)
            and (
                f.max_min_investment_minor is None
                or d.min_investment_minor <= f.max_min_investment_minor
            )
        ]
    return sorted(deals, key=lambda d: d.projected_yield_pct, reverse=True)
