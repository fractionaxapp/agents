"""Deal Sourcing Agent core: a seed catalogue of alternative-asset opportunities
plus deterministic aggregation/filtering. The seed data stands in for live
sourcing connectors until those land; the filtering logic is what the Copilot
calls to narrow opportunities by the user's intent.
"""

from __future__ import annotations

import json
from pathlib import Path

from fractionax_core import Asset, Deal, DealFilter
from fractionax_core.domain import InvoiceAsset, IpRoyaltyAsset, RevenueShareAsset

# A dated rwa.xyz snapshot used to seed the discovery catalogue for the demo
# (these deals have no backing Asset, so they list but don't generate a memo).
# Replaced by a licensed live connector in Milestone 2.
_SEED_FILE = Path(__file__).with_name("seed_deals.json")


def _load_catalogue_seed() -> list[Deal]:
    if not _SEED_FILE.exists():
        return []
    try:
        return [Deal(**row) for row in json.loads(_SEED_FILE.read_text())]
    except (ValueError, OSError):
        return []

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


def source_deals(deal_filter: DealFilter | None = None) -> list[Deal]:
    """Aggregate and filter opportunities by yield, risk, and jurisdiction.

    This is the Deal Sourcing Agent's deterministic core — the Copilot translates
    a parsed intent into a ``DealFilter`` and calls this. Results are sorted by
    projected yield (descending) so the best opportunities surface first.
    """
    deals = list(SEED_DEALS)
    if deal_filter is not None:
        f = deal_filter
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
