"""Deal Sourcing Agent core: a seed catalogue of alternative-asset opportunities
plus deterministic aggregation/filtering. The seed data stands in for live
sourcing connectors until those land; the filtering logic is what the Copilot
calls to narrow opportunities by the user's intent.
"""

from __future__ import annotations

from fractionax_core import Asset, Deal, DealFilter
from fractionax_core.domain import InvoiceAsset, IpRoyaltyAsset, RevenueShareAsset

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

# --- Seed deals (investable wrappers around the assets) ---------------------

SEED_DEALS: list[Deal] = [
    Deal(
        id="deal_catalog_a",
        asset_id="ast_catalog_a",
        title="Music catalogue A — royalty participation",
        jurisdiction="US",
        currency="USD",
        min_investment_minor=100_000,  # $1,000
        target_raise_minor=4_000_000,
        projected_yield_pct=8.5,
        risk_tier="low",
        status="open",
        sourced_at="2026-06-20T00:00:00Z",
    ),
    Deal(
        id="deal_patent_my",
        asset_id="ast_patent_my",
        title="Agritech patent licence — Malaysia",
        jurisdiction="MY",
        currency="USD",
        min_investment_minor=100_000,  # $1,000
        target_raise_minor=2_500_000,
        projected_yield_pct=7.2,
        risk_tier="low",
        status="open",
        sourced_at="2026-06-21T00:00:00Z",
    ),
    Deal(
        id="deal_invoice_sg",
        asset_id="ast_invoice_sg",
        title="Logistics receivable — Singapore",
        jurisdiction="SG",
        currency="USD",
        min_investment_minor=50_000,  # $500
        target_raise_minor=5_000_000,
        projected_yield_pct=11.0,
        risk_tier="medium",
        status="open",
        sourced_at="2026-06-21T00:00:00Z",
    ),
    Deal(
        id="deal_invoice_us",
        asset_id="ast_invoice_us",
        title="SaaS receivable — United States",
        jurisdiction="US",
        currency="USD",
        min_investment_minor=200_000,  # $2,000
        target_raise_minor=8_000_000,
        projected_yield_pct=10.0,
        risk_tier="medium",
        status="open",
        sourced_at="2026-06-22T00:00:00Z",
    ),
    Deal(
        id="deal_revshare_my",
        asset_id="ast_revshare_my",
        title="F&B franchise revenue share — Malaysia",
        jurisdiction="MY",
        currency="USD",
        min_investment_minor=25_000,  # $250
        target_raise_minor=3_000_000,
        projected_yield_pct=18.0,
        risk_tier="high",
        status="open",
        sourced_at="2026-06-22T00:00:00Z",
    ),
]


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
            and (f.min_yield_pct is None or d.projected_yield_pct >= f.min_yield_pct)
            and (
                f.max_min_investment_minor is None
                or d.min_investment_minor <= f.max_min_investment_minor
            )
        ]
    return sorted(deals, key=lambda d: d.projected_yield_pct, reverse=True)
