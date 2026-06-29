from collections import defaultdict

from fastapi.testclient import TestClient
from fractionax_core import DealFilter

from fractionax_agents.deals import SEED_DEALS, _refine_risk, source_deals
from fractionax_agents.server import app

client = TestClient(app)


def test_refine_risk_nudges_at_most_one_tier() -> None:
    assert _refine_risk("medium", 0.0, 50_000_000_000) == "low"  # large/established -> safer
    assert _refine_risk("medium", 0.0, 1_000_000) == "high"  # opaque size -> riskier
    assert _refine_risk("medium", 20.0, 1_000_000_000) == "high"  # high yield -> riskier
    assert _refine_risk("low", 0.0, 1_000_000) == "medium"  # capped one tier up
    assert _refine_risk("high", 0.0, 50_000_000_000) == "medium"  # capped one tier down


def test_risk_is_not_uniform_within_classes() -> None:
    # The class anchors risk, but per-deal signals spread most classes across tiers.
    tiers: dict[str, set[str]] = defaultdict(set)
    for d in SEED_DEALS:
        tiers[d.asset_class].add(d.risk_tier)
    assert sum(1 for t in tiers.values() if len(t) > 1) >= 5


def test_source_deals_returns_all_when_unfiltered() -> None:
    assert len(source_deals()) == len(SEED_DEALS)


def test_source_deals_sorted_by_yield_desc() -> None:
    yields = [d.projected_yield_pct for d in source_deals()]
    assert yields == sorted(yields, reverse=True)


def test_source_deals_filters_by_jurisdiction() -> None:
    juris = SEED_DEALS[0].jurisdiction
    deals = source_deals(DealFilter(jurisdiction=juris))
    assert deals and all(d.jurisdiction == juris for d in deals)


def test_source_deals_filters_by_asset_class() -> None:
    deals = source_deals(DealFilter(asset_class="stablecoins"))
    assert deals and all(d.asset_class == "stablecoins" for d in deals)


def test_source_deals_filters_by_risk_and_yield() -> None:
    deals = source_deals(DealFilter(risk_tier="high", min_yield_pct=15.0))
    assert all(d.risk_tier == "high" and d.projected_yield_pct >= 15.0 for d in deals)


def test_source_deals_filters_by_affordability() -> None:
    deals = source_deals(DealFilter(max_min_investment_minor=50_000))
    assert deals and all(d.min_investment_minor <= 50_000 for d in deals)


def test_every_deal_has_an_asset_class() -> None:
    # Discovery is served from the rwa.xyz catalogue; deals carry a class but no
    # backing typed Asset (those are a memo-only concern for the demo assets).
    assert SEED_DEALS and all(d.asset_class for d in SEED_DEALS)


def test_source_deals_matches_a_named_deal() -> None:
    target = SEED_DEALS[5].title
    deals = source_deals(DealFilter(title_query=target))
    # The named deal is surfaced, and it narrows the set rather than returning all.
    assert deals and any(d.title == target for d in deals)
    assert len(deals) < len(SEED_DEALS)


def test_title_query_falls_back_when_nothing_matches() -> None:
    deals = source_deals(DealFilter(title_query="no such deal exists anywhere"))
    assert len(deals) == len(SEED_DEALS)  # unmatched name -> don't strand the result


def test_deals_endpoint_filters() -> None:
    resp = client.get("/deals", params={"risk_tier": "high"})
    assert resp.status_code == 200
    body = resp.json()
    assert body and all(d["risk_tier"] == "high" for d in body)
