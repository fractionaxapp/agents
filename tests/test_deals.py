from fastapi.testclient import TestClient
from fractionax_core import Deal, DealFilter

from fractionax_agents.deals import _refine_risk, source_deals
from fractionax_agents.server import app

client = TestClient(app)


def test_refine_risk_nudges_at_most_one_tier() -> None:
    assert _refine_risk("medium", 0.0, 50_000_000_000) == "low"  # large/established -> safer
    assert _refine_risk("medium", 0.0, 1_000_000) == "high"  # opaque size -> riskier
    assert _refine_risk("medium", 20.0, 1_000_000_000) == "high"  # high yield -> riskier
    assert _refine_risk("low", 0.0, 1_000_000) == "medium"  # capped one tier up
    assert _refine_risk("high", 0.0, 50_000_000_000) == "medium"  # capped one tier down


def test_source_deals_empty_without_import() -> None:
    # No seed fallback: an empty database means an empty catalogue.
    assert source_deals() == []


def test_source_deals_returns_all_when_unfiltered(populated_db: list[Deal]) -> None:
    assert len(source_deals()) == len(populated_db)


def test_source_deals_sorted_by_yield_desc(populated_db: list[Deal]) -> None:
    yields = [d.projected_yield_pct for d in source_deals()]
    assert yields == sorted(yields, reverse=True)


def test_source_deals_filters_by_jurisdiction(populated_db: list[Deal]) -> None:
    deals = source_deals(DealFilter(jurisdiction="US"))
    assert deals and all(d.jurisdiction == "US" for d in deals)


def test_source_deals_filters_by_asset_class(populated_db: list[Deal]) -> None:
    deals = source_deals(DealFilter(asset_class="stablecoins"))
    assert deals and all(d.asset_class == "stablecoins" for d in deals)


def test_source_deals_filters_by_risk_and_yield(populated_db: list[Deal]) -> None:
    deals = source_deals(DealFilter(risk_tier="high", min_yield_pct=15.0))
    assert all(d.risk_tier == "high" and d.projected_yield_pct >= 15.0 for d in deals)


def test_source_deals_filters_by_affordability(populated_db: list[Deal]) -> None:
    deals = source_deals(DealFilter(max_min_investment_minor=50_000))
    assert deals and all(d.min_investment_minor <= 50_000 for d in deals)


def test_every_deal_has_an_asset_class(populated_db: list[Deal]) -> None:
    assert all(d.asset_class for d in source_deals())


def test_source_deals_matches_a_named_deal(populated_db: list[Deal]) -> None:
    target = "US Credit Fund"
    deals = source_deals(DealFilter(title_query=target))
    assert deals and any(d.title == target for d in deals)
    assert len(deals) < len(populated_db)


def test_title_query_falls_back_when_nothing_matches(populated_db: list[Deal]) -> None:
    deals = source_deals(DealFilter(title_query="no such deal exists anywhere"))
    assert len(deals) == len(populated_db)  # unmatched name -> don't strand the result


def test_deals_endpoint_filters(populated_db: list[Deal]) -> None:
    resp = client.get("/deals", params={"risk_tier": "high"})
    assert resp.status_code == 200
    body = resp.json()
    assert body and all(d["risk_tier"] == "high" for d in body)
