from fastapi.testclient import TestClient
from fractionax_core import DealFilter

from fractionax_agents.deals import ASSETS_BY_ID, SEED_DEALS, source_deals
from fractionax_agents.server import app

client = TestClient(app)


def test_source_deals_returns_all_when_unfiltered() -> None:
    assert len(source_deals()) == len(SEED_DEALS)


def test_source_deals_sorted_by_yield_desc() -> None:
    yields = [d.projected_yield_pct for d in source_deals()]
    assert yields == sorted(yields, reverse=True)


def test_source_deals_filters_by_jurisdiction() -> None:
    deals = source_deals(DealFilter(jurisdiction="MY"))
    assert deals and all(d.jurisdiction == "MY" for d in deals)


def test_source_deals_filters_by_risk_and_yield() -> None:
    deals = source_deals(DealFilter(risk_tier="high", min_yield_pct=15.0))
    assert all(d.risk_tier == "high" and d.projected_yield_pct >= 15.0 for d in deals)


def test_source_deals_filters_by_affordability() -> None:
    deals = source_deals(DealFilter(max_min_investment_minor=50_000))
    assert deals and all(d.min_investment_minor <= 50_000 for d in deals)


def test_every_deal_has_a_backing_asset() -> None:
    for d in SEED_DEALS:
        assert d.asset_id in ASSETS_BY_ID


def test_deals_endpoint_filters() -> None:
    resp = client.get("/deals", params={"risk_tier": "high"})
    assert resp.status_code == 200
    body = resp.json()
    assert body and all(d["risk_tier"] == "high" for d in body)
