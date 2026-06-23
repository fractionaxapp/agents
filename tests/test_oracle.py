from datetime import date

from fastapi.testclient import TestClient
from fractionax_core.domain import InvoiceAsset

from fractionax_agents.deals import SEED_ASSETS
from fractionax_agents.oracle import (
    FundamentalNavOracle,
    PythNavOracle,
    get_nav_oracle,
    illiquidity_adjusted_valuation,
)
from fractionax_agents.server import app

AS_OF = date(2026, 6, 24)
client = TestClient(app)


def test_invoice_nav_is_discounted_below_face_value() -> None:
    inv = InvoiceAsset(
        id="x",
        name="x",
        currency="USD",
        jurisdiction="US",
        debtor="d",
        face_value_minor=1_000_000,
        due_date="2026-12-24",
    )
    q = FundamentalNavOracle().quote(inv, as_of=AS_OF)
    assert 0 < q.nav_minor < 1_000_000  # discounted for time to settlement
    assert q.source == "manual"
    assert q.currency == "USD"


def test_every_seed_asset_prices_positive() -> None:
    oracle = FundamentalNavOracle()
    for asset in SEED_ASSETS:
        q = oracle.quote(asset, as_of=AS_OF)
        assert q.nav_minor > 0
        assert q.asset_id == asset.id


def test_illiquidity_haircut_grows_with_risk() -> None:
    nav = 1_000_000
    low = illiquidity_adjusted_valuation(nav, "low")
    high = illiquidity_adjusted_valuation(nav, "high")
    assert low < nav
    assert high < low  # higher risk -> larger haircut -> lower fair value


def test_default_oracle_is_fundamental() -> None:
    assert isinstance(get_nav_oracle(), FundamentalNavOracle)


def test_pyth_oracle_falls_back_without_a_feed() -> None:
    # No feed mapped -> fundamental fallback, so no network call is made.
    pyth = PythNavOracle(
        hermes_url="https://hermes.pyth.network", feeds={}, fallback=FundamentalNavOracle()
    )
    q = pyth.quote(SEED_ASSETS[0], as_of=AS_OF)
    assert q.source == "manual"


def test_nav_endpoint_returns_catalogue() -> None:
    resp = client.get("/nav")
    assert resp.status_code == 200
    assert len(resp.json()) == len(SEED_ASSETS)


def test_nav_endpoint_single_asset_and_404() -> None:
    ok = client.get("/nav", params={"asset_id": "ast_catalog_a"})
    assert ok.status_code == 200
    assert len(ok.json()) == 1
    assert client.get("/nav", params={"asset_id": "nope"}).status_code == 404
