"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fractionax_core.domain import Deal, RiskTier

from fractionax_agents import db, deals
from fractionax_agents.config import Settings


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give every test a fresh, empty SQLite database (so the catalogue starts empty)
    independent of any ambient DATABASE_URL or the dev's local DB."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setattr(
        "fractionax_agents.db.get_settings",
        lambda: Settings(_env_file=None, database_url=url),
    )
    monkeypatch.setattr(db, "_engine", None, raising=False)
    monkeypatch.setattr(db, "_engine_url", None, raising=False)
    monkeypatch.setattr(deals, "_catalogue_cache", None, raising=False)


def _mk(
    id_: str,
    jurisdiction: str,
    asset_class: str,
    base: RiskTier,
    yield_pct: float,
    target_raise_minor: int,
    min_investment_minor: int,
    title: str,
) -> Deal:
    return Deal(
        id=id_,
        asset_id=id_.replace("deal", "ast"),
        title=title,
        jurisdiction=jurisdiction,
        currency="USD",
        min_investment_minor=min_investment_minor,
        target_raise_minor=target_raise_minor,
        projected_yield_pct=yield_pct,
        risk_tier=base,
        status="open",
        asset_class=asset_class,
        sourced_at="2026-01-01T00:00:00.000Z",
    )


# A small, deterministic catalogue standing in for the seed the tests used to rely
# on. Base risk tiers; deals.get_catalogue refines them on read. deal_rwa_1 is an
# offshore (SV) stablecoin so the compliance tests behave as before.
_SAMPLE: list[Deal] = [
    _mk("deal_rwa_1", "SV", "stablecoins", "low", 0.0, 5_000_000_000, 10_000_000, "Tether USDt"),
    _mk(
        "deal_us_tsy",
        "US",
        "us-treasury-debt",
        "low",
        4.0,
        20_000_000_000,
        100_000,
        "US Treasury Fund",
    ),
    _mk(
        "deal_us_credit",
        "US",
        "corporate-credit",
        "medium",
        9.0,
        3_000_000_000,
        100_000,
        "US Credit Fund",
    ),
    _mk("deal_my_re", "MY", "real-estate", "medium", 12.0, 2_000_000_000, 50_000, "KL Real Estate"),
    _mk(
        "deal_sg_pe",
        "SG",
        "private-equity",
        "high",
        18.0,
        8_000_000_000,
        25_000,
        "SG Growth Equity",
    ),
    _mk(
        "deal_my_commod",
        "MY",
        "commodities",
        "medium",
        6.0,
        1_500_000_000,
        40_000,
        "Palm Oil Commodity",
    ),
    _mk("deal_gb_stocks", "GB", "stocks", "high", 3.0, 500_000, 100_000, "London Equities"),
]


@pytest.fixture
def sample_catalogue() -> list[Deal]:
    return list(_SAMPLE)


@pytest.fixture
def populated_db(sample_catalogue: list[Deal]) -> list[Deal]:
    """Load the sample catalogue into the (temp) database for tests that need deals."""
    deals.set_catalogue(sample_catalogue)
    return sample_catalogue
