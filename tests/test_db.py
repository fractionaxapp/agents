"""The SQLAlchemy data layer: catalogue round-trip, snapshots, URL normalization."""

from __future__ import annotations

import pytest
from fractionax_core.domain import Deal

from fractionax_agents import db
from fractionax_agents.config import Settings


def _deal(id_: str) -> Deal:
    return Deal(
        id=id_,
        asset_id=f"ast_{id_}",
        title=f"Deal {id_}",
        jurisdiction="SG",
        currency="USD",
        min_investment_minor=100_000,
        target_raise_minor=9_000_000,
        projected_yield_pct=7.5,
        risk_tier="medium",
        status="open",
        asset_class="real-estate",
        sourced_at="2026-01-01T00:00:00.000Z",
    )


def test_replace_load_count_clear() -> None:
    assert db.count_deals() == 0
    n = db.replace_catalogue([_deal("d1"), _deal("d2")])
    assert n == 2
    assert db.count_deals() == 2
    loaded = db.load_deals()
    assert {d.id for d in loaded} == {"d1", "d2"}
    assert all(isinstance(d, Deal) for d in loaded)
    db.clear_deals()
    assert db.count_deals() == 0


def test_replace_is_a_full_swap() -> None:
    db.replace_catalogue([_deal("a"), _deal("b")])
    db.replace_catalogue([_deal("c")])
    assert {d.id for d in db.load_deals()} == {"c"}


def test_snapshots_accumulate_and_query() -> None:
    db.replace_catalogue([_deal("d1")])
    db.replace_catalogue([_deal("d1")])
    assert db.count_snapshots() == 2
    hist = db.snapshot_history("d1")
    assert len(hist) == 2
    assert hist[0]["deal_id"] == "d1"
    assert "observed_at" in hist[0]


@pytest.mark.parametrize(
    ("configured", "expected_prefix"),
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    ],
)
def test_database_url_normalizes_postgres(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected_prefix: str
) -> None:
    monkeypatch.setattr(
        "fractionax_agents.db.get_settings",
        lambda: Settings(_env_file=None, database_url=configured),
    )
    assert db._database_url() == expected_prefix


def test_database_url_defaults_to_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fractionax_agents.db.get_settings",
        lambda: Settings(_env_file=None, database_url=None),
    )
    assert db._database_url().startswith("sqlite:///")
