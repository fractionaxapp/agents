"""The database-backed catalogue in deals.py (set / get / reset / snapshots).

Each test runs against a fresh temp SQLite DB (see conftest._isolate_db), so the
catalogue starts empty and falls back to the seed."""

from __future__ import annotations

from fractionax_core.domain import Deal

from fractionax_agents import db, deals


def _deal(id_: str = "deal_x", risk: str = "medium") -> Deal:
    return Deal(
        id=id_,
        asset_id="ast_x",
        title="X",
        jurisdiction="MY",
        currency="USD",
        min_investment_minor=100_000,
        target_raise_minor=5_000_000,
        projected_yield_pct=6.0,
        risk_tier=risk,  # type: ignore[arg-type]
        status="open",
        asset_class="real-estate",
        sourced_at="2026-01-01T00:00:00.000Z",
    )


def test_empty_without_import() -> None:
    # No seed/JSON fallback: an empty database means an empty catalogue.
    assert deals.catalogue_source() == "empty"
    assert deals.get_catalogue() == []


def test_set_then_get_uses_database() -> None:
    n = deals.set_catalogue([_deal("deal_a"), _deal("deal_b")])
    assert n == 2
    assert deals.catalogue_source() == "database"
    assert {d.id for d in deals.get_catalogue()} == {"deal_a", "deal_b"}


def test_clear_empties_catalogue_but_keeps_snapshots() -> None:
    deals.set_catalogue([_deal()])
    assert deals.catalogue_source() == "database"
    assert db.count_snapshots() == 1
    deals.clear_catalogue()
    assert deals.catalogue_source() == "empty"
    assert deals.get_catalogue() == []
    # The timeseries history survives clearing the live catalogue.
    assert db.count_snapshots() == 1


def test_override_risk_is_refined_on_read() -> None:
    # A high projected yield nudges a medium base tier up to high on read.
    deals.set_catalogue(
        [_deal("deal_hot", risk="medium").model_copy(update={"projected_yield_pct": 20.0})]
    )
    assert deals.get_catalogue()[0].risk_tier == "high"


def test_reimport_appends_snapshots() -> None:
    deals.set_catalogue([_deal("deal_a")])
    deals.set_catalogue([_deal("deal_a")])  # re-import the same deal
    # Two imports -> two snapshot points for the deal (the timeseries).
    assert db.count_snapshots() == 2
    assert len(db.snapshot_history("deal_a")) == 2
    # ...but only one live deal row.
    assert len(deals.get_catalogue()) == 1
