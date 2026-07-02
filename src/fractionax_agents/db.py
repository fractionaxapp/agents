"""Relational store for the deal catalogue plus a timeseries of import snapshots.

One Postgres in deploy (Neon), a local SQLite file otherwise — the SQLAlchemy Core
schema below is identical on both, so only ``DATABASE_URL`` changes. Every catalogue
import replaces the live ``deals`` rows and appends one ``deal_snapshots`` row per
deal, so re-importing the rwa.xyz export over time accumulates a per-deal timeseries
(AUM / yield / minimum). The snapshot table is timeseries-shaped (a ``deal_id`` +
``observed_at`` key) so it can become a TimescaleDB hypertable later with no schema
change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fractionax_core import Deal
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from .config import get_settings

metadata = MetaData()

# Current catalogue. risk_tier stores the class-anchored *base* tier; callers refine
# it per deal on read (mirrors the seed handling in deals.py).
deals_table = Table(
    "deals",
    metadata,
    Column("id", String, primary_key=True),
    Column("asset_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("jurisdiction", String(2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("min_investment_minor", BigInteger, nullable=False),
    Column("target_raise_minor", BigInteger, nullable=False),
    Column("projected_yield_pct", Float, nullable=False),
    Column("risk_tier", String, nullable=False),
    Column("status", String, nullable=False),
    Column("sourced_at", String, nullable=False),
    Column("asset_class", String, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

# Point-in-time observations captured on each import (the timeseries).
deal_snapshots = Table(
    "deal_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("deal_id", String, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column("target_raise_minor", BigInteger, nullable=False),
    Column("projected_yield_pct", Float, nullable=False),
    Column("min_investment_minor", BigInteger, nullable=False),
    Column("risk_tier", String, nullable=False),
)
Index("ix_deal_snapshots_deal_observed", deal_snapshots.c.deal_id, deal_snapshots.c.observed_at)

_DEAL_FIELDS = (
    "id",
    "asset_id",
    "title",
    "jurisdiction",
    "currency",
    "min_investment_minor",
    "target_raise_minor",
    "projected_yield_pct",
    "risk_tier",
    "status",
    "sourced_at",
    "asset_class",
)


def _database_url() -> str:
    """Resolve the SQLAlchemy URL. Normalizes a plain postgres URL to the psycopg
    driver; falls back to a local SQLite file when DATABASE_URL is unset."""
    url = get_settings().database_url
    if url:
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        return url
    return f"sqlite:///{Path('fractionax.db').resolve()}"


_engine: Engine | None = None
_engine_url: str | None = None


def get_engine() -> Engine:
    """Lazily create the engine and ensure the schema exists. Recreates the engine
    if the resolved URL changed (e.g. tests pointing at a temp SQLite file)."""
    global _engine, _engine_url
    url = _database_url()
    if _engine is None or _engine_url != url:
        # pool_pre_ping keeps Neon's idle-closed connections from surfacing as errors.
        _engine = create_engine(url, future=True, pool_pre_ping=True)
        _engine_url = url
        metadata.create_all(_engine)
    return _engine


def _to_row(deal: Deal, observed_at: datetime) -> dict[str, object]:
    row = deal.model_dump(include=set(_DEAL_FIELDS))
    row["updated_at"] = observed_at
    return row


def _to_snapshot(deal: Deal, observed_at: datetime) -> dict[str, object]:
    return {
        "deal_id": deal.id,
        "observed_at": observed_at,
        "target_raise_minor": deal.target_raise_minor,
        "projected_yield_pct": deal.projected_yield_pct,
        "min_investment_minor": deal.min_investment_minor,
        "risk_tier": deal.risk_tier,
    }


def replace_catalogue(deals: list[Deal], *, observed_at: datetime | None = None) -> int:
    """Replace the live catalogue with ``deals`` and append a snapshot row per deal.
    Returns the number of deals stored."""
    observed_at = observed_at or datetime.now(UTC)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(delete(deals_table))
        if deals:
            conn.execute(insert(deals_table), [_to_row(d, observed_at) for d in deals])
            conn.execute(insert(deal_snapshots), [_to_snapshot(d, observed_at) for d in deals])
    return len(deals)


def load_deals() -> list[Deal]:
    """Load the current catalogue as Deals (base risk tiers; refined by the caller)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(deals_table)).mappings().all()
    return [Deal(**{k: row[k] for k in _DEAL_FIELDS}) for row in rows]


def count_deals() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(deals_table)).scalar_one())


def count_snapshots() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(deal_snapshots)).scalar_one())


def clear_deals() -> None:
    """Remove the live catalogue (falls back to seed). Snapshot history is retained."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(delete(deals_table))


def snapshot_history(deal_id: str, *, limit: int = 100) -> list[dict[str, object]]:
    """Return a deal's snapshot points, most recent first (the timeseries)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(deal_snapshots)
                .where(deal_snapshots.c.deal_id == deal_id)
                .order_by(deal_snapshots.c.observed_at.desc())
                .limit(limit)
            )
            .mappings()
            .all()
        )
    return [
        {
            "deal_id": r["deal_id"],
            "observed_at": r["observed_at"].isoformat(),
            "target_raise_minor": r["target_raise_minor"],
            "projected_yield_pct": r["projected_yield_pct"],
            "min_investment_minor": r["min_investment_minor"],
            "risk_tier": r["risk_tier"],
        }
        for r in rows
    ]
