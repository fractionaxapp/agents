"""NAV (net-asset-value) pricing oracle.

Produces a :class:`NavQuote` for an asset. Two providers:

- :class:`FundamentalNavOracle` (default, source ``manual``) values an asset from
  its fundamentals — a discounted cash-flow for royalty/revenue-share streams and
  a time-value discount for invoices. This is the working oracle that grounds the
  Underwriting Agent's valuation today.
- :class:`PythNavOracle` (source ``pyth``) is the live-feed seam: once an asset is
  tokenized and has a price feed, map ``asset_id -> feed_id`` and the NAV is read
  from Pyth's Hermes endpoint. Unmapped assets fall back to the fundamental oracle.

Switchboard is a future provider that slots in behind the same ``NavOracle``
protocol. Select the provider via ``NAV_ORACLE_PROVIDER``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from fractionax_core import Asset, Deal, NavQuote
from fractionax_core.domain import (
    InvoiceAsset,
    IpRoyaltyAsset,
    RevenueShareAsset,
    RiskTier,
)

from .config import Settings, get_settings

# Annual discount rate applied to fundamental cash flows.
_DISCOUNT_RATE = 0.12
# Revenue-share streams have no stated term; value a fixed horizon.
_REVSHARE_HORIZON_MONTHS = 36
# Illiquidity haircut applied to NAV by risk tier, to get a mark-to-fair value.
_ILLIQUIDITY_DISCOUNT: dict[RiskTier, float] = {"low": 0.05, "medium": 0.12, "high": 0.20}


class NavOracle(Protocol):
    """Returns a net-asset-value quote for an asset."""

    def quote(self, asset: Asset, *, as_of: date | None = None) -> NavQuote: ...


def _pv_annuity(payment: float, monthly_rate: float, months: int) -> float:
    """Present value of a level monthly ``payment`` over ``months``."""
    if monthly_rate == 0:
        return payment * months
    return payment * (1 - (1 + monthly_rate) ** (-months)) / monthly_rate


def _fundamental_nav_minor(asset: Asset, as_of: date) -> int:
    """Value an asset from its fundamentals, in minor units."""
    monthly_rate = _DISCOUNT_RATE / 12
    if isinstance(asset, IpRoyaltyAsset):
        monthly = asset.annual_royalty_minor / 12
        return round(_pv_annuity(monthly, monthly_rate, asset.term_months))
    if isinstance(asset, RevenueShareAsset):
        monthly = asset.projected_monthly_revenue_minor * asset.share_pct / 100
        return round(_pv_annuity(monthly, monthly_rate, _REVSHARE_HORIZON_MONTHS))
    if isinstance(asset, InvoiceAsset):
        days = max((date.fromisoformat(asset.due_date) - as_of).days, 0)
        return round(asset.face_value_minor / (1 + _DISCOUNT_RATE * days / 365))
    raise ValueError(f"Unsupported asset kind: {asset!r}")


def illiquidity_adjusted_valuation(nav_minor: int, risk_tier: RiskTier) -> int:
    """Haircut a gross NAV for illiquidity/lock-up, by risk tier."""
    return round(nav_minor * (1 - _ILLIQUIDITY_DISCOUNT[risk_tier]))


def deal_implied_nav_minor(deal: Deal) -> int:
    """Gross NAV proxy for a catalogue deal that has no typed asset to price.

    A tokenised offering is issued at par against its underlying pool, so the target
    raise is the natural gross NAV. The same illiquidity haircut as the asset-backed
    path then yields the fair valuation, and the underwriting memo reasons about the
    deal's projected yield and risk around that anchor. Used when discovery surfaces
    catalogue deals (which carry no fundamental cash-flow inputs to discount).
    """
    return deal.target_raise_minor


def _as_of_iso(as_of: date) -> str:
    return f"{as_of.isoformat()}T00:00:00Z"


class FundamentalNavOracle:
    """Values assets from their fundamentals. Deterministic; no external calls."""

    def quote(self, asset: Asset, *, as_of: date | None = None) -> NavQuote:
        on = as_of or date.today()
        return NavQuote(
            asset_id=asset.id,
            nav_minor=_fundamental_nav_minor(asset, on),
            currency=asset.currency,
            as_of=_as_of_iso(on),
            source="manual",
        )


def _fetch_pyth_nav_minor(feed_id: str, hermes_url: str) -> int:
    """Read the latest price for ``feed_id`` from Pyth's Hermes endpoint.

    The price is normalized to minor units (value * 100). Used once an asset is
    tokenized and has a dedicated NAV feed.
    """
    import httpx

    resp = httpx.get(
        f"{hermes_url}/v2/updates/price/latest",
        params={"ids[]": feed_id},
        timeout=10,
    )
    resp.raise_for_status()
    parsed: dict[str, Any] = resp.json()
    price = parsed["parsed"][0]["price"]
    value = int(price["price"]) * (10 ** int(price["expo"]))
    return round(value * 100)


class PythNavOracle:
    """Reads NAV from Pyth feeds for mapped assets; falls back otherwise."""

    def __init__(self, *, hermes_url: str, feeds: dict[str, str], fallback: NavOracle) -> None:
        self._hermes_url = hermes_url
        self._feeds = feeds
        self._fallback = fallback

    def quote(self, asset: Asset, *, as_of: date | None = None) -> NavQuote:
        feed_id = self._feeds.get(asset.id)
        if feed_id is None:
            return self._fallback.quote(asset, as_of=as_of)
        on = as_of or date.today()
        return NavQuote(
            asset_id=asset.id,
            nav_minor=_fetch_pyth_nav_minor(feed_id, self._hermes_url),
            currency=asset.currency,
            as_of=_as_of_iso(on),
            source="pyth",
        )


def get_nav_oracle(settings: Settings | None = None) -> NavOracle:
    """Build the configured NAV oracle (defaults to the fundamental valuation)."""
    settings = settings or get_settings()
    fundamental = FundamentalNavOracle()
    if settings.nav_oracle_provider == "pyth":
        return PythNavOracle(
            hermes_url=settings.pyth_hermes_url,
            feeds=settings.pyth_feeds,
            fallback=fundamental,
        )
    # "switchboard" is a future provider; until it lands, value fundamentally.
    return fundamental
