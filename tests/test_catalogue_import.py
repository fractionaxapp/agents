"""Transforming an rwa.xyz asset-screener export into deals."""

from __future__ import annotations

import pytest

from fractionax_agents.catalogue_import import (
    CatalogueImportError,
    asset_to_deal,
    parse_screener_payload,
)

# A minimal asset shaped like the rwa.xyz export.
_ASSET = {
    "id": 51,
    "name": "Some Treasury Fund",
    "asset_class": {"slug": "us-treasury-debt", "name": "US Treasury Debt"},
    "jurisdiction": {"name": "El Salvador"},
    "minInvestment": {"amount": 100000, "currency": "USD"},
    "subscriptionAssets": ["USD"],
    "stats": {"return": 3.374, "aum": 3107314245.77, "inception": "5/1/2023"},
}


def test_asset_maps_to_deal() -> None:
    deal = asset_to_deal(_ASSET)
    assert deal is not None
    assert deal.id == "deal_rwa_51"
    assert deal.asset_id == "ast_rwa_51"
    assert deal.title == "Some Treasury Fund"
    assert deal.jurisdiction == "SV"  # El Salvador -> SV
    assert deal.currency == "USD"
    assert deal.min_investment_minor == 10_000_000  # 100000 * 100
    assert deal.target_raise_minor == round(3107314245.77 * 100)
    assert deal.projected_yield_pct == 3.37  # rounded return
    assert deal.risk_tier == "low"  # us-treasury-debt base tier
    assert deal.asset_class == "us-treasury-debt"
    assert deal.sourced_at == "2023-05-01T00:00:00.000Z"


def test_unknown_jurisdiction_defaults_to_us() -> None:
    a = {**_ASSET, "jurisdiction": {"name": "Atlantis"}}
    assert asset_to_deal(a).jurisdiction == "US"


def test_missing_min_and_aum_get_floors() -> None:
    a = {
        "id": 9,
        "name": "No Numbers",
        "asset_class": {"slug": "real-estate"},
        "jurisdiction": {"name": "Singapore"},
        "stats": {},
    }
    deal = asset_to_deal(a)
    assert deal is not None
    assert deal.min_investment_minor > 0
    assert deal.target_raise_minor > 0
    assert deal.projected_yield_pct == 0.0
    assert deal.risk_tier == "medium"  # real-estate base tier


def test_asset_without_essentials_is_skipped() -> None:
    assert asset_to_deal({"id": 1, "name": "No class"}) is None
    assert asset_to_deal({"name": "No id", "asset_class": {"slug": "stocks"}}) is None


def test_parse_accepts_pageprops_and_bare_shapes() -> None:
    assert len(parse_screener_payload({"pageProps": {"assets": [_ASSET]}})) == 1
    assert len(parse_screener_payload({"assets": [_ASSET]})) == 1
    assert len(parse_screener_payload([_ASSET])) == 1


def test_parse_rejects_unrecognized_payload() -> None:
    with pytest.raises(CatalogueImportError):
        parse_screener_payload({"nope": True})


def test_parse_rejects_empty_result() -> None:
    with pytest.raises(CatalogueImportError):
        parse_screener_payload([{"id": 1, "name": "x"}])  # no asset_class -> skipped -> empty
