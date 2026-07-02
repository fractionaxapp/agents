"""Import a deal catalogue from an rwa.xyz asset-screener export.

Accepts the Next.js ``_next/data/.../asset-screener.json`` payload (or a bare
``{assets: [...]}`` / list of assets) and transforms each asset into a ``Deal``.
The lookup tables below (country name → ISO 3166-1 alpha-2, asset-class slug →
base risk tier) were derived from the existing seed catalogue so imports stay
consistent with it; the base risk tier is refined per-deal at read time by
``deals._refine_risk`` exactly like the seed.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from fractionax_core import Deal
from fractionax_core.domain import RiskTier

# rwa.xyz jurisdiction display name -> ISO 3166-1 alpha-2. "N/A"/"-"/unknown fall
# back to US (matches how the seed treated undisclosed jurisdictions).
COUNTRY_ISO2: dict[str, str] = {
    "El Salvador": "SV",
    "Bermuda": "BM",
    "United States of America": "US",
    "British Virgin Islands": "VG",
    "United Kingdom": "GB",
    "Cayman Islands": "KY",
    "France": "FR",
    "Luxembourg": "LU",
    "Hong Kong": "HK",
    "Republic of the Marshall Islands": "MH",
    "Panama": "PA",
    "Brazil": "BR",
    "Germany": "DE",
    "Jersey, Channel Islands": "JE",
    "Singapore": "SG",
    "Spain": "ES",
    "Canada": "CA",
    "Seychelles": "SC",
    "Ireland": "IE",
    "Malta": "MT",
    "Kyrgyzstan": "KG",
    "Australia": "AU",
    "United Arab Emirates": "AE",
    "Switzerland": "CH",
    "Netherlands": "NL",
    "Liechtenstein": "LI",
    "Mexico": "MX",
    "Iceland": "IS",
    "Cook Islands": "CK",
    "Turkey": "TR",
    "South Korea": "KR",
    "Finland": "FI",
    "Republic of Georgia": "GE",
}
_DEFAULT_JURISDICTION = "US"

# asset-class slug -> base risk tier (class-anchored; refined per deal at read time).
CLASS_RISK: dict[str, RiskTier] = {
    "stablecoins": "low",
    "us-treasury-debt": "low",
    "non-us-government-debt": "low",
    "active-strategies": "medium",
    "asset-backed-credit": "medium",
    "commodities": "medium",
    "corporate-credit": "medium",
    "diversified-credit": "medium",
    "real-estate": "medium",
    "specialty-finance": "medium",
    "private-equity": "high",
    "stocks": "high",
    "venture-capital": "high",
}
_DEFAULT_RISK: RiskTier = "medium"

_MIN_INVESTMENT_FLOOR_MINOR = 100_000  # $1,000 when the source discloses no minimum
_DEFAULT_SOURCED_AT = "2024-01-01T00:00:00.000Z"


class CatalogueImportError(ValueError):
    """Raised when the payload isn't a recognizable asset-screener export."""


def _extract_assets(payload: Any) -> list[dict[str, Any]]:
    """Pull the assets array out of the various shapes the export can take."""
    if isinstance(payload, list):
        assets = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("pageProps"), dict):
            assets = payload["pageProps"].get("assets")
        elif "assets" in payload:
            assets = payload["assets"]
        else:
            assets = None
    else:
        assets = None
    if not isinstance(assets, list):
        raise CatalogueImportError(
            "Expected an rwa.xyz asset-screener export with a pageProps.assets (or assets) array."
        )
    return [a for a in assets if isinstance(a, dict)]


def _iso_date(inception: Any) -> str:
    """Convert an rwa.xyz 'M/D/YYYY' inception date to an ISO timestamp."""
    if not isinstance(inception, str):
        return _DEFAULT_SOURCED_AT
    parts = inception.split("/")
    if len(parts) != 3:
        return _DEFAULT_SOURCED_AT
    try:
        month, day, year = (int(p) for p in parts)
        return f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"
    except ValueError:
        return _DEFAULT_SOURCED_AT


def _currency(asset: dict[str, Any]) -> str:
    min_inv = asset.get("minInvestment") or {}
    cur = min_inv.get("currency")
    if not cur:
        subs = asset.get("subscriptionAssets") or []
        cur = subs[0] if subs else None
    if isinstance(cur, str) and len(cur) == 3:
        return cur.upper()
    return "USD"


def _minor(amount: Any) -> int | None:
    if isinstance(amount, (int, float)) and amount > 0:
        return round(amount * 100)
    return None


def asset_to_deal(asset: dict[str, Any]) -> Deal | None:
    """Transform one rwa.xyz asset into a ``Deal``, or ``None`` if it lacks the
    essentials (id, name, asset class)."""
    aid = asset.get("id")
    name = asset.get("name")
    asset_class = asset.get("asset_class") or asset.get("assetClass") or {}
    slug = asset_class.get("slug") if isinstance(asset_class, dict) else None
    if aid is None or not name or not slug:
        return None

    jur = asset.get("jurisdiction") or {}
    jur_name = jur.get("name") if isinstance(jur, dict) else None
    jurisdiction = COUNTRY_ISO2.get(jur_name or "", _DEFAULT_JURISDICTION)

    stats = asset.get("stats") or {}
    min_inv = asset.get("minInvestment") or {}
    min_minor = _minor(min_inv.get("amount")) or _MIN_INVESTMENT_FLOOR_MINOR
    target_minor = _minor(stats.get("aum")) or min_minor

    ret = stats.get("return")
    projected_yield = round(float(ret), 2) if isinstance(ret, (int, float)) else 0.0

    return Deal(
        id=f"deal_rwa_{aid}",
        asset_id=f"ast_rwa_{aid}",
        title=str(name),
        jurisdiction=jurisdiction,
        currency=_currency(asset),
        min_investment_minor=min_minor,
        target_raise_minor=target_minor,
        projected_yield_pct=projected_yield,
        risk_tier=CLASS_RISK.get(slug, _DEFAULT_RISK),
        status="open",
        sourced_at=_iso_date(stats.get("inception")),
        asset_class=slug,
    )


def parse_screener_payload(payload: Any) -> list[Deal]:
    """Parse an asset-screener export into deals (base risk tiers; refined at read time)."""
    deals: list[Deal] = []
    for asset in _extract_assets(payload):
        deal = asset_to_deal(asset)
        if deal is not None:
            deals.append(deal)
    if not deals:
        raise CatalogueImportError("No importable deals found in the payload.")
    return deals


def fetch_payload(url: str, *, timeout: float = 30.0, max_bytes: int = 32 * 1024 * 1024) -> Any:
    """Fetch an asset-screener JSON from a URL (e.g. an rwa.xyz _next/data link)."""
    if not url.lower().startswith(("http://", "https://")):
        raise CatalogueImportError("URL must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": "FractionAX-Admin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (scheme checked above)
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise CatalogueImportError("Response too large.")
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise CatalogueImportError("URL did not return valid JSON.") from exc
