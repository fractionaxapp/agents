from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fractionax_core import (
    ComplianceDecision,
    Deal,
    DealFilter,
    Investor,
    JurisdictionRule,
    NavQuote,
    TransferCheck,
)
from pydantic import BaseModel

from .admin_store import InvestorRecord, get_admin_store
from .agent import run_agent
from .catalogue_import import CatalogueImportError, fetch_payload, parse_screener_payload
from .compliance import check_transfer, jurisdiction_rules, verify_investor
from .config import get_settings
from .copilot import CopilotResult, run_copilot, stream_copilot
from .db import count_snapshots, snapshot_history
from .deals import (
    ASSETS_BY_ID,
    SEED_ASSETS,
    catalogue_source,
    clear_catalogue,
    get_catalogue,
    set_catalogue,
    source_deals,
)
from .oracle import get_nav_oracle

app = FastAPI(title="FractionAX Agents", version="0.0.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class CopilotRequest(BaseModel):
    message: str
    with_memo: bool = True


def _require_api_key() -> None:
    settings = get_settings()
    if not (settings.anthropic_api_key or settings.minimax_api_key):
        raise HTTPException(
            status_code=503,
            detail="Agent not configured: set ANTHROPIC_API_KEY and/or MINIMAX_API_KEY",
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    _require_api_key()
    return ChatResponse(reply=run_agent(request.message))


@app.post("/copilot", response_model=CopilotResult)
def copilot(request: CopilotRequest) -> CopilotResult:
    """Natural language -> structured intent -> matching deals -> investment memo."""
    _require_api_key()
    return run_copilot(request.message, with_memo=request.with_memo)


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/copilot/stream")
def copilot_stream(request: CopilotRequest) -> StreamingResponse:
    """Stream the Copilot pipeline stage by stage as Server-Sent Events.

    Emits `intent`, then `deals`, then `memo`, then `done` (or a terminal `error`).
    """
    _require_api_key()

    def gen() -> Iterator[str]:
        try:
            for event, payload in stream_copilot(request.message, with_memo=request.with_memo):
                yield _sse(event, payload)
        except Exception as exc:  # surface failures as a terminal SSE event
            yield _sse("error", {"error": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/deals", response_model=list[Deal])
def deals(
    jurisdiction: str | None = None,
    risk_tier: str | None = None,
    min_yield_pct: float | None = None,
) -> list[Deal]:
    """Deal discovery without the LLM — direct filtered access to the catalogue."""
    return source_deals(
        DealFilter(
            jurisdiction=jurisdiction,
            risk_tier=risk_tier,  # type: ignore[arg-type]
            min_yield_pct=min_yield_pct,
        )
    )


class ComplianceVerifyRequest(BaseModel):
    investor: Investor
    deal_id: str


class ComplianceTransferRequest(BaseModel):
    deal_id: str
    seller: Investor
    buyer: Investor


def _require_deal(deal_id: str) -> Deal:
    deal = next((d for d in get_catalogue() if d.id == deal_id), None)
    if deal is None:
        raise HTTPException(status_code=404, detail=f"Unknown deal: {deal_id}")
    return deal


@app.get("/compliance/rules", response_model=list[JurisdictionRule])
def compliance_rules() -> list[JurisdictionRule]:
    """The active jurisdiction-aware rules engine (one rule per exemption regime)."""
    return jurisdiction_rules()


@app.post("/compliance/verify", response_model=ComplianceDecision)
def compliance_verify(request: ComplianceVerifyRequest) -> ComplianceDecision:
    """Screen an investor (KYC/AML + sanctions) and decide eligibility for a deal.

    The ALLOW/DENY outcome is deterministic; the rationale narrative uses the LLM
    when a provider key is set and falls back to a deterministic sentence otherwise,
    so this endpoint works with no API key.

    Side effect: the decision is logged and the investor upserted into the admin
    store, so the super-admin directory and decision log populate from real usage.
    """
    decision = verify_investor(request.investor, _require_deal(request.deal_id))
    store = get_admin_store()
    store.upsert_investor(
        request.investor,
        kyc_status=decision.kyc_status,
        accreditation_tier=decision.accreditation_tier,
        updated_at=decision.decided_at,
    )
    store.log_decision(decision)
    return decision


@app.post("/compliance/transfer", response_model=TransferCheck)
def compliance_transfer(request: ComplianceTransferRequest) -> TransferCheck:
    """Secondary-market transfer restriction: may the buyer receive the deal's token?"""
    return check_transfer(_require_deal(request.deal_id), request.seller, request.buyer)


# --- Admin (super-admin dashboard) ------------------------------------------
# Gated by a shared key in the X-Admin-Key header. The web dashboard adds it
# server-side after its own team-allowlist session check; these endpoints are the
# off-chain data plane (investor directory + decision log).


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    configured = get_settings().admin_api_key
    if not configured:
        raise HTTPException(status_code=503, detail="Admin API disabled: set ADMIN_API_KEY")
    if x_admin_key != configured:
        raise HTTPException(status_code=401, detail="Invalid admin key")


class SetCredentialStatusRequest(BaseModel):
    status: str  # "issued" | "revoked" | "none"
    tx: str | None = None
    updated_at: str = "1970-01-01T00:00:00Z"


@app.get(
    "/admin/investors", response_model=list[InvestorRecord], dependencies=[Depends(require_admin)]
)
def admin_investors() -> list[InvestorRecord]:
    """The investor directory: everyone screened, with KYC/tier and credential state."""
    return get_admin_store().list_investors()


@app.post("/admin/investors", response_model=InvestorRecord, dependencies=[Depends(require_admin)])
def admin_add_investor(investor: Investor) -> InvestorRecord:
    """Manually add/screen an investor into the directory (no deal context — screens
    KYC/AML + accreditation only)."""
    from .compliance import MockKycProvider

    profile = MockKycProvider().screen(investor)
    return get_admin_store().upsert_investor(
        investor,
        kyc_status=profile.kyc_status,
        accreditation_tier=profile.accreditation_tier,
        updated_at=profile.screened_at,
    )


@app.post(
    "/admin/investors/{investor_id}/credential",
    response_model=InvestorRecord,
    dependencies=[Depends(require_admin)],
)
def admin_set_credential(investor_id: str, request: SetCredentialStatusRequest) -> InvestorRecord:
    """Mirror an on-chain credential issue/revoke onto the investor record (called by
    the web admin after the authority-signed transaction confirms)."""
    if request.status not in ("issued", "revoked", "none"):
        raise HTTPException(status_code=400, detail="status must be issued|revoked|none")
    record = get_admin_store().set_credential_status(
        investor_id,
        request.status,  # type: ignore[arg-type]
        tx=request.tx,
        updated_at=request.updated_at,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown investor: {investor_id}")
    return record


@app.get(
    "/admin/decisions",
    response_model=list[ComplianceDecision],
    dependencies=[Depends(require_admin)],
)
def admin_decisions(limit: int = 100) -> list[ComplianceDecision]:
    """The compliance decision log, most recent first."""
    return get_admin_store().list_decisions(limit=limit)


class CatalogueStatus(BaseModel):
    source: str  # "seed" | "database"
    count: int
    snapshots: int  # total timeseries points captured across all imports


class CatalogueImportRequest(BaseModel):
    # Provide exactly one: a URL to an rwa.xyz asset-screener export, or the parsed
    # JSON payload itself (from an uploaded file).
    url: str | None = None
    payload: dict | list | None = None


class CatalogueImportResult(BaseModel):
    imported: int
    source: str
    sample: list[Deal]


@app.get(
    "/admin/deals/catalogue",
    response_model=CatalogueStatus,
    dependencies=[Depends(require_admin)],
)
def admin_catalogue_status() -> CatalogueStatus:
    """Whether deals come from the seed snapshot or the imported database catalogue,
    how many deals, and how many timeseries snapshot points exist."""
    return CatalogueStatus(
        source=catalogue_source(),
        count=len(get_catalogue()),
        snapshots=count_snapshots(),
    )


@app.post(
    "/admin/deals/import",
    response_model=CatalogueImportResult,
    dependencies=[Depends(require_admin)],
)
def admin_import_catalogue(request: CatalogueImportRequest) -> CatalogueImportResult:
    """Replace the active deal catalogue from an rwa.xyz asset-screener export —
    either fetched from a URL or supplied inline as the parsed JSON of an upload."""
    if not request.url and request.payload is None:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'payload'.")
    try:
        payload = fetch_payload(request.url) if request.url else request.payload
        deals = parse_screener_payload(payload)
    except CatalogueImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"Couldn't fetch the URL: {exc}") from exc
    count = set_catalogue(deals)
    return CatalogueImportResult(imported=count, source=catalogue_source(), sample=deals[:5])


@app.post(
    "/admin/deals/reset",
    response_model=CatalogueStatus,
    dependencies=[Depends(require_admin)],
)
def admin_reset_catalogue() -> CatalogueStatus:
    """Clear all deals from the database (the catalogue goes empty). Snapshot history
    is retained."""
    clear_catalogue()
    return CatalogueStatus(
        source=catalogue_source(), count=len(get_catalogue()), snapshots=count_snapshots()
    )


@app.get("/admin/deals/history", dependencies=[Depends(require_admin)])
def admin_deal_history(deal_id: str, limit: int = 100) -> list[dict]:
    """A deal's snapshot timeseries (AUM / yield / minimum over successive imports),
    most recent first."""
    return snapshot_history(deal_id, limit=limit)


@app.get("/nav", response_model=list[NavQuote])
def nav(asset_id: str | None = None) -> list[NavQuote]:
    """NAV quotes from the pricing oracle, for one asset or the whole catalogue."""
    oracle = get_nav_oracle()
    if asset_id is not None:
        asset = ASSETS_BY_ID.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Unknown asset: {asset_id}")
        return [oracle.quote(asset)]
    return [oracle.quote(a) for a in SEED_ASSETS]
