from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fractionax_core import Deal, DealFilter, NavQuote
from pydantic import BaseModel

from .agent import run_agent
from .config import get_settings
from .copilot import CopilotResult, run_copilot, stream_copilot
from .deals import ASSETS_BY_ID, SEED_ASSETS, source_deals
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
