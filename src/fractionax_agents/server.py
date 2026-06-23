from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fractionax_core import Deal, DealFilter
from pydantic import BaseModel

from .agent import run_agent
from .config import get_settings
from .copilot import CopilotResult, run_copilot
from .deals import source_deals

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
