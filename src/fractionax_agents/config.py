from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ANTHROPIC_API_KEY — primary provider. Calls fail over to MiniMax when unset
    # or when Claude is unavailable.
    anthropic_api_key: str | None = None
    # Default to Anthropic's most capable model for agentic work.
    agent_model: str = "claude-opus-4-8"
    max_tokens: int = 8192

    # MINIMAX_API_KEY — fallback provider via MiniMax's OpenAI-compatible API.
    # Set it to enable failover (you can also run on MiniMax alone by leaving
    # ANTHROPIC_API_KEY unset).
    minimax_api_key: str | None = None
    minimax_model: str = "MiniMax-M2"
    minimax_base_url: str = "https://api.minimax.io/v1"

    # NAV oracle: 'fundamental' values assets from their cash flows (default);
    # 'pyth' reads live price feeds for tokenized assets; 'switchboard' is future.
    nav_oracle_provider: Literal["fundamental", "pyth", "switchboard"] = "fundamental"
    pyth_hermes_url: str = "https://hermes.pyth.network"
    pyth_feeds: dict[str, str] = {}  # asset_id -> Pyth price feed id

    host: str = "0.0.0.0"
    port: int = 8000

    # ADMIN_API_KEY — shared key the web dashboard sends (X-Admin-Key header) to
    # reach admin-only endpoints (investor directory, decision log). When unset,
    # admin endpoints are disabled (503) so they are never inadvertently open.
    admin_api_key: str | None = None
    # File that backs the admin store (investors + decision log). Defaults to
    # admin_store.json next to the package data.
    admin_store_path: str | None = None
    # Relational database for the deal catalogue + import snapshots (timeseries).
    # Postgres (Neon/Render) in deploy, e.g. postgresql://user:pass@host/db; when
    # unset, falls back to a local SQLite file (sqlite:///fractionax.db).
    database_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
