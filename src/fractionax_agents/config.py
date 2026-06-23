from __future__ import annotations

from functools import lru_cache

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

    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
