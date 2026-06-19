from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ANTHROPIC_API_KEY — required at runtime to call Claude.
    anthropic_api_key: str | None = None
    # Default to Anthropic's most capable model for agentic work.
    agent_model: str = "claude-opus-4-8"
    max_tokens: int = 8192
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
