"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 500
    DATABASE_URL: str
    CONFIDENCE_THRESHOLD: float = 0.85
    LOG_LEVEL: str = "INFO"
    PROMPT_VERSION: str = "v1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()  # type: ignore[call-arg]
