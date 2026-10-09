"""Application settings, read from environment variables and the repo-root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Beyond the Fine Print"
    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = Field(default="json", description="'json' or 'console'")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    llm_max_retries: int = Field(default=4, ge=0)
    llm_backoff_base_seconds: float = Field(default=1.0, gt=0)
    llm_backoff_max_seconds: float = Field(default=30.0, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    # Server-side refusal fallback (Claude API only): a declined request is re-run on
    # Anthropic's recommended fallback model instead of returning a refusal.
    llm_server_fallback: bool = True

    anthropic_api_key: SecretStr | None = None

    # Groq (OpenAI-compatible API)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_text_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str | None = None
    # Per-agent text model override, e.g. {"classifier": "openai/gpt-oss-20b"}.
    groq_agent_models: dict[str, str] = Field(default_factory=dict)
    # Completion tokens requested per call. Groq counts this toward the tokens-per-minute
    # limit, so it is capped well below the free-tier TPM.
    groq_max_output_tokens: int = Field(default=2048, gt=0)
    # Requests per minute is not reported in response headers, so it is configured here.
    groq_requests_per_minute: int = Field(default=30, gt=0)
    # The client-side limiter raises instead of waiting longer than this.
    groq_max_limiter_wait_seconds: float = Field(default=90.0, gt=0)
    # "model": each model has its own limits; "shared": one budget across all models.
    groq_rate_limit_scope: str = Field(default="model", pattern="^(model|shared)$")


@lru_cache
def get_settings() -> Settings:
    return Settings()
