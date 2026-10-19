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
    # Per-agent text model override. The small model is enough for classification and key
    # facts, and it has its own free-tier token budget, so it runs alongside the large one.
    groq_agent_models: dict[str, str] = Field(
        default_factory=lambda: {"classifier": "openai/gpt-oss-20b", "key_facts": "openai/gpt-oss-20b"}
    )
    # Completion tokens requested per call. Groq counts this toward the tokens-per-minute
    # limit, so it is capped well below the free-tier TPM.
    groq_max_output_tokens: int = Field(default=2048, gt=0)
    # Reasoning effort for gpt-oss models (low | medium | high; empty = provider default).
    # Reasoning tokens count toward the output cap above, so "low" leaves room for the answer.
    groq_reasoning_effort: str | None = Field(default="low", pattern="^(low|medium|high)?$")
    # Requests per minute is not reported in response headers, so it is configured here.
    groq_requests_per_minute: int = Field(default=30, gt=0)
    # The client-side limiter raises instead of waiting longer than this.
    groq_max_limiter_wait_seconds: float = Field(default=90.0, gt=0)
    # "model": each model has its own limits; "shared": one budget across all models.
    groq_rate_limit_scope: str = Field(default="model", pattern="^(model|shared)$")

    # Storage and retention. Documents and everything derived from them (including the
    # agent output cache) are deleted after ``retention_hours``.
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'app.db').as_posix()}"
    retention_hours: float = Field(default=24.0, gt=0)
    cleanup_interval_minutes: float = Field(default=30.0, gt=0)

    # Upload limits and ingestion
    max_upload_mb: int = Field(default=25, gt=0)
    max_pages: int = Field(default=20, gt=0)
    # A page with fewer characters than this is flagged as low confidence.
    low_text_chars: int = Field(default=40, ge=0)
    # Images are downscaled to this many pixels on the long side before vision OCR.
    ocr_max_image_side: int = Field(default=1600, gt=0)
    # Tesseract is only a fallback, used if this command is found on PATH.
    tesseract_cmd: str = "tesseract"

    # Agents
    classifier_min_confidence: float = Field(default=0.6, ge=0, le=1)
    # Risk score weights (w1..w4) for clause type, patterns, legality and money at stake.
    risk_weights: tuple[float, float, float, float] = (0.30, 0.50, 0.10, 0.10)
    # Approximate text budget per LLM call, in bytes of document text. Keeps each request
    # well under Groq's free-tier 8K tokens/minute; longer inputs are split into batches.
    llm_input_budget_bytes: int = Field(default=9000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
