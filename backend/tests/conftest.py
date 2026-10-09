import pytest

from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the developer's .env and environment."""
    return Settings(
        _env_file=None,
        llm_provider="anthropic",
        llm_model="claude-sonnet-5-5",
        anthropic_api_key="test-key-not-real",
        llm_max_retries=3,
        llm_backoff_base_seconds=1.0,
        llm_backoff_max_seconds=8.0,
    )
