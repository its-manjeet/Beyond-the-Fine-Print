from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    llm_model: str
    llm_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    if settings.llm_provider == "groq":
        model, key = settings.groq_text_model, settings.groq_api_key
    else:
        model, key = settings.llm_model, settings.anthropic_api_key
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=settings.llm_provider,
        llm_model=model,
        llm_configured=bool(key and key.get_secret_value()),
    )
