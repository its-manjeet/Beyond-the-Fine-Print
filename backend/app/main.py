import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app import __version__
from app.api import documents, health
from app.core.config import Settings, get_settings
from app.core.llm import get_llm_provider
from app.core.logging import configure_logging, get_logger
from app.pipeline.events import EventBus
from app.pipeline.orchestrator import Pipeline, ProviderFactory
from app.storage.cleanup import purge_expired
from app.storage.db import Database
from app.storage.models import Document

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, provider_for: ProviderFactory | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings (e.g. an in-memory database) and a mock provider."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    db = Database(settings.database_url)
    bus = EventBus()
    pipeline = Pipeline(db, bus, settings, provider_for or get_llm_provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.ensure_schema()
        _fail_interrupted(db)
        cleanup = asyncio.create_task(_cleanup_loop(db, settings.cleanup_interval_minutes * 60))
        try:
            yield
        finally:
            cleanup.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup
            await pipeline.shutdown()
            db.dispose()

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.bus = bus
    app.state.pipeline = pipeline
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    # The frontend's dev server proxies /api/* unchanged, so the same routes live there too.
    app.include_router(documents.router, prefix="/api", include_in_schema=False)

    logger.info(
        "app.created",
        extra={"environment": settings.environment, "llm_provider": settings.llm_provider, "llm_model": settings.llm_model},
    )
    return app


def _fail_interrupted(db: Database) -> None:
    """Documents left 'processing' by a previous run (e.g. a restart) will never finish."""
    with db.session() as session:
        for doc in session.scalars(select(Document).where(Document.status == "processing")):
            doc.status = "failed"
            doc.error = "Processing was interrupted. Please upload the document again."


async def _cleanup_loop(db: Database, interval_seconds: float) -> None:
    while True:
        try:
            purge_expired(db)
        except Exception as exc:  # keep the loop alive; the next run retries
            logger.error("retention.purge_failed", extra={"error": type(exc).__name__})
        await asyncio.sleep(interval_seconds)


app = create_app()
