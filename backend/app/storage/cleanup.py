"""Retention: delete documents, everything derived from them, and cached agent output."""

from datetime import datetime

from sqlalchemy import delete

from app.core.logging import get_logger
from app.storage.db import Database
from app.storage.models import AgentCache, Document, utcnow

logger = get_logger(__name__)


def purge_expired(db: Database, now: datetime | None = None) -> dict[str, int]:
    """Delete every document (and, by cascade, its derived rows) and cache entry past its expiry."""
    now = now or utcnow()
    with db.session() as session:
        documents = session.execute(delete(Document).where(Document.expires_at <= now)).rowcount
        cache = session.execute(delete(AgentCache).where(AgentCache.expires_at <= now)).rowcount
    if documents or cache:
        logger.info("retention.purged", extra={"documents": documents, "cache_entries": cache})
    return {"documents": documents, "cache_entries": cache}
