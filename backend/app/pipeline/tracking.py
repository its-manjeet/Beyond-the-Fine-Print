"""An ``LLMProvider`` wrapper that caches agent output and records every call in AGENT_RUN.

Agents receive this wrapper as their ``LLMProvider`` and never know about it. Each call is
cached under (document sha256, agent, prompt_version, model) plus a fingerprint of the
exact request, so an agent that makes several calls (batches, a corrective retry) gets
one entry per call, and re-running the same document is answered from the cache.
"""

import hashlib
import json
import time
from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import select

from app.core.llm import ContextPart, ImageInput, LLMProvider, capture_usage
from app.core.llm.base import T
from app.core.logging import get_logger
from app.storage.db import Database
from app.storage.models import AgentCache, AgentRun, utcnow

logger = get_logger(__name__)


class TrackedLLM(LLMProvider):
    def __init__(
        self,
        inner: LLMProvider,
        db: Database,
        *,
        document_id: str,
        document_sha256: str,
        agent: str,
        prompt_version: str,
        retention_hours: float,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self._db = db
        self._document_id = document_id
        self._sha = document_sha256
        self._agent = agent
        self._version = prompt_version
        self._retention = timedelta(hours=retention_hours)

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        images: Sequence[ImageInput] | None = None,
        context: str | Sequence[ContextPart] | None = None,
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> T:
        has_images = bool(images) or (context is not None and not isinstance(context, str) and any(
            isinstance(part, ImageInput) for part in context
        ))
        model = (getattr(self.inner, "vision_model", None) or self.model) if has_images else self.model
        key = self._cache_key(model, schema, prompt, images, context, system)
        start = time.perf_counter()

        cached = self._lookup(key)
        if cached is not None:
            result = schema.model_validate(cached)
            self._record(model, start, cached=True)
            logger.info("agent.cache_hit", extra={"agent": self._agent, "schema": schema.__name__})
            return result

        with capture_usage() as usage:
            try:
                result = await self.inner.generate_structured(
                    prompt, schema, images=images, context=context, system=system, max_tokens=max_tokens
                )
            except Exception as exc:
                self._record(_used_model(usage, model), start, usage=usage, error=type(exc).__name__)
                raise
        used_model = _used_model(usage, model)
        self._record(used_model, start, usage=usage)
        self._store(key, model, result.model_dump(mode="json"))
        return result

    # --- cache -----------------------------------------------------------------------

    def _cache_key(
        self,
        model: str,
        schema: type[T],
        prompt: str,
        images: Sequence[ImageInput] | None,
        context: str | Sequence[ContextPart] | None,
        system: str | None,
    ) -> str:
        parts: list[ContextPart] = [] if context is None else [context] if isinstance(context, str) else list(context)
        request = hashlib.sha256()
        for item in [system or "", prompt, schema.__name__, *parts, *(images or ())]:
            if isinstance(item, ImageInput):
                request.update(b"img:" + hashlib.sha256(item.data).digest())
            else:
                request.update(b"txt:" + item.encode("utf-8"))
            request.update(b"\x00")
        identity = json.dumps([self._sha, self._agent, self._version, model, request.hexdigest()])
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _lookup(self, key: str) -> dict | None:
        with self._db.session() as session:
            entry = session.get(AgentCache, key)
            if entry is None or entry.expires_at <= utcnow():
                return None
            return entry.output

    def _store(self, key: str, model: str, output: dict) -> None:
        now = utcnow()
        with self._db.session() as session:
            session.merge(
                AgentCache(
                    cache_key=key,
                    document_sha256=self._sha,
                    agent_name=self._agent,
                    prompt_version=self._version,
                    model=model,
                    output=output,
                    created_at=now,
                    expires_at=now + self._retention,
                )
            )

    # --- AGENT_RUN -------------------------------------------------------------------

    def _record(self, model: str, start: float, *, cached: bool = False, usage=(), error: str | None = None) -> None:
        with self._db.session() as session:
            session.add(
                AgentRun(
                    document_id=self._document_id,
                    agent_name=self._agent,
                    provider=self.name,
                    model=model,
                    prompt_version=self._version,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    input_tokens=sum(u.input_tokens for u in usage),
                    output_tokens=sum(u.output_tokens for u in usage),
                    cached=cached,
                    ok=error is None,
                    error=error,
                )
            )


def _used_model(usage: list, default: str) -> str:
    """The model the provider reports having used (it may differ, e.g. a fallback model)."""
    for item in reversed(usage):
        if item.model:
            return item.model
    return default


def agent_runs(db: Database, document_id: str) -> list[AgentRun]:
    with db.session() as session:
        return list(
            session.scalars(select(AgentRun).where(AgentRun.document_id == document_id).order_by(AgentRun.id)).all()
        )
