"""In-process publish/subscribe for pipeline progress, consumed by the SSE endpoint."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def publish(self, document_id: str, event: dict[str, Any]) -> None:
        for queue in self._subscribers.get(document_id, ()):
            queue.put_nowait(event)

    @asynccontextmanager
    async def subscribe(self, document_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(document_id, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(document_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[document_id]
