"""Client-side rate limiter driven by OpenAI-style ``x-ratelimit-*`` response headers.

Each bucket (one per model, or one shared bucket) tracks:
- tokens per minute: limit / remaining / reset, refreshed from every response's headers
- requests per day: limit / remaining / reset, refreshed from every response's headers
- requests per minute: a local sliding window (providers don't report RPM in headers)

Before a request is sent, ``acquire`` waits until the bucket can take it, so we stay
under the limits rather than relying on 429 retries.
"""

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from app.core.llm.base import LLMRateLimitError, LLMRequestError
from app.core.logging import get_logger

logger = get_logger(__name__)

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_reset(value: str | None) -> float | None:
    """Parse a reset duration such as ``7.66s``, ``2m59.56s``, ``1h2m``, ``500ms`` or ``12``."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        return None
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts)


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    raw = headers.get(name)
    try:
        return int(float(raw)) if raw is not None else None
    except ValueError:
        return None


@dataclass
class _Bucket:
    tokens_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_reset_at: float = 0.0
    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset_at: float = 0.0
    sent_at: deque[float] = field(default_factory=deque)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimiter:
    def __init__(
        self,
        *,
        requests_per_minute: int,
        max_wait_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._rpm = requests_per_minute
        self._max_wait = max_wait_seconds
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, _Bucket] = {}

    def _bucket(self, key: str) -> _Bucket:
        return self._buckets.setdefault(key, _Bucket())

    def snapshot(self, key: str) -> dict[str, int | float | None]:
        b = self._bucket(key)
        now = self._clock()
        return {
            "tokens_limit": b.tokens_limit,
            "tokens_remaining": b.tokens_remaining,
            "tokens_reset_in_s": max(0.0, b.tokens_reset_at - now),
            "requests_limit": b.requests_limit,
            "requests_remaining": b.requests_remaining,
            "requests_reset_in_s": max(0.0, b.requests_reset_at - now),
        }

    async def acquire(self, key: str, estimated_tokens: int) -> None:
        """Wait until ``key`` can accept a request of ``estimated_tokens``, then reserve it."""
        bucket = self._bucket(key)
        async with bucket.lock:
            if bucket.tokens_limit is not None and estimated_tokens > bucket.tokens_limit:
                raise LLMRequestError(
                    f"Request needs ~{estimated_tokens} tokens but the tokens-per-minute limit for "
                    f"'{key}' is {bucket.tokens_limit}; reduce the context or max output tokens"
                )
            while True:
                wait, reason = self._required_wait(bucket, estimated_tokens)
                if wait <= 0:
                    break
                if wait > self._max_wait:
                    raise LLMRateLimitError(
                        f"Rate limit for '{key}' ({reason}) resets in {wait:.0f}s, "
                        f"longer than the {self._max_wait:.0f}s maximum wait"
                    )
                logger.info(
                    "llm.rate_limiter_wait",
                    extra={"bucket": key, "reason": reason, "wait_s": round(wait, 2), "estimated_tokens": estimated_tokens},
                )
                await self._sleep(wait)

            now = self._clock()
            bucket.sent_at.append(now)
            if bucket.tokens_remaining is not None:
                bucket.tokens_remaining -= estimated_tokens
            if bucket.requests_remaining is not None:
                bucket.requests_remaining -= 1

    def _required_wait(self, bucket: _Bucket, estimated_tokens: int) -> tuple[float, str]:
        now = self._clock()

        # Refill once a reset time has passed.
        if bucket.tokens_remaining is not None and now >= bucket.tokens_reset_at:
            bucket.tokens_remaining = bucket.tokens_limit
        if bucket.requests_remaining is not None and now >= bucket.requests_reset_at:
            bucket.requests_remaining = bucket.requests_limit

        while bucket.sent_at and now - bucket.sent_at[0] >= 60.0:
            bucket.sent_at.popleft()

        waits: list[tuple[float, str]] = []
        if len(bucket.sent_at) >= self._rpm:
            waits.append((bucket.sent_at[0] + 60.0 - now, "requests_per_minute"))
        if bucket.tokens_remaining is not None and bucket.tokens_remaining < estimated_tokens:
            waits.append((bucket.tokens_reset_at - now, "tokens_per_minute"))
        if bucket.requests_remaining is not None and bucket.requests_remaining <= 0:
            waits.append((bucket.requests_reset_at - now, "requests_per_day"))
        if not waits:
            return 0.0, ""
        wait, reason = max(waits)
        return max(wait, 0.01), reason

    def update_from_headers(self, key: str, headers: Mapping[str, str]) -> None:
        """Make the provider's reported limits authoritative for ``key``."""
        bucket = self._bucket(key)
        now = self._clock()

        tokens_limit = _int_header(headers, "x-ratelimit-limit-tokens")
        tokens_remaining = _int_header(headers, "x-ratelimit-remaining-tokens")
        tokens_reset = parse_reset(headers.get("x-ratelimit-reset-tokens"))
        if tokens_limit is not None:
            bucket.tokens_limit = tokens_limit
        if tokens_remaining is not None:
            bucket.tokens_remaining = tokens_remaining
            bucket.tokens_reset_at = now + (tokens_reset if tokens_reset is not None else 60.0)

        requests_limit = _int_header(headers, "x-ratelimit-limit-requests")
        requests_remaining = _int_header(headers, "x-ratelimit-remaining-requests")
        requests_reset = parse_reset(headers.get("x-ratelimit-reset-requests"))
        if requests_limit is not None:
            bucket.requests_limit = requests_limit
        if requests_remaining is not None:
            bucket.requests_remaining = requests_remaining
            bucket.requests_reset_at = now + (requests_reset if requests_reset is not None else 86400.0)

    def note_retry_after(self, key: str, seconds: float) -> None:
        """After a 429, block the token budget until ``retry-after`` has elapsed.

        The server's ``retry-after`` takes precedence over any earlier reset estimate.
        """
        bucket = self._bucket(key)
        bucket.tokens_remaining = 0
        bucket.tokens_reset_at = self._clock() + seconds
