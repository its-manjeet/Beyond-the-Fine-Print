import logging

import pytest

from app.core.llm import LLMRateLimitError, LLMRequestError
from app.core.llm.rate_limiter import RateLimiter, parse_reset


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 2))
        self.now += seconds


def make_limiter(clock: FakeClock, rpm: int = 30, max_wait: float = 90.0) -> RateLimiter:
    return RateLimiter(requests_per_minute=rpm, max_wait_seconds=max_wait, clock=clock, sleep=clock.sleep)


def headers(tokens_remaining: int, reset_tokens: str = "7.66s", requests_remaining: int = 990) -> dict[str, str]:
    return {
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": str(tokens_remaining),
        "x-ratelimit-reset-tokens": reset_tokens,
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": str(requests_remaining),
        "x-ratelimit-reset-requests": "2m59.56s",
    }


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0), ("500ms", 0.5), ("12", 12.0), ("soon", None), (None, None)],
)
def test_parse_reset(raw: str | None, seconds: float | None) -> None:
    assert parse_reset(raw) == (pytest.approx(seconds) if seconds is not None else None)


async def test_no_wait_before_any_headers_are_known() -> None:
    clock = FakeClock()

    await make_limiter(clock).acquire("m", 5000)

    assert clock.sleeps == []


async def test_waits_for_token_reset_when_budget_is_short(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.update_from_headers("m", headers(tokens_remaining=1000, reset_tokens="7.66s"))

    with caplog.at_level(logging.INFO):
        await limiter.acquire("m", 3000)

    assert clock.sleeps == [7.66]
    record = next(r for r in caplog.records if r.msg == "llm.rate_limiter_wait")
    assert (record.bucket, record.reason, record.wait_s) == ("m", "tokens_per_minute", 7.66)


async def test_reserves_tokens_so_back_to_back_requests_wait() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.update_from_headers("m", headers(tokens_remaining=8000, reset_tokens="30s"))

    await limiter.acquire("m", 5000)
    assert clock.sleeps == []
    await limiter.acquire("m", 5000)  # only 3000 left until reset

    assert clock.sleeps == [30.0]


async def test_buckets_are_independent() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.update_from_headers("model-a", headers(tokens_remaining=0, reset_tokens="30s"))

    await limiter.acquire("model-b", 5000)

    assert clock.sleeps == []


async def test_requests_per_minute_window() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=2)

    await limiter.acquire("m", 10)
    clock.now += 10
    await limiter.acquire("m", 10)
    await limiter.acquire("m", 10)

    assert clock.sleeps == [50.0]  # until the first request leaves the 60s window


async def test_daily_request_limit_longer_than_max_wait_raises() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, max_wait=90)
    limiter.update_from_headers("m", headers(tokens_remaining=8000, requests_remaining=0))

    with pytest.raises(LLMRateLimitError, match="requests_per_day"):
        await limiter.acquire("m", 10)


async def test_request_larger_than_tpm_limit_raises_immediately() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.update_from_headers("m", headers(tokens_remaining=8000))

    with pytest.raises(LLMRequestError, match="tokens-per-minute limit"):
        await limiter.acquire("m", 9000)


async def test_retry_after_blocks_bucket() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock)
    limiter.update_from_headers("m", headers(tokens_remaining=8000))

    limiter.note_retry_after("m", 4.0)
    await limiter.acquire("m", 100)

    assert clock.sleeps == [4.0]
