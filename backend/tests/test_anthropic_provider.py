"""AnthropicProvider behaviour against a fake SDK client. No network calls."""

import json
import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from app.core.config import Settings
from app.core.llm import (
    ImageInput,
    LLMOutputError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMUnavailableError,
)
from app.core.llm import anthropic_provider as module
from app.core.llm.anthropic_provider import AnthropicProvider
from app.core.schemas import Claim

VALID_CLAIM = {"text": "Rent is due on the 5th.", "clause_ids": ["c-002"], "legal_refs": [], "confidence": 0.8, "verified": False}


def make_response(payload: Any = VALID_CLAIM, *, stop_reason: str = "end_turn", **usage: int) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 100),
            output_tokens=usage.get("output_tokens", 20),
            cache_read_input_tokens=usage.get("cache_read", 0),
            cache_creation_input_tokens=usage.get("cache_write", 0),
        ),
    )


def status_error(cls: type[anthropic.APIStatusError], status: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls(f"HTTP {status}", response=response, body=None)


class FakeClient:
    """Mimics ``AsyncAnthropic().messages.create``: returns or raises scripted outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: high)  # deterministic jitter
    return recorded


async def test_returns_validated_object_and_sends_schema(settings: Settings) -> None:
    client = FakeClient(make_response())
    provider = AnthropicProvider(settings, client=client)

    claim = await provider.generate_structured("Explain the rent clause", Claim)

    assert claim == Claim(**VALID_CLAIM)
    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "confidence" in request["output_config"]["format"]["schema"]["properties"]
    assert request["messages"][0]["content"][-1] == {"type": "text", "text": "Explain the rent clause"}


async def test_context_is_marked_for_prompt_caching(settings: Settings) -> None:
    client = FakeClient(make_response())
    provider = AnthropicProvider(settings, client=client)
    page_image = ImageInput(data=b"page-1-bytes", media_type="image/png")
    call_image = ImageInput(data=b"photo-bytes", media_type="image/jpeg")

    await provider.generate_structured(
        "List the parties",
        Claim,
        context=["DOCUMENT TEXT ...", page_image],
        images=[call_image],
        system="You are a careful assistant.",
    )

    content = client.requests[0]["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "image", "image", "text"]
    # Exactly one breakpoint, on the last context block; nothing after it is cached.
    assert "cache_control" not in content[0]
    assert content[1]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["source"]["media_type"] == "image/png"
    assert all("cache_control" not in block for block in content[2:])
    assert content[3]["text"] == "List the parties"


async def test_string_context_becomes_single_cached_block(settings: Settings) -> None:
    client = FakeClient(make_response())

    await AnthropicProvider(settings, client=client).generate_structured("q", Claim, context="DOC")

    content = client.requests[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "DOC", "cache_control": {"type": "ephemeral"}}


async def test_no_cache_marker_without_context(settings: Settings) -> None:
    client = FakeClient(make_response())

    await AnthropicProvider(settings, client=client).generate_structured("q", Claim)

    assert all("cache_control" not in block for block in client.requests[0]["messages"][0]["content"])


async def test_logs_model_latency_and_token_usage_including_cache(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(make_response(input_tokens=50, output_tokens=12, cache_read=4000, cache_write=0))

    with caplog.at_level(logging.INFO, logger=module.__name__):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim, context="DOC")

    record = next(r for r in caplog.records if r.msg == "llm.call")
    assert record.model == "claude-sonnet-5-5"
    assert record.schema == "Claim"
    assert record.latency_ms >= 0
    assert (record.input_tokens, record.output_tokens) == (50, 12)
    assert record.cache_read_input_tokens == 4000
    assert record.cache_creation_input_tokens == 0


async def test_retries_rate_limits_with_exponential_backoff(settings: Settings, sleeps: list[float]) -> None:
    client = FakeClient(
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.OverloadedError, 529),
        status_error(anthropic.InternalServerError, 500),
        make_response(),
    )

    claim = await AnthropicProvider(settings, client=client).generate_structured("q", Claim)

    assert claim.text == VALID_CLAIM["text"]
    assert len(client.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


async def test_backoff_honours_retry_after_and_cap(settings: Settings, sleeps: list[float]) -> None:
    client = FakeClient(
        status_error(anthropic.RateLimitError, 429, {"retry-after": "3"}),
        status_error(anthropic.RateLimitError, 429, {"retry-after": "120"}),
        make_response(),
    )

    await AnthropicProvider(settings, client=client).generate_structured("q", Claim)

    assert sleeps == [3.0, 8.0]  # 120s capped at llm_backoff_max_seconds


async def test_rate_limit_exhausted_raises(settings: Settings, sleeps: list[float]) -> None:
    client = FakeClient(*[status_error(anthropic.RateLimitError, 429) for _ in range(4)])

    with pytest.raises(LLMRateLimitError):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)
    assert len(client.requests) == settings.llm_max_retries + 1


async def test_connection_errors_exhausted_raise_unavailable(settings: Settings, sleeps: list[float]) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(*[anthropic.APIConnectionError(request=request) for _ in range(4)])

    with pytest.raises(LLMUnavailableError):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)


async def test_bad_request_is_not_retried(settings: Settings, sleeps: list[float]) -> None:
    client = FakeClient(status_error(anthropic.BadRequestError, 400))

    with pytest.raises(LLMRequestError):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)
    assert len(client.requests) == 1
    assert sleeps == []


async def test_invalid_output_gets_one_corrective_retry(settings: Settings) -> None:
    client = FakeClient(make_response({**VALID_CLAIM, "confidence": 7}), make_response())

    claim = await AnthropicProvider(settings, client=client).generate_structured("q", Claim, context="DOC")

    assert claim.confidence == 0.8
    retry_messages = client.requests[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert "failed validation" in retry_messages[2]["content"][0]["text"]
    # The cached context prefix is unchanged on the retry.
    assert retry_messages[0] == client.requests[0]["messages"][0]


async def test_invalid_output_twice_raises(settings: Settings) -> None:
    bad = {**VALID_CLAIM, "confidence": 7}
    client = FakeClient(make_response(bad), make_response(bad))

    with pytest.raises(LLMOutputError):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)


async def test_refusal_raises(settings: Settings) -> None:
    client = FakeClient(make_response("", stop_reason="refusal"))

    with pytest.raises(LLMRefusalError):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)


async def test_truncated_output_raises_without_retry(settings: Settings) -> None:
    client = FakeClient(make_response('{"text": "Rent is', stop_reason="max_tokens"))

    with pytest.raises(LLMOutputError, match="max_tokens"):
        await AnthropicProvider(settings, client=client).generate_structured("q", Claim)
    assert len(client.requests) == 1


async def test_server_fallback_toggle(settings: Settings) -> None:
    client = FakeClient(make_response(), make_response())
    await AnthropicProvider(settings, client=client).generate_structured("q", Claim)
    settings.llm_server_fallback = False
    await AnthropicProvider(settings, client=client).generate_structured("q", Claim)

    assert client.requests[0]["extra_body"] == {"fallbacks": "default"}
    assert client.requests[0]["extra_headers"] == {"anthropic-beta": module.SERVER_FALLBACK_BETA}
    assert "extra_body" not in client.requests[1]
