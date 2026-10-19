"""GroqProvider behaviour against a fake OpenAI-compatible client. No network calls."""

import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.llm import (
    ImageInput,
    LLMOutputError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMUnavailableError,
    LLMUsage,
    capture_usage,
    create_provider,
)
from app.core.llm import groq_provider as module
from app.core.llm.groq_provider import GroqProvider, drop_optional_nulls, to_strict_schema
from app.core.llm.rate_limiter import RateLimiter
from app.core.schemas import Claim, Document

VALID_CLAIM = {"text": "Rent is due on the 5th.", "clause_ids": ["c-002"], "legal_refs": [], "confidence": 0.8, "verified": False}
OK_HEADERS = {
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7000",
    "x-ratelimit-reset-tokens": "7.5s",
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-reset-requests": "1m26.4s",
}


def completion(payload: Any = VALID_CLAIM, *, finish_reason: str = "stop", refusal: str | None = None) -> SimpleNamespace:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        model="openai/gpt-oss-120b",
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )


class FakeRaw:
    def __init__(self, result: SimpleNamespace, headers: dict[str, str]) -> None:
        self._result = result
        self.headers = httpx2.Headers(headers)

    def parse(self) -> SimpleNamespace:  # synchronous, like the SDK's legacy raw response
        return self._result


def status_error(cls: type[openai.APIStatusError], status: int, headers: dict[str, str] | None = None, body: Any = None):
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=request, headers=headers or {}), body=body)


class FakeClient:
    """Mimics ``AsyncOpenAI().chat.completions.with_raw_response.create``."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))
        )

    async def _create(self, **kwargs: Any) -> FakeRaw:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, FakeRaw):
            return outcome
        return FakeRaw(outcome, OK_HEADERS)


class RecordingLimiter(RateLimiter):
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.now = 0.0

        async def sleep(seconds: float) -> None:
            self.sleeps.append(round(seconds, 2))
            self.now += seconds

        super().__init__(requests_per_minute=30, max_wait_seconds=90, clock=lambda: self.now, sleep=sleep)


@pytest.fixture
def groq_settings(settings: Settings) -> Settings:
    settings.llm_provider = "groq"
    settings.groq_api_key = SecretStr("test-key-not-real")
    settings.groq_text_model = "openai/gpt-oss-120b"
    settings.groq_vision_model = "qwen/qwen3.8-27b"
    settings.groq_agent_models = {"classifier": "openai/gpt-oss-20b"}
    settings.groq_max_output_tokens = 2048
    return settings


@pytest.fixture
def backoff_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: high)
    return recorded


def provider(settings: Settings, client: FakeClient, limiter: RateLimiter | None = None, agent: str | None = None):
    return GroqProvider(settings, agent=agent, client=client, limiter=limiter or RecordingLimiter())


# --- requests & parsing --------------------------------------------------------------


async def test_usage_is_reported_to_capture_including_corrective_retry(groq_settings: Settings) -> None:
    client = FakeClient(completion({"text": "missing confidence"}), completion())

    with capture_usage() as usage:
        await provider(groq_settings, client).generate_structured("p", Claim)

    assert usage == [LLMUsage(input_tokens=120, output_tokens=30, model="openai/gpt-oss-120b")] * 2



async def test_returns_validated_object_with_strict_schema(groq_settings: Settings) -> None:
    client = FakeClient(completion())

    claim = await provider(groq_settings, client).generate_structured("Explain the rent clause", Claim)

    assert claim == Claim(**VALID_CLAIM)
    request = client.requests[0]
    assert request["model"] == "openai/gpt-oss-120b"
    assert request["max_completion_tokens"] == 2048  # capped from the 16000 default
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "Claim"
    assert request["messages"] == [{"role": "user", "content": "Explain the rent clause"}]


async def test_context_is_prepended_to_prompt(groq_settings: Settings) -> None:
    client = FakeClient(completion())

    await provider(groq_settings, client).generate_structured("Who is the tenant?", Claim, context="DOC TEXT", system="Be careful.")

    assert client.requests[0]["messages"] == [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "DOC TEXT\n\nWho is the tenant?"},
    ]
    assert "cache_control" not in json.dumps(client.requests[0])


async def test_images_go_to_vision_model(groq_settings: Settings) -> None:
    client = FakeClient(completion())
    page = ImageInput(data=b"page", media_type="image/png")
    photo = ImageInput(data=b"photo", media_type="image/jpeg")

    await provider(groq_settings, client).generate_structured("Read it", Claim, context=["DOC", page], images=[photo])

    request = client.requests[0]
    assert request["model"] == "qwen/qwen3.8-27b"
    content = request["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["text"] == "DOC\n\nRead it"


async def test_images_without_vision_model_raise_clear_error(groq_settings: Settings) -> None:
    groq_settings.groq_vision_model = None
    client = FakeClient()

    with pytest.raises(LLMRequestError, match="GROQ_VISION_MODEL is not set"):
        await provider(groq_settings, client).generate_structured(
            "Read it", Claim, images=[ImageInput(data=b"x", media_type="image/png")]
        )
    assert client.requests == []


async def test_more_than_three_images_rejected(groq_settings: Settings) -> None:
    images = [ImageInput(data=b"x", media_type="image/png")] * 4

    with pytest.raises(LLMRequestError, match="at most 3 images"):
        await provider(groq_settings, FakeClient()).generate_structured("Read", Claim, images=images)


async def test_per_agent_model_override(groq_settings: Settings) -> None:
    client = FakeClient(completion(), completion())

    await provider(groq_settings, client, agent="classifier").generate_structured("q", Claim)
    await provider(groq_settings, client, agent="explainer").generate_structured("q", Claim)

    assert [r["model"] for r in client.requests] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def test_factory_selects_groq_with_agent(groq_settings: Settings) -> None:
    llm = create_provider(groq_settings, agent="classifier")

    assert isinstance(llm, GroqProvider)
    assert llm.model == "openai/gpt-oss-20b"


def test_factory_without_groq_key_raises(groq_settings: Settings) -> None:
    groq_settings.groq_api_key = None

    with pytest.raises(LLMRequestError, match="GROQ_API_KEY"):
        create_provider(groq_settings)


async def test_logs_model_latency_and_usage(groq_settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=module.__name__):
        await provider(groq_settings, FakeClient(completion())).generate_structured("q", Claim)

    record = next(r for r in caplog.records if r.msg == "llm.call")
    assert (record.provider, record.model, record.schema) == ("groq", "openai/gpt-oss-120b", "Claim")
    assert (record.input_tokens, record.output_tokens) == (120, 30)
    assert record.latency_ms >= 0
    assert record.ratelimit_tokens_remaining == 7000


# --- strict schema --------------------------------------------------------------------


def test_strict_schema_closes_objects_and_requires_all_fields() -> None:
    schema = to_strict_schema(Document)

    def objects(node: Any):
        if isinstance(node, dict):
            if "properties" in node:
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from objects(value)

    found = list(objects(schema))
    assert len(found) >= 3  # Document, Page, Clause
    for node in found:
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    assert "default" not in json.dumps(schema)
    # A defaulted list becomes nullable rather than optional.
    assert {"type": "null"} in schema["properties"]["clauses"]["anyOf"]


def test_optional_nulls_are_dropped_so_defaults_apply() -> None:
    raw = {"text": "x", "clause_ids": None, "legal_refs": ["s.106"], "confidence": 0.5, "verified": None}

    cleaned = drop_optional_nulls(raw, Claim.model_json_schema())

    assert Claim.model_validate(cleaned) == Claim(text="x", legal_refs=["s.106"], confidence=0.5)


def test_nested_nulls_are_dropped_through_refs() -> None:
    raw = {
        "filename": "lease.pdf",
        "id": None,
        "doc_type": None,
        "language": None,
        "created_at": None,
        "pages": [{"number": 1, "lines": None}],
        "clauses": None,
    }

    doc = Document.model_validate(drop_optional_nulls(raw, Document.model_json_schema()))

    assert doc.pages[0].lines == []
    assert doc.clauses == []
    assert doc.id  # default factory ran


async def test_strict_null_output_validates(groq_settings: Settings) -> None:
    client = FakeClient(completion({"text": "x", "clause_ids": None, "legal_refs": None, "confidence": 0.4, "verified": None}))

    claim = await provider(groq_settings, client).generate_structured("q", Claim)

    assert claim == Claim(text="x", confidence=0.4)


# --- retries, backoff, rate limiting ------------------------------------------------------


async def test_429_honours_retry_after_via_limiter(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    limiter = RecordingLimiter()
    client = FakeClient(status_error(openai.RateLimitError, 429, {"retry-after": "6"}), completion())

    claim = await provider(groq_settings, client, limiter).generate_structured("q", Claim)

    assert claim.text == VALID_CLAIM["text"]
    assert limiter.sleeps == [6.0]
    assert backoff_sleeps == []  # no extra exponential backoff on top of retry-after


async def test_5xx_and_connection_errors_use_exponential_backoff(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = FakeClient(
        status_error(openai.InternalServerError, 500),
        openai.APIConnectionError(request=request),
        status_error(openai.RateLimitError, 429),  # no retry-after header
        completion(),
    )

    await provider(groq_settings, client).generate_structured("q", Claim)

    assert backoff_sleeps == [1.0, 2.0, 4.0]


async def test_rate_limit_exhausted_raises(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    client = FakeClient(*[status_error(openai.RateLimitError, 429) for _ in range(4)])

    with pytest.raises(LLMRateLimitError):
        await provider(groq_settings, client).generate_structured("q", Claim)


async def test_unavailable_after_retries(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    client = FakeClient(*[status_error(openai.InternalServerError, 503) for _ in range(4)])

    with pytest.raises(LLMUnavailableError):
        await provider(groq_settings, client).generate_structured("q", Claim)


async def test_bad_request_not_retried_and_message_surfaced(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    body = {"error": {"message": "model does not support strict mode"}}
    client = FakeClient(status_error(openai.BadRequestError, 400, body=body))

    with pytest.raises(LLMRequestError, match="does not support strict mode"):
        await provider(groq_settings, client).generate_structured("q", Claim)
    assert len(client.requests) == 1


JSON_FAILED = {
    "error": {
        "message": "Failed to generate JSON. Please adjust your prompt.",
        "code": "json_validate_failed",
        "failed_generation": "partial model output that may contain document text",
    }
}


async def test_json_generation_failure_is_retried(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    client = FakeClient(status_error(openai.BadRequestError, 400, body=JSON_FAILED), completion())

    claim = await provider(groq_settings, client).generate_structured("q", Claim)

    assert claim == Claim(**VALID_CLAIM)
    assert len(client.requests) == 2


async def test_any_failed_generation_400_is_retried(groq_settings: Settings, backoff_sleeps: list[float]) -> None:
    body = {"error": {"message": "Parsing failed. The model generated output that could not be parsed.",
                      "code": "some_new_code", "failed_generation": "..."}}
    client = FakeClient(status_error(openai.BadRequestError, 400, body=body), completion())

    assert await provider(groq_settings, client).generate_structured("q", Claim) == Claim(**VALID_CLAIM)
    assert len(client.requests) == 2


async def test_json_generation_failure_exhausted_is_an_output_error(
    groq_settings: Settings, backoff_sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(*[status_error(openai.BadRequestError, 400, body=JSON_FAILED) for _ in range(4)])

    with pytest.raises(LLMOutputError, match="valid JSON after 4 attempts"):
        await provider(groq_settings, client).generate_structured("q", Claim)
    assert "partial model output" not in caplog.text  # failed_generation is never logged


async def test_reasoning_effort_sent_to_gpt_oss_models_only(groq_settings: Settings) -> None:
    client = FakeClient(completion(), completion())
    image = ImageInput(data=b"fake-png-bytes", media_type="image/png")

    await provider(groq_settings, client).generate_structured("q", Claim)
    await provider(groq_settings, client).generate_structured("q", Claim, images=[image])

    assert client.requests[0]["reasoning_effort"] == "low"
    assert client.requests[1]["model"] == "qwen/qwen3.8-27b" and "reasoning_effort" not in client.requests[1]


async def test_reasoning_effort_can_be_turned_off(groq_settings: Settings) -> None:
    groq_settings.groq_reasoning_effort = ""
    client = FakeClient(completion())

    await provider(groq_settings, client).generate_structured("q", Claim)

    assert "reasoning_effort" not in client.requests[0]


async def test_limiter_waits_using_headers_from_previous_response(groq_settings: Settings) -> None:
    limiter = RecordingLimiter()
    low = {**OK_HEADERS, "x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "12s"}
    client = FakeClient(FakeRaw(completion(), low), completion())
    llm = provider(groq_settings, client, limiter)

    await llm.generate_structured("q", Claim)
    await llm.generate_structured("q", Claim)  # needs ~2048+ tokens; only 500 left

    assert limiter.sleeps == [12.0]


async def test_shared_scope_uses_one_bucket(groq_settings: Settings) -> None:
    groq_settings.groq_rate_limit_scope = "shared"
    limiter = RecordingLimiter()
    low = {**OK_HEADERS, "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "5s"}
    client = FakeClient(FakeRaw(completion(), low), completion())

    await provider(groq_settings, client, limiter, agent="explainer").generate_structured("q", Claim)
    await provider(groq_settings, client, limiter, agent="classifier").generate_structured("q", Claim)

    assert limiter.sleeps == [5.0]


# --- corrective retry, refusals, truncation ------------------------------------------------


async def test_invalid_output_gets_one_corrective_retry(groq_settings: Settings) -> None:
    client = FakeClient(completion({**VALID_CLAIM, "confidence": 7}), completion())

    claim = await provider(groq_settings, client).generate_structured("q", Claim)

    assert claim.confidence == 0.8
    retry = client.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "failed validation" in retry[2]["content"]


async def test_invalid_output_twice_raises(groq_settings: Settings) -> None:
    bad = completion("not json")

    with pytest.raises(LLMOutputError, match="not valid JSON"):
        await provider(groq_settings, FakeClient(bad, completion("still not json"))).generate_structured("q", Claim)


async def test_refusal_raises(groq_settings: Settings) -> None:
    client = FakeClient(completion("", refusal="I can't help with that."))

    with pytest.raises(LLMRefusalError):
        await provider(groq_settings, client).generate_structured("q", Claim)


async def test_truncated_output_raises_without_retry(groq_settings: Settings) -> None:
    client = FakeClient(completion('{"text": "Rent', finish_reason="length"))

    with pytest.raises(LLMOutputError, match="truncated"):
        await provider(groq_settings, client).generate_structured("q", Claim)
    assert len(client.requests) == 1

