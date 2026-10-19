"""Groq implementation of ``LLMProvider`` via Groq's OpenAI-compatible API.

This is the only module in the codebase allowed to import the OpenAI SDK;
``tests/test_no_direct_sdk_import.py`` enforces that.

- Structured output: strict JSON-schema mode (constrained decoding). Pydantic schemas are
  rewritten to Groq's strict subset (see ``to_strict_schema``) and the response is
  validated with Pydantic. On validation failure there is one corrective retry.
- Images: only sent to ``GROQ_VISION_MODEL``; at most ``MAX_IMAGES`` per request.
- ``context``: prepended to the prompt as text. Groq has no explicit cache marker.
- Rate limits: a client-side limiter (shared per process) tracks requests and tokens per
  model from ``x-ratelimit-*`` headers and waits before sending a request that would
  exceed them. 429s are still retried, honouring ``retry-after``.
"""

import asyncio
import base64
import copy
import inspect
import json
import math
import random
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import openai
from pydantic import ValidationError

from app.core.config import Settings
from app.core.llm.base import (
    ContextPart,
    ImageInput,
    LLMOutputError,
    LLMProvider,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMUnavailableError,
    LLMUsage,
    T,
    report_usage,
)
from app.core.llm.rate_limiter import RateLimiter
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_IMAGES = 3
IMAGE_TOKENS = 2048  # Groq bills each image as 2048 input tokens
SHARED_BUCKET = "groq:shared"
# Only these models accept reasoning_effort low/medium/high.
REASONING_EFFORT_MODEL_PREFIX = "openai/gpt-oss"

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
# A 400 that carries "failed_generation" means the model's output could not be turned into
# valid JSON this time (codes seen include json_validate_failed and output parse failures).
# The same request usually succeeds when sent again.
_RETRYABLE_ERROR_CODES = frozenset({"json_validate_failed"})


@lru_cache
def _shared_limiter(requests_per_minute: int, max_wait_seconds: float) -> RateLimiter:
    """One limiter per process, so every provider instance shares the same budget."""
    return RateLimiter(requests_per_minute=requests_per_minute, max_wait_seconds=max_wait_seconds)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        settings: Settings,
        *,
        agent: str | None = None,
        client: Any | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self.model = settings.groq_agent_models.get(agent, settings.groq_text_model) if agent else settings.groq_text_model
        self.vision_model = settings.groq_vision_model or None
        if client is None:
            api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
            if not api_key:
                raise LLMRequestError("GROQ_API_KEY is not set")
            # SDK retries are disabled so backoff, rate limiting and logging live here.
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=settings.groq_base_url,
                max_retries=0,
                timeout=settings.llm_timeout_seconds,
            )
        self._client = client
        self._limiter = limiter or _shared_limiter(
            settings.groq_requests_per_minute, settings.groq_max_limiter_wait_seconds
        )

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
        context_text, context_images = _split_context(context)
        all_images = [*context_images, *(images or ())]
        model = self._select_model(all_images)

        text = f"{context_text}\n\n{prompt}" if context_text else prompt
        if all_images:
            user_content: Any = [*(_image_part(img) for img in all_images), {"type": "text", "text": text}]
        else:
            user_content = text
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        output_tokens = min(max_tokens, self._settings.groq_max_output_tokens)
        effort = self._reasoning_effort(model)
        request = _build_request(model, messages, schema, output_tokens, effort)
        estimate = _estimate_tokens(messages, len(all_images), output_tokens)

        message, finish_reason = await self._create_with_retries(request, schema, estimate)
        try:
            return _parse(message, finish_reason, schema)
        except LLMOutputError as first_error:
            # One corrective retry: show the model its output and the validation error.
            if finish_reason != "stop":
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": message.content or ""},
                {
                    "role": "user",
                    "content": "That output failed validation against the required schema:\n"
                    f"{first_error}\nReturn corrected output that satisfies the schema.",
                },
            ]
            request = _build_request(model, messages, schema, output_tokens, effort)
            estimate = _estimate_tokens(messages, len(all_images), output_tokens)
            logger.warning("llm.output_invalid_retrying", extra={"schema": schema.__name__, "model": model})
            message, finish_reason = await self._create_with_retries(request, schema, estimate)
            return _parse(message, finish_reason, schema)

    def _select_model(self, images: Sequence[ImageInput]) -> str:
        if not images:
            return self.model
        if not self.vision_model:
            raise LLMRequestError(
                "This request includes images but GROQ_VISION_MODEL is not set. "
                "Set GROQ_VISION_MODEL (e.g. qwen/qwen3.8-27b) in .env."
            )
        if len(images) > MAX_IMAGES:
            raise LLMRequestError(f"Groq accepts at most {MAX_IMAGES} images per request; got {len(images)}")
        return self.vision_model

    def _reasoning_effort(self, model: str) -> str | None:
        """Reasoning tokens count toward the output cap, so gpt-oss models are asked to keep it low."""
        effort = self._settings.groq_reasoning_effort
        return effort if effort and model.startswith(REASONING_EFFORT_MODEL_PREFIX) else None

    def _bucket(self, model: str) -> str:
        return SHARED_BUCKET if self._settings.groq_rate_limit_scope == "shared" else model

    async def _create_with_retries(self, request: dict[str, Any], schema: type[T], estimate: int) -> tuple[Any, str]:
        model = request["model"]
        bucket = self._bucket(model)
        max_attempts = self._settings.llm_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            await self._limiter.acquire(bucket, estimate)
            start = time.perf_counter()
            try:
                raw = await self._client.chat.completions.with_raw_response.create(**request)
            except (openai.APIConnectionError, openai.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)  # None for connection errors/timeouts
                response = getattr(exc, "response", None)
                if response is not None:
                    self._limiter.update_from_headers(bucket, response.headers)
                if status is not None and status not in _RETRYABLE_STATUS and not _retryable_code(exc):
                    logger.error(
                        "llm.request_rejected",
                        extra={
                            "provider": self.name,
                            "model": model,
                            "status": status,
                            "code": _error_body(exc).get("code"),
                            "error": _error_message(exc),
                        },
                    )
                    raise LLMRequestError(f"LLM request rejected ({status}): {_error_message(exc)}") from exc
                if attempt == max_attempts:
                    logger.error(
                        "llm.call_failed",
                        extra={"provider": self.name, "model": model, "attempts": attempt, "error": type(exc).__name__},
                    )
                    if status == 429:
                        raise LLMRateLimitError(f"Rate limited after {attempt} attempts") from exc
                    if _retryable_code(exc):
                        raise LLMOutputError(f"The model failed to produce valid JSON after {attempt} attempts") from exc
                    raise LLMUnavailableError(f"LLM unavailable after {attempt} attempts: {exc}") from exc

                retry_after = _retry_after(response)
                if status == 429 and retry_after is not None:
                    # The limiter blocks the bucket and acquire() waits it out (and logs it).
                    self._limiter.note_retry_after(bucket, retry_after)
                    delay = 0.0
                else:
                    delay = self._backoff_delay(attempt)
                logger.warning(
                    "llm.retrying",
                    extra={
                        "provider": self.name,
                        "model": model,
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "status": status,
                        "code": _error_body(exc).get("code"),
                        "retry_after_s": retry_after,
                        "delay_s": round(delay, 2),
                    },
                )
                if delay:
                    await asyncio.sleep(delay)
                continue

            latency_ms = (time.perf_counter() - start) * 1000
            self._limiter.update_from_headers(bucket, raw.headers)
            # The SDK's legacy raw response parses synchronously; newer ones return an awaitable.
            completion = raw.parse()
            if inspect.isawaitable(completion):
                completion = await completion
            self._log_call(completion, model, schema, attempt, latency_ms, bucket)
            choice = completion.choices[0]
            return choice.message, choice.finish_reason
        raise AssertionError("unreachable")

    def _backoff_delay(self, attempt: int) -> float:
        cap = self._settings.llm_backoff_max_seconds
        exp = self._settings.llm_backoff_base_seconds * (2 ** (attempt - 1))
        return random.uniform(0, min(exp, cap))

    def _log_call(self, completion: Any, model: str, schema: type[T], attempt: int, latency_ms: float, bucket: str) -> None:
        usage = getattr(completion, "usage", None)
        report_usage(
            LLMUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                model=getattr(completion, "model", None) or model,
            )
        )
        limits = self._limiter.snapshot(bucket)
        logger.info(
            "llm.call",
            extra={
                "provider": self.name,
                "model": getattr(completion, "model", model),
                "schema": schema.__name__,
                "latency_ms": round(latency_ms, 1),
                "attempt": attempt,
                "stop_reason": completion.choices[0].finish_reason,
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "ratelimit_tokens_remaining": limits["tokens_remaining"],
                "ratelimit_requests_remaining": limits["requests_remaining"],
            },
        )


def _build_request(
    model: str, messages: list[dict[str, Any]], schema: type[T], max_tokens: int, reasoning_effort: str | None = None
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": to_strict_schema(schema)},
        },
    }
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    return request


def _parse(message: Any, finish_reason: str | None, schema: type[T]) -> T:
    if getattr(message, "refusal", None):
        raise LLMRefusalError(f"Model declined the request: {message.refusal}")
    if finish_reason == "length":
        raise LLMOutputError("Output was truncated at the max output tokens; increase GROQ_MAX_OUTPUT_TOKENS")
    if finish_reason == "content_filter":
        raise LLMRefusalError("Model output was blocked by a content filter")
    try:
        data = json.loads(message.content or "")
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"Output is not valid JSON: {exc}") from exc
    try:
        return schema.model_validate(drop_optional_nulls(data, schema.model_json_schema()))
    except ValidationError as exc:
        raise LLMOutputError(str(exc)) from exc


# --- strict-schema support -------------------------------------------------------------


def to_strict_schema(schema: type[T]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON schema into Groq's strict-mode subset.

    Strict mode requires every property to be ``required`` and every object to set
    ``additionalProperties: false``. A property that was optional becomes nullable, and
    ``drop_optional_nulls`` removes those nulls before validation, so Pydantic applies the
    field's default.
    """
    result = copy.deepcopy(schema.model_json_schema())
    _make_strict(result)
    return result


def _make_strict(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _make_strict(item)
        return
    if not isinstance(node, dict):
        return
    node.pop("default", None)
    for key in ("$defs", "properties"):
        for child in node.get(key, {}).values():
            _make_strict(child)
    for key in ("items", "anyOf", "allOf", "oneOf", "additionalProperties"):
        if key in node:
            _make_strict(node[key])
    if node.get("type") == "object" or "properties" in node:
        properties: dict[str, Any] = node.setdefault("properties", {})
        required = set(node.get("required", []))
        for name, prop in properties.items():
            if name not in required and not _is_nullable(prop):
                properties[name] = {"anyOf": [prop, {"type": "null"}]}
        node["required"] = list(properties)
        node["additionalProperties"] = False


def _is_nullable(prop: dict[str, Any]) -> bool:
    if prop.get("type") == "null" or (isinstance(prop.get("type"), list) and "null" in prop["type"]):
        return True
    return any(option.get("type") == "null" for option in prop.get("anyOf", []))


def drop_optional_nulls(data: Any, schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:
    """Remove ``null`` values for properties that were optional in the original schema."""
    root = root or schema
    schema = _resolve(schema, root)
    if isinstance(data, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        cleaned = {}
        for key, value in data.items():
            if value is None and key in properties and key not in required:
                continue
            sub = properties.get(key)
            cleaned[key] = drop_optional_nulls(value, sub, root) if sub else value
        return cleaned
    if isinstance(data, list) and "items" in schema:
        return [drop_optional_nulls(item, schema["items"], root) for item in data]
    if "anyOf" in schema and data is not None:
        # Recurse into the first object-like branch (e.g. Optional[Model]).
        for option in schema["anyOf"]:
            resolved = _resolve(option, root)
            if "properties" in resolved or "items" in resolved:
                return drop_optional_nulls(data, resolved, root)
    return data


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), schema)
    return schema


# --- helpers --------------------------------------------------------------------------


def _split_context(context: str | Sequence[ContextPart] | None) -> tuple[str, list[ImageInput]]:
    if not context:
        return "", []
    parts: Sequence[ContextPart] = [context] if isinstance(context, str) else context
    texts = [p for p in parts if isinstance(p, str)]
    images = [p for p in parts if isinstance(p, ImageInput)]
    return "\n\n".join(texts), images


def _image_part(image: ImageInput) -> dict[str, Any]:
    data = base64.standard_b64encode(image.data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{data}"}}


def _estimate_tokens(messages: list[dict[str, Any]], image_count: int, output_tokens: int) -> int:
    """Conservative token estimate for the limiter (UTF-8 bytes / 3 covers Indic scripts)."""
    text_bytes = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
        else:
            text_bytes += sum(len(p["text"].encode("utf-8")) for p in content if p["type"] == "text")
    return math.ceil(text_bytes / 3) + image_count * IMAGE_TOKENS + output_tokens


def _retry_after(response: Any) -> float | None:
    if response is None:
        return None
    try:
        value = response.headers.get("retry-after")
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error_body(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _retryable_code(exc: Exception) -> bool:
    error = _error_body(exc)
    return error.get("code") in _RETRYABLE_ERROR_CODES or "failed_generation" in error


def _error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)
