"""Anthropic (Claude) implementation of ``LLMProvider``.

This is the only module in the codebase allowed to import the Anthropic SDK;
``tests/test_no_direct_sdk_import.py`` enforces that.
"""

import asyncio
import base64
import random
import time
from collections.abc import Sequence
from typing import Any

import anthropic
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
)
from app.core.logging import get_logger

logger = get_logger(__name__)

CACHE_CONTROL = {"type": "ephemeral"}
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Status codes retried with exponential backoff: timeout, conflict, rate limit,
# and the 5xx family (500 internal, 503 unavailable, 504 deadline, 529 overloaded).
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_RATE_LIMIT_STATUS = frozenset({429, 529})


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.model = settings.llm_model
        self._settings = settings
        if client is None:
            api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
            # SDK retries are disabled so backoff, logging and attempt counts live here.
            client = anthropic.AsyncAnthropic(
                api_key=api_key,
                max_retries=0,
                timeout=settings.llm_timeout_seconds,
            )
        self._client = client

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
        content = [*_context_blocks(context), *(_image_block(img) for img in images or ()), _text_block(prompt)]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        request = self._build_request(messages, schema, system, max_tokens)

        response = await self._create_with_retries(request, schema)
        try:
            return self._parse(response, schema)
        except LLMOutputError as first_error:
            # One corrective retry: show the model its output and the validation error.
            if response.stop_reason != "end_turn":
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": [{"type": "text", "text": _response_text(response)}]},
                {
                    "role": "user",
                    "content": [
                        _text_block(
                            "That output failed validation against the required schema:\n"
                            f"{first_error}\nReturn corrected output that satisfies the schema."
                        )
                    ],
                },
            ]
            request = self._build_request(messages, schema, system, max_tokens)
            logger.warning("llm.output_invalid_retrying", extra={"schema": schema.__name__, "model": self.model})
            response = await self._create_with_retries(request, schema)
            return self._parse(response, schema)

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        system: str | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": {
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
            },
        }
        if system:
            request["system"] = system
        if self._settings.llm_server_fallback:
            request["extra_headers"] = {"anthropic-beta": SERVER_FALLBACK_BETA}
            request["extra_body"] = {"fallbacks": "default"}
        return request

    async def _create_with_retries(self, request: dict[str, Any], schema: type[T]) -> Any:
        max_attempts = self._settings.llm_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            try:
                response = await self._client.messages.create(**request)
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)  # None for connection errors/timeouts
                if status is not None and status not in _RETRYABLE_STATUS:
                    logger.error(
                        "llm.request_rejected",
                        extra={"provider": self.name, "model": self.model, "status": status},
                    )
                    raise LLMRequestError(f"LLM request rejected ({status}): {exc.message}") from exc
                if attempt == max_attempts:
                    logger.error(
                        "llm.call_failed",
                        extra={
                            "provider": self.name,
                            "model": self.model,
                            "attempts": attempt,
                            "error": type(exc).__name__,
                        },
                    )
                    if status in _RATE_LIMIT_STATUS:
                        raise LLMRateLimitError(f"Rate limited after {attempt} attempts") from exc
                    raise LLMUnavailableError(f"LLM unavailable after {attempt} attempts: {exc}") from exc
                delay = self._backoff_delay(attempt, exc)
                logger.warning(
                    "llm.retrying",
                    extra={
                        "provider": self.name,
                        "model": self.model,
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "status": status,
                        "delay_s": round(delay, 2),
                    },
                )
                await asyncio.sleep(delay)
                continue

            self._log_call(response, schema, attempt, (time.perf_counter() - start) * 1000)
            return response
        raise AssertionError("unreachable")

    def _backoff_delay(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with full jitter, honouring ``retry-after`` when sent."""
        cap = self._settings.llm_backoff_max_seconds
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), cap)
            except ValueError:
                pass
        exp = self._settings.llm_backoff_base_seconds * (2 ** (attempt - 1))
        return random.uniform(0, min(exp, cap))

    def _log_call(self, response: Any, schema: type[T], attempt: int, latency_ms: float) -> None:
        usage = _usage(response)
        logger.info(
            "llm.call",
            extra={
                "provider": self.name,
                "model": getattr(response, "model", self.model),
                "schema": schema.__name__,
                "latency_ms": round(latency_ms, 1),
                "attempt": attempt,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            },
        )

    def _parse(self, response: Any, schema: type[T]) -> T:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(f"Model declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMOutputError("Output was truncated at max_tokens; increase max_tokens")
        text = _response_text(response)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMOutputError(str(exc)) from exc


def _usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _response_text(response: Any) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: ImageInput) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": base64.standard_b64encode(image.data).decode("ascii"),
        },
    }


def _context_blocks(context: str | Sequence[ContextPart] | None) -> list[dict[str, Any]]:
    """Render shared context, with a cache breakpoint on its last block.

    The breakpoint caches everything up to and including the context (system prompt
    plus context), so later calls that differ only in images/prompt read it from cache.
    """
    if not context:
        return []
    parts: Sequence[ContextPart] = [context] if isinstance(context, str) else context
    blocks = [_image_block(p) if isinstance(p, ImageInput) else _text_block(p) for p in parts]
    blocks[-1]["cache_control"] = dict(CACHE_CONTROL)
    return blocks
