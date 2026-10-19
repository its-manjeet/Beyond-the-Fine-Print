"""Provider-neutral LLM interface.

Agents depend only on ``LLMProvider``. Concrete providers (Anthropic today; Gemini,
Groq, ... later) live next to this file and are selected by ``LLM_PROVIDER``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


@dataclass(frozen=True)
class ImageInput:
    """Raw image bytes, e.g. a phone photo or a rendered PDF page."""

    data: bytes
    media_type: ImageMediaType


ContextPart = str | ImageInput
"""One piece of shared context: a block of text or an image."""


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str | None = None


_usage_sink: ContextVar[list[LLMUsage] | None] = ContextVar("llm_usage_sink", default=None)


def report_usage(usage: LLMUsage) -> None:
    """Called by providers after every successful API call; a no-op unless captured."""
    sink = _usage_sink.get()
    if sink is not None:
        sink.append(usage)


@contextmanager
def capture_usage() -> Iterator[list[LLMUsage]]:
    """Collect the usage of every API call made inside the block (including retries).

    Context variables are per asyncio task, so concurrent agents each see only their own calls.
    """
    sink: list[LLMUsage] = []
    token = _usage_sink.set(sink)
    try:
        yield sink
    finally:
        _usage_sink.reset(token)


class LLMError(Exception):
    """Base class for every error raised by an LLM provider."""


class LLMRateLimitError(LLMError):
    """Rate limit / overload persisted after all retries."""


class LLMUnavailableError(LLMError):
    """Network failure, timeout or server error persisted after all retries."""


class LLMRequestError(LLMError):
    """The request was rejected (bad request, auth, permissions); retrying won't help."""


class LLMRefusalError(LLMError):
    """The model declined to answer."""


class LLMOutputError(LLMError):
    """The model's output could not be validated against the requested schema."""


class LLMProvider(ABC):
    """Generates validated structured output from a prompt."""

    name: str
    model: str

    @abstractmethod
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
        """Return an instance of ``schema`` produced by the model.

        Args:
            prompt: The task-specific instruction / question.
            schema: Pydantic model the output must validate against.
            images: Images specific to this call (sent after the context).
            context: Large content shared by many calls, such as the document text or
                its page images. It is sent before ``images`` and ``prompt``, so
                providers that support prompt caching can cache it. Pass it byte-for-byte
                identical across calls to get cache hits.
            system: Optional system instructions. Providers place the system prompt
                ahead of ``context``, so a different ``system`` per agent stops agents
                from sharing the cached context. For cross-agent reuse, keep ``system``
                the same (or ``None``) and put agent instructions in ``prompt``.
            max_tokens: Upper bound on generated tokens.

        Raises:
            LLMError: or one of its subclasses on failure.
        """
