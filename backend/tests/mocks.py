"""Test doubles shared by agent, pipeline and API tests. No network."""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.llm import ContextPart, ImageInput, LLMProvider, LLMUsage
from app.core.llm.base import report_usage

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = REPO_ROOT / "data" / "samples" / "rental_agreement_sample.pdf"
SAMPLE_PNG = REPO_ROOT / "data" / "samples" / "rental_page1.png"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# A response is a payload dict, an exception to raise, or a function of the call.
Response = dict[str, Any] | Exception | Callable[[dict[str, Any]], dict[str, Any]]


class MockProvider(LLMProvider):
    """Answers by schema name from canned responses and records every call.

    ``responses[schema_name]`` is one response (reused for every call) or a list consumed
    in order. Each answered call reports fixed token usage, like a real provider.
    """

    name = "mock"

    def __init__(self, responses: dict[str, Response | list[Response]], model: str = "mock-model") -> None:
        self.responses = responses
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        images: Sequence[ImageInput] | None = None,
        context: str | Sequence[ContextPart] | None = None,
        system: str | None = None,
        max_tokens: int = 16000,
    ):
        await asyncio.sleep(0)  # yield like a real network call, so concurrent agents interleave
        call = {"prompt": prompt, "schema": schema.__name__, "images": images, "context": context, "system": system}
        self.calls.append(call)
        if schema.__name__ not in self.responses:
            raise AssertionError(f"MockProvider has no response for {schema.__name__}")
        response = self.responses[schema.__name__]
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        payload = response(call) if callable(response) else response
        report_usage(LLMUsage(input_tokens=100, output_tokens=20, model=self.model))
        return schema.model_validate(payload)

    def calls_for(self, schema_name: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["schema"] == schema_name]


def sample_pdf_bytes() -> bytes:
    if SAMPLE_PDF.exists():
        return SAMPLE_PDF.read_bytes()
    from scripts.make_rental_sample import build_pdf

    return build_pdf()


def load_golden(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
