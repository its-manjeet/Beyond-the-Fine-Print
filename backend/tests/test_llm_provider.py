"""Contract tests for the LLMProvider interface, using an in-memory mock provider."""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.llm import (
    ContextPart,
    ImageInput,
    LLMError,
    LLMProvider,
    available_providers,
    create_provider,
)
from app.core.llm.anthropic_provider import AnthropicProvider
from app.core.schemas import Claim


class MockProvider(LLMProvider):
    """Returns canned payloads and records every call. No network."""

    name = "mock"
    model = "mock-model"

    def __init__(self, payloads: dict[type[BaseModel], dict]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

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
        self.calls.append({"prompt": prompt, "schema": schema, "images": images, "context": context, "system": system})
        return schema.model_validate(self.payloads[schema])


async def explain_clause(llm: LLMProvider, clause_text: str, document_text: str) -> Claim:
    """Stand-in for an agent: depends only on the interface."""
    return await llm.generate_structured(f"Explain: {clause_text}", Claim, context=document_text)


async def test_agent_receives_validated_pydantic_object() -> None:
    llm = MockProvider(
        {Claim: {"text": "You must give 30 days notice.", "clause_ids": ["c-004"], "confidence": 0.9}}
    )

    claim = await explain_clause(llm, "Notice period is one month.", "full document text")

    assert isinstance(claim, Claim)
    assert claim.clause_ids == ["c-004"]
    assert claim.verified is False
    assert llm.calls[0]["context"] == "full document text"


async def test_mock_provider_passes_images_through() -> None:
    llm = MockProvider({Claim: {"text": "x", "confidence": 0.5}})
    page = ImageInput(data=b"\x89PNG", media_type="image/png")

    await llm.generate_structured("read this page", Claim, images=[page])

    assert llm.calls[0]["images"] == [page]


async def test_invalid_payload_raises_validation_error() -> None:
    llm = MockProvider({Claim: {"text": "x", "confidence": 1.5}})

    with pytest.raises(ValueError):
        await llm.generate_structured("p", Claim)


def test_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_factory_selects_anthropic(settings: Settings) -> None:
    provider = create_provider(settings)

    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-5-5"
    assert "anthropic" in available_providers()


def test_factory_rejects_unknown_provider(settings: Settings) -> None:
    settings.llm_provider = "nope"

    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        create_provider(settings)
