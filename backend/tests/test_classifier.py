"""F2: classifier agent. The LLM is mocked; no network."""

import pytest

from app.agents.classifier import HEAD_BYTES, classify, is_supported
from app.core.schemas import DocLine
from tests.mocks import MockProvider


def lines(count: int, text: str = "x" * 99) -> list[DocLine]:
    return [DocLine(page=1, line_number=i, text=text) for i in range(1, count + 1)]


async def test_sends_only_the_opening_lines_and_normalises_language() -> None:
    llm = MockProvider({"ClassifierOutput": {"doc_type": "rental_agreement", "language": " EN ", "confidence": 0.93}})

    result = await classify(llm, lines(500))

    assert (result.doc_type, result.language, result.confidence) == ("rental_agreement", "en", 0.93)
    context = llm.calls[0]["context"]
    assert len(context.encode()) <= HEAD_BYTES
    assert "p1 L" not in context  # plain text; line refs are not needed to classify


async def test_rejects_unknown_doc_type() -> None:
    llm = MockProvider({"ClassifierOutput": {"doc_type": "recipe", "language": "en", "confidence": 0.9}})
    with pytest.raises(ValueError):
        await classify(llm, lines(3))


@pytest.mark.parametrize(
    ("doc_type", "language", "supported"),
    [
        ("rental_agreement", "en", True),
        ("rental_agreement", "en-IN", True),
        ("rental_agreement", "hi", False),
        ("insurance_policy", "en", False),
        (None, None, False),
    ],
)
def test_phase_1_supports_english_rental_agreements_only(doc_type, language, supported) -> None:
    assert is_supported(doc_type, language) is supported
