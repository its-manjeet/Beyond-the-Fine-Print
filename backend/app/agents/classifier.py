"""F2 Document classifier: type, language and confidence from the opening lines."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.agents.text import head_text
from app.core.llm import LLMProvider
from app.core.schemas import DocLine, DocType
from app.prompts import classifier as prompt

AGENT = "classifier"
HEAD_BYTES = 2500  # the opening of a document is enough to tell its type


class ClassifierOutput(BaseModel):
    doc_type: DocType
    language: str = Field(description="ISO 639-1 code, e.g. en, hi, mr")
    confidence: float = Field(ge=0, le=1)


async def classify(llm: LLMProvider, lines: Sequence[DocLine]) -> ClassifierOutput:
    result = await llm.generate_structured(
        prompt.PROMPT, ClassifierOutput, context=head_text(lines, HEAD_BYTES), max_tokens=1500
    )
    result.language = result.language.strip().lower()
    return result


def is_supported(doc_type: str | None, language: str | None) -> bool:
    """Phase 1 handles English rental agreements only."""
    return doc_type == "rental_agreement" and (language or "").split("-")[0] == "en"
