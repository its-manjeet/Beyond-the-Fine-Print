"""Core domain schemas shared by every agent."""

from datetime import UTC, datetime
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Clause(BaseModel):
    """A clause/section of a document, with a stable ID and its source span."""

    id: str = Field(description="Stable clause ID, e.g. 'c-012'")
    text: str
    page: int = Field(ge=1, description="1-indexed page number")
    line_start: int = Field(ge=1, description="1-indexed first line on the page")
    line_end: int = Field(ge=1, description="1-indexed last line on the page (inclusive)")

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Page(BaseModel):
    number: int = Field(ge=1, description="1-indexed page number")
    lines: list[str] = Field(default_factory=list, description="Text lines in reading order")

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    filename: str
    doc_type: str | None = None
    language: str | None = Field(default=None, description="BCP-47 code, e.g. 'en', 'hi'")
    pages: list[Page] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Claim(BaseModel):
    """A single user-facing statement, traceable to clauses and legal sources."""

    text: str
    clause_ids: list[str] = Field(default_factory=list)
    legal_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = False
