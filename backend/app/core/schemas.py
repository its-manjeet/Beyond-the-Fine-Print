"""Core domain schemas shared by every agent."""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


DocType = Literal[
    "rental_agreement",
    "insurance_policy",
    "loan_agreement",
    "legal_notice",
    "scheme_rules",
    "bank_circular",
    "other",
]

RentalCategory = Literal[
    "parties_property",
    "rent_payment",
    "security_deposit",
    "rent_escalation",
    "maintenance",
    "lock_in",
    "termination_notice",
    "penalty_late_fee",
    "renewal",
    "use_restrictions",
    "dispute_resolution",
    "other",
]


class DocLine(BaseModel):
    """One numbered line of the document, as extracted by ingestion."""

    page: int = Field(ge=1)
    line_number: int = Field(ge=1, description="1-indexed, restarting on every page")
    text: str

    @property
    def ref(self) -> str:
        return f"p{self.page} L{self.line_number}"


class Clause(BaseModel):
    """A clause/section of a document, with a stable code and its source span.

    The span starts at (page, line_start) and ends at (page_end, line_end), inclusive.
    ``text`` is always copied from the stored lines, never written by a model.
    """

    clause_code: str = Field(description="Stable clause code, e.g. 'C-01'")
    title: str
    category: str
    page: int = Field(ge=1, description="1-indexed page the clause starts on")
    line_start: int = Field(ge=1)
    page_end: int = Field(ge=1, description="Page the clause ends on (usually the same page)")
    line_end: int = Field(ge=1, description="Last line on page_end (inclusive)")
    text: str

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if (self.page_end, self.line_end) < (self.page, self.line_start):
            raise ValueError("the span must end at or after its start")
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
