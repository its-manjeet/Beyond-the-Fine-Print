"""SQLAlchemy tables for documents and everything derived from them.

Every table below ``documents`` has a foreign key with ``ON DELETE CASCADE``, so deleting
a document row deletes all of its pages, lines, clauses, results and agent runs.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Stores timezone-aware UTC datetimes; SQLite drops tzinfo, so it is restored on load."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class Base(DeclarativeBase):
    pass


def _document_fk() -> Mapped[str]:
    return mapped_column(ForeignKey("documents.document_id", ondelete="CASCADE"), index=True)


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    doc_type: Mapped[str | None] = mapped_column(String(32))
    doc_type_confidence: Mapped[float | None] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(16))
    # processing | needs_confirmation | unsupported | completed | partial | failed
    status: Mapped[str] = mapped_column(String(32), default="processing")
    # stage name -> {"status": pending|running|done|failed|skipped, "error": str | None}
    stages: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)

    pages: Mapped[list["Page"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True, order_by="Page.page_number"
    )


class DocumentFile(Base):
    """The uploaded bytes, kept so the UI can show the original document (F8)."""

    __tablename__ = "document_files"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[bytes] = mapped_column(LargeBinary)


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    page_number: Mapped[int] = mapped_column(Integer)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)
    # text_layer | vision | tesseract | none
    source: Mapped[str] = mapped_column(String(16))
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped[Document] = relationship(back_populates="pages")


class Line(Base):
    __tablename__ = "lines"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    page: Mapped[int] = mapped_column(Integer)
    line_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # [x0, y0, x1, y1] as fractions (0-1) of the page width/height, or None when the
    # text came from vision OCR, which does not report positions.
    bbox: Mapped[list[float] | None] = mapped_column(JSON)


class Clause(Base):
    __tablename__ = "clauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    clause_code: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(32))
    page_number: Mapped[int] = mapped_column(Integer)
    line_start: Mapped[int] = mapped_column(Integer)
    # A clause may run onto the next page; then page_end > page_number and line_end is on page_end.
    page_end: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    clause_code: Mapped[str] = mapped_column(String(16))
    level: Mapped[str] = mapped_column(String(8))  # green | amber | red
    score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    factors: Mapped[dict[str, Any]] = mapped_column(JSON)


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    kind: Mapped[str] = mapped_column(String(16))  # explanation | answer
    text: Mapped[str] = mapped_column(Text)
    clause_codes: Mapped[list[str]] = mapped_column(JSON)
    # explanation: the clause being explained; answer: None
    subject_clause_code: Mapped[str | None] = mapped_column(String(16))
    # answer: the question it answers (groups the sentences of one answer)
    question: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class KeyFact(Base):
    __tablename__ = "key_facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    clause_code: Mapped[str] = mapped_column(String(16))


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = _document_fk()
    agent_name: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AgentCache(Base):
    """Agent outputs keyed by (document sha256, agent, prompt_version, model).

    Not linked to a document row (two uploads of the same file share entries), so it has
    its own expiry and is purged by the same cleanup job.
    """

    __tablename__ = "agent_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_sha256: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(32))
    prompt_version: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    output: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
