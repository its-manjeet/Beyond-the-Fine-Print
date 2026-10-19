"""24-hour retention: expired documents and everything derived from them are deleted."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.storage.cleanup import purge_expired
from app.storage.models import (
    AgentCache,
    AgentRun,
    Claim,
    Clause,
    Document,
    DocumentFile,
    KeyFact,
    Line,
    Page,
    RiskAssessment,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
DERIVED = (DocumentFile, Page, Line, Clause, RiskAssessment, Claim, KeyFact, AgentRun)


def add_document(session, created_at: datetime) -> str:
    doc = Document(
        filename="a.pdf",
        content_type="application/pdf",
        sha256="f" * 64,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=24),
    )
    session.add(doc)
    session.flush()
    doc_id = doc.document_id
    session.add_all(
        [
            DocumentFile(document_id=doc_id, data=b"%PDF"),
            Page(document_id=doc_id, page_number=1, width=595, height=842, source="text_layer"),
            Line(document_id=doc_id, page=1, line_number=1, text="x", bbox=[0, 0, 1, 1]),
            Clause(
                document_id=doc_id, clause_code="C-01", title="t", category="other",
                page_number=1, line_start=1, page_end=1, line_end=1, text="x",
            ),
            RiskAssessment(document_id=doc_id, clause_code="C-01", level="green", score=0.1, reason="r", factors={}),
            Claim(document_id=doc_id, kind="explanation", text="x", clause_codes=["C-01"]),
            KeyFact(document_id=doc_id, name="monthly_rent", value="Rs. 1", clause_code="C-01"),
            AgentRun(
                document_id=doc_id, agent_name="classifier", provider="mock", model="m",
                prompt_version="v1", latency_ms=1.0,
            ),
        ]
    )
    return doc_id


def count(session, table) -> int:
    return session.scalar(select(func.count()).select_from(table))


def test_documents_older_than_24_hours_are_deleted_with_everything_derived(db) -> None:
    with db.session() as session:
        old = add_document(session, NOW - timedelta(hours=25))
        fresh = add_document(session, NOW - timedelta(hours=23))
        session.add_all(
            [
                AgentCache(cache_key="a" * 64, document_sha256="f" * 64, agent_name="risk", prompt_version="v1",
                           model="m", output={}, expires_at=NOW - timedelta(minutes=1)),
                AgentCache(cache_key="b" * 64, document_sha256="f" * 64, agent_name="risk", prompt_version="v1",
                           model="m", output={}, expires_at=NOW + timedelta(hours=1)),
            ]
        )

    result = purge_expired(db, now=NOW)

    assert result == {"documents": 1, "cache_entries": 1}
    with db.session() as session:
        assert session.get(Document, old) is None
        assert session.get(Document, fresh) is not None
        for table in DERIVED:
            remaining = session.scalars(select(table.document_id)).all()
            assert remaining == [fresh], table.__tablename__
        assert session.scalars(select(AgentCache.cache_key)).all() == ["b" * 64]


def test_document_exactly_at_expiry_is_deleted(db) -> None:
    with db.session() as session:
        add_document(session, NOW - timedelta(hours=24))

    assert purge_expired(db, now=NOW)["documents"] == 1
    with db.session() as session:
        assert all(count(session, table) == 0 for table in (Document, *DERIVED))


def test_nothing_to_purge(db) -> None:
    with db.session() as session:
        add_document(session, NOW)

    assert purge_expired(db, now=NOW) == {"documents": 0, "cache_entries": 0}
