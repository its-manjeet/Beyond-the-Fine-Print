"""Reads and writes pipeline results, and builds the document snapshot the API returns."""

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select

from app.agents.citations import CitedSentence
from app.agents.explainer import ClauseExplanation
from app.agents.ingestion import IngestResult, UploadInfo
from app.agents.key_facts import KeyFactItem
from app.agents.risk import RiskResult
from app.core.schemas import Clause as ClauseSchema
from app.core.schemas import DocLine
from app.storage.db import Database
from app.storage.models import (
    AgentRun,
    Claim,
    Clause,
    Document,
    DocumentFile,
    KeyFact,
    Line,
    Page,
    RiskAssessment,
    utcnow,
)

STAGES = ("ingest", "classify", "clauses", "key_facts", "risk", "explain")
ANALYSIS_STAGES = ("key_facts", "risk", "explain")
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "unsupported", "needs_confirmation"})

NOTICE = (
    "This is information to help you understand your document, not legal advice. "
    "For advice on your situation, contact a lawyer or free legal aid (NALSA / your State Legal Services Authority)."
)
KEY_FACT_LABELS = {
    "owner": "Owner (licensor)",
    "tenant": "Tenant (licensee)",
    "property": "Property",
    "monthly_rent": "Monthly rent",
    "security_deposit": "Security deposit",
    "start_date": "Start date",
    "duration": "Duration",
    "lock_in": "Lock-in",
    "notice_period_owner": "Notice the owner must give",
    "notice_period_tenant": "Notice you must give",
    "late_payment_penalty": "Late payment penalty",
    "early_exit_penalty": "Early exit penalty",
    "other_penalty": "Other penalty",
    "rent_increase": "Rent increase",
}


def create_document(db: Database, filename: str, info: UploadInfo, data: bytes, sha256: str, retention_hours: float) -> str:
    now = utcnow()
    with db.session() as session:
        doc = Document(
            filename=filename[:255] or "upload",
            content_type=info.content_type,
            sha256=sha256,
            status="processing",
            stages={stage: {"status": "pending"} for stage in STAGES},
            created_at=now,
            expires_at=now + timedelta(hours=retention_hours),
        )
        session.add(doc)
        session.flush()
        session.add(DocumentFile(document_id=doc.document_id, data=data))
        return doc.document_id


def get_document(db: Database, document_id: str) -> Document | None:
    with db.session() as session:
        return session.get(Document, document_id)


def get_file(db: Database, document_id: str) -> bytes | None:
    with db.session() as session:
        row = session.get(DocumentFile, document_id)
        return row.data if row else None


def delete_document(db: Database, document_id: str) -> bool:
    with db.session() as session:
        return session.execute(delete(Document).where(Document.document_id == document_id)).rowcount > 0


def update_stage(db: Database, document_id: str, stage: str, status: str, **extra: Any) -> dict[str, Any]:
    with db.session() as session:
        doc = session.get(Document, document_id)
        stages = {**doc.stages}
        # A stage that (re)starts begins clean; otherwise keep details recorded earlier.
        previous = stages.get(stage, {})
        kept = {} if status in ("pending", "running") else {k: v for k, v in previous.items() if k not in ("status", "error")}
        stages[stage] = {**kept, "status": status, **{k: v for k, v in extra.items() if v is not None}}
        doc.stages = stages  # reassign so SQLAlchemy sees the JSON change
        return stages[stage]


def update_document(db: Database, document_id: str, **fields: Any) -> None:
    with db.session() as session:
        doc = session.get(Document, document_id)
        for key, value in fields.items():
            setattr(doc, key, value)


def save_ingest(db: Database, document_id: str, result: IngestResult) -> None:
    with db.session() as session:
        for page in result.pages:
            session.add(
                Page(
                    document_id=document_id,
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    source=page.source,
                    low_confidence=page.low_confidence,
                )
            )
            session.add_all(
                Line(document_id=document_id, page=page.page_number, line_number=l.line_number, text=l.text, bbox=l.bbox)
                for l in page.lines
            )


def load_lines(db: Database, document_id: str) -> list[DocLine]:
    with db.session() as session:
        rows = session.scalars(
            select(Line).where(Line.document_id == document_id).order_by(Line.page, Line.line_number)
        ).all()
        return [DocLine(page=r.page, line_number=r.line_number, text=r.text) for r in rows]


def replace_results(db: Database, document_id: str, *tables: type) -> None:
    """Remove earlier results (e.g. before re-running stages after a type confirmation)."""
    with db.session() as session:
        for table in tables:
            criteria = [table.document_id == document_id]
            if table is Claim:
                criteria.append(Claim.kind == "explanation")
            session.execute(delete(table).where(*criteria))


def save_clauses(db: Database, document_id: str, clauses: Sequence[ClauseSchema]) -> None:
    with db.session() as session:
        session.add_all(
            Clause(
                document_id=document_id,
                clause_code=c.clause_code,
                title=c.title,
                category=c.category,
                page_number=c.page,
                line_start=c.line_start,
                page_end=c.page_end,
                line_end=c.line_end,
                text=c.text,
            )
            for c in clauses
        )


def load_clauses(db: Database, document_id: str) -> list[ClauseSchema]:
    with db.session() as session:
        rows = session.scalars(select(Clause).where(Clause.document_id == document_id).order_by(Clause.id)).all()
        return [
            ClauseSchema(
                clause_code=r.clause_code,
                title=r.title,
                category=r.category,
                page=r.page_number,
                line_start=r.line_start,
                page_end=r.page_end,
                line_end=r.line_end,
                text=r.text,
            )
            for r in rows
        ]


def save_key_facts(db: Database, document_id: str, facts: Sequence[KeyFactItem]) -> None:
    with db.session() as session:
        session.add_all(
            KeyFact(document_id=document_id, name=f.name, value=f.value, clause_code=f.clause_code) for f in facts
        )


def save_risks(db: Database, document_id: str, results: Sequence[RiskResult]) -> None:
    with db.session() as session:
        session.add_all(
            RiskAssessment(
                document_id=document_id,
                clause_code=r.clause_code,
                level=r.level,
                score=r.score,
                reason=r.reason,
                factors=r.factors,
            )
            for r in results
        )


def save_explanations(db: Database, document_id: str, explanations: Sequence[ClauseExplanation]) -> None:
    with db.session() as session:
        session.add_all(
            Claim(
                document_id=document_id,
                kind="explanation",
                text=s.text,
                clause_codes=s.clause_codes,
                subject_clause_code=e.clause_code,
            )
            for e in explanations
            for s in e.sentences
        )


def save_answer(db: Database, document_id: str, question: str, sentences: Sequence[CitedSentence]) -> None:
    with db.session() as session:
        session.add_all(
            Claim(document_id=document_id, kind="answer", text=s.text, clause_codes=s.clause_codes, question=question)
            for s in sentences
        )


def stage_ready(doc: Document, stage: str) -> bool:
    return doc.stages.get(stage, {}).get("status") == "done"


def snapshot(db: Database, document_id: str) -> dict[str, Any] | None:
    """Everything known about a document so far, with a status per stage."""
    with db.session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        pages = session.scalars(select(Page).where(Page.document_id == document_id).order_by(Page.page_number)).all()
        lines = session.scalars(
            select(Line).where(Line.document_id == document_id).order_by(Line.page, Line.line_number)
        ).all()
        clauses = session.scalars(select(Clause).where(Clause.document_id == document_id).order_by(Clause.id)).all()
        risks = {
            r.clause_code: r
            for r in session.scalars(select(RiskAssessment).where(RiskAssessment.document_id == document_id)).all()
        }
        claims = session.scalars(select(Claim).where(Claim.document_id == document_id).order_by(Claim.id)).all()
        facts = session.scalars(select(KeyFact).where(KeyFact.document_id == document_id).order_by(KeyFact.id)).all()
        runs = session.scalars(select(AgentRun).where(AgentRun.document_id == document_id).order_by(AgentRun.id)).all()

        lines_by_page: dict[int, list[dict[str, Any]]] = {}
        for line in lines:
            lines_by_page.setdefault(line.page, []).append(
                {"line_number": line.line_number, "text": line.text, "bbox": line.bbox}
            )
        explanations: dict[str, list[dict[str, Any]]] = {}
        answers: list[dict[str, Any]] = []
        for claim in claims:
            item = {"text": claim.text, "clause_codes": claim.clause_codes}
            if claim.kind == "explanation":
                explanations.setdefault(claim.subject_clause_code or "", []).append(item)
            elif answers and answers[-1]["question"] == claim.question:
                answers[-1]["sentences"].append(item)
            else:
                answers.append({"question": claim.question, "sentences": [item]})

        risk_counts = {"red": 0, "amber": 0, "green": 0}
        for risk in risks.values():
            risk_counts[risk.level] += 1

        return {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "status": doc.status,
            "error": doc.error,
            "doc_type": doc.doc_type,
            "doc_type_confidence": doc.doc_type_confidence,
            "language": doc.language,
            "created_at": doc.created_at.isoformat(),
            "expires_at": doc.expires_at.isoformat(),
            "stages": {stage: doc.stages.get(stage, {"status": "pending"}) for stage in STAGES},
            "notice": NOTICE,
            "pages": [
                {
                    "page_number": p.page_number,
                    "width": p.width,
                    "height": p.height,
                    "source": p.source,
                    "low_confidence": p.low_confidence,
                    "lines": lines_by_page.get(p.page_number, []),
                }
                for p in pages
            ],
            "clauses": [
                {
                    "clause_code": c.clause_code,
                    "title": c.title,
                    "category": c.category,
                    "page": c.page_number,
                    "line_start": c.line_start,
                    "page_end": c.page_end,
                    "line_end": c.line_end,
                    "text": c.text,
                    "risk": _risk(risks.get(c.clause_code)),
                    "explanation": explanations.get(c.clause_code, []),
                }
                for c in clauses
            ],
            "risk_summary": risk_counts if risks else None,
            "key_facts": [
                {"name": f.name, "label": KEY_FACT_LABELS.get(f.name, f.name), "value": f.value, "clause_code": f.clause_code}
                for f in facts
            ],
            "answers": answers,
            "agent_runs": [
                {
                    "agent": r.agent_name,
                    "provider": r.provider,
                    "model": r.model,
                    "prompt_version": r.prompt_version,
                    "latency_ms": r.latency_ms,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached": r.cached,
                    "ok": r.ok,
                }
                for r in runs
            ],
        }


def _risk(row: RiskAssessment | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"level": row.level, "score": row.score, "reason": row.reason, "factors": row.factors}
