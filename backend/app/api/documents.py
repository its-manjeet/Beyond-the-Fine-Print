"""Document endpoints: upload, results, live progress (SSE), type confirmation, Q&A."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents import qa
from app.agents.ingestion import UploadError, validate_upload
from app.core.logging import get_logger
from app.core.schemas import DocType
from app.pipeline import store
from app.pipeline.orchestrator import Pipeline, user_message
from app.pipeline.tracking import TrackedLLM
from app.prompts import qa as qa_prompt
from app.storage.db import Database

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

SSE_KEEPALIVE_SECONDS = 15.0


class UploadResponse(BaseModel):
    document_id: str
    status: str
    stages: dict[str, Any]


class ConfirmTypeRequest(BaseModel):
    doc_type: DocType


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class CitedText(BaseModel):
    text: str
    clause_codes: list[str]


class ChatResponse(BaseModel):
    question: str
    covered: bool
    answer: list[CitedText]
    removed_sentences: int


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    db.ensure_schema()
    return db


def _pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _require(db: Database, document_id: str):
    doc = store.get_document(db, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found. It may have been deleted after 24 hours.")
    return doc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = request.app.state.settings
    limit = settings.max_upload_mb * 1024 * 1024
    # Read one byte past the limit, so an oversized upload is rejected without reading it all.
    data = await file.read(limit + 1)
    try:
        info = validate_upload(data, settings)
    except UploadError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc

    db = _db(request)
    document_id = store.create_document(
        db, file.filename or "upload", info, data, hashlib.sha256(data).hexdigest(), settings.retention_hours
    )
    logger.info(
        "document.uploaded",
        extra={"document_id": document_id, "content_type": info.content_type, "pages": info.page_count, "bytes": len(data)},
    )
    _pipeline(request).start(document_id)
    doc = store.get_document(db, document_id)
    return UploadResponse(document_id=document_id, status=doc.status, stages=doc.stages)


@router.get("/{document_id}")
async def get_document(request: Request, document_id: str) -> dict[str, Any]:
    result = store.snapshot(_db(request), document_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found. It may have been deleted after 24 hours.")
    return result


@router.get("/{document_id}/file")
async def get_document_file(request: Request, document_id: str) -> Response:
    """The original upload, for the document viewer."""
    db = _db(request)
    doc = _require(db, document_id)
    return Response(store.get_file(db, document_id), media_type=doc.content_type, headers={"Cache-Control": "no-store"})


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(request: Request, document_id: str) -> Response:
    """Delete the document and everything derived from it now, without waiting 24 hours."""
    db = _db(request)
    _require(db, document_id)
    await request.app.state.pipeline.cancel(document_id)
    store.delete_document(db, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/events")
async def document_events(request: Request, document_id: str) -> StreamingResponse:
    """Server-Sent Events: a snapshot of every stage, then updates until processing stops.

    Events: ``stage`` ({stage, status, ...}), ``status`` ({status, error}) and a final
    ``done``. Clients should close the stream on ``done``.
    """
    db = _db(request)
    _require(db, document_id)
    bus = request.app.state.bus

    async def stream() -> AsyncIterator[str]:
        async with bus.subscribe(document_id) as queue:
            doc = store.get_document(db, document_id)
            yield _sse("snapshot", {"status": doc.status, "error": doc.error, "stages": doc.stages})
            if doc.status in store.TERMINAL_STATUSES:
                yield _sse("done", {"status": doc.status})
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), SSE_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(event["type"], {k: v for k, v in event.items() if k != "type"})
                if event["type"] == "status" and event["status"] in store.TERMINAL_STATUSES:
                    yield _sse("done", {"status": event["status"]})
                    return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{document_id}/confirm-type", status_code=status.HTTP_202_ACCEPTED)
async def confirm_type(request: Request, document_id: str, body: ConfirmTypeRequest) -> dict[str, Any]:
    db = _db(request)
    doc = _require(db, document_id)
    if doc.status not in ("needs_confirmation", "unsupported") or not store.stage_ready(doc, "classify"):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"The type can be confirmed only after classification; the status is '{doc.status}'."
        )
    _pipeline(request).confirm_type(document_id, body.doc_type)
    return store.snapshot(db, document_id)


@router.post("/{document_id}/chat", response_model=ChatResponse)
async def chat(request: Request, document_id: str, body: ChatRequest) -> ChatResponse:
    db = _db(request)
    doc = _require(db, document_id)
    if not store.stage_ready(doc, "clauses"):
        raise HTTPException(status.HTTP_409_CONFLICT, "The document's clauses are not ready yet. Try again when processing finishes.")
    pipeline = _pipeline(request)
    try:
        llm = TrackedLLM(
            pipeline.provider_for(qa.AGENT),
            db,
            document_id=document_id,
            document_sha256=doc.sha256,
            agent=qa.AGENT,
            prompt_version=qa_prompt.VERSION,
            retention_hours=request.app.state.settings.retention_hours,
        )
        answer = await qa.answer_question(
            llm, store.load_clauses(db, document_id), body.question, request.app.state.settings.llm_input_budget_bytes
        )
    except Exception as exc:
        logger.warning("chat.failed", extra={"document_id": document_id, "error": type(exc).__name__})
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, user_message(exc)) from exc
    store.save_answer(db, document_id, body.question.strip(), answer.sentences)
    return ChatResponse(
        question=body.question.strip(),
        covered=answer.covered,
        answer=[CitedText(text=s.text, clause_codes=s.clause_codes) for s in answer.sentences],
        removed_sentences=answer.removed,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
