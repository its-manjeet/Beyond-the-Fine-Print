"""Runs the agents for one document in the background and reports progress per stage.

    ingest -> classify -> clauses -> (key_facts, risk, explain in parallel)

A low-confidence classification pauses the pipeline (status ``needs_confirmation``) until
the user confirms the type; a document Phase 1 cannot handle stops as ``unsupported``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from app.agents import classifier, clauses, explainer, key_facts, risk
from app.agents.classifier import classify, is_supported
from app.agents.clauses import extract_clauses
from app.agents.explainer import explain_clauses
from app.agents.ingestion import CONTENT_TYPES, UploadInfo, ingest
from app.agents.key_facts import extract_key_facts
from app.agents.risk import analyze_risk
from app.core.config import Settings
from app.core.llm import (
    LLMError,
    LLMOutputError,
    LLMProvider,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMUnavailableError,
)
from app.core.logging import get_logger
from app.pipeline import store
from app.pipeline.events import EventBus
from app.pipeline.tracking import TrackedLLM
from app.prompts import classifier as classifier_prompt
from app.prompts import clauses as clauses_prompt
from app.prompts import explainer as explainer_prompt
from app.prompts import key_facts as key_facts_prompt
from app.prompts import ocr as ocr_prompt
from app.prompts import risk as risk_prompt
from app.storage.db import Database
from app.storage.models import Claim, Clause, KeyFact, RiskAssessment

logger = get_logger(__name__)

ProviderFactory = Callable[[str], LLMProvider]
OCR_AGENT = "ocr"


class ProcessingError(Exception):
    """A stage cannot continue; the message is shown to the user."""


def user_message(exc: Exception) -> str:
    if isinstance(exc, ProcessingError):
        return str(exc)
    if isinstance(exc, LLMRateLimitError):
        return "The AI service is busy (rate limit reached). Please try again in a minute."
    if isinstance(exc, LLMUnavailableError):
        return "The AI service could not be reached. Please try again later."
    if isinstance(exc, LLMRequestError):
        return f"The AI service rejected the request: {exc}"
    if isinstance(exc, LLMRefusalError):
        return "The AI model declined to process this document."
    if isinstance(exc, LLMOutputError):
        return "The AI model returned a result that could not be used. Please try again."
    if isinstance(exc, LLMError):
        return f"AI error: {exc}"
    return f"Internal error ({type(exc).__name__})."


class Pipeline:
    def __init__(self, db: Database, bus: EventBus, settings: Settings, provider_for: ProviderFactory) -> None:
        self.db = db
        self.bus = bus
        self.settings = settings
        self.provider_for = provider_for
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # --- task control ----------------------------------------------------------------

    def start(self, document_id: str) -> None:
        self._launch(document_id, self._run(document_id))

    def resume(self, document_id: str) -> None:
        """Continue after the user confirmed the document type."""
        self._launch(document_id, self._after_classification(document_id))

    async def wait(self, document_id: str) -> None:
        task = self._tasks.get(document_id)
        if task is not None:
            await asyncio.shield(task)

    async def cancel(self, document_id: str) -> None:
        task = self._tasks.get(document_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def _launch(self, document_id: str, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(self._guard(document_id, coro), name=f"pipeline:{document_id}")
        self._tasks[document_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(document_id, None) if self._tasks.get(document_id) is t else None)

    async def _guard(self, document_id: str, coro: Awaitable[None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never leave a document stuck in "processing"
            logger.error("pipeline.crashed", extra={"document_id": document_id, "error": type(exc).__name__})
            self._set_status(document_id, "failed", user_message(exc))

    # --- stages ----------------------------------------------------------------------

    async def _run(self, document_id: str) -> None:
        if not await self._stage(document_id, "ingest", self._ingest):
            return self._fail_after(document_id, "ingest")
        if not await self._stage(document_id, "classify", self._classify):
            return self._fail_after(document_id, "classify")
        # Decided after the classify stage is reported done, so its event comes first.
        doc = store.get_document(self.db, document_id)
        if doc.doc_type_confidence < self.settings.classifier_min_confidence:
            return self._set_status(document_id, "needs_confirmation")
        if not is_supported(doc.doc_type, doc.language):
            return self._mark_unsupported(document_id, doc.doc_type, doc.language)
        await self._after_classification(document_id)

    async def _after_classification(self, document_id: str) -> None:
        if not await self._stage(document_id, "clauses", self._clauses):
            return self._fail_after(document_id, "clauses")
        results = await asyncio.gather(
            self._stage(document_id, "key_facts", self._key_facts),
            self._stage(document_id, "risk", self._risk),
            self._stage(document_id, "explain", self._explain),
        )
        self._set_status(document_id, "completed" if all(results) else "partial")

    async def _stage(self, document_id: str, stage: str, run: Callable[[str], Awaitable[dict[str, Any] | None]]) -> bool:
        self._set_stage(document_id, stage, "running")
        try:
            details = await run(document_id) or {}
        except Exception as exc:
            logger.warning(
                "pipeline.stage_failed", extra={"document_id": document_id, "stage": stage, "error": type(exc).__name__}
            )
            self._set_stage(document_id, stage, "failed", error=user_message(exc))
            return False
        self._set_stage(document_id, stage, "done", **details)
        return True

    async def _ingest(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        data = store.get_file(self.db, document_id)
        kind = next(k for k, v in CONTENT_TYPES.items() if v == doc.content_type)
        info = UploadInfo(kind, doc.content_type, 0)
        try:
            llm: LLMProvider | None = self._llm(doc.document_id, doc.sha256, OCR_AGENT, ocr_prompt.VERSION)
        except Exception as exc:  # no API key etc.: digital PDFs can still be read
            logger.warning("pipeline.ocr_provider_unavailable", extra={"error": type(exc).__name__})
            llm = None
        result = await ingest(data, info, llm, self.settings)
        if result.line_count == 0:
            raise ProcessingError(
                "No readable text was found. If this is a photo or scan, check that the AI provider has a vision "
                "model (on Groq set GROQ_VISION_MODEL=qwen/qwen3.8-27b) or install Tesseract, and upload a clear image."
            )
        store.save_ingest(self.db, document_id, result)
        return {
            "pages": len(result.pages),
            "lines": result.line_count,
            "low_confidence_pages": [p.page_number for p in result.pages if p.low_confidence],
            "warnings": result.warnings,
        }

    async def _classify(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        llm = self._llm(document_id, doc.sha256, classifier.AGENT, classifier_prompt.VERSION)
        result = await classify(llm, store.load_lines(self.db, document_id))
        store.update_document(
            self.db, document_id, doc_type=result.doc_type, doc_type_confidence=result.confidence, language=result.language
        )
        return {"doc_type": result.doc_type, "language": result.language, "confidence": result.confidence}

    async def _clauses(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        llm = self._llm(document_id, doc.sha256, clauses.AGENT, clauses_prompt.VERSION)
        result = await extract_clauses(llm, store.load_lines(self.db, document_id), self.settings.llm_input_budget_bytes)
        if not result.clauses:
            raise ProcessingError("No clauses could be found in this document.")
        store.replace_results(self.db, document_id, Clause)
        store.save_clauses(self.db, document_id, result.clauses)
        return {"clauses": len(result.clauses), "retried": result.retried, "dropped_spans": result.dropped}

    async def _key_facts(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        llm = self._llm(document_id, doc.sha256, key_facts.AGENT, key_facts_prompt.VERSION)
        facts = await extract_key_facts(llm, store.load_clauses(self.db, document_id), self.settings.llm_input_budget_bytes)
        store.replace_results(self.db, document_id, KeyFact)
        store.save_key_facts(self.db, document_id, facts)
        return {"facts": len(facts)}

    async def _risk(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        llm = self._llm(document_id, doc.sha256, risk.AGENT, risk_prompt.VERSION)
        results = await analyze_risk(
            llm, store.load_clauses(self.db, document_id), self.settings.llm_input_budget_bytes, self.settings.risk_weights
        )
        store.replace_results(self.db, document_id, RiskAssessment)
        store.save_risks(self.db, document_id, results)
        return {level: sum(r.level == level for r in results) for level in ("red", "amber", "green")}

    async def _explain(self, document_id: str) -> dict[str, Any]:
        doc = store.get_document(self.db, document_id)
        llm = self._llm(document_id, doc.sha256, explainer.AGENT, explainer_prompt.VERSION)
        result = await explain_clauses(llm, store.load_clauses(self.db, document_id), self.settings.llm_input_budget_bytes)
        store.replace_results(self.db, document_id, Claim)
        store.save_explanations(self.db, document_id, result.explanations)
        return {"explained_clauses": len(result.explanations), "removed_sentences": result.removed}

    # --- confirmation ----------------------------------------------------------------

    def confirm_type(self, document_id: str, doc_type: str) -> None:
        """Apply the user's choice of type, then resume or stop. The caller checks the state."""
        doc = store.get_document(self.db, document_id)
        store.update_document(self.db, document_id, doc_type=doc_type)
        self._set_stage(document_id, "classify", "done", confirmed_by_user=True, doc_type=doc_type)
        if not is_supported(doc_type, doc.language):
            self._mark_unsupported(document_id, doc_type, doc.language)
            return
        for stage in ("clauses", *store.ANALYSIS_STAGES):
            self._set_stage(document_id, stage, "pending")
        self._set_status(document_id, "processing")
        self.resume(document_id)

    # --- helpers ---------------------------------------------------------------------

    def _llm(self, document_id: str, sha256: str, agent: str, prompt_version: str) -> TrackedLLM:
        return TrackedLLM(
            self.provider_for(agent),
            self.db,
            document_id=document_id,
            document_sha256=sha256,
            agent=agent,
            prompt_version=prompt_version,
            retention_hours=self.settings.retention_hours,
        )

    def _mark_unsupported(self, document_id: str, doc_type: str | None, language: str | None) -> None:
        for stage in ("clauses", *store.ANALYSIS_STAGES):
            self._set_stage(document_id, stage, "skipped")
        what = (doc_type or "unknown").replace("_", " ")
        self._set_status(
            document_id,
            "unsupported",
            f"This looks like a {what} in language '{language or 'unknown'}'. "
            "This version explains English rental agreements only. If this is a rental agreement, confirm the type.",
        )

    def _fail_after(self, document_id: str, failed_stage: str) -> None:
        doc = store.get_document(self.db, document_id)
        later = list(store.STAGES)[list(store.STAGES).index(failed_stage) + 1 :]
        for stage in later:
            self._set_stage(document_id, stage, "skipped")
        self._set_status(document_id, "failed", doc.stages[failed_stage].get("error"))

    def _set_stage(self, document_id: str, stage: str, status: str, **extra: Any) -> None:
        entry = store.update_stage(self.db, document_id, stage, status, **extra)
        self.bus.publish(document_id, {"type": "stage", "stage": stage, **entry})

    def _set_status(self, document_id: str, status: str, error: str | None = None) -> None:
        store.update_document(self.db, document_id, status=status, error=error)
        self.bus.publish(document_id, {"type": "status", "status": status, "error": error})
