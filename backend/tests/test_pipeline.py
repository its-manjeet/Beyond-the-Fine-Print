"""Orchestrator: stage order, statuses, caching and AGENT_RUN logging. LLM mocked."""

import asyncio
import hashlib

from app.agents.ingestion import validate_upload
from app.core.config import Settings
from app.core.llm import LLMRateLimitError, LLMRequestError
from app.pipeline import store
from app.pipeline.events import EventBus
from app.pipeline.orchestrator import Pipeline
from app.pipeline.tracking import agent_runs
from tests.mocks import MockProvider, sample_pdf_bytes
from tests.pipeline_fixtures import golden_provider


def make_pipeline(db, settings: Settings, llm: MockProvider) -> tuple[Pipeline, EventBus]:
    bus = EventBus()
    return Pipeline(db, bus, settings, lambda agent: llm), bus


def upload(db, settings: Settings, data: bytes | None = None) -> str:
    data = data or sample_pdf_bytes()
    info = validate_upload(data, settings)
    return store.create_document(db, "sample.pdf", info, data, hashlib.sha256(data).hexdigest(), 24)


async def run_and_collect(pipeline: Pipeline, bus: EventBus, document_id: str) -> list[dict]:
    events: list[dict] = []
    async with bus.subscribe(document_id) as queue:
        pipeline.start(document_id)
        await pipeline.wait(document_id)
        while not queue.empty():
            events.append(queue.get_nowait())
    return events


async def test_stages_run_in_order_and_analysis_runs_in_parallel(db, settings: Settings) -> None:
    pipeline, bus = make_pipeline(db, settings, golden_provider())
    document_id = upload(db, settings)

    events = await run_and_collect(pipeline, bus, document_id)

    stage_events = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    first = {}
    for i, item in enumerate(stage_events):
        first.setdefault(item, i)
    assert first[("ingest", "running")] < first[("ingest", "done")] < first[("classify", "running")]
    assert first[("classify", "done")] < first[("clauses", "running")] < first[("clauses", "done")]
    analysis = ("key_facts", "risk", "explain")
    # All three start before any of them finishes: they run concurrently.
    assert max(first[(s, "running")] for s in analysis) < min(first[(s, "done")] for s in analysis)
    assert first[("clauses", "done")] < min(first[(s, "running")] for s in analysis)
    assert events[-1] == {"type": "status", "status": "completed", "error": None}

    doc = store.get_document(db, document_id)
    assert doc.status == "completed"
    assert {stage: info["status"] for stage, info in doc.stages.items()} == dict.fromkeys(store.STAGES, "done")
    assert doc.stages["ingest"]["lines"] == 73 and doc.stages["clauses"]["clauses"] == 15
    assert doc.stages["risk"] == {"status": "done", "red": 1, "amber": 2, "green": 12}
    assert doc.stages["explain"]["removed_sentences"] == 1


async def test_every_call_is_logged_and_a_rerun_is_served_from_cache(db, settings: Settings) -> None:
    llm = golden_provider()
    pipeline, bus = make_pipeline(db, settings, llm)

    first = upload(db, settings)
    await run_and_collect(pipeline, bus, first)
    calls_first = len(llm.calls)
    second = upload(db, settings)
    await run_and_collect(pipeline, bus, second)

    assert len(llm.calls) == calls_first == 9  # classifier, clauses, key facts, 2 risk and 4 explainer batches
    runs_first, runs_second = agent_runs(db, first), agent_runs(db, second)
    assert sorted(r.agent_name for r in runs_first) == sorted(
        ["classifier", "clauses", "key_facts", "risk", "risk", "explainer", "explainer", "explainer", "explainer"]
    )
    assert all(not r.cached and r.ok and r.input_tokens == 100 and r.output_tokens == 20 for r in runs_first)
    assert all(r.provider == "mock" and r.model == "mock-model" for r in runs_first)
    from app.prompts import classifier, clauses, risk

    assert {r.prompt_version for r in runs_first} >= {classifier.VERSION, clauses.VERSION, risk.VERSION}
    assert len(runs_second) == 9 and all(r.cached and r.input_tokens == 0 for r in runs_second)
    assert store.snapshot(db, second)["clauses"] == store.snapshot(db, first)["clauses"]


async def test_cache_key_includes_model_and_prompt_version(db, settings: Settings, monkeypatch) -> None:
    llm = golden_provider()
    pipeline, bus = make_pipeline(db, settings, llm)
    await run_and_collect(pipeline, bus, upload(db, settings))

    llm.model = "another-model"
    await run_and_collect(pipeline, bus, upload(db, settings))
    assert len(llm.calls) == 18

    from app.prompts import risk as risk_prompt

    monkeypatch.setattr(risk_prompt, "VERSION", risk_prompt.VERSION + "-changed")
    await run_and_collect(pipeline, bus, upload(db, settings))
    assert len(llm.calls) == 20  # only the risk agent's two calls ran again


async def test_low_confidence_pauses_for_confirmation_then_resumes(db, settings: Settings) -> None:
    llm = golden_provider(ClassifierOutput={"doc_type": "rental_agreement", "language": "en", "confidence": 0.45})
    pipeline, bus = make_pipeline(db, settings, llm)
    document_id = upload(db, settings)

    await run_and_collect(pipeline, bus, document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "needs_confirmation"
    assert doc.stages["classify"]["status"] == "done" and doc.stages["clauses"]["status"] == "pending"
    assert not llm.calls_for("ClauseExtraction")

    pipeline.confirm_type(document_id, "rental_agreement")
    await pipeline.wait(document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "completed" and doc.stages["classify"]["confirmed_by_user"] is True


async def test_unsupported_type_stops_with_skipped_stages(db, settings: Settings) -> None:
    llm = golden_provider(ClassifierOutput={"doc_type": "insurance_policy", "language": "en", "confidence": 0.9})
    pipeline, bus = make_pipeline(db, settings, llm)
    document_id = upload(db, settings)

    events = await run_and_collect(pipeline, bus, document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "unsupported" and "English rental agreements only" in doc.error
    assert [doc.stages[s]["status"] for s in ("clauses", "key_facts", "risk", "explain")] == ["skipped"] * 4
    assert events[-1]["status"] == "unsupported"


async def test_failed_analysis_stage_gives_partial_result(db, settings: Settings) -> None:
    llm = golden_provider(RiskOutput=LLMRateLimitError("rate limited"))
    pipeline, bus = make_pipeline(db, settings, llm)
    document_id = upload(db, settings)

    await run_and_collect(pipeline, bus, document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "partial"
    assert doc.stages["risk"] == {"status": "failed", "error": "The AI service is busy (rate limit reached). Please try again in a minute."}
    assert doc.stages["key_facts"]["status"] == doc.stages["explain"]["status"] == "done"
    failed_run = next(r for r in agent_runs(db, document_id) if r.agent_name == "risk")
    assert (failed_run.ok, failed_run.error) == (False, "LLMRateLimitError")


async def test_failed_clause_stage_fails_the_document(db, settings: Settings) -> None:
    llm = golden_provider(ClauseExtraction=LLMRequestError("GROQ_API_KEY is not set"))
    pipeline, bus = make_pipeline(db, settings, llm)
    document_id = upload(db, settings)

    await run_and_collect(pipeline, bus, document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "failed" and "GROQ_API_KEY is not set" in doc.error
    assert [doc.stages[s]["status"] for s in ("key_facts", "risk", "explain")] == ["skipped"] * 3


async def test_unreadable_upload_fails_at_ingest(db, settings: Settings, monkeypatch) -> None:
    import io

    from PIL import Image

    from app.agents import ingestion

    monkeypatch.setattr(ingestion, "tesseract_available", lambda cmd: False)
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(buffer, format="PNG")
    llm = golden_provider(OcrOutput=LLMRequestError("GROQ_VISION_MODEL is not set"))
    pipeline, bus = make_pipeline(db, settings, llm)
    document_id = upload(db, settings, buffer.getvalue())

    await run_and_collect(pipeline, bus, document_id)

    doc = store.get_document(db, document_id)
    assert doc.status == "failed" and "No readable text was found" in doc.error
    assert doc.stages["classify"]["status"] == "skipped"


async def test_unexpected_crash_never_leaves_document_processing(db, settings: Settings, monkeypatch) -> None:
    pipeline, bus = make_pipeline(db, settings, golden_provider())
    document_id = upload(db, settings)

    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline, "_fail_after", boom)
    monkeypatch.setattr(pipeline, "_ingest", boom)
    await run_and_collect(pipeline, bus, document_id)

    assert store.get_document(db, document_id).status == "failed"


async def test_cancel_stops_a_running_pipeline(db, settings: Settings) -> None:
    started = asyncio.Event()

    class SlowProvider(MockProvider):
        async def generate_structured(self, prompt, schema, **kwargs):
            started.set()
            await asyncio.sleep(10)

    pipeline, bus = make_pipeline(db, settings, SlowProvider({}))
    document_id = upload(db, settings)
    pipeline.start(document_id)
    await asyncio.wait_for(started.wait(), 5)

    await pipeline.cancel(document_id)

    assert document_id not in pipeline._tasks
