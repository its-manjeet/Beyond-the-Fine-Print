"""End-to-end API test on the sample PDF with golden mock responses. No network."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app
from tests.mocks import MockProvider, sample_pdf_bytes
from tests.pipeline_fixtures import golden_provider


@pytest.fixture
def api_settings(settings: Settings) -> Settings:
    settings.database_url = "sqlite://"
    return settings


@pytest.fixture
def llm() -> MockProvider:
    return golden_provider()


@pytest.fixture
async def app(api_settings: Settings, llm: MockProvider):
    application = create_app(api_settings, provider_for=lambda agent: llm)
    application.state.db.ensure_schema()
    yield application
    await application.state.pipeline.shutdown()
    application.state.db.dispose()


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http


async def upload_sample(client: httpx.AsyncClient, app, data: bytes | None = None, name: str = "rental.pdf") -> str:
    response = await client.post("/documents", files={"file": (name, data or sample_pdf_bytes(), "application/pdf")})
    assert response.status_code == 202, response.text
    document_id = response.json()["document_id"]
    await app.state.pipeline.wait(document_id)
    return document_id


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_sample_pdf_end_to_end(client: httpx.AsyncClient, app) -> None:
    response = await client.post("/documents", files={"file": ("rental.pdf", sample_pdf_bytes(), "application/pdf")})
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "processing" and set(accepted["stages"]) == {
        "ingest", "classify", "clauses", "key_facts", "risk", "explain"
    }
    document_id = accepted["document_id"]
    await app.state.pipeline.wait(document_id)

    body = (await client.get(f"/documents/{document_id}")).json()

    assert body["status"] == "completed" and body["error"] is None
    assert (body["doc_type"], body["language"], body["doc_type_confidence"]) == ("rental_agreement", "en", 0.97)
    assert all(stage["status"] == "done" for stage in body["stages"].values())
    assert "not legal advice" in body["notice"]

    # F1: pages and numbered lines with boxes for click-to-source (F8)
    assert [len(p["lines"]) for p in body["pages"]] == [35, 33, 5]
    assert body["pages"][1]["lines"][0] == {
        "line_number": 1, "text": "7. Termination", "bbox": body["pages"][1]["lines"][0]["bbox"]
    }
    assert len(body["pages"][1]["lines"][0]["bbox"]) == 4

    # F3: clauses with spans and text copied from the lines
    clauses = {c["clause_code"]: c for c in body["clauses"]}
    assert len(clauses) == 15
    termination = clauses["C-08"]
    assert (termination["title"], termination["page"], termination["line_start"], termination["line_end"]) == (
        "Termination", 2, 1, 4
    )
    assert termination["text"].startswith("7. Termination\nThe Licensor may terminate this licence at any time")

    # F6: risk computed in code
    assert clauses["C-07"]["risk"]["level"] == "amber"
    assert clauses["C-08"]["risk"]["level"] == "red"
    assert clauses["C-08"]["risk"]["reason"] == "Main factor: one-sided termination"
    assert clauses["C-04"]["risk"]["level"] == "green"
    assert body["risk_summary"] == {"red": 1, "amber": 2, "green": 12}

    # F4: explanations with citations; the sentence citing C-42 was removed
    assert len(termination["explanation"]) == 2
    assert all(
        set(s["clause_codes"]) <= set(clauses) for c in body["clauses"] for s in c["explanation"]
    )

    # F5: key facts, each tied to a clause; the fact citing C-99 was dropped
    facts = {f["name"]: f for f in body["key_facts"]}
    assert facts["monthly_rent"]["clause_code"] == "C-04" and facts["monthly_rent"]["label"] == "Monthly rent"
    assert facts["notice_period_owner"]["value"] == "7 days"
    assert all(f["clause_code"] in clauses for f in body["key_facts"])

    # AGENT_RUN is exposed for the "why this?" panel
    assert {r["agent"] for r in body["agent_runs"]} == {"classifier", "clauses", "key_facts", "risk", "explainer"}

    # The original file is available for the viewer
    file = await client.get(f"/documents/{document_id}/file")
    assert file.status_code == 200 and file.headers["content-type"] == "application/pdf"
    assert file.content == sample_pdf_bytes()


async def test_events_stream_reports_stages_and_ends(client: httpx.AsyncClient, app) -> None:
    document_id = await upload_sample(client, app)

    response = await client.get(f"/documents/{document_id}/events")

    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["snapshot", "done"]
    snapshot = events[0][1]
    assert snapshot["status"] == "completed"
    assert list(snapshot["stages"]) == ["ingest", "classify", "clauses", "key_facts", "risk", "explain"]
    assert events[1][1] == {"status": "completed"}


async def test_events_stream_live_updates(client: httpx.AsyncClient, app) -> None:
    import asyncio

    response = await client.post("/documents", files={"file": ("r.pdf", sample_pdf_bytes(), "application/pdf")})
    document_id = response.json()["document_id"]

    # Connect while the pipeline is still running; the stream ends when it finishes.
    stream = await asyncio.wait_for(client.get(f"/documents/{document_id}/events"), timeout=10)

    events = parse_sse(stream.text)
    names = [name for name, _ in events]
    assert names[0] == "snapshot" and names[-1] == "done"
    stage_updates = [(data["stage"], data["status"]) for name, data in events if name == "stage"]
    assert ("explain", "done") in stage_updates
    assert events[-2] == ("status", {"status": "completed", "error": None})


async def test_chat_answers_with_citations_and_refuses_when_not_covered(client: httpx.AsyncClient, app) -> None:
    document_id = await upload_sample(client, app)

    answer = await client.post(f"/documents/{document_id}/chat", json={"question": "Can I leave after 6 months?"})
    refusal = await client.post(f"/documents/{document_id}/chat", json={"question": "Is there a parking space?"})

    assert answer.status_code == 200
    body = answer.json()
    assert body["covered"] is True and body["removed_sentences"] == 1
    assert [s["clause_codes"] for s in body["answer"]] == [["C-07"], ["C-07", "C-06"], ["C-08"]]
    refused = refusal.json()
    assert refused["covered"] is False
    assert refused["answer"][0]["clause_codes"] == [] and "legal aid" in refused["answer"][0]["text"]

    history = (await client.get(f"/documents/{document_id}")).json()["answers"]
    assert [a["question"] for a in history] == ["Can I leave after 6 months?", "Is there a parking space?"]
    assert len(history[0]["sentences"]) == 3


async def test_reupload_is_served_from_cache(client: httpx.AsyncClient, app, llm: MockProvider) -> None:
    await upload_sample(client, app)
    calls = len(llm.calls)

    second = await upload_sample(client, app)

    assert len(llm.calls) == calls
    runs = (await client.get(f"/documents/{second}")).json()["agent_runs"]
    assert runs and all(r["cached"] for r in runs)


async def test_low_confidence_needs_confirmation(api_settings: Settings) -> None:
    llm = golden_provider(ClassifierOutput={"doc_type": "rental_agreement", "language": "en", "confidence": 0.4})
    app = create_app(api_settings, provider_for=lambda agent: llm)
    app.state.db.ensure_schema()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        document_id = await upload_sample(client, app)
        body = (await client.get(f"/documents/{document_id}")).json()
        assert body["status"] == "needs_confirmation" and body["clauses"] == []

        chat = await client.post(f"/documents/{document_id}/chat", json={"question": "Rent?"})
        assert chat.status_code == 409

        bad = await client.post(f"/documents/{document_id}/confirm-type", json={"doc_type": "recipe"})
        assert bad.status_code == 422

        confirmed = await client.post(f"/documents/{document_id}/confirm-type", json={"doc_type": "rental_agreement"})
        assert confirmed.status_code == 202
        await app.state.pipeline.wait(document_id)
        body = (await client.get(f"/documents/{document_id}")).json()
        assert body["status"] == "completed" and len(body["clauses"]) == 15

        again = await client.post(f"/documents/{document_id}/confirm-type", json={"doc_type": "rental_agreement"})
        assert again.status_code == 409
    app.state.db.dispose()


async def test_confirming_another_type_marks_it_unsupported(api_settings: Settings) -> None:
    llm = golden_provider(ClassifierOutput={"doc_type": "other", "language": "en", "confidence": 0.3})
    app = create_app(api_settings, provider_for=lambda agent: llm)
    app.state.db.ensure_schema()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        document_id = await upload_sample(client, app)

        response = await client.post(f"/documents/{document_id}/confirm-type", json={"doc_type": "loan_agreement"})

        assert response.status_code == 202
        assert response.json()["status"] == "unsupported" and response.json()["doc_type"] == "loan_agreement"
    app.state.db.dispose()


@pytest.mark.parametrize(
    ("content", "status", "message"),
    [
        (b"", 400, "The file is empty."),
        (b"just some text", 400, "Unsupported file type. Upload a PDF, JPG or PNG file."),
        (b"%PDF-1.4 broken", 400, "This PDF could not be opened. It may be damaged."),
    ],
)
async def test_upload_validation_errors(client: httpx.AsyncClient, content: bytes, status: int, message: str) -> None:
    response = await client.post("/documents", files={"file": ("x.pdf", content, "application/pdf")})

    assert response.status_code == status
    assert response.json() == {"detail": message}


async def test_upload_too_large_is_413(client: httpx.AsyncClient, app) -> None:
    app.state.settings.max_upload_mb = 1

    response = await client.post("/documents", files={"file": ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), "application/pdf")})

    assert response.status_code == 413
    assert response.json()["detail"].endswith("The limit is 1 MB.")


async def test_upload_over_page_limit_is_400(client: httpx.AsyncClient, app) -> None:
    app.state.settings.max_pages = 2

    response = await client.post("/documents", files={"file": ("long.pdf", sample_pdf_bytes(), "application/pdf")})

    assert response.status_code == 400 and response.json()["detail"] == "The PDF has 3 pages. The limit is 2 pages."


async def test_unknown_document_is_404(client: httpx.AsyncClient) -> None:
    for path in ("", "/events", "/file"):
        assert (await client.get(f"/documents/nope{path}")).status_code == 404
    assert (await client.post("/documents/nope/chat", json={"question": "hi"})).status_code == 404


async def test_delete_removes_everything(client: httpx.AsyncClient, app) -> None:
    document_id = await upload_sample(client, app)

    assert (await client.delete(f"/documents/{document_id}")).status_code == 204
    assert (await client.get(f"/documents/{document_id}")).status_code == 404


async def test_routes_are_also_served_under_api_prefix(client: httpx.AsyncClient, app) -> None:
    response = await client.post("/api/documents", files={"file": ("r.pdf", sample_pdf_bytes(), "application/pdf")})
    assert response.status_code == 202
    document_id = response.json()["document_id"]
    await app.state.pipeline.wait(document_id)

    assert (await client.get(f"/api/documents/{document_id}")).json()["status"] == "completed"
