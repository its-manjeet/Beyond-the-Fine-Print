"""Privacy: a full run on the sample, including Q&A and failures, logs no document text."""

import logging

import httpx

from app.core.config import Settings
from app.core.llm import LLMOutputError
from app.main import create_app
from tests.mocks import sample_pdf_bytes
from tests.pipeline_fixtures import golden_provider

# Phrases that occur only in the document, its clauses, the model's answers or the user's question.
DOCUMENT_PHRASES = [
    "Kulkarni",
    "Ananya",
    "Sai Kiran",
    "Gulmohar",
    "licence fee of Rs. 18,000",
    "terminate this licence",
    "Courts usually do not allow",  # a removed explainer sentence
    "Most courts would let you",  # a removed Q&A sentence
    "leave after 6 months",  # the user's question
]


async def test_full_run_logs_no_document_text(settings: Settings, caplog) -> None:
    settings.database_url = "sqlite://"
    llm = golden_provider(KeyFactsOutput=LLMOutputError("value 'Suresh Vasant Kulkarni' failed validation"))
    app = create_app(settings, provider_for=lambda agent: llm)
    app.state.db.ensure_schema()

    with caplog.at_level(logging.DEBUG):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/documents", files={"file": ("r.pdf", sample_pdf_bytes(), "application/pdf")})
            document_id = response.json()["document_id"]
            await app.state.pipeline.wait(document_id)
            await client.post(f"/documents/{document_id}/chat", json={"question": "What if I leave after 6 months?"})

    app.state.db.dispose()
    assert caplog.records, "expected the run to log something"
    logged = "\n".join(
        f"{record.getMessage()} {vars(record)}" for record in caplog.records if record.name.startswith("app")
    )
    leaked = [phrase for phrase in DOCUMENT_PHRASES if phrase in logged]
    assert leaked == []
