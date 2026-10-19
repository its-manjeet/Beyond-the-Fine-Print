# Beyond the Fine Print

An explainable, agentic AI system that helps people in India understand government, legal and financial documents in their own language. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full spec.

**Status: Phase 1 backend (F1–F8) for English rental agreements.** Upload a rental agreement (PDF or phone photo) and the backend extracts its lines, classifies it, splits it into clauses, and produces a key facts card, a green/amber/red risk radar, plain-English explanations and grounded Q&A. Every statement cites the clause it came from, and every clause points to its page and lines, so the UI can highlight the source (F8). The React frontend still shows only the health check; the document UI is the next step. Phase 0 built the scaffold: config, PII-masked logging, and the `LLMProvider` interface with Anthropic and Groq implementations.

## Prerequisites

- Python 3.11+ (developed on 3.12)
- Node.js 20.17+ and npm
- An API key for your chosen LLM provider: Anthropic, or Groq (which has a free tier). You only need it for real LLM calls; the tests and `/health` run without one.

## Setup

Run all commands from the repository root. They're shown for Windows (PowerShell); macOS and Linux equivalents follow.

```powershell
# 1. Environment file
Copy-Item .env.example .env        # then set LLM_PROVIDER and that provider's API key

# 2. Backend
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements-dev.txt

# 3. Frontend
npm --prefix frontend install
```

macOS / Linux:

```bash
cp .env.example .env
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements-dev.txt
npm --prefix frontend install
```

## Run

Use two terminals.

```powershell
# Terminal 1: backend on http://127.0.0.1:8000  (API docs at /docs)
cd backend
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: frontend on http://localhost:5173
cd frontend
npm run dev
```

In development, Vite proxies `/health` and `/api/*` to the backend. To point it at a backend somewhere else, set `VITE_BACKEND_URL`. For a production build served separately from the API, set `VITE_API_BASE_URL`.

To test on a phone on the same Wi-Fi, run `npm run dev -- --host` and open the "Network" URL it prints.

## Test

```powershell
# Backend (pytest). No real API calls; the LLM is mocked.
cd backend
.venv\Scripts\python -m pytest

# Frontend (Vitest), plus typecheck and build
cd frontend
npm test
npm run build
```

## How a document is processed

After `POST /documents`, a background pipeline runs these stages and reports each one's status:

```
ingest -> classify -> clauses -> key_facts ┐
                                 risk      ├ (in parallel)
                                 explain   ┘
```

| Stage | Agent | What it does |
|---|---|---|
| `ingest` (F1) | `agents/ingestion.py` | Digital PDF pages: PyMuPDF words grouped into lines by vertical position, numbered from 1 on each page, with a bounding box per line (fractions of the page size). Scanned pages and photos go to the provider's vision model, which transcribes them line by line; Tesseract is used as a fallback if it is installed. Pages with very little text are flagged `low_confidence`. |
| `classify` (F2) | `agents/classifier.py` | Document type, language and confidence from the opening lines. Below 0.6 the document waits in `needs_confirmation` for `POST /documents/{id}/confirm-type`. Anything other than an English rental agreement stops as `unsupported`. |
| `clauses` (F3) | `agents/clauses.py` | The model sees numbered lines (`p1 L14: text`) and returns only each clause's code, title, category and span. The backend rejects spans that don't exist, retries once with the errors, and copies clause text from the stored lines, so clause text is never written by a model. Codes are `C-01`, `C-02`, ... in document order. |
| `key_facts` (F5) | `agents/key_facts.py` | Parties, property, rent, deposit, dates, lock-in, notice for each side, penalties, rent increase. Each fact cites a clause. Facts citing an unknown clause are dropped, and a fact whose numbers don't appear in its cited clause is moved to the clause that has them, or dropped. |
| `risk` (F6) | `agents/risk.py` | The model only reports each clause's category, risk patterns and rupee amount at stake. The score is computed in code (below). |
| `explain` (F4) | `agents/explainer.py` | 1–3 plain-English sentences per clause (Grade 6–8), each citing clause codes. |

**Risk score** (`agents/risk.py`): `R = w1*C_type + w2*P + w3*L + w4*M`, with weights from `RISK_WEIGHTS` (default 0.30, 0.50, 0.10, 0.10).

- `P = 1 - prod(1 - p_j)` over the patterns present, with severities such as one-sided termination 0.9 and lock-in with penalty 0.7.
- `C_type` is a fixed value per clause category.
- `L` is 0.3 ("unknown") until Phase 2 adds the legality checker.
- `M = min(1, amount / (3 * monthly rent))`.

Green below 0.40, amber below 0.70, red from 0.70. The one-line reason names the largest weighted factor, e.g. "Main factor: one-sided termination".

**Checks on every user-facing sentence** (`agents/citations.py`): a sentence that cites no clause, or a clause code that doesn't exist, is removed and the removal is logged (codes only, never text). A sentence that tells the user to sign or not sign is removed as a backstop to the prompts.

**Caching and audit:** every model call goes through `pipeline/tracking.py`. It caches the output by (document SHA-256, agent, prompt version, model) plus a fingerprint of the exact request, so uploading the same file again is answered from the cache. It also writes one `AGENT_RUN` row per call: agent, provider, model, prompt version, latency (including rate-limiter waits), tokens, and whether it was cached.

**Privacy:** documents, their derived results and the cache are deleted 24 hours after upload (`RETENTION_HOURS`); a cleanup job runs every 30 minutes. `DELETE /documents/{id}` deletes one immediately. Logs never contain document text (`tests/test_no_document_text_in_logs.py`).

Prompts live in `backend/app/prompts/`, one file per agent, each with a `VERSION`. Changing a prompt means bumping its version, which also invalidates its cache entries.

## API

All routes are also served under `/api` (for the Vite dev proxy). Interactive docs: `http://127.0.0.1:8000/docs`.

| Method and path | Purpose |
|---|---|
| `POST /documents` | Upload a PDF, JPG or PNG (multipart field `file`). Max 25 MB and 20 pages. Returns `202` with `document_id` and the stage list. Bad files get `400`, oversized files `413`, each with a clear `detail` message. |
| `GET /documents/{id}` | Everything so far: `status`, `stages` (each `pending`/`running`/`done`/`failed`/`skipped`), `pages` with numbered lines and boxes, `clauses` (span, text, `risk`, `explanation`), `key_facts`, `risk_summary`, `answers`, `agent_runs`, and the "not legal advice" `notice`. |
| `GET /documents/{id}/events` | Server-Sent Events. Sends a `snapshot` first, then `stage` and `status` events, and ends with `done` when processing stops (`completed`, `partial`, `failed`, `unsupported` or `needs_confirmation`). Close the stream on `done`. |
| `POST /documents/{id}/confirm-type` | `{"doc_type": "rental_agreement"}`. Allowed when the status is `needs_confirmation` or `unsupported`; processing resumes if the type is supported. |
| `POST /documents/{id}/chat` | `{"question": "..."}`. Answers only from the document, each sentence citing clauses. If the document doesn't cover it, `covered` is `false` and the answer suggests asking the other party or free legal aid. Returns `409` until clauses are ready. |
| `GET /documents/{id}/file` | The original upload, for the document viewer. |
| `DELETE /documents/{id}` | Delete the document and everything derived from it now. |

Document `status` is one of `processing`, `needs_confirmation`, `unsupported`, `completed`, `partial` (some analysis stages failed) or `failed`.

## Try it on the sample

The repo includes a fictitious 11-month leave and licence agreement for a flat in Pune: `data/samples/rental_agreement_sample.pdf` (3 pages, 14 numbered clauses), plus `data/samples/rental_page1.png`, a photo-like image of page 1. To regenerate them, run `.venv\Scripts\python -m scripts.make_rental_sample` from `backend/`.

With Groq (PowerShell; `GROQ_API_KEY` set in `.env`):

```powershell
# Terminal 1: backend using Groq. The vision model is only needed for photos and scans.
cd backend
$env:LLM_PROVIDER = "groq"
$env:GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
.venv\Scripts\python -m uvicorn app.main:app --port 8000

# Terminal 2, from the repo root: upload, watch progress, save results, ask a question
$doc = (curl.exe -s -F "file=@data/samples/rental_agreement_sample.pdf" http://127.0.0.1:8000/documents | ConvertFrom-Json).document_id
curl.exe -sN "http://127.0.0.1:8000/documents/$doc/events"
curl.exe -s "http://127.0.0.1:8000/documents/$doc" -o result.json
'{"question": "What happens if I leave after 6 months?"}' | Out-File -Encoding ascii question.json
curl.exe -s -X POST "http://127.0.0.1:8000/documents/$doc/chat" -H "Content-Type: application/json" --data-binary "@question.json"
```

To keep these settings, put `LLM_PROVIDER=groq` and `GROQ_VISION_MODEL=qwen/qwen3.8-27b` in `.env` instead. The photo works the same way with `file=@data/samples/rental_page1.png`.

On the Groq free tier the sample takes about a minute: the rate limiter waits between calls to stay under 8K tokens per minute. Uploading the same file again is answered from the cache in about a second.

## Configuration

All settings come from `.env` at the repo root; see `.env.example`. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Which `LLMProvider` implementation to use: `anthropic` or `groq` |
| `LLM_MAX_RETRIES` | `4` | Retries on rate limits, overload, 5xx and network errors (all providers) |
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-5-5` | Anthropic model |
| `LLM_SERVER_FALLBACK` | `true` | Re-run refused requests on Anthropic's fallback model |
| `GROQ_API_KEY` | (none) | Groq API key |
| `GROQ_TEXT_MODEL` | `openai/gpt-oss-120b` | Groq text model (must support strict structured outputs) |
| `GROQ_VISION_MODEL` | (none) | Groq model for requests with images. Only `qwen/qwen3.8-27b` accepts images. If unset, image requests raise an error. |
| `GROQ_AGENT_MODELS` | classifier and key_facts on `openai/gpt-oss-20b` | Per-agent text model (JSON) |
| `GROQ_MAX_OUTPUT_TOKENS` | `2048` | Completion tokens requested per call |
| `GROQ_REASONING_EFFORT` | `low` | Reasoning effort for gpt-oss models; reasoning tokens count toward the output cap |
| `DATABASE_URL` | `sqlite:///<repo>/data/app.db` | SQLAlchemy database URL |
| `RETENTION_HOURS` | `24` | Documents and everything derived from them are deleted after this |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | `25` / `20` | Upload limits |
| `CLASSIFIER_MIN_CONFIDENCE` | `0.6` | Below this, the user confirms the document type |
| `RISK_WEIGHTS` | `[0.30, 0.50, 0.10, 0.10]` | Risk weights for clause type, patterns, legality, money at stake |
| `TESSERACT_CMD` | `tesseract` | Optional OCR fallback, used only if found on PATH |
| `LOG_FORMAT` | `json` | `json` or `console` |

## Using Groq

Set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and optionally `GROQ_VISION_MODEL=qwen/qwen3.8-27b`. How the Groq provider works:

- **Structured output:** it uses Groq's strict structured outputs. Pydantic schemas are rewritten to Groq's strict subset. Optional fields become nullable, and the nulls are dropped before validation, so defaults still apply.
- **Context:** `context` is prepended to the prompt as plain text. Groq has no prompt caching here.
- **Images:** at most 3 per request. Each one counts as about 2,048 input tokens.
- **Reasoning:** gpt-oss models are sent `reasoning_effort` (`GROQ_REASONING_EFFORT`, default `low`). Reasoning tokens count toward `GROQ_MAX_OUTPUT_TOKENS`, and with Groq's default effort a 15-clause risk call used the whole 2,048-token cap and was cut off.
- **Strict-mode failures:** Groq sometimes answers `400 json_validate_failed` ("Failed to generate JSON"). That request is retried like a 5xx, and if it keeps failing it raises `LLMOutputError`. Agents then split the batch in half and try again.
- **Rate limiting:** a client-side limiter tracks each model's tokens-per-minute and requests-per-day from Groq's `x-ratelimit-*` headers. Requests per minute are tracked locally. Before sending a request that would go over a limit, it waits and logs `llm.rate_limiter_wait`. If the wait would be longer than `GROQ_MAX_LIMITER_WAIT_SECONDS`, it raises an error instead. A 429 is retried after the server's `retry-after`.
- **Smoke test:** makes 4 small real calls.
  ```powershell
  cd backend
  .venv\Scripts\python -m scripts.smoke_groq
  ```

> **The Groq free tier can't meet the spec's 60-second target.** Free-tier limits are about 8K tokens/minute, 30 requests/minute and 1K requests/day per model (checked 2026-10-02). The 3-page sample uses about 12K tokens on `gpt-oss-120b` across 7 calls (plus about 3K on `gpt-oss-20b`), so the limiter waits between them. Expect about a minute for the sample (66 seconds measured on 2026-10-09) and several minutes for a 10-page document. The 60-second target applies to Anthropic or a paid Groq plan.
>
> To stay within the limit, agents send only what they need: the classifier sees the opening ~2.5 KB, and key facts skip clauses that hold none. Each request carries at most about 9 KB of document text (`LLM_INPUT_BUDGET_BYTES`), and each call covers only a few clauses.

## Calling the LLM from an agent

```python
from pydantic import BaseModel
from app.core.llm import get_llm_provider

class Parties(BaseModel):
    landlord: str
    tenant: str

llm = get_llm_provider("key_facts")   # agent name selects a per-agent model where configured
parties = await llm.generate_structured(
    "Who are the landlord and tenant?",
    Parties,
    context=document_text,   # shared across agents and prompt-cached; keep it byte-identical
)
```

Agents must never import `anthropic` (or any other model SDK) directly. A test enforces this.

## Logging and privacy

Logs are structured JSON. Before a record is written, Aadhaar, PAN, phone and bank account numbers are masked (`[AADHAAR]`, `[PAN]`, `[PHONE]`, `[ACCOUNT]`). The mask applies to the message, its arguments and any `extra` fields. LLM calls log the model, latency and token counts (including cache reads and writes), never prompt or document text.

## Layout

```
backend/app/
  agents/         ingestion, classifier, clauses, key_facts, risk, explainer, qa, citations, grounding
  api/            health, documents
  core/           config, logging, schemas, llm/ (LLMProvider, Anthropic, Groq, rate limiter)
  pipeline/       orchestrator, tracking (cache + AGENT_RUN), store, events (SSE)
  prompts/        one versioned prompt per agent
  providers/ocr/  Tesseract fallback
  storage/        SQLAlchemy models, database, 24-hour cleanup
backend/scripts/  make_rental_sample.py, smoke_groq.py
backend/tests/    pytest suite; golden/ holds the mock model responses for the sample
frontend/         React + Vite + TypeScript + Tailwind (mobile-first)
data/             legal_corpus/, samples/, benchmark/  (team-collected; anonymised only)
eval/             evaluation scripts (Phase 6)
```
