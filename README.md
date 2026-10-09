# Beyond the Fine Print

An explainable, agentic AI system that helps people in India understand government, legal and financial documents in their own language. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full spec.

**Status: Phase 0 (scaffold).** The repo has a FastAPI backend with `/health` and a React frontend that calls it. It also has config, PII-masked structured logging, core schemas, and the `LLMProvider` interface with two implementations: Anthropic and Groq. No agents exist yet.

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
| `GROQ_AGENT_MODELS` | `{}` | Per-agent text model, e.g. `{"classifier": "openai/gpt-oss-20b"}` |
| `GROQ_MAX_OUTPUT_TOKENS` | `2048` | Completion tokens requested per call |
| `LOG_FORMAT` | `json` | `json` or `console` |

## Using Groq

Set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and optionally `GROQ_VISION_MODEL=qwen/qwen3.8-27b`. How the Groq provider works:

- **Structured output:** it uses Groq's strict structured outputs. Pydantic schemas are rewritten to Groq's strict subset. Optional fields become nullable, and the nulls are dropped before validation, so defaults still apply.
- **Context:** `context` is prepended to the prompt as plain text. Groq has no prompt caching here.
- **Images:** at most 3 per request. Each one counts as about 2,048 input tokens.
- **Rate limiting:** a client-side limiter tracks each model's tokens-per-minute and requests-per-day from Groq's `x-ratelimit-*` headers. Requests per minute are tracked locally. Before sending a request that would go over a limit, it waits and logs `llm.rate_limiter_wait`. If the wait would be longer than `GROQ_MAX_LIMITER_WAIT_SECONDS`, it raises an error instead. A 429 is retried after the server's `retry-after`.
- **Smoke test:** makes 4 small real calls.
  ```powershell
  cd backend
  .venv\Scripts\python -m scripts.smoke_groq
  ```

> **The Groq free tier can't meet the spec's 60-second target.** Free-tier limits are about 8K tokens/minute, 30 requests/minute and 1K requests/day per model (checked 2026-10-02). One 10-page document is roughly 5–8K tokens, so a full pipeline run means the limiter waits between agents. Expect several minutes per document. The 60-second target applies to Anthropic or a paid Groq plan.

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
backend/app/      FastAPI app: api/, core/ (config, logging, schemas, llm/), agents/, prompts/, providers/, kb/
backend/tests/    pytest suite
frontend/         React + Vite + TypeScript + Tailwind (mobile-first)
data/             legal_corpus/, samples/, benchmark/  (team-collected; anonymised only)
eval/             evaluation scripts (Phase 6)
```
