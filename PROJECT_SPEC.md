# Beyond the Fine Print — Project Spec

An explainable, agentic AI system that helps ordinary people in India understand government, legal and financial documents in their own language, check whether the terms are fair and legal, and know exactly what to do next.

This file is the single source of truth for the build. Read it fully before writing code, and re-read the relevant phase before starting it.

---

## 1. Goals

1. **Understand**: explain any supported document in plain language, in the user's language, at a simple reading level.
2. **Trust**: every explanation is traced to the exact line in the document and, where relevant, to the exact legal source. A verifier checks claims before they reach the user. The system says "I'm not sure" instead of guessing.
3. **Protect**: flag risky, unusual, or possibly illegal clauses, and detect fake or scam documents.
4. **Act**: turn the document into concrete next steps (deadlines, documents needed, who to contact) and help draft replies or complaints.
5. **Reach**: work from a phone photo, with voice input and output, in Indian languages, for people with low literacy.

## 2. Supported document types

| Type | Examples |
|---|---|
| Rental agreement | Leave and licence, lease deed |
| Insurance policy | Health, life, motor policy wordings |
| Loan agreement | Personal, home, gold, app-based digital loans |
| Legal notice / summons | Advocate's notice, court summons, recovery notice |
| Government scheme | Scheme guidelines, eligibility rules, application forms |
| Banking / RBI circular | Bank letters, RBI notifications affecting customers |

Build depth-first: rental agreements and insurance policies fully first, then extend.

## 3. Feature list

### Phase 1 — Core understanding (MVP)
- **F1 Upload & ingest**: PDF, image (JPG/PNG), or phone camera photo. Multi-page support. Extract text with page and line positions preserved.
- **F2 Document classifier**: detect document type and language; ask the user to confirm if confidence is low.
- **F3 Clause extractor**: split the document into clauses/sections with stable IDs and source spans (page, line range).
- **F4 Plain-language explainer**: per-clause and whole-document explanation; reading level set to roughly class 6–8; every sentence carries a citation to the clause ID it came from.
- **F5 Key facts card**: parties, amounts, dates, duration, notice period, penalties, renewal terms, extracted into a structured summary.
- **F6 Risk radar**: label each clause green / amber / red with a one-line reason (e.g. lock-in with penalty, auto-renewal, one-sided termination, broad exclusions, unlimited liability, arbitration in a distant city).
- **F7 Grounded Q&A chat**: user asks questions about their document; answers cite clause IDs; refuses to answer from outside the document unless using the legal knowledge base (Phase 2).
- **F8 Click-to-source UI**: tapping any explanation highlights the original text in the document viewer.

### Phase 2 — Legal grounding & verification
- **F9 Legal knowledge base (RAG)**: indexed corpus of bare acts, regulations and circulars, chunked by section with metadata (act name, section number, year, jurisdiction, source URL).
- **F10 Legality checker**: for each clause, retrieve relevant law and state whether it appears consistent, possibly inconsistent, or unknown, citing the specific section.
- **F11 Verifier agent**: independently checks every generated claim against the document text and retrieved law; removes or flags unsupported claims; produces a confidence score per claim.
- **F12 Uncertainty & disclaimers**: low-confidence items are shown as "uncertain — consider asking a lawyer / legal aid". Persistent "not legal advice" notice. Link to free legal aid (NALSA / State Legal Services Authority).

### Phase 3 — Action
- **F13 Action planner**: extract deadlines, required documents, offices/authorities, and steps into a checklist; export to calendar (.ics).
- **F14 What-if simulator**: user asks a scenario ("what if I miss 2 EMIs", "what if I leave after 6 months") and gets consequences computed from the document's actual terms, with rupee amounts where possible.
- **F15 Response drafter**: draft a reply to a legal notice, an RTI application, a bank/insurance grievance, a banking ombudsman complaint, or a consumer commission complaint, pre-filled from the document. Always marked as a draft for the user to review.

### Phase 4 — Accessibility
- **F16 Multilingual output**: Hindi, Marathi, Tamil, Telugu, Bengali, Kannada, Gujarati, plus English. Language selectable; legal terms kept with a simple gloss.
- **F17 Voice in / voice out**: speak questions and listen to explanations. Pluggable speech provider.
- **F18 Simple mode**: icon-led, large-text, audio-first UI for low-literacy users.

### Phase 5 — Protection & comparison
- **F19 Scam / fake document detector**: flags signs of fake summons, fake RBI/government letters, and predatory loan-app notices (wrong authority names, impossible section numbers, payment demands to personal UPI IDs, threats, urgency language, mismatched letterheads). Outputs signals with reasons, never a definitive "fake" verdict.
- **F20 Scheme eligibility engine**: short voice/form questionnaire → list of schemes the user may qualify for, with the rule each answer satisfied, and documents needed to apply.
- **F21 Document comparison**: compare two policies or two loan offers side by side; highlight differences in costs, exclusions and penalties.

### Phase 6 — Evaluation & research
- **F22 Benchmark dataset**: ~50 real, anonymised documents with hand-labelled clauses, risk labels and key facts.
- **F23 Evaluation harness**: scripts measuring clause extraction accuracy, risk-label precision/recall, key-fact accuracy, citation correctness, and hallucination rate (claims rejected by verifier or by human review).
- **F24 User study support**: comprehension quiz before/after using the system; anonymous result logging.

## 4. Architecture

### Agent pipeline
```
Upload → Ingestion/OCR → Classifier → Clause Extractor
       → Key Facts Extractor
       → Risk Analyzer ──┐
       → Legal Retriever ┼→ Legality Checker
                         ↓
                   Explainer (language + reading level)
                         ↓
                      Verifier  ──→  rejected claims dropped / flagged
                         ↓
                  Action Planner
                         ↓
                     Response to UI
```
- Each agent is a separate module with a typed input/output schema (Pydantic). No agent passes free text to another where structured data is possible.
- Orchestration in plain Python (async), with each step's inputs, outputs, model, prompt version and latency logged for explainability and debugging.
- Every claim object carries: `text`, `clause_ids[]`, `legal_refs[]`, `confidence`, `verified: bool`.

### LLMProvider interface
All model calls go through one provider-neutral interface in `backend/app/core/llm/`. **No agent imports a model SDK directly.** Each SDK may be imported only by its own provider module (`anthropic` by `anthropic_provider.py`; `openai`/`groq` by `groq_provider.py`). A test (`tests/test_no_direct_sdk_import.py`) fails the build otherwise.

```python
class LLMProvider(ABC):
    async def generate_structured(
        self,
        prompt: str,                                   # task-specific instruction / question
        schema: type[T],                               # Pydantic model; result is validated against it
        *,
        images: Sequence[ImageInput] | None = None,    # images specific to this call
        context: str | Sequence[str | ImageInput] | None = None,  # large shared content (document text, page images)
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> T: ...
```

- **Selection**: `LLM_PROVIDER` in `.env` (`anthropic` | `groq`) picks the implementation from a registry in `factory.py`. Agents get it with `get_llm_provider("<agent name>")`. The agent name lets providers apply a per-agent model override; providers without overrides ignore it. To add another provider (e.g. Gemini), write one module implementing `LLMProvider` and add one registry entry. Agent code does not change.
- **Request order**: `system` → `context` → `images` → `prompt`. `context` is the content many agents reuse for one document. Providers that support prompt caching cache it, and other providers may ignore caching. For agents to share the cached context, pass it byte-for-byte identical and keep `system` identical (or `None`), with agent-specific instructions in `prompt`.
- **Errors**: every failure is raised as a subclass of `LLMError`:
  - `LLMRateLimitError` / `LLMUnavailableError`: retries exhausted
  - `LLMRequestError`: rejected request, not retried
  - `LLMRefusalError`: the model declined
  - `LLMOutputError`: output failed schema validation or was truncated
- **Anthropic provider** (`anthropic_provider.py`, the only module importing the SDK):
  - **Structured output**: uses native structured outputs (`output_config.format` with the JSON schema from the Pydantic model), then validates with Pydantic. If validation fails, it retries once and shows the model the validation error.
  - **Prompt caching**: marks the last `context` block with `cache_control`.
  - **Retries**: the SDK's own retries are disabled. The provider retries 408/409/429/5xx/529 and network errors with exponential backoff plus full jitter, honouring `retry-after` and capped by `LLM_BACKOFF_MAX_SECONDS`.
  - **Refusal fallback**: opts into server-side refusal fallback (`LLM_SERVER_FALLBACK`).
  - **Logging**: every call logs provider, model, schema, latency, attempt, stop reason, input/output tokens, and cache read/write tokens. Prompt and document text are never logged.
- **Groq provider** (`groq_provider.py`, Groq's OpenAI-compatible API via the `openai` SDK):
  - **Models**: `GROQ_TEXT_MODEL` (default `openai/gpt-oss-120b`), with per-agent overrides in `GROQ_AGENT_MODELS`. Requests with images go to `GROQ_VISION_MODEL`; only `qwen/qwen3.8-27b` accepts images. Images with no vision model configured raise `LLMRequestError`, as do more than 3 images.
  - **Structured output**: strict JSON-schema mode (constrained decoding). The Pydantic schema is rewritten to Groq's strict subset: every property required, objects closed, optional fields made nullable. Those nulls are dropped before Pydantic validation, so defaults apply. Same corrective retry as Anthropic. Strict mode with images was confirmed to work on 2026-10-02.
  - **Context**: prepended to the prompt as text. No caching.
  - **Rate limiting**: a process-wide client-side limiter, keyed per model (`GROQ_RATE_LIMIT_SCOPE`). Tokens-per-minute and requests-per-day come from each response's `x-ratelimit-*` headers, and requests-per-minute from a local sliding window.
    - Before sending, it estimates the request's tokens (input, ~2,048 per image, and the requested output) and waits if that would exceed a limit, logging `llm.rate_limiter_wait`.
    - If the wait would exceed `GROQ_MAX_LIMITER_WAIT_SECONDS`, it raises `LLMRateLimitError`. A request larger than the whole tokens-per-minute limit raises `LLMRequestError`.
    - A 429 with `retry-after` blocks the bucket for exactly that long. Other retryable errors use the same exponential backoff as Anthropic.
    - Live headers on 2026-10-02 showed the limits are tracked separately per model: 8K tokens/minute and 1K requests/day on the free tier.
  - **Output size**: requested output is capped at `GROQ_MAX_OUTPUT_TOKENS` (default 2048), because it counts toward the tokens-per-minute limit.
  - **Reasoning**: gpt-oss models are sent `reasoning_effort` (`GROQ_REASONING_EFFORT`, default `low`). Their reasoning tokens count toward the output cap; on 2026-10-09, the default effort used the whole 2,048 tokens on a 15-clause risk call and cut the answer off.
  - **Generation failures**: a 400 that carries `failed_generation` (e.g. `json_validate_failed`, "Parsing failed") is retried like a 5xx. If it persists, it raises `LLMOutputError`, and agents split the batch in half and retry.
  - **Performance**: the 60-second processing target (section 6) does not apply on the Groq free tier.
- **Testing**: agent tests use a mock `LLMProvider`. Provider tests inject a fake SDK client. No test makes a real API call.

### Tech stack
- **Backend**: Python 3.11+, FastAPI, Pydantic v2, SQLite (dev) / PostgreSQL (prod) via SQLAlchemy.
- **LLM**: Accessed only through the `LLMProvider` interface above. The default provider is Claude via the Anthropic Python SDK. Model name read from env `LLM_MODEL` (default `claude-sonnet-5-5`). Use structured outputs for every agent. Use vision input for scanned pages.
- **OCR**: Claude vision as primary; Tesseract (with `hin`, `mar`, `tam` etc. language packs) as offline fallback. Keep page/line coordinates.
- **PDF**: PyMuPDF for text-layer PDFs and page rendering.
- **Vector store**: ChromaDB with a multilingual embedding model (e.g. `BAAI/bge-m3` or `intfloat/multilingual-e5-large`).
- **Translation / speech**: provider interface with implementations for (a) Claude for translation, (b) Bhashini / AI4Bharat models (IndicTrans2, Indic ASR/TTS), (c) browser Web Speech API as frontend fallback.
- **Frontend**: React + Vite + TypeScript + Tailwind, mobile-first, installable PWA. PDF viewer with highlight overlay (react-pdf / pdf.js).
- **Testing**: pytest for backend, Vitest for frontend. Golden-file tests for agents using fixture documents.

### Repository layout
```
/backend
  /app
    /agents        ingestion.py classifier.py clauses.py key_facts.py risk.py
                   retriever.py legality.py explainer.py verifier.py actions.py
                   whatif.py drafter.py scam.py eligibility.py compare.py
    /api           routes for documents, chat, actions, voice, eval
    /core          config, logging (PII-masked), schemas
      /llm         LLMProvider interface, provider factory, rate_limiter.py,
                   anthropic_provider.py, groq_provider.py
    /prompts       one versioned prompt file per agent
    /providers     ocr/, translation/, speech/ (pluggable)
    /kb            legal corpus loader, chunker, indexer
  /tests
/frontend
/data
  /legal_corpus    source files + metadata.json (collected by the team)
  /samples         anonymised sample documents
  /benchmark       labelled evaluation set
/eval              evaluation scripts and reports
PROJECT_SPEC.md
```

## 5. Legal knowledge base sources (to be collected by the team)

Initial corpus, downloaded from official sources (India Code, RBI, IRDAI, state government sites):
- Indian Contract Act, 1872
- Model Tenancy Act, 2021, and the relevant state rent control act(s)
- Consumer Protection Act, 2019
- IRDAI regulations on protection of policyholders' interests and health insurance
- RBI digital lending rules, fair practices code, and guidelines on recovery agents
- RBI Integrated Ombudsman Scheme
- Bharatiya Nagarik Suraksha Sanhita, 2023 (summons provisions) and Code of Civil Procedure, 1908 (civil summons)
- Right to Information Act, 2005
- Legal Services Authorities Act, 1987 (free legal aid)
- Guidelines of the selected government schemes

**Rule: the system must never state legal facts from the model's memory.** Every legal statement must come from a retrieved corpus chunk with its citation. If nothing relevant is retrieved, the answer is "unknown". Always check that the corpus is current, since laws and regulations get amended and replaced.

## 6. Non-functional requirements

- **Privacy**: documents processed and then deleted by default (configurable retention, default 24h). Mask Aadhaar, PAN, phone numbers and account numbers before any logging. No document text in logs.
- **Safety**: persistent "not legal advice" notice; never tell a user to sign or not sign; frame outputs as information and options. Scam detector gives signals, not verdicts.
- **Explainability**: every user-facing statement is clickable back to its source span and legal reference; a "why this?" panel shows which agent produced it and its confidence.
- **Performance**: 10-page document fully processed in under 60 seconds; stream partial results to the UI.
- **Cost control**: cache per-document agent outputs; use smaller/cheaper models for classification where accuracy allows.
- **Code quality**: typed code, small modules, tests for every agent, README with setup steps, `.env.example`, no secrets in the repo.

## 7. Build order

Build one phase at a time. At the end of each phase: all tests pass, the app runs end to end, and the README is updated. Do not start the next phase until told.

1. Phase 0 — Scaffold: repo layout, FastAPI + React skeleton, config, LLM client wrapper, logging, CI-friendly test setup, sample documents.
2. Phase 1 — F1–F8, rental agreements and insurance policies first.
3. Phase 2 — F9–F12.
4. Phase 3 — F13–F15.
5. Phase 4 — F16–F18.
6. Phase 5 — F19–F21.
7. Phase 6 — F22–F24.
