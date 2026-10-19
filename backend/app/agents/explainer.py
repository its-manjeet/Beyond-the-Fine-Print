"""F4 Plain-language explainer: 1-3 sentences per clause, each with clause citations."""

from collections.abc import Sequence

from pydantic import BaseModel

from app.agents.batching import chunks, run_split
from app.agents.citations import CitedSentence, check_sentences, normalize_code
from app.agents.text import batches, clause_block, clauses_text, utf8_len
from app.core.llm import LLMProvider
from app.core.logging import get_logger
from app.core.schemas import Clause
from app.prompts import explainer as prompt

logger = get_logger(__name__)

AGENT = "explainer"
MAX_SENTENCES = 3
# Clauses per call: bounds the output size as well as the input.
MAX_CLAUSES_PER_CALL = 4


class ClauseExplanation(BaseModel):
    clause_code: str
    sentences: list[CitedSentence]


class ExplainerOutput(BaseModel):
    explanations: list[ClauseExplanation]


class ExplainResult(BaseModel):
    explanations: list[ClauseExplanation]
    removed: int = 0


async def explain_clauses(llm: LLMProvider, clauses: Sequence[Clause], budget_bytes: int) -> ExplainResult:
    known = {c.clause_code for c in clauses}
    found: dict[str, ClauseExplanation] = {}
    removed = 0

    async def run(group: list[Clause]) -> ExplainerOutput:
        return await llm.generate_structured(prompt.PROMPT, ExplainerOutput, context=clauses_text(group), max_tokens=2048)

    outputs: list[ExplainerOutput] = []
    for batch in batches(list(clauses), lambda c: utf8_len(clause_block(c)), budget_bytes):
        for group in chunks(batch, MAX_CLAUSES_PER_CALL):
            outputs.extend(await run_split(group, run, agent=AGENT))
    for output in outputs:
        for item in output.explanations:
            code = normalize_code(item.clause_code)
            if code not in known or code in found:
                continue
            kept, dropped = check_sentences(item.sentences[:MAX_SENTENCES], known, agent=AGENT)
            removed += dropped + max(0, len(item.sentences) - MAX_SENTENCES)
            if kept:
                found[code] = ClauseExplanation(clause_code=code, sentences=kept)

    order = {c.clause_code: i for i, c in enumerate(clauses)}
    explanations = sorted(found.values(), key=lambda e: order[e.clause_code])
    missing = len(known) - len(explanations)
    if missing:
        logger.info("explainer.clauses_without_explanation", extra={"count": missing})
    return ExplainResult(explanations=explanations, removed=removed)
