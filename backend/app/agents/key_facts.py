"""F5 Key facts card: parties, money, dates, lock-in, notice and penalties, each with its clause."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.agents.batching import run_split
from app.agents.citations import normalize_code
from app.agents.grounding import ground_citation
from app.agents.text import batches, clause_block, clauses_text, utf8_len
from app.core.llm import LLMProvider
from app.core.logging import get_logger
from app.core.schemas import Clause
from app.prompts import key_facts as prompt

logger = get_logger(__name__)

AGENT = "key_facts"

FactName = Literal[
    "owner",
    "tenant",
    "property",
    "monthly_rent",
    "security_deposit",
    "start_date",
    "duration",
    "lock_in",
    "notice_period_owner",
    "notice_period_tenant",
    "late_payment_penalty",
    "early_exit_penalty",
    "other_penalty",
    "rent_increase",
]
# Facts that may appear more than once; every other name keeps its first value.
REPEATABLE: frozenset[str] = frozenset({"late_payment_penalty", "early_exit_penalty", "other_penalty"})
# Clause categories that never hold a key fact, left out to keep the request small.
SKIP_CATEGORIES = frozenset({"maintenance", "use_restrictions", "dispute_resolution"})


class KeyFactItem(BaseModel):
    name: FactName
    value: str
    clause_code: str


class KeyFactsOutput(BaseModel):
    facts: list[KeyFactItem]


async def extract_key_facts(llm: LLMProvider, clauses: Sequence[Clause], budget_bytes: int) -> list[KeyFactItem]:
    known = {c.clause_code for c in clauses}
    relevant = [c for c in clauses if c.category not in SKIP_CATEGORIES]
    facts: list[KeyFactItem] = []
    seen: set[str] = set()
    seen_values: set[tuple[str, str]] = set()
    dropped = recited = ungrounded = 0

    async def run(group: list[Clause]) -> KeyFactsOutput:
        return await llm.generate_structured(prompt.PROMPT, KeyFactsOutput, context=clauses_text(group), max_tokens=2048)

    outputs: list[KeyFactsOutput] = []
    for batch in batches(relevant, lambda c: utf8_len(clause_block(c)), budget_bytes):
        outputs.extend(await run_split(batch, run, agent=AGENT))
    for output in outputs:
        for fact in output.facts:
            fact.clause_code = normalize_code(fact.clause_code)
            if fact.clause_code not in known:
                dropped += 1
                continue
            if fact.name in seen and fact.name not in REPEATABLE:
                continue
            value = fact.value.strip()
            if not value or (fact.name, value.lower()) in seen_values:
                continue
            seen_values.add((fact.name, value.lower()))
            # The value's numbers must appear in the clause it cites (see grounding.py).
            code = ground_citation(value, fact.clause_code, clauses)
            if code is None:
                ungrounded += 1
                continue
            recited += code != fact.clause_code
            seen.add(fact.name)
            facts.append(KeyFactItem(name=fact.name, value=value, clause_code=code))
    if dropped:
        logger.warning("key_facts.unknown_clause_dropped", extra={"dropped": dropped})
    if recited or ungrounded:
        logger.warning("key_facts.citation_corrected", extra={"recited": recited, "dropped": ungrounded})
    return facts
