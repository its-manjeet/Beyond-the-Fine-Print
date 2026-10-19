"""F7 Grounded Q&A: answers come only from the document's clauses, with clause citations."""

import re
from collections.abc import Sequence

from pydantic import BaseModel

from app.agents.citations import CitedSentence, check_sentences
from app.agents.text import clause_block, clauses_text, utf8_len
from app.core.llm import LLMProvider
from app.core.schemas import Clause
from app.prompts import qa as prompt

AGENT = "qa"
MAX_SENTENCES = 4

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are can do does for from have how i if in is it me my of on or the to what when where which who why will with you your".split()
)
# Everyday words mapped to the words rental agreements use.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "rent": ("licence", "fee", "rent"),
    "landlord": ("licensor", "owner"),
    "owner": ("licensor", "landlord"),
    "tenant": ("licensee",),
    "leave": ("vacate", "terminate", "termination", "notice", "lock"),
    "move": ("vacate", "terminate", "notice"),
    "early": ("lock", "terminate", "notice"),
    "deposit": ("deposit", "refund"),
    "increase": ("increase", "increased", "escalation"),
    "repair": ("repairs", "maintenance"),
    "pet": ("pets",),
    "late": ("late", "delay"),
}


class QAOutput(BaseModel):
    covered: bool
    sentences: list[CitedSentence]


class Answer(BaseModel):
    covered: bool
    sentences: list[CitedSentence]
    removed: int = 0


def select_clauses(clauses: Sequence[Clause], question: str, budget_bytes: int) -> list[Clause]:
    """All clauses if they fit the budget; otherwise the best keyword matches that fit."""
    if sum(utf8_len(clause_block(c)) for c in clauses) <= budget_bytes:
        return list(clauses)
    words = {w for w in _WORD.findall(question.lower()) if w not in _STOPWORDS}
    for word in list(words):
        words.update(_SYNONYMS.get(word.rstrip("s"), ()))
    terms = {_stem(w) for w in words}

    def score(clause: Clause) -> int:
        title = {_stem(w) for w in _WORD.findall(clause.title.lower())}
        body = {_stem(w) for w in _WORD.findall(clause.text.lower())}
        return sum(2 * (t in title) + (t in body) for t in terms)

    ranked = sorted(clauses, key=score, reverse=True)
    chosen: list[Clause] = []
    used = 0
    for clause in ranked:
        size = utf8_len(clause_block(clause))
        if used + size > budget_bytes:
            continue
        chosen.append(clause)
        used += size
    order = {c.clause_code: i for i, c in enumerate(clauses)}
    return sorted(chosen, key=lambda c: order[c.clause_code])


def _stem(word: str) -> str:
    """Crude prefix stemming: 'vacates'/'vacate' and 'terminate'/'termination' match."""
    return word[:5]


async def answer_question(llm: LLMProvider, clauses: Sequence[Clause], question: str, budget_bytes: int) -> Answer:
    selected = select_clauses(clauses, question, budget_bytes)
    output = await llm.generate_structured(
        prompt.PROMPT.format(question=question.strip()), QAOutput, context=clauses_text(selected), max_tokens=1500
    )
    # Citations are checked against every clause in the document, not just those sent.
    kept, removed = check_sentences(output.sentences[:MAX_SENTENCES], {c.clause_code for c in clauses}, agent=AGENT)
    if not output.covered or not kept:
        return Answer(covered=False, sentences=[CitedSentence(text=prompt.NOT_COVERED, clause_codes=[])], removed=removed)
    return Answer(covered=True, sentences=kept, removed=removed)
