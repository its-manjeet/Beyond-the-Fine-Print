"""F3 Clause extractor.

The model sees numbered lines ("p1 L14: text") and returns only each clause's code,
title, category and span. The backend checks every span against the stored lines,
retries once with the errors, drops spans that are still invalid, and copies the
clause text from the lines itself, so clause text can never be invented.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.agents.batching import run_split
from app.agents.text import batches, numbered_lines, utf8_len
from app.core.llm import LLMProvider
from app.core.logging import get_logger
from app.core.schemas import Clause, DocLine, RentalCategory
from app.prompts import clauses as prompt

logger = get_logger(__name__)

AGENT = "clauses"


class Span(BaseModel):
    page: int
    line_start: int
    line_end: int
    page_end: int | None = Field(default=None, description="Only if the clause continues onto a later page")


class ExtractedClause(BaseModel):
    clause_code: str
    title: str
    category: RentalCategory
    span: Span


class ClauseExtraction(BaseModel):
    clauses: list[ExtractedClause]


@dataclass
class ClauseResult:
    clauses: list[Clause]
    dropped: int = 0
    retried: bool = False
    errors: list[str] = field(default_factory=list)


class LineIndex:
    """Positions of the stored lines, for span checks and copying clause text."""

    def __init__(self, lines: Sequence[DocLine]) -> None:
        self.lines = list(lines)
        self.position = {(line.page, line.line_number): i for i, line in enumerate(self.lines)}
        self.page_sizes: dict[int, int] = {}
        for line in self.lines:
            self.page_sizes[line.page] = max(self.page_sizes.get(line.page, 0), line.line_number)

    def span_error(self, item: ExtractedClause) -> str | None:
        span = item.span
        page_end = span.page_end or span.page
        start, end = (span.page, span.line_start), (page_end, span.line_end)
        label = f"{item.clause_code}: span p{span.page} L{span.line_start} to p{page_end} L{span.line_end}"
        for page, line in (start, end):
            if page not in self.page_sizes:
                return f"{label} uses page {page}, which is not in the lines given"
            if (page, line) not in self.position:
                return f"{label} uses line {line} on page {page}, which has lines 1 to {self.page_sizes[page]}"
        if self.position[end] < self.position[start]:
            return f"{label} ends before it starts"
        return None

    def text(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        return "\n".join(line.text for line in self.lines[self.position[start] : self.position[end] + 1])


async def extract_clauses(llm: LLMProvider, lines: Sequence[DocLine], budget_bytes: int) -> ClauseResult:
    """Extract clauses page-batch by page-batch, keeping each request within ``budget_bytes``."""
    pages: dict[int, list[DocLine]] = {}
    for line in lines:
        pages.setdefault(line.page, []).append(line)
    page_batches = batches(
        list(pages.values()), lambda page: sum(utf8_len(l.text) + 10 for l in page), budget_bytes
    )

    result = ClauseResult(clauses=[])
    accepted: list[ExtractedClause] = []
    full_index = LineIndex(lines)

    async def run(batch: list[list[DocLine]]) -> tuple[list[ExtractedClause], int, bool, list[str]]:
        return await _extract_batch(llm, [line for page in batch for line in page])

    for batch in page_batches:
        # A batch whose output is cut off is split by pages and tried again.
        for items, dropped, retried, errors in await run_split(batch, run, agent=AGENT):
            accepted.extend(items)
            result.dropped += dropped
            result.retried |= retried
            result.errors.extend(errors)

    accepted.sort(key=lambda c: full_index.position[(c.span.page, c.span.line_start)])
    for number, item in enumerate(accepted, start=1):
        span = item.span
        page_end = span.page_end or span.page
        result.clauses.append(
            Clause(
                # Codes are reassigned in document order so they stay unique across batches.
                clause_code=f"C-{number:02d}",
                title=item.title.strip() or f"Clause {number}",
                category=item.category,
                page=span.page,
                line_start=span.line_start,
                page_end=page_end,
                line_end=span.line_end,
                text=full_index.text((span.page, span.line_start), (page_end, span.line_end)),
            )
        )
    return result


async def _extract_batch(
    llm: LLMProvider, batch_lines: list[DocLine]
) -> tuple[list[ExtractedClause], int, bool, list[str]]:
    index = LineIndex(batch_lines)
    context = numbered_lines(batch_lines)
    output = await llm.generate_structured(prompt.PROMPT, ClauseExtraction, context=context, max_tokens=4096)
    valid, errors = _check(output.clauses, index)
    if not errors:
        return valid, 0, False, []

    logger.warning("clauses.invalid_spans_retrying", extra={"invalid": len(errors)})
    retry_prompt = prompt.PROMPT + prompt.RETRY.format(errors="\n".join(f"- {e}" for e in errors))
    output = await llm.generate_structured(retry_prompt, ClauseExtraction, context=context, max_tokens=4096)
    valid, still_invalid = _check(output.clauses, index)
    if still_invalid:
        logger.warning("clauses.invalid_spans_dropped", extra={"dropped": len(still_invalid)})
    return valid, len(still_invalid), True, still_invalid


def _check(items: list[ExtractedClause], index: LineIndex) -> tuple[list[ExtractedClause], list[str]]:
    """Split items into those with valid, non-overlapping spans and error messages for the rest."""
    valid: list[ExtractedClause] = []
    errors: list[str] = []
    for item in items:
        error = index.span_error(item)
        if error:
            errors.append(error)
        else:
            valid.append(item)

    valid.sort(key=lambda c: index.position[(c.span.page, c.span.line_start)])
    kept: list[ExtractedClause] = []
    last_end = -1
    for item in valid:
        start = index.position[(item.span.page, item.span.line_start)]
        if start <= last_end:
            errors.append(f"{item.clause_code}: span overlaps the previous clause")
            continue
        kept.append(item)
        last_end = index.position[(item.span.page_end or item.span.page, item.span.line_end)]
    return kept, errors
