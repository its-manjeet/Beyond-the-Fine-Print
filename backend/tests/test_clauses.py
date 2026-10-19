"""F3: clause extraction, span validation and the corrective retry. The LLM is mocked."""

import copy

import pytest

from app.agents.clauses import extract_clauses
from app.agents.ingestion import ingest, validate_upload
from app.core.config import Settings
from app.core.schemas import DocLine
from tests.mocks import MockProvider, load_golden, sample_pdf_bytes

BUDGET = 9000


@pytest.fixture
async def sample_lines(settings: Settings) -> list[DocLine]:
    data = sample_pdf_bytes()
    result = await ingest(data, validate_upload(data, settings), None, settings)
    return [DocLine(page=p.page_number, line_number=l.line_number, text=l.text) for p in result.pages for l in p.lines]


def golden() -> dict:
    return copy.deepcopy(load_golden("clauses"))


async def test_sample_clauses_get_text_copied_from_lines(sample_lines: list[DocLine]) -> None:
    llm = MockProvider({"ClauseExtraction": golden()})

    result = await extract_clauses(llm, sample_lines, BUDGET)

    assert len(llm.calls) == 1 and not result.retried and result.dropped == 0
    codes = [c.clause_code for c in result.clauses]
    assert codes == [f"C-{n:02d}" for n in range(1, 16)]
    rent = result.clauses[3]
    assert (rent.title, rent.category, rent.page, rent.line_start, rent.line_end) == ("Licence Fee", "rent_payment", 1, 18, 21)
    assert rent.text.splitlines()[0] == "3. Licence Fee"
    assert "Rs. 18,000/-" in rent.text and "on or before the 5th day" in rent.text
    termination = result.clauses[7]
    assert termination.page == 2 and termination.text.startswith("7. Termination\nThe Licensor may terminate")
    # The model saw numbered lines, and was never asked to write clause text.
    context = llm.calls[0]["context"]
    assert "p1 L18: 3. Licence Fee" in context and "p2 L1: 7. Termination" in context


async def test_invalid_spans_trigger_one_retry_with_the_errors(sample_lines: list[DocLine]) -> None:
    bad = golden()
    bad["clauses"][7]["span"] = {"page": 2, "line_start": 40, "line_end": 44}  # page 2 has 33 lines
    bad["clauses"][8]["span"] = {"page": 4, "line_start": 1, "line_end": 2}  # no page 4
    llm = MockProvider({"ClauseExtraction": [bad, golden()]})

    result = await extract_clauses(llm, sample_lines, BUDGET)

    assert len(llm.calls) == 2 and result.retried and result.dropped == 0
    assert len(result.clauses) == 15
    retry_prompt = llm.calls[1]["prompt"]
    assert "C-08: span p2 L40 to p2 L44 uses line 40 on page 2, which has lines 1 to 33" in retry_prompt
    assert "C-09: span p4 L1 to p4 L2 uses page 4, which is not in the lines given" in retry_prompt


async def test_spans_still_invalid_after_retry_are_dropped(sample_lines: list[DocLine]) -> None:
    bad = golden()
    bad["clauses"][2]["span"] = {"page": 1, "line_start": 17, "line_end": 15}  # ends before it starts
    llm = MockProvider({"ClauseExtraction": [copy.deepcopy(bad), bad]})

    result = await extract_clauses(llm, sample_lines, BUDGET)

    assert len(llm.calls) == 2 and result.dropped == 1
    assert len(result.clauses) == 14
    assert [c.title for c in result.clauses][:3] == ["Parties", "Premises", "Licence Fee"]
    assert [c.clause_code for c in result.clauses] == [f"C-{n:02d}" for n in range(1, 15)]  # renumbered


async def test_overlapping_span_is_an_error(sample_lines: list[DocLine]) -> None:
    bad = golden()
    bad["clauses"][4]["span"] = {"page": 1, "line_start": 20, "line_end": 24}  # overlaps Licence Fee
    llm = MockProvider({"ClauseExtraction": [bad, golden()]})

    result = await extract_clauses(llm, sample_lines, BUDGET)

    assert "C-05: span overlaps the previous clause" in llm.calls[1]["prompt"]
    assert result.dropped == 0


async def test_long_documents_are_split_into_page_batches_and_renumbered(sample_lines: list[DocLine]) -> None:
    def respond(call: dict) -> dict:
        page = int(call["context"].split(" ", 1)[0].removeprefix("p"))  # batch's first line ref
        items = [c for c in golden()["clauses"] if c["span"]["page"] == page]
        for i, item in enumerate(items, start=1):
            item["clause_code"] = f"C-{i:02d}"  # each batch numbers from 1
        return {"clauses": items}

    llm = MockProvider({"ClauseExtraction": respond})

    result = await extract_clauses(llm, sample_lines, budget_bytes=2600)

    assert len(llm.calls) == 3  # one per page
    assert "p2 L" not in llm.calls[0]["context"]
    assert [c.clause_code for c in result.clauses] == [f"C-{n:02d}" for n in range(1, 16)]
    assert result.clauses[7].title == "Termination"


async def test_clause_can_continue_onto_the_next_page(sample_lines: list[DocLine]) -> None:
    llm = MockProvider(
        {
            "ClauseExtraction": {
                "clauses": [
                    {"clause_code": "C-01", "title": "Lock-in and termination", "category": "lock_in",
                     "span": {"page": 1, "line_start": 30, "page_end": 2, "line_end": 4}}
                ]
            }
        }
    )

    result = await extract_clauses(llm, sample_lines, BUDGET)

    clause = result.clauses[0]
    assert (clause.page, clause.line_start, clause.page_end, clause.line_end) == (1, 30, 2, 4)
    assert "Page 1 of 3" in clause.text and clause.text.endswith("by giving one (1) month's written notice to the Licensor.")
