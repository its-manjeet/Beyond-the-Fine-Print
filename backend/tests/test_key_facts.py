"""F5: key facts carry clause codes; facts citing unknown clauses are dropped. LLM mocked."""

import pytest

from app.agents.clauses import extract_clauses
from app.agents.ingestion import ingest, validate_upload
from app.agents.key_facts import extract_key_facts
from app.core.config import Settings
from app.core.schemas import Clause, DocLine
from tests.mocks import MockProvider, load_golden, sample_pdf_bytes


def clause(code: str, category: str, text: str = "Some clause text.") -> Clause:
    return Clause(clause_code=code, title=code, category=category, page=1, line_start=1, page_end=1, line_end=1, text=text)


CLAUSES = [clause(f"C-{n:02d}", "other") for n in range(1, 15)] + [clause("C-15", "dispute_resolution")]


@pytest.fixture
async def sample_clauses(settings: Settings) -> list[Clause]:
    data = sample_pdf_bytes()
    pages = (await ingest(data, validate_upload(data, settings), None, settings)).pages
    lines = [DocLine(page=p.page_number, line_number=l.line_number, text=l.text) for p in pages for l in p.lines]
    return (await extract_clauses(MockProvider({"ClauseExtraction": load_golden("clauses")}), lines, 9000)).clauses


async def test_facts_keep_their_clause_and_unknown_clauses_are_dropped(sample_clauses: list[Clause]) -> None:
    llm = MockProvider({"KeyFactsOutput": load_golden("key_facts")})

    facts = await extract_key_facts(llm, sample_clauses, budget_bytes=9000)

    by_name = {f.name: f for f in facts}
    assert by_name["monthly_rent"].value == "Rs. 18,000 per month, due by the 5th"  # first value wins
    assert by_name["tenant"].clause_code == "C-01"  # the C-99 duplicate was dropped
    assert all(f.clause_code != "C-99" for f in facts)
    assert by_name["notice_period_owner"].value == "7 days"
    assert by_name["notice_period_tenant"].clause_code == "C-08"
    assert by_name["rent_increase"].clause_code == "C-09"
    assert len(facts) == 14


async def test_only_relevant_clauses_are_sent() -> None:
    llm = MockProvider({"KeyFactsOutput": {"facts": []}})

    await extract_key_facts(llm, CLAUSES, budget_bytes=9000)

    context = llm.calls[0]["context"]
    assert "[C-01]" in context and "[C-15]" not in context  # dispute resolution holds no key facts


async def test_large_input_is_batched_and_repeatable_facts_merge() -> None:
    long_clauses = [clause(f"C-{n:02d}", "penalty_late_fee", f"Late fee Rs. {n}00 a day. " + "x" * 400) for n in range(1, 7)]
    responses = [
        {"facts": [{"name": "late_payment_penalty", "value": "Rs. 100 a day", "clause_code": "C-01"}]},
        {"facts": [
            {"name": "late_payment_penalty", "value": "Rs. 500 a day", "clause_code": "C-05"},
            {"name": "late_payment_penalty", "value": "Rs. 500 a day", "clause_code": "C-05"},  # repeated: kept once
        ]},
    ]
    llm = MockProvider({"KeyFactsOutput": responses})

    facts = await extract_key_facts(llm, long_clauses, budget_bytes=1500)

    assert len(llm.calls) == 2
    assert [(f.value, f.clause_code) for f in facts] == [("Rs. 100 a day", "C-01"), ("Rs. 500 a day", "C-05")]


# --- grounding: a fact's numbers must appear in the clause it cites ---------------------

GROUNDED = [
    clause("C-04", "rent_payment", "The Licensee shall pay a monthly licence fee of Rs. 18,000/- by the 5th day."),
    clause("C-08", "termination_notice", "The Licensor may terminate by giving seven (7) days' notice; the Licensee one (1) month."),
    clause("C-09", "rent_escalation", "The licence fee shall stand increased by ten per cent (10%) every twelve months."),
    clause("C-06", "security_deposit", "A refundable security deposit of Rs. 1,00,000/- (Rupees One Lakh only)."),
]


async def test_fact_citing_the_wrong_clause_is_moved_to_the_clause_with_its_numbers() -> None:
    llm = MockProvider({"KeyFactsOutput": {"facts": [
        # Cited by the printed clause number "8." instead of its code C-09: a real Groq output.
        {"name": "rent_increase", "value": "10% every 12 months", "clause_code": "C-08"},
        {"name": "security_deposit", "value": "Rs. 1,00,000", "clause_code": "C-06"},
        {"name": "notice_period_owner", "value": "7 days", "clause_code": "C-08"},
        {"name": "owner", "value": "Suresh Kulkarni", "clause_code": "[C-04]"},  # no numbers: not checkable
        {"name": "late_payment_penalty", "value": "Rs. 18,000 rent due by the 5th or 3rd", "clause_code": "C-04"},  # partly matches
        {"name": "monthly_rent", "value": "Rs. 25,000", "clause_code": "C-04"},  # in no clause: dropped
    ]}})

    facts = {f.name: f.clause_code for f in await extract_key_facts(llm, GROUNDED, 9000)}

    assert facts == {
        "rent_increase": "C-09", "security_deposit": "C-06", "notice_period_owner": "C-08", "owner": "C-04",
        "late_payment_penalty": "C-04",
    }


def test_numbers_in_reads_digits_grouping_and_words() -> None:
    from app.agents.grounding import numbers_in

    assert numbers_in("Rs. 1,00,000 and 18,000/- by the 5th, ten per cent every twelve months") == {100000, 18000, 5, 10, 12}
    assert numbers_in("third floor") == {3}
