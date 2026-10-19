"""F7: grounded Q&A with citations, and refusal when the document does not cover it. LLM mocked."""

from app.agents.qa import answer_question, select_clauses
from app.core.schemas import Clause
from app.prompts.qa import NOT_COVERED
from tests.mocks import MockProvider, load_golden


def clause(code: str, title: str, text: str) -> Clause:
    return Clause(clause_code=code, title=title, category="other", page=1, line_start=1, page_end=1, line_end=1, text=text)


CLAUSES = [
    clause("C-06", "Security Deposit", "The Licensee has paid a refundable security deposit of Rs. 1,00,000."),
    clause("C-07", "Lock-in Period", "If the Licensee vacates before 11 months, two months' licence fee is payable."),
    clause("C-08", "Termination", "The Licensee may terminate after the lock-in with one month's notice."),
    clause("C-11", "Use of Premises", "No pets without written consent."),
]


async def test_answer_keeps_cited_sentences_and_drops_unknown_citations() -> None:
    llm = MockProvider({"QAOutput": load_golden("qa")["leave_early"]})

    answer = await answer_question(llm, CLAUSES, "What happens if I leave after 6 months?", 9000)

    assert answer.covered
    assert [s.clause_codes for s in answer.sentences] == [["C-07"], ["C-07", "C-06"], ["C-08"]]
    assert answer.removed == 1  # "Most courts would let you leave anyway." cited C-31
    assert "What happens if I leave after 6 months?" in llm.calls[0]["prompt"]


async def test_refuses_when_the_document_does_not_cover_it() -> None:
    llm = MockProvider({"QAOutput": load_golden("qa")["parking"]})

    answer = await answer_question(llm, CLAUSES, "Do I get a parking space?", 9000)

    assert not answer.covered
    assert [s.text for s in answer.sentences] == [NOT_COVERED]
    assert "legal aid" in NOT_COVERED and "ask the other party" in NOT_COVERED


async def test_answer_whose_citations_all_fail_becomes_a_refusal() -> None:
    llm = MockProvider(
        {"QAOutput": {"covered": True, "sentences": [{"text": "The law gives you 30 days.", "clause_codes": ["C-50"]}]}}
    )

    answer = await answer_question(llm, CLAUSES, "How long do I have?", 9000)

    assert not answer.covered and answer.sentences[0].text == NOT_COVERED and answer.removed == 1


async def test_sign_advice_is_never_returned() -> None:
    llm = MockProvider(
        {"QAOutput": {"covered": True, "sentences": [
            {"text": "You should not sign this agreement.", "clause_codes": ["C-08"]},
            {"text": "You must give one month's notice after the lock-in.", "clause_codes": ["C-08"]},
        ]}}
    )

    answer = await answer_question(llm, CLAUSES, "Should I sign?", 9000)

    assert [s.text for s in answer.sentences] == ["You must give one month's notice after the lock-in."]


def test_small_documents_send_every_clause() -> None:
    assert select_clauses(CLAUSES, "anything", 9000) == CLAUSES


def test_large_documents_send_the_best_matching_clauses_in_document_order() -> None:
    chosen = select_clauses(CLAUSES, "Can my landlord keep my deposit if I leave early?", budget_bytes=300)

    assert [c.clause_code for c in chosen] == ["C-06", "C-07", "C-08"]  # not the pets clause
