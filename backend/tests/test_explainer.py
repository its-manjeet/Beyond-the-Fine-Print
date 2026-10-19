"""F4: explanations, the citation check and the sign-advice guard. LLM mocked."""

import logging
import re

import pytest

from app.agents.citations import CitedSentence, check_sentences, normalize_code, tells_user_to_sign
from app.agents.explainer import explain_clauses
from app.core.schemas import Clause
from tests.mocks import MockProvider, load_golden

CODES = [f"C-{n:02d}" for n in range(1, 16)]
CLAUSES = [
    Clause(clause_code=c, title=c, category="other", page=1, line_start=1, page_end=1, line_end=1, text=f"Text of {c}.")
    for c in CODES
]


def golden_for_request(call: dict) -> dict:
    """Answer only for the clauses in this request, as the real model would."""
    asked = set(re.findall(r"^\[(C-\d+)\]", call["context"], re.MULTILINE))
    return {"explanations": [e for e in load_golden("explainer")["explanations"] if e["clause_code"] in asked]}


async def test_explains_every_clause_in_batches_and_removes_unknown_citation(caplog) -> None:
    llm = MockProvider({"ExplainerOutput": golden_for_request})

    with caplog.at_level(logging.WARNING):
        result = await explain_clauses(llm, CLAUSES, budget_bytes=9000)

    assert len(llm.calls) == 4  # 15 clauses, at most 4 per call
    assert [e.clause_code for e in result.explanations] == CODES
    termination = result.explanations[7]
    assert len(termination.sentences) == 2  # the sentence citing C-42 was removed
    assert all("C-42" not in s.clause_codes for e in result.explanations for s in e.sentences)
    assert result.removed == 1
    removal = next(r for r in caplog.records if r.getMessage() == "citations.sentence_removed")
    assert (removal.reason, removal.unknown) == ("unknown_clause", ["C-42"])
    assert "Courts" not in caplog.text  # removed sentence text is never logged


async def test_explanations_for_unknown_clauses_and_extra_sentences_are_dropped() -> None:
    llm = MockProvider(
        {
            "ExplainerOutput": {
                "explanations": [
                    {"clause_code": "C-77", "sentences": [{"text": "Made up.", "clause_codes": ["C-77"]}]},
                    {"clause_code": "c-4", "sentences": [{"text": f"Sentence {i}.", "clause_codes": ["c4"]} for i in range(5)]},
                ]
            }
        }
    )

    result = await explain_clauses(llm, CLAUSES[:5], budget_bytes=9000)

    assert [e.clause_code for e in result.explanations] == ["C-04"]
    assert [s.text for s in result.explanations[0].sentences] == ["Sentence 0.", "Sentence 1.", "Sentence 2."]
    assert result.explanations[0].sentences[0].clause_codes == ["C-04"]
    assert result.removed == 2


async def test_truncated_batch_is_split_and_retried() -> None:
    from app.core.llm import LLMOutputError

    def respond(call: dict) -> dict:
        asked = re.findall(r"^\[(C-\d+)\]", call["context"], re.MULTILINE)
        if len(asked) > 2:
            raise LLMOutputError("Output was truncated at the max output tokens")
        return golden_for_request(call)

    llm = MockProvider({"ExplainerOutput": respond})

    result = await explain_clauses(llm, CLAUSES[:4], budget_bytes=9000)

    assert [len(re.findall(r"^\[C-", c["context"], re.MULTILINE)) for c in llm.calls] == [4, 2, 2]
    assert [e.clause_code for e in result.explanations] == CODES[:4]


def test_citation_check_rules() -> None:
    sentences = [
        CitedSentence(text="Fine.", clause_codes=["C-01"]),
        CitedSentence(text="Cites nothing.", clause_codes=[]),
        CitedSentence(text="Cites a ghost.", clause_codes=["C-01", "C-99"]),
        CitedSentence(text="  ", clause_codes=["C-01"]),
        CitedSentence(text="You should not sign this agreement.", clause_codes=["C-01"]),
        CitedSentence(text="Normalised.", clause_codes=["c1", "C-01"]),
    ]

    kept, removed = check_sentences(sentences, {"C-01"}, agent="test")

    assert [(s.text, s.clause_codes) for s in kept] == [("Fine.", ["C-01"]), ("Normalised.", ["C-01"])]
    assert removed == 4


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("You should not sign this.", True),
        ("Don't sign until the owner changes clause 8.", True),
        ("We recommend signing it.", True),
        ("It is safe to sign.", True),
        ("Both parties signed the agreement.", False),
        ("The agreement must be signed by two witnesses.", False),
        ("You must pay Rs. 18,000 by the 5th.", False),
    ],
)
def test_sign_advice_guard(text: str, flagged: bool) -> None:
    assert tells_user_to_sign(text) is flagged


@pytest.mark.parametrize(
    ("raw", "code"),
    [("C-04", "C-04"), ("c-4", "C-04"), ("C4", "C-04"), (" C - 12 ", "C-12"), ("[C-01]", "C-01"), ("X-1", "X-1")],
)
def test_normalize_code(raw: str, code: str) -> None:
    assert normalize_code(raw) == code
