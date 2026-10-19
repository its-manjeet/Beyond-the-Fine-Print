"""F6: the risk score is computed in code from the model's pattern detection. LLM mocked."""

import pytest

from app.agents.clauses import extract_clauses
from app.agents.ingestion import ingest, validate_upload
from app.agents.risk import analyze_risk, format_rupees, level_for, score_clause
from app.core.config import Settings
from app.core.schemas import Clause, DocLine
from tests.mocks import MockProvider, load_golden, sample_pdf_bytes

WEIGHTS = (0.30, 0.50, 0.10, 0.10)
RENT = 18000


@pytest.fixture
async def sample_clauses(settings: Settings) -> list[Clause]:
    data = sample_pdf_bytes()
    pages = (await ingest(data, validate_upload(data, settings), None, settings)).pages
    lines = [DocLine(page=p.page_number, line_number=l.line_number, text=l.text) for p in pages for l in p.lines]
    llm = MockProvider({"ClauseExtraction": load_golden("clauses")})
    return (await extract_clauses(llm, lines, 9000)).clauses


async def test_sample_lock_in_is_amber_termination_red_rent_due_green(sample_clauses: list[Clause]) -> None:
    llm = MockProvider({"RiskOutput": load_golden("risk")})

    results = {r.clause_code: r for r in await analyze_risk(llm, sample_clauses, 9000, WEIGHTS)}
    titles = {c.clause_code: c.title for c in sample_clauses}

    lock_in, termination, rent = results["C-07"], results["C-08"], results["C-04"]
    assert titles["C-07"] == "Lock-in Period" and titles["C-08"] == "Termination" and titles["C-04"] == "Licence Fee"
    assert (lock_in.level, lock_in.score) == ("amber", pytest.approx(0.6267, abs=1e-4))
    assert lock_in.reason == "Main factor: lock-in with penalty"
    assert (termination.level, termination.score) == ("red", 0.72)
    assert termination.reason == "Main factor: one-sided termination"
    assert (rent.level, rent.score) == ("green", 0.12)
    assert rent.reason == "Main factor: type of clause (rent payment)"
    assert len(results) == 15


def test_formula_matches_the_report() -> None:
    # Lock-in: C=0.6, P=0.7, L=0.3, M=36000/(3*18000)
    r = score_clause("C-07", "lock_in", ["lock_in_with_penalty"], 36000, RENT, WEIGHTS)
    assert r.score == pytest.approx(0.3 * 0.6 + 0.5 * 0.7 + 0.1 * 0.3 + 0.1 * (36000 / 54000), abs=1e-4)
    assert r.factors["contributions"] == {"C_type": 0.18, "P": 0.35, "L": 0.03, "M": 0.0667}


def test_patterns_combine_as_noisy_or() -> None:
    r = score_clause("C-01", "other", ["one_sided_termination", "lock_in_with_penalty"], None, RENT, WEIGHTS)
    assert r.factors["P"] == pytest.approx(1 - (1 - 0.9) * (1 - 0.7))
    assert r.reason == "Main factor: one-sided termination (also lock-in with penalty)"
    assert r.factors["patterns"] == ["one_sided_termination", "lock_in_with_penalty"]


def test_money_at_stake_is_capped_and_needs_a_rent() -> None:
    capped = score_clause("C-06", "security_deposit", [], 100000, RENT, WEIGHTS)
    assert capped.factors["M"] == 1.0
    no_rent = score_clause("C-06", "security_deposit", [], 100000, None, WEIGHTS)
    assert no_rent.factors["M"] == 0.0


def test_money_can_be_the_main_factor() -> None:
    r = score_clause("C-06", "other", [], 100000, RENT, (0.1, 0.5, 0.1, 0.3))
    assert r.reason == "Main factor: money at stake (Rs. 1,00,000)"


@pytest.mark.parametrize(
    ("score", "level"),
    [(0.0, "green"), (0.3999, "green"), (0.40, "amber"), (0.6999, "amber"), (0.70, "red"), (1.0, "red")],
)
def test_level_thresholds(score: float, level: str) -> None:
    assert level_for(score) == level


def test_weights_come_from_settings(settings: Settings) -> None:
    assert settings.risk_weights == WEIGHTS
    r = score_clause("C-08", "termination_notice", ["one_sided_termination"], None, RENT, (0.0, 1.0, 0.0, 0.0))
    assert r.score == 0.9


@pytest.mark.parametrize(("amount", "text"), [(500, "Rs. 500"), (36000, "Rs. 36,000"), (100000, "Rs. 1,00,000"), (12345678, "Rs. 1,23,45,678")])
def test_rupees_use_indian_grouping(amount: float, text: str) -> None:
    assert format_rupees(amount) == text


async def test_unknown_codes_are_dropped_and_skipped_clauses_scored_from_category(sample_clauses: list[Clause]) -> None:
    llm = MockProvider(
        {
            "RiskOutput": {
                "monthly_rent_rupees": 18000,
                "clauses": [
                    {"clause_code": "C-99", "category": "lock_in", "patterns": ["one_sided_termination"]},
                    {"clause_code": "C-08", "category": "termination_notice", "patterns": ["one_sided_termination"]},
                ],
            }
        }
    )

    results = {r.clause_code: r for r in await analyze_risk(llm, sample_clauses, 9000, WEIGHTS)}

    assert "C-99" not in results
    assert results["C-08"].level == "red"
    assert results["C-07"].factors["category"] == "lock_in" and results["C-07"].factors["patterns"] == []
