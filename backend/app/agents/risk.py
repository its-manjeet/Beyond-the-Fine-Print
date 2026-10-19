"""F6 Risk radar.

The model only reports, per clause, its category, which risk patterns are present and
the rupee amount at stake. The score is computed here:

    R(c) = w1*C_type + w2*P + w3*L + w4*M
    P    = 1 - prod_j(1 - p_j * b_j)     (b_j = 1 if pattern j is present)
    L    = 0.3 ("unknown") until Phase 2 adds the legality checker
    M    = min(1, amount / (3 * monthly_rent)), 0 if no amount

green if R < 0.40, amber if 0.40 <= R < 0.70, red if R >= 0.70.
"""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.batching import chunks, run_split
from app.agents.citations import normalize_code
from app.agents.text import batches, clause_block, clauses_text, utf8_len
from app.core.llm import LLMProvider
from app.core.logging import get_logger
from app.core.schemas import Clause, RentalCategory
from app.prompts import risk as prompt

logger = get_logger(__name__)

AGENT = "risk"

Pattern = Literal[
    "lock_in_with_penalty",
    "auto_renewal",
    "one_sided_termination",
    "broad_exclusion",
    "unlimited_liability",
    "distant_arbitration",
    "short_reply_deadline",
    "non_refundable_deposit",
    "unilateral_rent_increase",
]
Level = Literal["green", "amber", "red"]

SEVERITY: dict[str, float] = {
    "lock_in_with_penalty": 0.7,
    "auto_renewal": 0.5,
    "one_sided_termination": 0.9,
    "broad_exclusion": 0.6,
    "unlimited_liability": 0.8,
    "distant_arbitration": 0.6,
    "short_reply_deadline": 0.6,
    "non_refundable_deposit": 0.8,
    "unilateral_rent_increase": 0.7,
}
CATEGORY_RISK: dict[str, float] = {
    "parties_property": 0.1,
    "rent_payment": 0.3,
    "security_deposit": 0.5,
    "rent_escalation": 0.5,
    "maintenance": 0.4,
    "lock_in": 0.6,
    "termination_notice": 0.8,
    "penalty_late_fee": 0.7,
    "renewal": 0.4,
    "use_restrictions": 0.3,
    "dispute_resolution": 0.5,
    "other": 0.2,
}
LEGALITY_UNKNOWN = 0.3
# Clauses per call, so the output stays well inside the output-token cap.
MAX_CLAUSES_PER_CALL = 8
AMBER_FROM, RED_FROM = 0.40, 0.70


class ClauseRisk(BaseModel):
    clause_code: str
    category: RentalCategory
    patterns: list[Pattern]
    amount_at_stake_rupees: float | None = Field(default=None, ge=0)


class RiskOutput(BaseModel):
    monthly_rent_rupees: float | None = Field(default=None, ge=0)
    clauses: list[ClauseRisk]


class RiskResult(BaseModel):
    clause_code: str
    level: Level
    score: float
    reason: str
    factors: dict[str, Any]


def level_for(score: float) -> Level:
    if score >= RED_FROM:
        return "red"
    if score >= AMBER_FROM:
        return "amber"
    return "green"


def score_clause(
    clause_code: str,
    category: str,
    patterns: Sequence[str],
    amount: float | None,
    monthly_rent: float | None,
    weights: Sequence[float],
) -> RiskResult:
    w1, w2, w3, w4 = weights
    present = sorted(set(patterns), key=lambda p: (-SEVERITY[p], p))
    c_type = CATEGORY_RISK.get(category, CATEGORY_RISK["other"])
    survive = 1.0
    for pattern in present:
        survive *= 1 - SEVERITY[pattern]
    p = 1 - survive
    legality = LEGALITY_UNKNOWN
    m = min(1.0, amount / (3 * monthly_rent)) if amount and monthly_rent else 0.0

    contributions = {"C_type": w1 * c_type, "P": w2 * p, "L": w3 * legality, "M": w4 * m}
    score = sum(contributions.values())
    # Ties go to the more specific factor: patterns, then money, then clause type, then legality.
    main = max(("P", "M", "C_type", "L"), key=lambda k: contributions[k])
    return RiskResult(
        clause_code=clause_code,
        level=level_for(score),
        score=round(score, 4),
        reason=_reason(main, category, present, amount),
        factors={
            "C_type": c_type,
            "P": round(p, 4),
            "L": legality,
            "M": round(m, 4),
            "weights": list(weights),
            "contributions": {k: round(v, 4) for k, v in contributions.items()},
            "category": category,
            "patterns": present,
            "amount_at_stake_rupees": amount,
            "monthly_rent_rupees": monthly_rent,
            "legality": "unknown",
        },
    )


def _reason(main: str, category: str, patterns: list[str], amount: float | None) -> str:
    if main == "P" and patterns:
        reason = f"Main factor: {_label(patterns[0])}"
        if len(patterns) > 1:
            reason += f" (also {', '.join(_label(p) for p in patterns[1:])})"
        return reason
    if main == "M" and amount:
        return f"Main factor: money at stake ({format_rupees(amount)})"
    if main == "L":
        return "Main factor: legality not checked yet"
    return f"Main factor: type of clause ({_label(category)})"


LABELS: dict[str, str] = {
    "lock_in_with_penalty": "lock-in with penalty",
    "auto_renewal": "automatic renewal",
    "one_sided_termination": "one-sided termination",
    "broad_exclusion": "broad exclusion of the owner's responsibility",
    "unlimited_liability": "unlimited liability",
    "distant_arbitration": "arbitration in a distant city",
    "short_reply_deadline": "short deadline to respond",
    "non_refundable_deposit": "non-refundable deposit",
    "unilateral_rent_increase": "rent increase set by one side",
    "parties_property": "parties and property",
    "rent_payment": "rent payment",
    "security_deposit": "security deposit",
    "rent_escalation": "rent increase",
    "maintenance": "maintenance",
    "lock_in": "lock-in",
    "termination_notice": "termination and notice",
    "penalty_late_fee": "penalty or late fee",
    "renewal": "renewal",
    "use_restrictions": "restrictions on use",
    "dispute_resolution": "dispute resolution",
    "other": "other",
}


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def format_rupees(amount: float) -> str:
    """Indian digit grouping: 100000 -> 'Rs. 1,00,000'."""
    whole = str(int(round(amount)))
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    return f"Rs. {whole}"


async def analyze_risk(
    llm: LLMProvider, clauses: Sequence[Clause], budget_bytes: int, weights: Sequence[float]
) -> list[RiskResult]:
    by_code = {c.clause_code: c for c in clauses}
    found: dict[str, ClauseRisk] = {}
    monthly_rent: float | None = None
    unknown = 0

    async def run(group: list[Clause]) -> RiskOutput:
        return await llm.generate_structured(prompt.PROMPT, RiskOutput, context=clauses_text(group), max_tokens=2048)

    outputs: list[RiskOutput] = []
    for batch in batches(list(clauses), lambda c: utf8_len(clause_block(c)), budget_bytes):
        for group in chunks(batch, MAX_CLAUSES_PER_CALL):
            outputs.extend(await run_split(group, run, agent=AGENT))
    for output in outputs:
        if monthly_rent is None and output.monthly_rent_rupees:
            monthly_rent = output.monthly_rent_rupees
        for item in output.clauses:
            item.clause_code = normalize_code(item.clause_code)
            if item.clause_code not in by_code:
                unknown += 1
            else:
                found.setdefault(item.clause_code, item)
    if unknown:
        logger.warning("risk.unknown_clause_dropped", extra={"dropped": unknown})

    results = []
    for clause in clauses:
        item = found.get(clause.clause_code)
        if item is None:
            # The model skipped it: score it from its extracted category alone.
            results.append(score_clause(clause.clause_code, clause.category, [], None, monthly_rent, weights))
        else:
            results.append(
                score_clause(
                    clause.clause_code, item.category, item.patterns, item.amount_at_stake_rupees, monthly_rent, weights
                )
            )
    return results
