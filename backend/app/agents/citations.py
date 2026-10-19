"""Checks applied to every user-facing sentence before it is stored or shown.

1. Citation check: every sentence must cite at least one clause, and every cited clause
   code must exist in the document. Otherwise the sentence is removed and the removal
   is logged (codes and reason only, never the sentence text)
2. Advice guard: a sentence that tells the user to sign or not sign is removed. The
   prompts already forbid this; this is the backstop
"""

import re
from collections.abc import Collection, Iterable

from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger(__name__)

_CODE = re.compile(r"^\s*\[?\s*C\s*-?\s*(\d{1,3})\s*\]?\s*$", re.IGNORECASE)
_SIGN = re.compile(r"\bsign(ing)?\b", re.IGNORECASE)
_DIRECTIVE = re.compile(
    r"\b(should|shouldn't|must|do not|don't|never|recommend\w*|advis\w*|safe to|okay to|ok to|go ahead|avoid|better (not )?to)\b",
    re.IGNORECASE,
)


class CitedSentence(BaseModel):
    text: str
    clause_codes: list[str]


def normalize_code(code: str) -> str:
    """'c-4', 'C4', '[C-04]' and 'C-04' all become 'C-04'; anything else is returned stripped."""
    match = _CODE.match(code)
    return f"C-{int(match.group(1)):02d}" if match else code.strip()


def tells_user_to_sign(text: str) -> bool:
    return bool(_SIGN.search(text) and _DIRECTIVE.search(text))


def check_sentences(
    sentences: Iterable[CitedSentence], known_codes: Collection[str], *, agent: str
) -> tuple[list[CitedSentence], int]:
    """Return the sentences that pass both checks, with codes normalised, and how many were removed."""
    kept: list[CitedSentence] = []
    removed = 0
    for sentence in sentences:
        text = sentence.text.strip()
        codes = list(dict.fromkeys(normalize_code(c) for c in sentence.clause_codes))
        reason = None
        if not text:
            reason = "empty"
        elif not codes:
            reason = "no_citation"
        elif unknown := [c for c in codes if c not in known_codes]:
            reason = "unknown_clause"
        elif tells_user_to_sign(text):
            reason = "sign_advice"
        if reason:
            removed += 1
            logger.warning(
                "citations.sentence_removed",
                extra={"agent": agent, "reason": reason, "cited": codes, "unknown": unknown if reason == "unknown_clause" else []},
            )
            continue
        kept.append(CitedSentence(text=text, clause_codes=codes))
    return kept, removed
