"""Cheap checks that a value really comes from the clause it cites.

Numbers are the most checkable part of a fact: "Rs. 18,000", "10%", "7 days". If none of
a value's numbers appear in the cited clause, the citation is probably wrong (a common
slip is citing the document's printed clause number instead of its code).
"""

import re
from collections.abc import Sequence

from app.core.schemas import Clause

_NUMBER_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "tenth": 10, "twelfth": 12,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90,
}
_WORD_PATTERN = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
_DIGIT_GROUPING = re.compile(r"(?<=\d),(?=\d)")  # "1,00,000" -> "100000"
_NUMBER = re.compile(r"\d+")


def numbers_in(text: str) -> set[int]:
    text = _DIGIT_GROUPING.sub("", text)
    found = {int(n) for n in _NUMBER.findall(text)}
    found.update(_NUMBER_WORDS[w.lower()] for w in _WORD_PATTERN.findall(text))
    return found


def ground_citation(value: str, clause_code: str, clauses: Sequence[Clause]) -> str | None:
    """The clause code that supports ``value``: the cited one, a corrected one, or None.

    A value without numbers cannot be checked this way and keeps its citation. Otherwise
    the citation stands if the cited clause contains any of the value's numbers (values
    are paraphrased: "3rd floor" for "third floor"). If it contains none, the citation
    moves to the one clause sharing the most numbers with the value, or is None when no
    clause shares any or several tie.
    """
    wanted = numbers_in(value)
    if not wanted:
        return clause_code
    by_code = {c.clause_code: c for c in clauses}
    cited = by_code.get(clause_code)
    if cited is not None and wanted & numbers_in(cited.text):
        return clause_code
    overlap = {c.clause_code: len(wanted & numbers_in(c.text)) for c in clauses}
    best = max(overlap.values(), default=0)
    winners = [code for code, count in overlap.items() if count == best]
    return winners[0] if best and len(winners) == 1 else None
