"""Helpers that turn stored lines and clauses into compact prompt text."""

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from app.core.schemas import Clause, DocLine

T = TypeVar("T")


def numbered_lines(lines: Iterable[DocLine]) -> str:
    """'p1 L14: text' per line: the format the clause extractor cites spans in."""
    return "\n".join(f"{line.ref}: {line.text}" for line in lines)


def clause_block(clause: Clause) -> str:
    return f"[{clause.clause_code}] {clause.title}\n{clause.text}"


def clauses_text(clauses: Iterable[Clause]) -> str:
    return "\n\n".join(clause_block(c) for c in clauses)


def head_text(lines: Iterable[DocLine], budget_bytes: int) -> str:
    """The document's first lines, up to ``budget_bytes`` of UTF-8."""
    out: list[str] = []
    used = 0
    for line in lines:
        size = len(line.text.encode("utf-8")) + 1
        if out and used + size > budget_bytes:
            break
        out.append(line.text)
        used += size
    return "\n".join(out)


def batches(items: Sequence[T], size_of: Callable[[T], int], budget: int) -> list[list[T]]:
    """Split ``items`` in order into batches whose total size stays within ``budget``.

    An item larger than the budget gets a batch of its own.
    """
    result: list[list[T]] = []
    current: list[T] = []
    used = 0
    for item in items:
        size = size_of(item)
        if current and used + size > budget:
            result.append(current)
            current, used = [], 0
        current.append(item)
        used += size
    if current:
        result.append(current)
    return result


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))
