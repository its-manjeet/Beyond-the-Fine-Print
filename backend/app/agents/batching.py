"""Run an LLM call over a batch, splitting the batch if the output does not fit.

A long output can hit the provider's output-token cap (reasoning tokens count toward it
on gpt-oss), which raises ``LLMOutputError``. Halving the batch and trying each half is
cheaper than failing the stage, and a single item that still fails raises as before.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from app.core.llm import LLMOutputError
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def chunks(items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def run_split(items: list[T], run: Callable[[list[T]], Awaitable[R]], *, agent: str) -> list[R]:
    """``[await run(items)]``, or the results for each half if that output was unusable."""
    try:
        return [await run(items)]
    except LLMOutputError:
        if len(items) <= 1:
            raise
        logger.warning("agent.batch_split", extra={"agent": agent, "items": len(items)})
        middle = len(items) // 2
        return [*await run_split(items[:middle], run, agent=agent), *await run_split(items[middle:], run, agent=agent)]
