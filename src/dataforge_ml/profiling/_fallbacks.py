"""Profiling fallbacks: computations that degraded, recorded for the caller.

A profiler that cannot compute a signal leaves it unset and records a
fallback here instead of warning. The ``profile`` Operation collects them and
returns one coded Result Warning per code (ADR-0102). Outside a collection
block, recording is a no-op.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Fallback:
    """One degraded computation: its registered warning code and its column."""

    code: str
    column: str | None = None


_COLLECTOR: ContextVar[list[Fallback] | None] = ContextVar(
    "dataforge_ml_profiling_fallbacks", default=None
)


def record_fallback(code: str, column: str | None = None) -> None:
    """Record a degraded computation, when a collection block is active.

    Parameters
    ----------
    code : str
        A Result Warning code registered in the Error Code registry.
    column : str, optional
        The column whose computation degraded.
    """
    collector = _COLLECTOR.get()
    if collector is not None:
        collector.append(Fallback(code, column))


@contextmanager
def collect_fallbacks() -> Generator[list[Fallback]]:
    """Collect every fallback recorded in the block, in order.

    Yields
    ------
    list of Fallback
        Filled as the block runs.
    """
    collected: list[Fallback] = []
    token = _COLLECTOR.set(collected)
    try:
        yield collected
    finally:
        _COLLECTOR.reset(token)
