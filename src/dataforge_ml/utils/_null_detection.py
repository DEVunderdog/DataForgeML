"""
_null_detection  –  shared dtype-driven null primitives for Phase 1 and Phase 2.

Single authority for what counts as "effectively null" across the entire
implementation. No config, no SemanticType overrides, no state.
"""

from __future__ import annotations

import polars as pl

_SENTINEL_STRINGS: frozenset[str] = frozenset({"NA", "NAN", "NULL", "NONE", "?"})


def _sentinel_eligible(dtype: pl.DataType) -> bool:
    """True when sentinel-string detection should run for this column (String/Utf8 only)."""
    return dtype in (pl.Utf8, pl.String)


def _inf_eligible(dtype: pl.DataType) -> bool:
    """True when Inf/NaN expansion should run (Float32/Float64 only)."""
    return dtype in (pl.Float32, pl.Float64)
