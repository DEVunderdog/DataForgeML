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


_NUMERIC_SENTINEL_DTYPES: frozenset[pl.DataType] = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})


def _numeric_sentinel_eligible(dtype: pl.DataType) -> bool:
    """True when user-declared numeric sentinel normalization should run for this column.

    Covers all integer and float Polars dtypes. String sentinels are handled
    by ``_sentinel_eligible``; Inf/NaN by ``_inf_eligible``.

    Parameters
    ----------
    dtype : pl.DataType
        Polars column dtype to test.

    Returns
    -------
    bool
        ``True`` for Int8/16/32/64, UInt8/16/32/64, Float32, Float64;
        ``False`` for all other dtypes.
    """
    return dtype in _NUMERIC_SENTINEL_DTYPES
