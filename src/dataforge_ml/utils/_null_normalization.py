"""
_null_normalization  –  boundary normalisation for Phase 2.

Converts every effective null in a DataFrame to a Polars-native null before
any Phase 2 operation touches it, so all downstream code works exclusively
with pl.Null and never needs to handle sentinels, empty strings, or Inf.
"""

from __future__ import annotations

import polars as pl

from ._null_detection import (
    _SENTINEL_STRINGS,
    _inf_eligible,
    _sentinel_eligible,
)


def _resolve_effective_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame with all effective nulls converted to Polars null.

    Applies dtype-driven rules identical to Phase 1:
    - String/Utf8: empty/whitespace strings and case-insensitive sentinel
      strings → null.
    - Float32/Float64: NaN and Inf (positive and negative) → null.
    - All other dtypes: unchanged.

    Returns ``df`` unchanged (same object) if no eligible-dtype columns exist.
    """
    exprs: list[pl.Expr] = []

    for col_name in df.columns:
        dtype = df[col_name].dtype

        if _sentinel_eligible(dtype):
            condition = (
                pl.col(col_name).is_null()
                | (pl.col(col_name).str.strip_chars() == "")
                | pl.col(col_name).str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
            )
            exprs.append(
                pl.when(condition)
                .then(pl.lit(None, dtype=dtype))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )

        elif _inf_eligible(dtype):
            condition = (
                pl.col(col_name).is_nan()
                | pl.col(col_name).is_infinite()
            )
            exprs.append(
                pl.when(condition)
                .then(pl.lit(None, dtype=dtype))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )

    if not exprs:
        return df

    return df.with_columns(exprs)
