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
    _numeric_sentinel_eligible,
    _sentinel_eligible,
)


def _resolve_effective_nulls(
    df: pl.DataFrame,
    numeric_sentinels: dict[str, list[float]] | None = None,
    string_sentinels: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Return a DataFrame with all effective nulls converted to Polars null.

    Applies dtype-driven rules identical to Phase 1:

    - String/Utf8: empty/whitespace strings always → null; sentinel string
      matching uses **replace semantics** when the column name appears in
      ``string_sentinels`` (only declared values converted, hardcoded defaults
      suppressed for that column), or falls back to the hardcoded
      ``_SENTINEL_STRINGS`` set when no declaration exists.
    - Float32/Float64: NaN and Inf (positive and negative) → null.
    - Any numeric dtype (int or float): user-declared sentinel values → null,
      when the column name appears in ``numeric_sentinels``.

    Columns absent from ``numeric_sentinels`` or ``string_sentinels``, or
    whose dtype does not pass the relevant eligibility check, are not affected
    by those mappings.

    Returns ``df`` unchanged (same object) when no eligible-dtype columns exist
    and both sentinel dicts are ``None`` or empty.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to normalize.
    numeric_sentinels : dict[str, list[float]] or None, optional
        Mapping from column name to a list of sentinel float values to replace
        with null.  ``None`` or an empty dict disables numeric sentinel
        normalization entirely, preserving current behaviour.
    string_sentinels : dict[str, list[str]] or None, optional
        Mapping from column name to a list of string sentinel values.  When a
        column name is present, only the declared values are matched
        (case-insensitive); the hardcoded defaults are suppressed for that
        column.  Empty/whitespace detection always applies regardless.
        ``None`` or an empty dict preserves current hardcoded-default behaviour
        for all string columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with all effective nulls replaced by Polars-native null.
    """
    exprs: list[pl.Expr] = []
    sentinels: dict[str, list[float]] = numeric_sentinels or {}
    str_decls: dict[str, list[str]] = string_sentinels or {}

    for col_name in df.columns:
        dtype = df[col_name].dtype
        col_sentinels = sentinels.get(col_name)

        if _sentinel_eligible(dtype):
            col_str_decl = str_decls.get(col_name)
            if col_str_decl is not None:
                # Replace semantics: declared values only (case-insensitive),
                # hardcoded defaults suppressed for this column.
                sentinel_set = [s.upper() for s in col_str_decl]
                condition = (
                    pl.col(col_name).is_null()
                    | (pl.col(col_name).str.strip_chars() == "")
                    | pl.col(col_name).str.to_uppercase().is_in(sentinel_set)
                )
            else:
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
            # Float32/Float64: combine Inf/NaN + any user-declared sentinels in one
            # expression so both rules apply even though they share the same alias.
            condition: pl.Expr = (
                pl.col(col_name).is_nan() | pl.col(col_name).is_infinite()
            )
            if col_sentinels:
                for v in col_sentinels:
                    condition = condition | (pl.col(col_name) == pl.lit(v, dtype=dtype))
            exprs.append(
                pl.when(condition)
                .then(pl.lit(None, dtype=dtype))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )

        elif _numeric_sentinel_eligible(dtype) and col_sentinels:
            # Integer columns: sentinel replacement only.
            condition = pl.col(col_name) == pl.lit(col_sentinels[0]).cast(dtype)
            for v in col_sentinels[1:]:
                condition = condition | (pl.col(col_name) == pl.lit(v).cast(dtype))
            exprs.append(
                pl.when(condition)
                .then(pl.lit(None, dtype=dtype))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )

    if not exprs:
        return df

    return df.with_columns(exprs)
