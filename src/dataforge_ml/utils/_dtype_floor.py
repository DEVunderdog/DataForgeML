"""Dtype-floor enforcement for phase-entry frames (ADR-0085).

The **Dtype Floor** is the guarantee every consumer of a frame may rely on:
*no active column reaches a consumer as* ``pl.String`` *unless its semantic
type is* ``Text`` *or* ``Identifier``.  Each other semantic type names the
dtype it materialises as::

    Categorical -> pl.Categorical
    Boolean     -> pl.Boolean
    Datetime    -> pl.Datetime

``Numeric`` is already enforced by the type detector's coercion path, and
int-coded categoricals (``TypeFlag.EncodedCategory``) stay ``pl.Int*`` — the
rule is a floor keyed off the *dtype*, not a biconditional keyed off the
semantic type, so a column that is not ``pl.String`` is never touched.

The floor is enforced constructively at each phase entry, immediately after
``_resolve_effective_nulls``, driven by the profile's semantic types.  Nothing
is cached: every phase re-derives a working copy and throws it away, so the
user's frame is never mutated.  The cast is silent — no raise, no warning, no
public predicate — and a failed Boolean/Datetime coercion becomes an Effective
Null, which is why a phase-entry null count may legitimately exceed the
profile's recorded missingness.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ..config import SemanticType
from ..models._data_types import _FALSE_STRINGS, _TRUE_STRINGS

# The semantic types that materialise as a non-string dtype. A column whose
# semantic type is absent from this mapping (Text, Identifier, Numeric, or an
# unprofiled column) keeps whatever dtype it arrived with.
_FLOOR_DTYPES: dict[SemanticType, pl.DataType] = {
    SemanticType.Categorical: pl.Categorical,
    SemanticType.Boolean: pl.Boolean,
    SemanticType.Datetime: pl.Datetime,
}


def _to_boolean(series: pl.Series) -> pl.Series:
    """Map a string series onto ``pl.Boolean``, nulling unrecognised tokens."""
    lower = series.str.strip_chars().str.to_lowercase()
    return (
        pl.select(
            pl.when(lower.is_in(list(_TRUE_STRINGS)))
            .then(pl.lit(True))
            .when(lower.is_in(list(_FALSE_STRINGS)))
            .then(pl.lit(False))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias(series.name)
        )
        .to_series()
        .rename(series.name)
    )


def _to_datetime(series: pl.Series) -> pl.Series:
    """Map a string series onto ``pl.Datetime``, nulling unparseable values.

    Mirrors the detector's own coercion (``str.to_datetime(strict=False)``).
    A column no value of which parses — including one Polars cannot infer a
    format for at all — resolves to all-null, the same answer the strict=False
    path gives value by value.
    """
    try:
        return series.str.to_datetime(strict=False)
    except Exception:  # noqa: BLE001
        return pl.Series(series.name, [None] * series.len(), dtype=pl.Datetime)


def _apply_dtype_floor(
    df: pl.DataFrame,
    semantic_types: Mapping[str, SemanticType | None],
) -> pl.DataFrame:
    """Materialise every ``pl.String`` column at its semantic type's dtype.

    The phase-entry normaliser of the Dtype Floor (ADR-0085).  Only
    ``pl.String``/``pl.Utf8`` columns are considered: a column already carrying
    a non-string dtype — an int-coded categorical, a native ``pl.Datetime``, an
    already-``pl.Categorical`` column — is returned untouched, so calling this
    twice is a no-op.  Columns whose semantic type is ``Text``, ``Identifier``,
    ``Numeric``, ``None``, or absent from ``semantic_types`` are left as they
    are.

    Coercion failures are absorbed as Effective Nulls rather than raised: a
    boolean token outside the recognised vocabulary and an unparseable datetime
    both become null.  The phase-entry null count may therefore exceed the
    missingness the profile recorded.

    Parameters
    ----------
    df : pl.DataFrame
        The phase's working copy, already passed through
        ``_resolve_effective_nulls``.  Never mutated.
    semantic_types : Mapping[str, SemanticType or None]
        Column name to semantic type, read off whatever the phase holds — the
        profile's ``ColumnProfile``, the plan's ``ColumnImputationDecision``, or
        a fitted record's decision.

    Returns
    -------
    pl.DataFrame
        A new frame with the floor applied.  ``df`` itself is returned when no
        column needs casting.
    """
    replacements: list[pl.Series] = []
    for name, dtype in df.schema.items():
        if dtype not in (pl.Utf8, pl.String):
            continue
        target = _FLOOR_DTYPES.get(semantic_types.get(name))  # type: ignore[arg-type]
        if target is None:
            continue
        series = df.get_column(name)
        if target == pl.Categorical:
            replacements.append(series.cast(pl.Categorical))
        elif target == pl.Boolean:
            replacements.append(_to_boolean(series))
        else:
            replacements.append(_to_datetime(series))

    if not replacements:
        return df
    return df.with_columns(replacements)
