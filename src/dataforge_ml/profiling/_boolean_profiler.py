"""
BooleanProfiler  –  Phase 1 extension: Boolean Column Profiling.

Handles columns classified as SemanticType.Boolean, which includes:
  - Native Polars Boolean dtype
  - Integer {0, 1} columns with a Boolean override in ProfileConfig
  - Boolean-string columns ("true"/"false", "yes"/"no", "1"/"0") with override

Per-column metrics:
  1. true_count   – count of non-null truthy values
  2. false_count  – count of non-null falsy values
  3. true_ratio   – true_count / non_null_count  (nulls excluded)
  4. false_ratio  – false_count / non_null_count (nulls excluded)
  5. mode         – most frequent non-null value (True / False), or None if tied

Null values are NOT counted in ratios — missingness is already captured by
the upstream MissingnessProfiler pass and lives in ColumnProfile.missingness.
"""

from __future__ import annotations

import polars as pl

from ._base import ColumnBatchProfiler
from ._config import BooleanStats
from ._boolean_config import BooleanFlag, BooleanProfileResult
from ..models._data_types import _INT_DTYPES

# ---------------------------------------------------------------------------
# String values that represent True / False
# ---------------------------------------------------------------------------

_TRUE_STRINGS: frozenset[str] = frozenset({"true", "yes", "1", "t", "y"})
_FALSE_STRINGS: frozenset[str] = frozenset({"false", "no", "0", "f", "n"})


class BooleanProfiler(ColumnBatchProfiler[BooleanProfileResult]):
    """
    Boolean column profiler for Polars DataFrames.

    Profiles every column passed to profile(df, columns) — no config,
    no internal eligibility gate.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str],
        user_overrides: set[str] | None = None,
    ) -> BooleanProfileResult:
        """
        Profile the specified boolean columns in a DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            The input Polars DataFrame containing the columns to profile.
        columns : list[str]
            A list of column names to profile.
        user_overrides : set[str] | None, optional
            A set of column names that have been manually overridden by the user.

        Returns
        -------
        BooleanProfileResult
            A result object containing distribution statistics for the profiled columns.

        Raises
        ------
        OverrideCoercionError
            If a column in user_overrides completely fails coercion to Boolean.
        """
        return self._run(data, columns, user_overrides)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(
        self,
        df: pl.DataFrame,
        columns: list[str],
        user_overrides: set[str] | None = None,
    ) -> BooleanProfileResult:
        result = BooleanProfileResult()
        user_overrides = user_overrides or set()

        available = self._resolve_columns(df.columns, columns)
        result.analysed_columns = available

        for col_name in available:
            result.columns[col_name] = self._profile_column(df[col_name], col_name, df.height, user_overrides)

        return result

    # ------------------------------------------------------------------
    # Per-column driver
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pl.Series,
        col_name: str,
        n_rows: int,
        user_overrides: set[str],
    ) -> BooleanStats:
        profile = BooleanStats()

        # Coerce to a clean boolean series (drop nulls)
        bool_series = self._to_bool_series(series)
        non_null_count = bool_series.len()

        # FormatMismatch: a value that is present (non-null after the
        # orchestrator's Effective-Null normalization) but falls outside the
        # recognized true/false vocabulary is dropped by coercion.  A shortfall
        # in the non-null count means the column holds dirty, uncoercible data.
        if non_null_count < series.drop_nulls().len():
            profile.flags.append(BooleanFlag.FormatMismatch)

        if non_null_count == 0:
            if series.drop_nulls().len() > 0 and col_name in user_overrides:
                from ._base import OverrideCoercionError
                raise OverrideCoercionError(
                    f"Column {col_name!r} with TypeFlag.UserOverride completely failed coercion to Boolean."
                )
            return profile

        true_count = int(bool_series.sum())
        false_count = non_null_count - true_count

        profile.true_count = true_count
        profile.false_count = false_count
        profile.true_ratio = true_count / non_null_count
        profile.false_ratio = false_count / non_null_count

        # Mode: True if more trues, False if more falses, None if perfectly tied
        if true_count > false_count:
            profile.mode = True
        elif false_count > true_count:
            profile.mode = False
        else:
            profile.mode = None  # tied — no single mode

        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bool_series(series: pl.Series) -> pl.Series:
        """
        Return a null-free Boolean Series regardless of the input dtype.

        Handles:
          - pl.Boolean      → drop nulls directly
          - integer {0, 1}  → cast to Boolean, drop nulls
          - string          → map known true/false strings, drop nulls
        """
        if series.dtype == pl.Boolean:
            return series.drop_nulls()

        if series.dtype in _INT_DTYPES:
            return series.cast(pl.Boolean).drop_nulls()

        if series.dtype == pl.Utf8:
            lower = series.str.to_lowercase().str.strip_chars()
            true_mask = lower.is_in(list(_TRUE_STRINGS))
            false_mask = lower.is_in(list(_FALSE_STRINGS))
            known_mask = true_mask | false_mask
            return true_mask.filter(known_mask)

        # Fallback: attempt a cast and drop nulls (covers e.g. Categorical)
        try:
            return series.cast(pl.Boolean).drop_nulls()
        except Exception:
            return pl.Series([], dtype=pl.Boolean)
