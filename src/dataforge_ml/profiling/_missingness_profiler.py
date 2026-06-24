"""
MissingnessProfiler  –  Phase 1 extension: Missingness Profiling.

Eligibility model
-----------------
Effective-null detection is purely dtype-driven — no SemanticType overrides:

sentinel-string detection  →  runs for every String/Utf8 column unconditionally
Inf / NaN expansion        →  runs for every Float32/Float64 column unconditionally
"""

from __future__ import annotations


import numpy as np
import polars as pl

from ._base import DatasetLevelProfiler
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingnessProfileConfig,
    MissingnessProfileResult,
    MissingSeverity,
    RowMissingnessDistribution,
)
from ..utils._null_detection import (
    _SENTINEL_STRINGS,
    _inf_eligible,
    _numeric_sentinel_eligible,
    _sentinel_eligible,
)


class MissingnessProfiler(DatasetLevelProfiler[MissingnessProfileResult]):
    """
    Phase 1 sub-processor that computes per-column missingness profiles.

    Detects effective nulls (native Polars nulls, string sentinels, Inf/NaN,
    and user-declared numeric and string sentinels) and classifies each column
    by severity, flags drop candidates and MAR suspects, and builds a binary
    missingness indicator used for column-pair correlation analysis.

    Parameters
    ----------
    config : MissingnessProfileConfig, optional
        Threshold configuration for severity bands, drop threshold, and MAR
        correlation threshold.  Defaults to ``MissingnessProfileConfig()``.
    numeric_sentinels : dict[str, list[float]], optional
        Per-column numeric sentinel declarations copied from
        ``ProfileConfig.numeric_sentinels``.  Keys are column names; values are
        float-compatible sentinel values treated as effective nulls.  Columns
        absent from this mapping are not affected.  Defaults to ``{}``.
    string_sentinels : dict[str, list[str]], optional
        Per-column string sentinel declarations copied from
        ``ProfileConfig.string_sentinels``.  When a column name is present,
        only the declared values are matched (case-insensitive) and the
        hardcoded defaults are suppressed for that column.  Empty/whitespace
        detection always applies regardless.  Defaults to ``{}``.
    """

    def __init__(
        self,
        config: MissingnessProfileConfig | None = None,
        numeric_sentinels: dict[str, list[float]] | None = None,
        string_sentinels: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._config = config if config is not None else MissingnessProfileConfig()
        self._numeric_sentinels: dict[str, list[float]] = numeric_sentinels or {}
        self._string_sentinels: dict[str, list[str]] = string_sentinels or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str] | None = None,
    ) -> MissingnessProfileResult:
        """
        Profile missingness across the specified columns of a DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame to profile.
        columns : list[str], optional
            Subset of columns to analyse.  When ``None``, all columns are used.

        Returns
        -------
        MissingnessProfileResult
            Per-column missingness profiles, severity classifications,
            drop-candidate and MAR-suspect flags, and a pairwise missingness
            correlation matrix for columns that have any effective nulls.
        """
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Scope resolution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame, cols: list[str]) -> MissingnessProfileResult:
        result = MissingnessProfileResult()
        result.analysed_columns = cols
        n_rows = df.height

        if n_rows == 0 or not cols:
            return result

        indicator_cols: list[pl.Series] = []

        for col_name in cols:
            col_profile, indicator = self._profile_column(
                series=df[col_name],
                col_name=col_name,
                n_rows=n_rows,
                config=self._config,
                sentinels=self._numeric_sentinels.get(col_name),
                string_sentinels=self._string_sentinels.get(col_name),
            )
            result.columns[col_name] = col_profile
            indicator_cols.append(indicator)

            ratio = col_profile.effective_null_ratio
            if ratio == 1.0:
                result.fully_null_columns.append(col_name)
                col_profile.flags.append(MissingnessFlag.FullyNull)
            elif ratio > self._config.col_drop_threshold:
                col_profile.flags.append(MissingnessFlag.DropCandidate)

        # ── Missingness correlation matrix ────────────────────────────
        cols_with_missing = [
            c for c in cols if result.columns[c].effective_null_count > 0
        ]
        if len(cols_with_missing) >= 2:
            indicator_frame = pl.DataFrame(
                {s.name: s for s in indicator_cols if s.name in cols_with_missing}
            )
            corr_matrix = self._compute_correlation_matrix(
                indicator_frame, cols_with_missing
            )
            result.correlation_matrix = corr_matrix

            for col_a in cols_with_missing:
                mar_peers = [
                    col_b
                    for col_b, r in corr_matrix.get(col_a, {}).items()
                    if col_b != col_a and r > self._config.mar_correlation_threshold
                ]
                if mar_peers:
                    result.columns[col_a].correlated_with = mar_peers
                    if MissingnessFlag.MARSuspect not in result.columns[col_a].flags:
                        result.columns[col_a].flags.append(MissingnessFlag.MARSuspect)

        # ── Row missingness p90 ────────────────────────────────────────
        if indicator_cols:
            row_missing = (
                pl.DataFrame({s.name: s for s in indicator_cols})
                .select(pl.sum_horizontal(pl.all()).alias("n"))["n"]
                .to_numpy()
            )
            p90 = int(np.percentile(row_missing, 90))
        else:
            p90 = 0
        result.row_distribution = RowMissingnessDistribution(row_missingness_p90=p90)

        return result

    # ------------------------------------------------------------------
    # Per-column profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_column(
        series: pl.Series,
        col_name: str,
        n_rows: int,
        config: MissingnessProfileConfig,
        sentinels: list[float] | None = None,
        string_sentinels: list[str] | None = None,
    ) -> tuple[ColumnMissingnessProfile, pl.Series]:
        profile = ColumnMissingnessProfile(column=col_name, total_rows=n_rows)
        dtype = series.dtype
        std_null = series.is_null()

        if _sentinel_eligible(dtype):
            if string_sentinels is not None:
                # Replace semantics: only declared values (case-insensitive);
                # hardcoded _SENTINEL_STRINGS suppressed for this column.
                declared_upper = [s.upper() for s in string_sentinels]
                eff_null = (
                    std_null
                    | (series.str.strip_chars() == "")
                    | series.str.to_uppercase().is_in(declared_upper)
                )
            else:
                eff_null = (
                    std_null
                    | (series.str.strip_chars() == "")
                    | series.str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
                )
        elif _inf_eligible(dtype):
            eff_null = std_null | series.is_nan() | series.is_infinite()
            if sentinels:
                for v in sentinels:
                    eff_null = eff_null | (series == v)
        else:
            eff_null = std_null
            if sentinels and _numeric_sentinel_eligible(dtype):
                # Cast to Float64 for a type-safe comparison that works for all
                # integer dtypes without requiring a sentinel-value cast per dtype.
                float_series = series.cast(pl.Float64)
                for v in sentinels:
                    eff_null = eff_null | (float_series == v)

        std_count = int(std_null.sum())
        eff_count = int(eff_null.sum())

        profile.standard_null_count = std_count
        profile.effective_null_count = eff_count
        profile.standard_null_ratio = std_count / n_rows if n_rows else 0.0
        profile.effective_null_ratio = eff_count / n_rows if n_rows else 0.0

        r = profile.effective_null_ratio

        if r == 0.0:
            profile.severity = None
        elif r < config.severity_minor:
            profile.severity = MissingSeverity.Minor
        elif r < config.severity_moderate:
            profile.severity = MissingSeverity.Moderate
        elif r < config.severity_high:
            profile.severity = MissingSeverity.High
        else:
            profile.severity = MissingSeverity.Severe

        indicator = eff_null.cast(pl.Int8).rename(col_name)
        return profile, indicator

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlation_matrix(
        indicator_frame: pl.DataFrame,
        cols: list[str],
    ) -> dict[str, dict[str, float]]:
        import itertools

        matrix: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        if len(cols) < 2:
            return matrix

        pairs = list(itertools.combinations(cols, 2))
        exprs = [
            pl.corr(col_a, col_b, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]
        result_row = indicator_frame.select(exprs).to_dicts()[0]

        for (col_a, col_b), r_value in zip(pairs, result_row.values()):
            r = max(-1.0, min(1.0, float(r_value)))
            matrix[col_a][col_b] = r
            matrix[col_b][col_a] = r

        return matrix
