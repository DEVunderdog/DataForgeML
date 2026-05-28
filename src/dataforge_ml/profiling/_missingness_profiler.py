"""
MissingnessProfiler  –  Phase 1 extension: Missingness Profiling.

Eligibility model
-----------------
Effective-null detection is purely dtype-driven — no SemanticType overrides:

sentinel-string detection  →  runs for every String/Utf8 column unconditionally
Inf / NaN expansion        →  runs for every Float32/Float64 column unconditionally
"""

from __future__ import annotations


import polars as pl

from ._base import DatasetLevelProfiler
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingnessProfileResult,
    MissingSeverity,
)
from ..utils._null_detection import _SENTINEL_STRINGS, _inf_eligible, _sentinel_eligible

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SEVERITY_MINOR = 0.01
_SEVERITY_MODERATE = 0.05
_SEVERITY_HIGH = 0.20

_MAR_CORRELATION_THRESHOLD = 0.60
_COL_DROP_THRESHOLD = 0.50


class MissingnessProfiler(DatasetLevelProfiler[MissingnessProfileResult]):
    """Missingness profiler for Polars DataFrames."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str] | None = None,
    ) -> MissingnessProfileResult:
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
            )
            result.columns[col_name] = col_profile
            indicator_cols.append(indicator)

            ratio = col_profile.effective_null_ratio
            if ratio == 1.0:
                result.fully_null_columns.append(col_name)
                col_profile.flags.append(MissingnessFlag.FullyNull)
            elif ratio > _COL_DROP_THRESHOLD:
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
                    if col_b != col_a and r > _MAR_CORRELATION_THRESHOLD
                ]
                if mar_peers:
                    result.columns[col_a].correlated_with = mar_peers
                    if MissingnessFlag.MARSuspect not in result.columns[col_a].flags:
                        result.columns[col_a].flags.append(MissingnessFlag.MARSuspect)

        return result

    # ------------------------------------------------------------------
    # Per-column profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_column(
        series: pl.Series,
        col_name: str,
        n_rows: int,
    ) -> tuple[ColumnMissingnessProfile, pl.Series]:
        profile = ColumnMissingnessProfile(column=col_name, total_rows=n_rows)
        dtype = series.dtype
        std_null = series.is_null()

        if _sentinel_eligible(dtype):
            eff_null = (
                std_null
                | (series.str.strip_chars() == "")
                | series.str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
            )
        elif _inf_eligible(dtype):
            eff_null = std_null | series.is_nan() | series.is_infinite()
        else:
            eff_null = std_null

        std_count = int(std_null.sum())
        eff_count = int(eff_null.sum())

        profile.standard_null_count = std_count
        profile.effective_null_count = eff_count
        profile.standard_null_ratio = std_count / n_rows if n_rows else 0.0
        profile.effective_null_ratio = eff_count / n_rows if n_rows else 0.0

        r = profile.effective_null_ratio

        if r == 0.0:
            profile.severity = None
        elif r < _SEVERITY_MINOR:
            profile.severity = MissingSeverity.Minor
        elif r < _SEVERITY_MODERATE:
            profile.severity = MissingSeverity.Moderate
        elif r < _SEVERITY_HIGH:
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
