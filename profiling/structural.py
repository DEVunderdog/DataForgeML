"""
StructuralProfiler  –  unified Phase 1 entry point.

Orchestrates TabularProfiler and (optionally) CategoricalProfiler,
returning a single StructuralProfileResult that contains both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl
import math

from ._tabular import TabularProfiler
from ._categorical import CategoricalProfiler
from .config import (
    ProfileConfig,
    ColumnProfile,
    DatasetStats,
    StructuralProfileResult,
    RowMissingnessDistribution,
    SemanticType,
    Modality
)
from ._categorical_config import CategoricalProfileResult
from ._numeric_config import NumericProfileResult
from ._numeric_profiler import NumericProfiler
from ._missingness_profiler import MissingnessProfiler
from ._missingness_config import MissingnessProfileResult
from ._target_config import TargetProfileResult
from ._target_profiler import TargetProfiler
from ._correlation_profiler import CorrelationProfiler
from ._correlation_config import CorrelationProfileResult

_ROW_DROP_THRESHOLD = 0.50


class StructuralProfiler:
    """
    Single entry point for Phase 1 structural profiling.

    Usage
    -----
    >>> cfg = ProfileConfig(
    ...     duplicate_columns=["user_id", "event_time"],
    ...     type_detection_columns=["age", "income"],
    ...     categorical_columns=["status", "country"],
    ... )
    >>> profiler = StructuralProfiler(config=cfg)
    >>> result = profiler.profile(df)
    >>> print(result.tabular.duplicate_ratio)
    >>> print(result.categorical.columns["status"].cardinality)
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()

        if self.config.modality == Modality.Tabular:
            self.modality_profiler = TabularProfiler(self.config)
        else:
            raise NotImplementedError(f"modality {self.config.modality} not supported yet")

    def profile(self, data: Any) -> StructuralProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        dataset_stats = self.modality_profiler.profile(data)
        result.dataset = dataset_stats

        missingness_result = MissingnessProfiler(config=self.config).profile(data)

        for col_name in missingness_result.analysed_columns:
            col_profile = result.columns.setdefault(
                col_name,
                ColumnProfile(name=col_name),
            )
            col_profile.missingness = missingness_result.columns.get(col_name)

        if missingness_result.correlation_matrix:
            result.dataset.missingness_matrix = missingness_result.correlation_matrix

        active_cols = missingness_result.analysed_columns
        result.dataset.row_distribution = self._compute_row_distribution(
            df=data,
            cols=active_cols,
            n_rows=data.height,
            overrides=self.config.column_overrides,
        )

        return result

    @staticmethod
    def _compute_row_distribution(
        df: pl.DataFrame,
        cols: list[str],
        n_rows: int,
        overrides: dict[str, SemanticType],
    ) -> RowMissingnessDistribution:
        from ._missingness_profiler import (
            _sentinel_eligible,
            _inf_eligible,
            _SENTINEL_STRINGS,
        )

        dist = RowMissingnessDistribution()
        if n_rows == 0 or not cols:
            return dist

        n_cols = len(cols)
        per_col_exprs = []

        for col_name in cols:
            dtype = df[col_name].dtype
            override = overrides.get(col_name)
            null_e = pl.col(col_name).is_null()

            if _sentinel_eligible(dtype, override):
                eff = (
                    null_e
                    | (pl.col(col_name).str.strip_chars() == "")
                    | pl.col(col_name).str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
                )
            elif _inf_eligible(dtype):
                eff = (
                    null_e | pl.col(col_name).is_nan() | pl.col(col_name).is_infinite()
                )
            else:
                eff = null_e

            per_col_exprs.append(eff.cast(pl.Int8).alias(col_name))

        row_missing: pl.Series = df.select(per_col_exprs).select(
            pl.sum_horizontal(pl.all()).alias("row_missing")
        )["row_missing"]

        half_threshold = math.ceil(n_cols * _ROW_DROP_THRESHOLD)

        dist.pct_zero_missing = float((row_missing == 0).sum()) / n_rows
        dist.pct_one_to_two = (
            float(((row_missing >= 1) & (row_missing <= 2)).sum()) / n_rows
        )
        dist.pct_three_to_five = (
            float(((row_missing >= 3) & (row_missing <= 5)).sum()) / n_rows
        )
        dist.pct_over_five = float((row_missing > 5).sum()) / n_rows
        dist.drop_candidate_row_count = int((row_missing >= half_threshold).sum())
        dist.pct_over_half_missing = dist.drop_candidate_row_count / n_rows

        return dist
