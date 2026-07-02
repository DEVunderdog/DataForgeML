"""
StructuralProfiler  –  unified Phase 1 entry point.

Execution order inside profile(df):
  1. ModalityProfiler      → result.dataset (DatasetStats)
  2. MissingnessProfiler   → ColumnProfile.missingness + dataset.missingness_matrix
  3. Row-missingness dist  → dataset.row_distribution
  4. TypeDetector          → ColumnProfile.semantic_type / type_flags / dtypes
  5. column_overrides      → replace SemanticType on existing ColumnProfiles
     numeric_kind_overrides→ replace NumericKind (validated; Numeric-only)
  6. ColumnTypeProfiler    → route each column to its profiler by SemanticType;
                            Identifier columns: skip, stats stays None
  7. target_columns        → TargetProfiler; mark ColumnProfile.is_target=True
  8. Correlation           → if compute_correlation=True:
       a. profile_features()  → dataset.feature_correlation  (computed once)
       b. profile_target()    → dataset.target_correlations[target]
                                (once per declared target column)
  9. Nonlinearity          → if compute_nonlinearity=True:
       NonlinearityProfiler  → NumericStats.nonlinearity_tag + four signal fields
       Reuses Pearson/Spearman matrices from step 8 when compute_correlation=True.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ._base import ModalityProfiler, ColumnBatchProfiler, OverrideCoercionError
from ._tabular import TabularProfiler
from ._categorical import CategoricalProfiler
from ._datetime_profiler import DatetimeProfiler
from ._numeric_profiler import NumericProfiler
from ._boolean_profiler import BooleanProfiler
from ._text_profiler import TextProfiler
from ._missingness_profiler import MissingnessProfiler
from ._target_profiler import TargetProfiler
from ._correlation_profiler import CorrelationProfiler
from ._nonlinearity_profiler import NonlinearityProfiler
from ._type_detector import TypeDetector
from ..config import PipelineConfig, PipelinePhase, SemanticType, Modality
from ._config import (
    ColumnProfile,
    StructuralProfileResult,
    RowMissingnessDistribution,
    TypeFlag,
)
from ..utils._null_normalization import _resolve_effective_nulls

# ---------------------------------------------------------------------------
# Registry: SemanticType → ColumnTypeProfiler class
#
# Stateless between profile(series, df) calls, so one instance per
# SemanticType safely handles all columns of that type in one run.
# Add Boolean / Text profilers here when implemented.
# ---------------------------------------------------------------------------
_COLUMN_PROFILER_REGISTRY: dict[SemanticType, type[ColumnBatchProfiler]] = {  # type: ignore[type-arg]
    SemanticType.Numeric: NumericProfiler,
    SemanticType.Categorical: CategoricalProfiler,
    SemanticType.Datetime: DatetimeProfiler,
    SemanticType.Boolean: BooleanProfiler,
    SemanticType.Text: TextProfiler,
}


class StructuralProfiler:
    """
    Phase 1 orchestrator — runs all sub-processors and assembles
    ``StructuralProfileResult``.

    Parameters
    ----------
    config : PipelineConfig, optional
        Master pipeline configuration.  Defaults to ``PipelineConfig()`` which
        applies all sub-processor defaults exactly as they were before Scope 15.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config: PipelineConfig = config or PipelineConfig()

        if self.config.profiling.modality == Modality.Tabular:
            self.modality_profiler: ModalityProfiler = TabularProfiler()
        else:
            raise NotImplementedError(
                f"modality {self.config.profiling.modality} not supported yet"
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> StructuralProfileResult:
        """
        Profile a Polars DataFrame and return a full ``StructuralProfileResult``.

        Runs all Phase 1 sub-processors in sequence: modality stats, missingness,
        type detection, per-column distribution profiling, target profiling, and
        (optionally) correlation analysis.  Sub-processor thresholds are drawn
        from the nested sub-configs on ``config.profiling``.

        Parameters
        ----------
        data : Any
            Must be a ``polars.DataFrame``.

        Returns
        -------
        StructuralProfileResult
            Fully populated profile result for the dataset.

        Raises
        ------
        TypeError
            When ``data`` is not a ``polars.DataFrame``.
        OverrideCoercionError
            When a column carrying ``TypeFlag.UserOverride`` completely fails
            coercion to its overridden ``SemanticType`` (zero usable values
            remain despite the original column having non-null data).
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        active_cols = self.config.resolve_active_columns(
            PipelinePhase.Profiling, list(data.columns)
        )

        # Columns soft-excluded for Profiling: skipped but retained in the result.
        hard_set = set(self.config.exclude_columns)
        soft_retained = [
            c for c in data.columns
            if c in set(self.config.phase_exclusions.get(PipelinePhase.Profiling, []))
            and c not in hard_set
        ]

        # ── 1. Modality profiler ─────────────────────────────────────────
        # Replaces default DatasetStats with the real one (row_count, memory,
        # duplicates, etc.).  Must run before anything writes to result.dataset.
        result.dataset = self.modality_profiler.profile(data)

        # ── 2. Missingness pre-pass ──────────────────────────────────────
        # setdefault creates ColumnProfile entries; subsequent steps mutate
        # the same objects via the same setdefault pattern.
        missingness_result = MissingnessProfiler(
            config=self.config.profiling.missingness,
            numeric_sentinels=self.config.profiling.numeric_sentinels,
            string_sentinels=self.config.profiling.string_sentinels,
        ).profile(data, columns=active_cols)
        for col_name in missingness_result.analysed_columns:
            cp = result.columns.setdefault(col_name, ColumnProfile(name=col_name))
            cp.missingness = missingness_result.columns.get(col_name)

        if missingness_result.correlation_matrix:
            result.dataset.missingness_matrix = missingness_result.correlation_matrix

        # ── 3. Row-missingness distribution ─────────────────────────────
        result.dataset.row_distribution = self._compute_row_distribution(
            df=data,
            cols=active_cols,
            n_rows=data.height,
            row_drop_threshold=self.config.profiling.row_drop_threshold,
            numeric_sentinels=self.config.profiling.numeric_sentinels,
            string_sentinels=self.config.profiling.string_sentinels,
        )

        # ── 4. Type detection ────────────────────────────────────────────
        # setdefault returns the existing ColumnProfile from step 2, so
        # missingness and type info land on the same object.
        type_info = TypeDetector(
            columns=active_cols,
            config=self.config.profiling.type_detection,
        ).detect(data)
        for col_name, info in type_info.items():
            cp = result.columns.setdefault(col_name, ColumnProfile(name=col_name))
            cp.semantic_type = info.semantic_type
            cp.numeric_kind = info.numeric_kind
            cp.type_flags = list(info.flags)
            cp.original_dtype = info.original_dtype
            cp.inferred_dtype = info.inferred_dtype

        # ── 5. Apply column_overrides then numeric_kind_overrides ────────
        # All active columns are in result.columns by now (steps 2 + 4).
        # Overrides for excluded / non-existent columns are silently ignored.
        # SemanticType overrides must be applied first so the NumericKind guard
        # checks the user's declared type, not the detector's raw type.
        for col_name, override_type in self.config.column_overrides.items():
            if col_name in result.columns:
                cp = result.columns[col_name]
                cp.semantic_type = override_type
                if TypeFlag.UserOverride not in cp.type_flags:
                    cp.type_flags.append(TypeFlag.UserOverride)

        for col_name, override_kind in self.config.numeric_kind_overrides.items():
            if col_name not in result.columns:
                continue
            cp = result.columns[col_name]
            if cp.semantic_type != SemanticType.Numeric:
                raise ValueError(
                    f"NumericKind override for column {col_name!r} is invalid — "
                    f"column has {cp.semantic_type!r}. "
                    f"NumericKind only applies to SemanticType.Numeric columns."
                )
            cp.numeric_kind = override_kind
            if TypeFlag.NumericKindOverride not in cp.type_flags:
                cp.type_flags.append(TypeFlag.NumericKindOverride)

        # ── 6. Per-column profiling routed by SemanticType ───────────────
        # Batch all columns of the same SemanticType together and call each
        # profiler once with (df, column_list) — matching the profiler API.
        type_to_cols: dict[SemanticType, list[str]] = {}
        for col_name in active_cols:
            cp = result.columns.get(col_name)
            if cp is None or cp.semantic_type is None:
                continue
            if cp.semantic_type == SemanticType.Identifier:
                continue
            sem_type = cp.semantic_type
            type_to_cols.setdefault(sem_type, []).append(col_name)

        pc = self.config.profiling
        for sem_type, cols in type_to_cols.items():
            if sem_type == SemanticType.Numeric:
                profiler = NumericProfiler(config=pc.numeric)
            elif sem_type == SemanticType.Categorical:
                profiler = CategoricalProfiler(config=pc.categorical)
            elif sem_type == SemanticType.Datetime:
                profiler = DatetimeProfiler(config=pc.datetime_, epoch_units=pc.datetime_epoch_units)
            else:
                profiler_cls = _COLUMN_PROFILER_REGISTRY.get(sem_type)  # type: ignore[arg-type]
                if profiler_cls is None:
                    continue
                profiler = profiler_cls()
            try:
                user_overrides = {
                    c for c in cols
                    if result.columns.get(c) and TypeFlag.UserOverride in result.columns[c].type_flags
                }
                batch = profiler.profile(data, columns=cols, user_overrides=user_overrides)
                for col_name in batch.analysed_columns:
                    if col_name in result.columns:
                        result.columns[col_name].stats = batch.columns.get(col_name)
            except OverrideCoercionError:
                raise
            except Exception:
                pass

        # ── 7. Target columns ────────────────────────────────────────────
        # TargetProfiler produces target-specific analysis stored in
        # result.targets.  cp.stats is NOT overwritten — step 6 already set it.
        if self.config.profiling.target_columns:
            for target in self.config.profiling.target_columns:
                if target not in data.columns:
                    continue
                target_result = TargetProfiler(
                    target_column=target,
                    config=self.config.profiling,
                ).profile(data)
                result.targets[target] = target_result

                # setdefault returns the existing ColumnProfile.
                cp = result.columns.setdefault(target, ColumnProfile(name=target))
                cp.is_target = True

        # ── 8. Correlation ───────────────────────────────────────────────
        if self.config.profiling.compute_correlation:
            # Resolve column lists by detected SemanticType (post-override).
            numeric_cols = [
                c
                for c in active_cols
                if result.columns.get(c)
                and result.columns[c].semantic_type == SemanticType.Numeric
            ]
            categorical_cols = [
                c
                for c in active_cols
                if result.columns.get(c)
                and result.columns[c].semantic_type == SemanticType.Categorical
            ]

            corr_profiler = CorrelationProfiler(
                numeric_columns=numeric_cols,
                categorical_columns=categorical_cols,
                config=self.config.profiling.correlation,
            )

            # 8a. Feature-feature matrices — computed ONCE, target-independent.
            feature_corr = corr_profiler.profile_features(
                data, numeric_cols, categorical_cols
            )
            result.dataset.feature_correlation = feature_corr

            # 8b. Per-target analysis — matrices are NOT recomputed; each call
            #     shallow-copies feature_corr and appends target-specific fields.
            for target in self.config.profiling.target_columns:
                if target not in data.columns:
                    continue
                result.dataset.target_correlations[target] = (
                    corr_profiler.profile_target(
                        data, feature_corr, numeric_cols, categorical_cols, target
                    )
                )

        # ── 9. Nonlinearity ──────────────────────────────────────────────
        if self.config.profiling.compute_nonlinearity:
            numeric_cols_nl = [
                c
                for c in active_cols
                if result.columns.get(c)
                and result.columns[c].semantic_type == SemanticType.Numeric
            ]
            if len(numeric_cols_nl) >= 2:
                p_mat = (
                    result.dataset.feature_correlation.pearson_matrix
                    if result.dataset.feature_correlation is not None
                    else None
                )
                s_mat = (
                    result.dataset.feature_correlation.spearman_matrix
                    if result.dataset.feature_correlation is not None
                    else None
                )
                nl_result = NonlinearityProfiler(
                    numeric_columns=numeric_cols_nl,
                    config=self.config.profiling.nonlinearity,
                ).profile(data, pearson_matrix=p_mat, spearman_matrix=s_mat)

                from ._numeric_config import NumericStats as _NumericStats

                for col_name, signals in nl_result.columns.items():
                    cp = result.columns.get(col_name)
                    if cp is not None and isinstance(cp.stats, _NumericStats):
                        cp.stats.nonlinearity_tag = signals.tag
                        cp.stats.spearman_pearson_discrepancy = (
                            signals.spearman_pearson_discrepancy
                        )
                        cp.stats.mean_mutual_information = signals.mean_mutual_information
                        cp.stats.r2_gap = signals.r2_gap
                        cp.stats.heteroscedasticity_p_value = (
                            signals.heteroscedasticity_p_value
                        )

        # ── Soft-excluded placeholders ───────────────────────────────────────
        # Columns soft-excluded for Profiling are not profiled but must still
        # appear in the result so downstream phases can reference them.
        for col in soft_retained:
            result.columns.setdefault(col, ColumnProfile(name=col))

        result.numeric_sentinels = dict(self.config.profiling.numeric_sentinels)
        result.string_sentinels = {
            k: list(v) for k, v in self.config.profiling.string_sentinels.items()
        }

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_row_distribution(
        df: pl.DataFrame,
        cols: list[str],
        n_rows: int,
        row_drop_threshold: float = 0.50,
        numeric_sentinels: dict[str, list[float]] | None = None,
        string_sentinels: dict[str, list[str]] | None = None,
    ) -> RowMissingnessDistribution:
        dist = RowMissingnessDistribution()
        if n_rows == 0 or not cols:
            return dist

        n_cols = len(cols)
        subset = _resolve_effective_nulls(
            df.select(cols),
            numeric_sentinels=numeric_sentinels,
            string_sentinels=string_sentinels,
        )
        row_missing: pl.Series = subset.select(
            pl.sum_horizontal(pl.all().is_null()).alias("row_missing")
        )["row_missing"]

        half_threshold = math.ceil(n_cols * row_drop_threshold)

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
        dist.row_missingness_p90 = int(np.percentile(row_missing.to_numpy(), 90))

        return dist
