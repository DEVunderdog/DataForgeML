"""
ImputationOrchestrator — Phase 2 stateless entry point.

fit(train_df, profile) routes columns to sub-processors via _IMPUTATION_REGISTRY,
emits SplitImbalanceWarning for unsafe splits, and assembles a FittedImputer.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl

from ..config import PipelineConfig, SemanticType
from ._config import ImputationResult
from ._fitted_imputer import FittedImputer
from ._numeric_imputer import NumericImputer, _NumericFitBundle
from ..utils._null_normalization import _resolve_effective_nulls

if TYPE_CHECKING:
    from ..profiling._config import StructuralProfileResult


class SplitImbalanceWarning(UserWarning):
    """
    Emitted when the training split has zero missing values for a column that
    the full-dataset profile reports as having missingness.

    This typically indicates a non-profile-stratified split was used, which
    means fill values will be computed from a "clean" slice. Use
    DataSplitter.profile_stratified_split() to ensure missing values appear
    in the training partition.
    """


_IMPUTATION_REGISTRY: dict[SemanticType, type] = {
    SemanticType.Numeric: NumericImputer,
}


class ImputationOrchestrator:
    """
    Stateless Phase 2 orchestrator.

    Does not store any fitted state itself — fit() returns a FittedImputer.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  Defaults to PipelineConfig() when omitted.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def fit(
        self,
        train_df: pl.DataFrame,
        profile: StructuralProfileResult,
    ) -> FittedImputer:
        """
        Select imputation strategies from the profile and learn fill values
        from train_df.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training split.  Fill parameters are computed exclusively here.
        profile : StructuralProfileResult
            Full-dataset profile from Phase 1.  Used for strategy routing only.

        Returns
        -------
        FittedImputer
            Stateless imputer that can transform any DataFrame.
        """
        train_df = _resolve_effective_nulls(train_df)
        _check_split_imbalance(train_df, profile)

        imp_cfg = self._config.imputation
        mnar_columns = set(imp_cfg.mnar_columns)

        all_records: dict = {}
        all_models: dict = {}
        all_model_cols: dict = {}

        # Group active columns by semantic type
        type_to_cols: dict[SemanticType, list[str]] = {}
        for col, cp in profile.columns.items():
            if cp.semantic_type is None:
                continue
            if col not in train_df.columns:
                continue
            type_to_cols.setdefault(cp.semantic_type, []).append(col)

        # Route each semantic type to its registered sub-processor
        for sem_type, cols in type_to_cols.items():
            imputer_cls = _IMPUTATION_REGISTRY.get(sem_type)
            if imputer_cls is None:
                # SemanticType.Text, Identifier, and unregistered types pass through
                continue
            result = imputer_cls().fit(
                train_df=train_df,
                columns=cols,
                profile=profile,
                config=imp_cfg.numeric,
                mnar_columns=mnar_columns,
            )
            if isinstance(result, _NumericFitBundle):
                recs = result.records
                all_models.update(result.models)
                all_model_cols.update(result.model_cols)
            else:
                recs = result
            for rec in recs:
                all_records[rec.column] = rec

        return FittedImputer(
            records=all_records,
            models=all_models,
            model_cols=all_model_cols,
        )

    def fit_transform(
        self,
        train_df: pl.DataFrame,
        profile: StructuralProfileResult,
    ) -> ImputationResult:
        """Convenience: fit on train_df, then transform train_df."""
        return self.fit(train_df, profile).transform(train_df)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _check_split_imbalance(
    train_df: pl.DataFrame,
    profile: StructuralProfileResult,
) -> None:
    imbalanced: list[str] = []
    for col, cp in profile.columns.items():
        if cp.missingness is None or cp.missingness.effective_null_count == 0:
            continue
        if col not in train_df.columns:
            continue
        if train_df[col].null_count() == 0:
            imbalanced.append(col)

    if imbalanced:
        warnings.warn(
            f"Training split has no missing values for {len(imbalanced)} column(s) "
            f"that the full-dataset profile reports as having missingness: "
            f"{imbalanced}. "
            f"Fill values will still be computed but the split may not be "
            f"representative. Consider using "
            f"DataSplitter.profile_stratified_split() to ensure missing values "
            f"appear in the training partition.",
            SplitImbalanceWarning,
            stacklevel=3,
        )
