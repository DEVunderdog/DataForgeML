"""
ImputationOrchestrator — Phase 2 stateless entry point.

fit(train_df, profile) routes columns to sub-processors via _IMPUTATION_REGISTRY
and assembles a FittedImputer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ..config import PipelineConfig, SemanticType
from ._config import ColumnImputationRecord, ImputationResult, ImputationStrategy
from ._fitted_imputer import FittedImputer
from ._numeric_imputer import NumericImputer, _NumericFitBundle
from ..utils._null_normalization import _resolve_effective_nulls

if TYPE_CHECKING:
    from ..profiling._config import StructuralProfileResult


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
        self._config.imputation.validate()

        train_df = _resolve_effective_nulls(
            train_df,
            numeric_sentinels=profile.numeric_sentinels,
            string_sentinels=profile.string_sentinels,
        )

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
                random_seed=self._config.random_seed,
            )
            if isinstance(result, _NumericFitBundle):
                recs = result.records
                all_models.update(result.models)
                all_model_cols.update(result.model_cols)
            else:
                recs = result
            for rec in recs:
                all_records[rec.column] = rec

        # Passthrough pass: register every train_df column not handled by a sub-processor
        for col in train_df.columns:
            if col in all_records:
                continue
            cp = profile.columns.get(col)
            if cp is None or cp.semantic_type is None:
                continue
            all_records[col] = ColumnImputationRecord(
                column=col,
                semantic_type=cp.semantic_type,
                strategy=ImputationStrategy.Passthrough,
                fill_value=None,
                indicator_added=False,
            )

        # Indicator pass: pre-register {col}_missing columns that transform() will produce
        for col, rec in list(all_records.items()):
            if not rec.indicator_added:
                continue
            indicator_col = f"{col}_missing"
            all_records[indicator_col] = ColumnImputationRecord(
                column=indicator_col,
                semantic_type=SemanticType.Boolean,
                strategy=ImputationStrategy.Indicator,
                fill_value=None,
                indicator_added=False,
            )

        return FittedImputer(
            records=all_records,
            models=all_models,
            model_cols=all_model_cols,
            numeric_sentinels=dict(profile.numeric_sentinels),
            string_sentinels=dict(profile.string_sentinels),
            random_seed=self._config.random_seed,
        )

    def fit_transform(
        self,
        train_df: pl.DataFrame,
        profile: StructuralProfileResult,
    ) -> tuple[FittedImputer, ImputationResult]:
        """Fit the imputer on train_df and transform it in one step.

        Returns both the fitted imputer and the imputed training result so that
        callers are not forced to call ``fit()`` a second time to obtain the
        imputer for test-set transformation.  The natural next step after
        unpacking is ``fitted_imputer.transform(test_df)``.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training split.  Fill parameters are computed exclusively here.
        profile : StructuralProfileResult
            Full-dataset profile from Phase 1.

        Returns
        -------
        tuple[FittedImputer, ImputationResult]
            A two-element tuple ``(fitted_imputer, imputation_result)``.
            ``fitted_imputer`` is identical to the object returned by a
            standalone ``fit()`` call on the same inputs.
            ``imputation_result`` contains the imputed DataFrame and audit
            records for the training split.
        """
        fitted = self.fit(train_df, profile)
        result = fitted.transform(train_df)
        return fitted, result

