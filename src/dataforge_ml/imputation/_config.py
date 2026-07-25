"""
Configuration and result dataclasses for the imputation phase — Phase 2.

ImputationConfig controls strategy thresholds and MNAR declarations.
Result dataclasses carry per-column audit records and the imputed DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Optional

import polars as pl

from ..config import SemanticType


class ImputationStrategy(StrEnum):
    """Imputation strategy assigned to a column after Phase 2 fitting.

    Members fall into two categories:

    **Input strategies** — may be declared in ``per_column_strategy`` to
    override automatic routing: ``Mean``, ``Median``, ``Mode``, ``KNN``,
    ``MICE``.

    **Output-only labels** — assigned by the engine after ``fit()`` and
    recorded in ``ColumnImputationRecord.strategy``; declaring them in
    ``per_column_strategy`` raises ``ValueError`` at construction time:
    ``Constant``, ``MNAR``, ``Dropped``, ``Passthrough``, ``Indicator``.
    ``Constant`` is produced when a column appears in
    ``per_column_constant_fill``; use that field instead of declaring it in
    ``per_column_strategy``.
    """

    Mean = "mean"
    Median = "median"
    Mode = "mode"
    KNN = "knn"
    MICE = "mice"
    MNAR = "mnar"
    Constant = "constant"
    Dropped = "dropped"
    Passthrough = "passthrough"  # output-only: assigned to columns with no missing values in training; cannot be declared in per_column_strategy
    ClusterConditional = "cluster_conditional"
    GMMSampling = "gmm_sampling"
    Indicator = "indicator"  # output-only: assigned to {col}_missing columns appended by the MNAR mechanism; cannot be declared in per_column_strategy


_MODEL_BASED_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.MICE,
        ImputationStrategy.KNN,
        ImputationStrategy.ClusterConditional,
        ImputationStrategy.GMMSampling,
    }
)


class ModelChoice(StrEnum):
    """Concrete estimator family selected for a model-based imputation column.

    A value-free *name* for the sklearn estimator family resolved at
    decide-time from ``(NonlinearityTag, n_rows, config)`` — the Recipe half of
    the Recipe/Learned split (ADR-0059) pulled one layer earlier so the plan is
    inspectable and overridable before anything trains. It is a label, never a
    live or fitted estimator object; the execution layer constructs and fits the
    actual estimator from this choice.

    Members map onto the branches of ``RegressionEstimatorFactory``:
    ``BayesianRidge`` for the ``Linear`` branch (wrapped in a scaling pipeline
    at fit-time), ``RandomForestRegressor`` for ``MonotonicNonlinear`` and the
    small-sample ``ComplexNonlinear`` branch, and ``GradientBoostingRegressor``
    for the large-sample ``ComplexNonlinear`` branch. An ``Unpredictable``
    column resolves to no model choice (``None``) and routes to a scalar
    fallback instead.
    """

    BayesianRidge = "bayesian_ridge"
    RandomForestRegressor = "random_forest_regressor"
    GradientBoostingRegressor = "gradient_boosting_regressor"


# ---------------------------------------------------------------------------
# Shared strategy-legality rules
#
# One source of truth for "which strategies may a user declare, and what does
# the redirect say when they cannot" — consumed both by
# ``NumericImputationConfig.set_per_column_strategy`` (declaring a strategy in
# config) and by ``ImputationDecision.with_strategy`` (editing a built plan), so
# both paths reject illegal strategies identically (ADR-0060).
# ---------------------------------------------------------------------------

# Output-only labels that a user may never declare directly — the engine assigns
# them after routing. ``Constant`` is handled separately: it is declarable in
# config (paired with a fill value) but not editable onto a plan.
_OUTPUT_ONLY_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.Passthrough,
        ImputationStrategy.Indicator,
        ImputationStrategy.Dropped,
        ImputationStrategy.MNAR,
        ImputationStrategy.ClusterConditional,
        ImputationStrategy.GMMSampling,
    }
)

# Signal recorded on the Passthrough decision of an Imputation-soft-excluded
# column. Written by the decision assembler; matched by transform to tell
# "excluded by declaration" (missing values ride through untouched) apart from
# "passed through because fit saw no missingness" (missing values raise).
_EXCLUSION_SIGNAL = "soft-excluded for Imputation phase"

# Input strategies a user may declare, keyed by the column's semantic type. Only
# semantic types with a registered imputer appear here; a strategy declared for
# any other type has no engine that can execute it and is rejected. Future phases
# extend this map as they gain imputers.
_DECLARABLE_STRATEGIES_BY_TYPE: dict[SemanticType, frozenset[ImputationStrategy]] = {
    SemanticType.Numeric: frozenset(
        {
            ImputationStrategy.Mean,
            ImputationStrategy.Median,
            ImputationStrategy.Mode,
            ImputationStrategy.KNN,
            ImputationStrategy.MICE,
        }
    ),
}


def _output_only_redirect(column: str, strategy: ImputationStrategy) -> str:
    """Build the redirect message for an output-only strategy declaration."""
    if strategy == ImputationStrategy.Dropped:
        return (
            f"Column '{column}': 'Dropped' cannot be used in per_column_strategy. "
            f"To exclude a column, use PipelineConfig.exclude_columns."
        )
    if strategy == ImputationStrategy.MNAR:
        return (
            f"Column '{column}': 'MNAR' cannot be used in per_column_strategy. "
            f"To declare MNAR semantics, use mnar_columns."
        )
    return (
        f"Column '{column}': '{strategy}' is an internal-only strategy and cannot "
        f"be used in per_column_strategy."
    )


def _constant_without_fill_redirect(column: str) -> str:
    """Build the redirect message for a ``Constant`` declaration with no fill."""
    return (
        f"Column '{column}': strategy is 'Constant' but no fill value was provided. "
        f"Add an entry to per_column_constant_fill."
    )


@dataclass
class NumericImputationConfig:
    """
    Operational thresholds for the numeric imputation sub-processor.

    Parameters
    ----------
    knn_max_rows : int
        Maximum number of rows before KNN is skipped in favour of MICE.
    knn_max_features : int
        Maximum number of features before KNN is skipped in favour of MICE.
    mice_min_rows : int
        Minimum number of rows required to fit a stable MICE (chained-equations)
        model. Applied as a uniform floor to every routing path that can enter
        the joint MICE block; a column below this floor diverts to KNN, then
        Median, instead (ADR-0079). Renamed from ``regression_min_rows``.
    gradient_boost_min_rows : int
        Row count threshold above which ``GradientBoostingRegressor`` is preferred
        over ``RandomForestRegressor`` for ``ComplexNonlinear`` columns. Below this
        threshold the cheaper ``RandomForestRegressor`` is used instead.
    base_max_iter : int
        Base number of ``IterativeImputer`` iterations before dynamic signal
        adjustments are applied.  Increase this value for columns that exhibit
        convergence warnings in ``ColumnImputationRecord.signals``.
    knn_min_neighbors : int
        Floor on the adaptively computed ``n_neighbors`` value passed to
        ``KNNImputer``. The computed k will never fall below this value.
    knn_max_neighbors : int
        Cap on the adaptively computed ``n_neighbors`` value passed to
        ``KNNImputer``. The computed k will never exceed this value.
    knn_distance_weight_max_null_ratio : float
        Feature-matrix missingness fraction below which distance weighting is
        considered reliable. When ``miss_frac`` exceeds this threshold,
        ``weights`` is forced to ``"uniform"``.
    knn_distance_weight_max_features : int
        Dimensionality threshold below which distance weighting is considered
        reliable. When the number of KNN feature columns exceeds this value,
        ``weights`` is forced to ``"uniform"``.
    mice_n_nearest_features_min_cols : int
        MICE block size at or below which ``n_nearest_features`` is left unset,
        meaning all columns in the block are used as predictors for every
        imputation target. Above this threshold, ``n_nearest_features`` is
        derived from value-level Pearson correlations.
    mice_max_nearest_features : int
        Upper cap on the ``n_nearest_features`` value computed for large MICE
        blocks. The correlation-derived count is clamped to this maximum before
        being passed to ``IterativeImputer``.
    mice_correlation_threshold : float
        Minimum absolute Pearson correlation ``|r|`` required for another MICE
        column to be counted as an informative predictor when computing
        ``n_nearest_features``. Columns below this threshold are excluded from
        the count.
    mcar_feature_predictability_threshold : float
        Maximum absolute Pearson correlation ``|r|`` below which MCAR
        model-based routing is skipped in favour of Median. When no numeric
        predictor exceeds this threshold against the target column, KNN and
        MICE are not attempted because the feature set contains no useful
        predictive signal. Applies only to MCAR paths; MAR paths are not
        affected. Default of ``0.2`` preserves existing behaviour (no check
        applied today).
    per_column_strategy : dict[str, ImputationStrategy]
        Explicit per-column strategy overrides that fire at Priority 1.5 in the
        routing chain — after ``DropCandidate`` but before MNAR routing.  A
        column listed here bypasses all routing priorities 2–7.  Defaults to
        empty dict (no overrides).  Allowed values: ``Mean``, ``Median``,
        ``Mode``, ``KNN``, ``MICE``.  To route a column to a
        constant fill, use ``per_column_constant_fill``
    per_column_constant_fill : dict[str, float]
        Self-sufficient constant fill declarations.  Each column listed here
        is routed to ``ImputationStrategy.Constant`` at Priority 1.5,
        bypassing all routing priorities 2–7.  No companion entry in
        ``per_column_strategy`` is required or allowed.  Keyed by column name.
        Defaults to empty dict.
    knn_n_neighbors : int, optional
        Overrides the dynamically-computed ``n_neighbors`` for the entire KNN
        block. A single value governs all KNN columns.
    mice_max_iter : int, optional
        Overrides the dynamically-computed ``max_iter`` for the entire MICE
        block. A single value governs all MICE columns.
    refit_r2_min_complete_rows : int
        Minimum number of complete rows required to attempt the held-out
        accuracy computation.  When fewer complete rows are available, the
        ``r2_cv``, ``rmse``, and ``mae`` fields on ``AccuracyDiagnostic`` are set
        to ``None``.  With k-fold CV (``refit_r2_cv_folds=5``), each validation
        fold contains 1/k of the complete rows; the floor of 50 ensures at least
        10 rows per fold.  Default ``50``.
    refit_r2_cv_folds : int
        Number of folds for the cross-validated accuracy computation
        (:meth:`EvaluationOrchestrator.score_accuracy`).  Applied uniformly
        across KNN and MICE columns.  Default ``5``.
    bimodal_grouping_variables : dict[str, str]
        Maps a bimodal column name to the name of the grouping column that
        explains the bimodal split (e.g. ``{"age": "employment_status"}``).
    bimodal_min_correlated_features : int
        Minimum number of numeric features with ``|r| > 0.2`` required to
        qualify the Bimodal Imputation Framework for branch 2 (MICE/KNN);
        columns with fewer correlated features fall to branch 3 (Cluster-Conditional).
    bimodal_correlation_threshold : float
        Minimum absolute Pearson correlation ``|r|`` a feature must have against
        a bimodal column for it to count toward the branch 2/3 feature tally in
        the Bimodal Imputation Framework.
    max_workers : int, optional
        Degree of thread parallelism for the numeric fit (ADR-0056).  The
        mutually-independent strategy blocks (MICE, KNN, GMM, cluster) and the
        independent columns within a per-column strategy are fitted
        concurrently on threads, capped at this many workers.
        ``None`` (the default) auto-sizes to the available CPU count; ``1``
        forces a fully sequential fit.  Concurrency never changes a result:
        the same ``random_seed`` yields a byte-identical ``FittedImputer``
        regardless of this value.

    Raises
    ------
    ValueError
        If any column in ``per_column_strategy`` is mapped to
        ``Passthrough``, ``Indicator``, ``Dropped``, or ``MNAR``.  ``Constant``
        columns should use ``per_column_constant_fill``; ``Dropped`` columns
        should use ``PipelineConfig.exclude_columns``; ``MNAR`` columns should
        use ``mnar_columns``; ``Passthrough`` and ``Indicator`` are
        internal-only.
    """

    knn_max_rows: int = 50_000
    knn_max_features: int = 50
    mice_min_rows: int = 500
    gradient_boost_min_rows: int = 10_000
    base_max_iter: int = 10
    knn_min_neighbors: int = 5
    knn_max_neighbors: int = 25
    knn_distance_weight_max_null_ratio: float = 0.15
    knn_distance_weight_max_features: int = 30
    mice_n_nearest_features_min_cols: int = 10
    mice_max_nearest_features: int = 20
    mice_correlation_threshold: float = 0.1
    mcar_feature_predictability_threshold: float = 0.2
    _per_column_strategy: dict[str, ImputationStrategy] = field(default_factory=dict)
    _per_column_constant_fill: dict[str, float] = field(default_factory=dict)
    knn_n_neighbors: int | None = None
    mice_max_iter: int | None = None
    refit_r2_min_complete_rows: int = 50
    refit_r2_cv_folds: int = 5
    _bimodal_grouping_variables: dict[str, str] = field(default_factory=dict)
    bimodal_min_correlated_features: int = 3
    bimodal_correlation_threshold: float = 0.2
    max_workers: int | None = None

    @property
    def per_column_strategy(self) -> MappingProxyType[str, ImputationStrategy]:
        """
        Explicit per-column strategy overrides that fire at Priority 1.5 in the
        routing chain — after ``DropCandidate`` but before MNAR routing.

        Returns
        -------
        MappingProxyType[str, ImputationStrategy]
            Read-only view of per-column strategy overrides.
        """
        return MappingProxyType(self._per_column_strategy)

    @property
    def per_column_constant_fill(self) -> MappingProxyType[str, float]:
        """
        Self-sufficient constant fill declarations.

        Returns
        -------
        MappingProxyType[str, float]
            Read-only view of per-column constant fill values.
        """
        return MappingProxyType(self._per_column_constant_fill)

    @property
    def bimodal_grouping_variables(self) -> MappingProxyType[str, str]:
        """
        Maps a bimodal column name to the name of the grouping column that
        explains the bimodal split.

        Returns
        -------
        MappingProxyType[str, str]
            Read-only view of bimodal grouping variables.
        """
        return MappingProxyType(self._bimodal_grouping_variables)

    def set_per_column_strategy(
        self, column: str | list[str], strategy: str | ImputationStrategy
    ) -> None:
        """
        Set the imputation strategy for one or more columns.

        Parameters
        ----------
        column : str | list[str]
            A single column name or list of column names.
        strategy : str | ImputationStrategy
            The strategy to assign to the column(s).

        Raises
        ------
        ValueError
            If the strategy is an output-only label (e.g. 'MNAR', 'Dropped').
            If 'Constant' is set but no corresponding fill value exists in
            ``per_column_constant_fill``.
        """
        if isinstance(column, str):
            column = [column]

        strategy = ImputationStrategy(strategy)

        if strategy in _OUTPUT_ONLY_STRATEGIES:
            for col in column:
                raise ValueError(_output_only_redirect(col, strategy))

        if strategy == ImputationStrategy.Constant:
            for col in column:
                if col not in self._per_column_constant_fill:
                    raise ValueError(_constant_without_fill_redirect(col))

        for col in column:
            self._per_column_strategy[col] = strategy

    def set_per_column_constant_fill(
        self, column: str | list[str], value: float
    ) -> None:
        """
        Set a constant fill value for one or more columns.

        Parameters
        ----------
        column : str | list[str]
            A single column name or list of column names.
        value : float
            The constant fill value.

        Raises
        ------
        ValueError
            If the value is NaN or infinity.
        """
        import math

        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Fill value cannot be NaN or infinity, got {value}.")

        if isinstance(column, str):
            column = [column]

        for col in column:
            self._per_column_constant_fill[col] = value

    def set_bimodal_grouping_variable(
        self, column: str | list[str], grouping_variable: str
    ) -> None:
        """
        Set the grouping variable for one or more bimodal columns.

        Parameters
        ----------
        column : str | list[str]
            A single column name or list of column names.
        grouping_variable : str
            The name of the grouping column.

        Raises
        ------
        ValueError
            If the grouping variable is empty or purely whitespace.
        """
        if not grouping_variable or not grouping_variable.strip():
            raise ValueError("Grouping variable cannot be empty or purely whitespace.")

        if isinstance(column, str):
            column = [column]

        for col in column:
            self._bimodal_grouping_variables[col] = grouping_variable

    def __post_init__(self) -> None:
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(
                f"max_workers must be None or >= 1, got {self.max_workers}."
            )
        _BLOCKED = {
            ImputationStrategy.Passthrough,
            ImputationStrategy.Indicator,
            ImputationStrategy.Dropped,
            ImputationStrategy.MNAR,
            ImputationStrategy.ClusterConditional,
            ImputationStrategy.GMMSampling,
        }
        for col, strategy in self._per_column_strategy.items():
            if strategy in _BLOCKED:
                if strategy == ImputationStrategy.Dropped:
                    raise ValueError(
                        f"Column '{col}': 'Dropped' cannot be used in per_column_strategy. "
                        f"To exclude a column, use PipelineConfig.exclude_columns."
                    )
                if strategy == ImputationStrategy.MNAR:
                    raise ValueError(
                        f"Column '{col}': 'MNAR' cannot be used in per_column_strategy. "
                        f"To declare MNAR semantics, use mnar_columns."
                    )
                raise ValueError(
                    f"Column '{col}': '{strategy}' is an internal-only strategy and cannot "
                    f"be used in per_column_strategy."
                )
            if (
                strategy == ImputationStrategy.Constant
                and col not in self._per_column_constant_fill
            ):
                raise ValueError(
                    f"Column '{col}': strategy is 'Constant' but no fill value was provided. "
                    f"Add an entry to per_column_constant_fill."
                )

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "knn_max_rows": self.knn_max_rows,
            "knn_max_features": self.knn_max_features,
            "mice_min_rows": self.mice_min_rows,
            "gradient_boost_min_rows": self.gradient_boost_min_rows,
            "base_max_iter": self.base_max_iter,
            "knn_min_neighbors": self.knn_min_neighbors,
            "knn_max_neighbors": self.knn_max_neighbors,
            "knn_distance_weight_max_null_ratio": self.knn_distance_weight_max_null_ratio,
            "knn_distance_weight_max_features": self.knn_distance_weight_max_features,
            "mice_n_nearest_features_min_cols": self.mice_n_nearest_features_min_cols,
            "mice_max_nearest_features": self.mice_max_nearest_features,
            "mice_correlation_threshold": self.mice_correlation_threshold,
            "mcar_feature_predictability_threshold": self.mcar_feature_predictability_threshold,
            "per_column_strategy": {
                k: str(v) for k, v in self._per_column_strategy.items()
            },
            "per_column_constant_fill": dict(self._per_column_constant_fill),
            "knn_n_neighbors": self.knn_n_neighbors,
            "mice_max_iter": self.mice_max_iter,
            "refit_r2_min_complete_rows": self.refit_r2_min_complete_rows,
            "refit_r2_cv_folds": self.refit_r2_cv_folds,
            "bimodal_grouping_variables": dict(self._bimodal_grouping_variables),
            "bimodal_min_correlated_features": self.bimodal_min_correlated_features,
            "bimodal_correlation_threshold": self.bimodal_correlation_threshold,
            "max_workers": self.max_workers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NumericImputationConfig:
        """
        Construct a ``NumericImputationConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        NumericImputationConfig
            Reconstructed config instance.
        """
        config = cls(
            knn_max_rows=int(data.get("knn_max_rows", 50_000)),
            knn_max_features=int(data.get("knn_max_features", 50)),
            mice_min_rows=int(data.get("mice_min_rows", 500)),
            gradient_boost_min_rows=int(data.get("gradient_boost_min_rows", 10_000)),
            base_max_iter=int(data.get("base_max_iter", 10)),
            knn_min_neighbors=int(data.get("knn_min_neighbors", 5)),
            knn_max_neighbors=int(data.get("knn_max_neighbors", 25)),
            knn_distance_weight_max_null_ratio=float(
                data.get("knn_distance_weight_max_null_ratio", 0.15)
            ),
            knn_distance_weight_max_features=int(
                data.get("knn_distance_weight_max_features", 30)
            ),
            mice_n_nearest_features_min_cols=int(
                data.get("mice_n_nearest_features_min_cols", 10)
            ),
            mice_max_nearest_features=int(data.get("mice_max_nearest_features", 20)),
            mice_correlation_threshold=float(
                data.get("mice_correlation_threshold", 0.1)
            ),
            mcar_feature_predictability_threshold=float(
                data.get("mcar_feature_predictability_threshold", 0.2)
            ),
            _per_column_strategy={},
            _per_column_constant_fill={},
            knn_n_neighbors=(
                int(data["knn_n_neighbors"])
                if data.get("knn_n_neighbors") is not None
                else None
            ),
            mice_max_iter=(
                int(data["mice_max_iter"])
                if data.get("mice_max_iter") is not None
                else None
            ),
            refit_r2_min_complete_rows=int(data.get("refit_r2_min_complete_rows", 50)),
            refit_r2_cv_folds=int(data.get("refit_r2_cv_folds", 5)),
            _bimodal_grouping_variables={},
            bimodal_min_correlated_features=int(
                data.get("bimodal_min_correlated_features", 3)
            ),
            bimodal_correlation_threshold=float(
                data.get("bimodal_correlation_threshold", 0.2)
            ),
            max_workers=(
                int(data["max_workers"])
                if data.get("max_workers") is not None
                else None
            ),
        )

        for col, val in data.get("per_column_constant_fill", {}).items():
            config.set_per_column_constant_fill(col, float(val))
        for col, val in data.get("per_column_strategy", {}).items():
            config.set_per_column_strategy(col, ImputationStrategy(val))
        for col, val in data.get("bimodal_grouping_variables", {}).items():
            config.set_bimodal_grouping_variable(col, str(val))

        return config


@dataclass
class ImputationConfig:
    """
    Cross-type Phase 2 configuration.

    Parameters
    ----------
    numeric : NumericImputationConfig
        Thresholds and fill values for numeric imputation.
    mnar_columns : list[str]
        Columns declared by the user as Missing Not At Random.
        These receive a data-derived fill (observed mean or median, skew-driven)
        plus a binary missingness indicator, regardless of Phase 1 signals.
    add_indicator_columns : list[str]
        Columns for which a binary missingness indicator should be added
        even when they are not MNAR.

    Raises
    ------
    ValueError
        If any column appears in both ``mnar_columns`` and
        ``numeric.per_column_strategy``.  These declarations are mutually
        exclusive: ``mnar_columns`` applies a data-derived fill plus an
        indicator; ``per_column_strategy`` directs the routing engine to a
        user-specified strategy.  Declaring the same column in both is
        contradictory and is caught at construction time before any data is
        touched.
    """

    numeric: NumericImputationConfig = field(default_factory=NumericImputationConfig)
    _mnar_columns: list[str] = field(default_factory=list, init=False)
    _add_indicator_columns: list[str] = field(default_factory=list, init=False)

    @property
    def mnar_columns(self) -> tuple[str, ...]:
        """
        Get the columns declared as Missing Not At Random.

        Returns
        -------
        tuple[str, ...]
            Columns declared by the user as MNAR.
        """
        return tuple(self._mnar_columns)

    @property
    def add_indicator_columns(self) -> tuple[str, ...]:
        """
        Get the columns for which a binary missingness indicator should be added.

        Returns
        -------
        tuple[str, ...]
            Columns for which a binary missingness indicator is forced.
        """
        return tuple(self._add_indicator_columns)

    def add_mnar_column(self, column: str | list[str]) -> None:
        """
        Declare one or more columns as Missing Not At Random.

        Parameters
        ----------
        column : str | list[str]
            Column name or list of column names to mark as MNAR.

        Raises
        ------
        ValueError
            If any specified column already has a strategy in ``numeric.per_column_strategy``.
        """
        if isinstance(column, str):
            column = [column]

        conflicts = sorted(set(column) & set(self.numeric.per_column_strategy.keys()))
        if conflicts:
            names = ", ".join(f"'{c}'" for c in conflicts)
            raise ValueError(
                f"Columns appear in both mnar_columns and numeric.per_column_strategy, "
                f"which are mutually exclusive: {names}. "
                f"Use mnar_columns for MNAR semantics (data-derived fill + indicator) "
                f"or per_column_strategy for a user-specified strategy, not both."
            )

        for c in column:
            if c not in self._mnar_columns:
                self._mnar_columns.append(c)

    def add_indicator_column(self, column: str | list[str]) -> None:
        """
        Force a binary missingness indicator for one or more columns.

        Parameters
        ----------
        column : str | list[str]
            Column name or list of column names.
        """
        if isinstance(column, str):
            column = [column]
        for c in column:
            if c not in self._add_indicator_columns:
                self._add_indicator_columns.append(c)

    def validate(self) -> None:
        """
        Validate the configuration for cross-field conflicts.

        Raises
        ------
        ValueError
            If any column appears in both ``mnar_columns`` and
            ``numeric.per_column_strategy``.
        """
        conflicts = sorted(
            set(self._mnar_columns) & set(self.numeric.per_column_strategy.keys())
        )
        if conflicts:
            names = ", ".join(f"'{c}'" for c in conflicts)
            raise ValueError(
                f"Columns appear in both mnar_columns and numeric.per_column_strategy, "
                f"which are mutually exclusive: {names}. "
                f"Use mnar_columns for MNAR semantics (data-derived fill + indicator) "
                f"or per_column_strategy for a user-specified strategy, not both."
            )

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name, with ``numeric`` nested.
        """
        return {
            "numeric": self.numeric.to_dict(),
            "mnar_columns": list(self._mnar_columns),
            "add_indicator_columns": list(self._add_indicator_columns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImputationConfig:
        """
        Construct an ``ImputationConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        ImputationConfig
            Reconstructed config instance.
        """
        config = cls(
            numeric=NumericImputationConfig.from_dict(data.get("numeric", {})),
        )
        if "mnar_columns" in data:
            config.add_mnar_column(data["mnar_columns"])
        if "add_indicator_columns" in data:
            config.add_indicator_column(data["add_indicator_columns"])
        return config


@dataclass
class InspectionDiagnostic:
    """Retrain-free inspection diagnostic for a single model-based column.

    Produced by :meth:`EvaluationOrchestrator.inspect` — the cheap, retrain-free
    check that reuses the models ``fit()`` already learned (ADR-0058).  It
    answers "do the imputed values look sensible?" by comparing the values the
    fitted model filled into the originally-null cells against the observed
    (non-null) values, and by surfacing the fitted model's own metadata.  It
    carries no held-out accuracy numbers — those live on
    :class:`AccuracyDiagnostic`, which is irreducibly a refit.

    Present for KNN, MICE, and the bimodal strategies
    (Cluster-Conditional, GMM-Sampling); absent (no report entry) for
    Passthrough, Dropped, Constant, MNAR, and the scalar strategies
    (Mean, Median, Mode).

    Parameters
    ----------
    imputed_mean : float
        Mean of the values the fitted model filled into the originally-null
        rows.  ``0.0`` when the inspected frame has no nulls in this column.
    imputed_std : float
        Standard deviation of those imputed values.  ``0.0`` when the inspected
        frame has no nulls in this column.
    observed_mean : float
        Mean of the non-null values in this column.
    observed_std : float
        Standard deviation of the non-null values in this column.
    variance_ratio : float
        ``imputed_std / observed_std`` (``0.0`` when ``observed_std`` is zero).
        A value near zero flags distribution collapse — the model is predicting
        near-constant fills.
    converged : bool, optional
        Whether ``IterativeImputer`` halted before reaching ``max_iter``.  Read
        from the fitted MICE model.  ``None`` for KNN and the
        bimodal strategies (convergence is not applicable).
    n_iter : int, optional
        Actual iteration count of the fitted ``IterativeImputer``.  ``None`` for
        KNN and the bimodal strategies.
    n_neighbors_used : int, optional
        Actual ``n_neighbors`` used by the fitted KNN block.  ``None`` for
        MICE and the bimodal strategies.
    k_capped : bool, optional
        ``True`` when the KNN neighbour count was forced down to ``n_rows − 1``
        (the model is averaging nearly every row).  ``None`` when a
        ``knn_n_neighbors`` override is active or the strategy is not KNN.
    """

    imputed_mean: float
    imputed_std: float
    observed_mean: float
    observed_std: float
    variance_ratio: float
    converged: Optional[bool] = None
    n_iter: Optional[int] = None
    n_neighbors_used: Optional[int] = None
    k_capped: Optional[bool] = None

    def to_dict(self) -> dict:
        """Serialise the diagnostic to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "imputed_mean": self.imputed_mean,
            "imputed_std": self.imputed_std,
            "observed_mean": self.observed_mean,
            "observed_std": self.observed_std,
            "variance_ratio": self.variance_ratio,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "n_neighbors_used": self.n_neighbors_used,
            "k_capped": self.k_capped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionDiagnostic":
        """Reconstruct an ``InspectionDiagnostic`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        InspectionDiagnostic
            Reconstructed diagnostic instance.
        """
        return cls(
            imputed_mean=float(data["imputed_mean"]),
            imputed_std=float(data["imputed_std"]),
            observed_mean=float(data["observed_mean"]),
            observed_std=float(data["observed_std"]),
            variance_ratio=float(data["variance_ratio"]),
            converged=data.get("converged"),
            n_iter=data.get("n_iter"),
            n_neighbors_used=data.get("n_neighbors_used"),
            k_capped=data.get("k_capped"),
        )


@dataclass
class InspectionReport:
    """Per-column inspection report returned by :meth:`EvaluationOrchestrator.inspect`.

    Holds one :class:`InspectionDiagnostic` per model-based column (KNN,
    MICE, and the bimodal strategies); columns handled by scalar
    strategies, Passthrough, Dropped, Constant, or MNAR carry no entry.
    Supports ``report[col]`` lookup and ``col in report`` membership tests
    (ADR-0058).

    Parameters
    ----------
    columns : dict[str, InspectionDiagnostic]
        Mapping from column name to its inspection diagnostic.
    """

    columns: dict[str, InspectionDiagnostic] = field(default_factory=dict)

    def __getitem__(self, column: str) -> InspectionDiagnostic:
        """Return the diagnostic for ``column``.

        Parameters
        ----------
        column : str
            Column name to look up.

        Returns
        -------
        InspectionDiagnostic
            The inspection diagnostic for ``column``.

        Raises
        ------
        KeyError
            If ``column`` has no diagnostic in this report.
        """
        return self.columns[column]

    def __contains__(self, column: object) -> bool:
        """Return whether ``column`` has a diagnostic in this report.

        Parameters
        ----------
        column : object
            Column name to test for membership.

        Returns
        -------
        bool
            ``True`` when a diagnostic is present for ``column``.
        """
        return column in self.columns

    def to_dict(self) -> dict:
        """Serialise the report to a plain dictionary.

        Returns
        -------
        dict
            Mapping with a single ``"columns"`` key whose value maps each
            column name to its serialised diagnostic.
        """
        return {"columns": {col: diag.to_dict() for col, diag in self.columns.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionReport":
        """Reconstruct an ``InspectionReport`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        InspectionReport
            Reconstructed report instance.
        """
        return cls(
            columns={
                col: InspectionDiagnostic.from_dict(raw)
                for col, raw in data.get("columns", {}).items()
            }
        )


@dataclass
class AccuracyDiagnostic:
    """Held-out accuracy diagnostic for a single model-based column.

    Produced by :meth:`EvaluationOrchestrator.score_accuracy` — the expensive,
    refit-based check (ADR-0058).  Honest held-out accuracy is *irreducibly* a
    refit: the model learned at fit time has already seen every cell, so scoring
    it in-sample is optimistically biased.  These numbers therefore come from
    cross-validating the column's recorded strategy on folds of the complete
    rows, never from the final fitted model.

    Present for KNN, MICE, and the bimodal strategies
    (Cluster-Conditional, GMM-Sampling); absent (no report entry) for
    Passthrough, Dropped, Constant, MNAR, and the scalar strategies
    (Mean, Median, Mode).

    Parameters
    ----------
    r2_cv : float, optional
        Mean R² across k cross-validation folds on complete rows (k =
        ``refit_r2_cv_folds``).  Named ``r2_cv`` — not ``r2_train`` — because it
        is always a held-out score, never an in-sample one.  ``None`` when fewer
        than ``refit_r2_min_complete_rows`` complete rows are available, when all
        folds are skipped due to zero variance in ``y_true``, or when the
        strategy has no held-out truth to score (GMM-Sampling and the
        grouping-variable Cluster-Conditional branch).
    rmse : float, optional
        Root-mean-squared error across the same held-out folds, in the column's
        own units.  ``None`` under the same conditions as ``r2_cv``.
    mae : float, optional
        Mean absolute error across the same held-out folds, in the column's own
        units.  ``None`` under the same conditions as ``r2_cv``.
    """

    r2_cv: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]

    def to_dict(self) -> dict:
        """Serialise the diagnostic to a plain dictionary.

        Returns
        -------
        dict
            All three field values keyed by field name.
        """
        return {
            "r2_cv": self.r2_cv,
            "rmse": self.rmse,
            "mae": self.mae,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccuracyDiagnostic":
        """Reconstruct an ``AccuracyDiagnostic`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        AccuracyDiagnostic
            Reconstructed diagnostic instance.
        """
        return cls(
            r2_cv=data.get("r2_cv"),
            rmse=data.get("rmse"),
            mae=data.get("mae"),
        )


@dataclass
class AccuracyReport:
    """Per-column accuracy report returned by :meth:`EvaluationOrchestrator.score_accuracy`.

    Holds one :class:`AccuracyDiagnostic` per model-based column (KNN,
    MICE, and the bimodal strategies); columns handled by scalar
    strategies, Passthrough, Dropped, Constant, or MNAR carry no entry.
    Supports ``report[col]`` lookup and ``col in report`` membership tests
    (ADR-0058).

    Parameters
    ----------
    columns : dict[str, AccuracyDiagnostic]
        Mapping from column name to its held-out accuracy diagnostic.
    """

    columns: dict[str, AccuracyDiagnostic] = field(default_factory=dict)

    def __getitem__(self, column: str) -> AccuracyDiagnostic:
        """Return the diagnostic for ``column``.

        Parameters
        ----------
        column : str
            Column name to look up.

        Returns
        -------
        AccuracyDiagnostic
            The held-out accuracy diagnostic for ``column``.

        Raises
        ------
        KeyError
            If ``column`` has no diagnostic in this report.
        """
        return self.columns[column]

    def __contains__(self, column: object) -> bool:
        """Return whether ``column`` has a diagnostic in this report.

        Parameters
        ----------
        column : object
            Column name to test for membership.

        Returns
        -------
        bool
            ``True`` when a diagnostic is present for ``column``.
        """
        return column in self.columns

    def to_dict(self) -> dict:
        """Serialise the report to a plain dictionary.

        Returns
        -------
        dict
            Mapping with a single ``"columns"`` key whose value maps each
            column name to its serialised diagnostic.
        """
        return {"columns": {col: diag.to_dict() for col, diag in self.columns.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "AccuracyReport":
        """Reconstruct an ``AccuracyReport`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        AccuracyReport
            Reconstructed report instance.
        """
        return cls(
            columns={
                col: AccuracyDiagnostic.from_dict(raw)
                for col, raw in data.get("columns", {}).items()
            }
        )


@dataclass(frozen=True)
class ColumnImputationDecision:
    """Pure, value-free per-column imputation plan entry.

    The decided half of the Decision/Execution split (ADR-0060): a complete,
    inspectable description of *what will happen* to one column and *why*,
    derived purely from ``(profile, shape, config)``. It structurally cannot
    hold anything the execution layer learns from training data — no fill value,
    no fitted model, no convergence/``n_iter`` — so "what was decided" is
    un-blurrably separate from "what was learned". ``ColumnImputationRecord``
    composes this decision with those learned values.

    The dataclass is frozen and every field holds an immutable value (enums,
    ``str``, ``bool``, or hashable tuples), so a decision is safe to share and
    edits produce new objects rather than mutating in place.

    Parameters
    ----------
    column : str
        Column name this decision applies to.
    semantic_type : SemanticType
        Detected semantic type of the column, carried from the profile.
    strategy : ImputationStrategy
        Strategy routed for this column.
    signals : tuple[str, ...], optional
        Human-readable routing rationale — the reasons that drove ``strategy``.
        A tuple so the decision stays immutable.
    model_choice : ModelChoice, optional
        Concrete estimator family resolved at decide-time for model-based
        strategies. ``None`` for scalar strategies and for columns whose
        nonlinearity is ``Unpredictable``. A value-free label, never a live or
        fitted estimator.
    domain_snap_bounds : tuple[float, float], optional
        ``(min, max)`` bounds used to snap model-based predictions for
        BoundedDiscrete columns. ``None`` for all other columns. Sourced from
        the profile, not learned from training data.
    indicator_flag : bool
        Whether a binary missingness indicator column will be appended.
    mnar : bool
        Whether the column is routed as Missing-Not-At-Random.
    drop : bool
        Whether the column will be dropped for exceeding the drop threshold.
    forced : bool
        Whether ``strategy`` was declared by the user rather than routed
        automatically — set both by the router (from ``per_column_strategy``)
        and by :meth:`ImputationDecision.with_strategy`. This is the machine
        predicate for forced-ness (ADR-0066): a forced strategy that cannot
        train raises instead of degrading, and because the fact lives on the
        plan, a plan loaded from a store answers "was this forced?" without the
        originating config. The ``per_column_strategy_override`` signal remains
        for human readers only and is never parsed.

    Notes
    -----
    Fill values, fitted coefficients, convergence counts, and any other
    training-derived state deliberately have no home on this type; they live on
    the execution layer's fitted units (ADR-0060).
    """

    column: str
    semantic_type: SemanticType
    strategy: ImputationStrategy
    signals: tuple[str, ...] = ()
    model_choice: Optional[ModelChoice] = None
    domain_snap_bounds: Optional[tuple[float, float]] = None
    indicator_flag: bool = False
    mnar: bool = False
    drop: bool = False
    forced: bool = False

    def to_dict(self) -> dict:
        """Serialise the decision to a plain dictionary.

        Enums are rendered by member *name* (per ADR-0063, so reordering an
        enum later cannot silently corrupt a saved plan) and tuples as lists,
        so the result is JSON-friendly and structurally round-trippable.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "column": self.column,
            "semantic_type": self.semantic_type.name,
            "strategy": self.strategy.name,
            "signals": list(self.signals),
            "model_choice": (
                self.model_choice.name if self.model_choice is not None else None
            ),
            "domain_snap_bounds": (
                list(self.domain_snap_bounds)
                if self.domain_snap_bounds is not None
                else None
            ),
            "indicator_flag": self.indicator_flag,
            "mnar": self.mnar,
            "drop": self.drop,
            "forced": self.forced,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnImputationDecision":
        """Reconstruct a ``ColumnImputationDecision`` from a plain dictionary.

        The inverse of :meth:`to_dict`: enums are rebuilt by member name and
        list fields are restored to their immutable tuple form so the
        reconstructed decision is structurally equal to the original.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ColumnImputationDecision
            Reconstructed decision instance.
        """
        raw_bounds = data.get("domain_snap_bounds")
        raw_model_choice = data.get("model_choice")
        return cls(
            column=data["column"],
            semantic_type=SemanticType[data["semantic_type"]],
            strategy=ImputationStrategy[data["strategy"]],
            signals=tuple(data.get("signals", ())),
            model_choice=(
                ModelChoice[raw_model_choice] if raw_model_choice is not None else None
            ),
            domain_snap_bounds=(tuple(raw_bounds) if raw_bounds is not None else None),
            indicator_flag=bool(data.get("indicator_flag", False)),
            mnar=bool(data.get("mnar", False)),
            drop=bool(data.get("drop", False)),
            forced=bool(data.get("forced", False)),
        )


@dataclass
class ColumnImputationRecord:
    """
    Per-column audit entry produced after fit().

    Composes the pure, value-free :class:`ColumnImputationDecision` (*what was
    decided*, reachable via ``record.decision``) with the values ``fit()``
    learned from the training data (*what was learned*): the scalar
    ``fill_value`` and the ``indicator_added`` fit-metadata flag (ADR-0060).
    Plan fields — ``column``, ``strategy``, ``signals``, ``domain_snap_bounds``
    and the rest — are read through ``record.decision.*``.

    Parameters
    ----------
    decision : ColumnImputationDecision
        The decided, value-free per-column plan entry.
    fill_value : Any, optional
        Scalar fill value learned from training data (None for model-based
        strategies).
    indicator_added : bool
        Whether a binary missingness indicator column was appended — a
        fit-time fact distinct from the decided ``decision.indicator_flag``.

    Notes
    -----
    Fit-quality metrics are no longer carried here.  ``fit()`` only learns
    fill values and models; quality measurement is a deliberate second step
    via the opt-in Evaluation phase, which returns an
    :class:`InspectionReport` or :class:`AccuracyReport` keyed by column
    (ADR-0058).
    """

    decision: ColumnImputationDecision
    fill_value: Optional[Any] = None
    indicator_added: bool = False

    def to_dict(self) -> dict:
        """
        Serialise the audit record to a plain dictionary.

        Flattens the decision's serialised fields alongside the learned
        ``fill_value`` and ``indicator_added`` so the whole record round-trips
        through :meth:`from_dict`.

        Returns
        -------
        dict
            The decision's fields plus ``fill_value`` and ``indicator_added``.
        """
        return {
            **self.decision.to_dict(),
            "fill_value": self.fill_value,
            "indicator_added": self.indicator_added,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnImputationRecord":
        """
        Reconstruct a ``ColumnImputationRecord`` from a plain dictionary.

        The inverse of :meth:`to_dict`: the decision is rebuilt from the same
        flat mapping, then paired with the learned ``fill_value`` and
        ``indicator_added``.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ColumnImputationRecord
            Reconstructed audit record.
        """
        return cls(
            decision=ColumnImputationDecision.from_dict(data),
            fill_value=data.get("fill_value"),
            indicator_added=bool(data.get("indicator_added", False)),
        )


# ---------------------------------------------------------------------------
# ImputationDecision — the pure, immutable plan (ADR-0060, issue #345)
# ---------------------------------------------------------------------------

# Strategies whose execution is one joint block over several columns; each
# collapses to a single unit keyed by the block id rather than one unit per
# column.
_JOINT_BLOCK_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {ImputationStrategy.MICE, ImputationStrategy.KNN}
)

# Structural strategies carry no learned value and train nothing, so the plan
# projects their records straight through without materialising a unit.
_STRUCTURAL_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.Dropped,
        ImputationStrategy.Passthrough,
        ImputationStrategy.Indicator,
    }
)


def _validate_declarable_strategy(
    column: str,
    semantic_type: SemanticType,
    strategy: ImputationStrategy,
) -> None:
    """Validate that ``strategy`` may be declared on ``column`` when editing a plan.

    Legality only — data-size feasibility (size guards) is deferred to execution
    (ADR-0060). Output-only labels are rejected with the exact redirect the
    config surfaces for ``per_column_strategy``; ``Constant`` redirects to
    ``per_column_constant_fill``; a strategy the column's semantic type has no
    imputer for is rejected as non-declarable.
    """
    if strategy in _OUTPUT_ONLY_STRATEGIES:
        raise ValueError(_output_only_redirect(column, strategy))
    if strategy == ImputationStrategy.Constant:
        raise ValueError(_constant_without_fill_redirect(column))
    declarable = _DECLARABLE_STRATEGIES_BY_TYPE.get(semantic_type, frozenset())
    if strategy not in declarable:
        raise ValueError(
            f"Column '{column}': '{strategy}' is not a declarable strategy for a "
            f"'{semantic_type}' column."
        )


def _hyperparameters_from_dict(
    hyperparameters: "dict[str, Any]",
) -> "tuple[tuple[str, Any], ...]":
    """Restore a hyperparameter mapping read off the wire to its in-memory form.

    JSON has no tuple, so a sequence-valued dial (e.g. a Cluster-Conditional
    unit's ``feature_cols``) arrives as a list. Restoring it to a tuple is what
    keeps ``load(save(plan)) == plan`` (ADR-0063) and keeps every dial hashable.
    """
    return tuple(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in hyperparameters.items()
    )


@dataclass(frozen=True)
class ImputationUnit:
    """One executable unit of an imputation plan.

    The plan's materialised, re-derived projection of *what execution trains
    together* (ADR-0060): the joint ``MICE`` block, the joint ``KNN`` block, and
    one unit per independent column (GMM-Sampling /
    Cluster-Conditional / scalar). Structural strategies (Dropped / Passthrough /
    Indicator) train nothing and produce no unit. A unit is a value-free
    projection — it names the columns and the estimator family, never a fitted
    model; the execution layer keys its checkpoints against ``unit_id``.

    Parameters
    ----------
    unit_id : str
        Stable identifier: ``"mice"`` / ``"knn"`` for the joint blocks (exactly
        one of each when present), ``"{strategy}:{column}"`` for a per-column
        unit.
    strategy : ImputationStrategy
        Strategy the unit executes.
    columns : tuple[str, ...]
        Columns trained by this unit — every column in the block for the joint
        units, a single-element tuple for a per-column unit.

    Notes
    -----
    The unit is value-free and estimator-agnostic: the concrete estimator
    family a model-based column trains with lives on its
    :class:`ColumnImputationDecision` (``model_choice``), reachable through the
    owning plan's ``column_decisions``, so it is never duplicated here.
    """

    unit_id: str
    strategy: ImputationStrategy
    columns: tuple[str, ...]
    hyperparameters: Optional[tuple[tuple[str, Any], ...]] = None

    def to_dict(self) -> dict:
        """Serialise the unit to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name, the strategy enum rendered by
            member name and ``columns`` as a list.
        """
        return {
            "unit_id": self.unit_id,
            "strategy": self.strategy.name,
            "columns": list(self.columns),
            "hyperparameters": (
                {k: v for k, v in self.hyperparameters}
                if self.hyperparameters is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImputationUnit":
        """Reconstruct an ``ImputationUnit`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationUnit
            Reconstructed unit instance.
        """
        return cls(
            unit_id=data["unit_id"],
            strategy=ImputationStrategy[data["strategy"]],
            columns=tuple(data.get("columns", ())),
            hyperparameters=(
                _hyperparameters_from_dict(data["hyperparameters"])
                if data.get("hyperparameters") is not None
                else None
            ),
        )


def _merge_unit_hyperparameters(
    decided_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]",
    override_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]",
    unit_id: str,
) -> "Optional[tuple[tuple[str, Any], ...]]":
    """Stamp ``merged = decided ⊕ delta`` for one unit (delta wins per key).

    The decided base is complete for the unit's strategy and the override delta
    is sparse (ADR-0073); merging them per key is what lets a fitter read one
    complete dict while the two-map split stays invisible below the plan surface.
    Returns ``None`` when neither map carries the unit — an empty hyperparameter
    set, matching a unit no strategy resolved dials for.
    """
    base = dict(decided_hyperparameters.get(unit_id, ()))
    base.update(dict(override_hyperparameters.get(unit_id, ())))
    return tuple(base.items()) if base else None


def _derive_units(
    column_decisions: "dict[str, ColumnImputationDecision]",
    decided_hyperparameters: "Optional[dict[str, tuple[tuple[str, Any], ...]]]" = None,
    override_hyperparameters: "Optional[dict[str, tuple[tuple[str, Any], ...]]]" = None,
) -> tuple[ImputationUnit, ...]:
    """Project the per-column decision map into its execution units.

    Called at every :class:`ImputationDecision` construction so the unit list
    can never drift from the map it describes (ADR-0060). Each unit is stamped
    with the per-key merge of the decided base and the override delta (ADR-0073).
    """
    mice_cols = tuple(
        c for c, d in column_decisions.items() if d.strategy == ImputationStrategy.MICE
    )
    knn_cols = tuple(
        c for c, d in column_decisions.items() if d.strategy == ImputationStrategy.KNN
    )

    decided_hyperparameters = decided_hyperparameters or {}
    override_hyperparameters = override_hyperparameters or {}
    units: list[ImputationUnit] = []
    mice_emitted = False
    knn_emitted = False
    for column, decision in column_decisions.items():
        strategy = decision.strategy
        if strategy == ImputationStrategy.MICE:
            if not mice_emitted:
                units.append(
                    ImputationUnit(
                        unit_id="mice",
                        strategy=ImputationStrategy.MICE,
                        columns=mice_cols,
                        hyperparameters=_merge_unit_hyperparameters(
                            decided_hyperparameters, override_hyperparameters, "mice"
                        ),
                    )
                )
                mice_emitted = True
        elif strategy == ImputationStrategy.KNN:
            if not knn_emitted:
                units.append(
                    ImputationUnit(
                        unit_id="knn",
                        strategy=ImputationStrategy.KNN,
                        columns=knn_cols,
                        hyperparameters=_merge_unit_hyperparameters(
                            decided_hyperparameters, override_hyperparameters, "knn"
                        ),
                    )
                )
                knn_emitted = True
        elif strategy in _STRUCTURAL_STRATEGIES:
            continue
        else:
            unit_id = f"{strategy}:{column}"
            units.append(
                ImputationUnit(
                    unit_id=unit_id,
                    strategy=strategy,
                    columns=(column,),
                    hyperparameters=_merge_unit_hyperparameters(
                        decided_hyperparameters, override_hyperparameters, unit_id
                    ),
                )
            )
    return tuple(units)


@dataclass(frozen=True)
class ImputationDecision:
    """The pure, immutable imputation plan produced by :func:`decide`.

    The keystone of the Decision/Execution split (ADR-0060): a complete,
    inspectable, editable description of *what will happen* to every column,
    derived purely from ``(profile, shape, config)`` and holding no value the
    execution layer learns from training data. The execution layer consumes it,
    persistence serialises it, and the user can inspect and edit it before
    anything trains.

    The plan is immutable: :meth:`with_strategy` and :meth:`with_model_choice`
    return a *new* ``ImputationDecision`` rather than mutating in place, and
    ``units`` is re-derived from ``column_decisions`` at every construction, so a
    stale unit list is structurally impossible and every plan that exists is
    valid by construction.

    Parameters
    ----------
    column_decisions : dict[str, ColumnImputationDecision]
        Per-column plan entries keyed by column name, in decision order. Held as
        an internal copy so the constructed plan is independent of the caller's
        mapping.
    decided_for_shape : tuple[int, int, tuple[str, ...]]
        The ``(n_rows, n_features, column set)`` shape the plan was decided for;
        the imputable-column population and size the routing depended on.
        ``n_rows`` is the row count passed to :func:`decide` — the train split's
        — so the plan is valid only for a split of that size (ADR-0066). It may
        legitimately differ from ``profile_provenance["row_count"]``.
    config_snapshot : dict
        Serialised :class:`~dataforge_ml.PipelineConfig` (``config.to_dict()``)
        the plan was decided under.
    profile_provenance : dict
        Descriptive, non-load-bearing provenance of the source profile — its
        ``row_count`` — carried for diagnosis only. ``row_count`` is the
        *full-dataset* count, since provenance describes the profile rather than
        the shape the plan was decided for (ADR-0066). Content-addressed identity
        was retired with cross-process resume (ADR-0072).
    decided_hyperparameters : dict[str, tuple[tuple[str, Any], ...]]
        The decide-time hyperparameter base per unit id, complete for each unit's
        strategy and written only by :func:`decide` (ADR-0073).
    override_hyperparameters : dict[str, tuple[tuple[str, Any], ...]]
        The sparse per-unit override delta, written only by
        :meth:`with_hyperparameters`. ``_derive_units`` stamps the per-key merge
        ``decided ⊕ delta`` onto each unit, so the two-map split is invisible
        below the plan surface (ADR-0073).
    numeric_sentinels : dict[str, list[float]]
        Declared numeric sentinel values per column, carried from the source
        profile so the execution layer can normalise effective nulls without it
        (ADR-0068).
    string_sentinels : dict[str, list[str]]
        Declared string sentinel values per column, carried from the source
        profile for the same reason as ``numeric_sentinels`` (ADR-0068).
    units : tuple[ImputationUnit, ...]
        Materialised execution units, re-derived from ``column_decisions`` at
        construction. Not an init argument.
    dropped_columns : tuple[str, ...]
        Convenience projection of the columns routed to ``Dropped``. Not an init
        argument.

    Notes
    -----
    ``column_decisions`` is the single source of truth; ``units`` and
    ``dropped_columns`` are always projections of it and are never set directly.
    """

    column_decisions: "dict[str, ColumnImputationDecision]"
    decided_for_shape: tuple
    config_snapshot: dict
    profile_provenance: dict
    decided_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]" = field(
        default_factory=dict
    )
    override_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]" = field(
        default_factory=dict
    )
    numeric_sentinels: "dict[str, list[float]]" = field(default_factory=dict)
    string_sentinels: "dict[str, list[str]]" = field(default_factory=dict)
    units: tuple = field(init=False, default=())
    dropped_columns: tuple = field(init=False, default=())

    def __post_init__(self) -> None:
        decisions = dict(self.column_decisions)
        decided_hyp = dict(self.decided_hyperparameters)
        override_hyp = dict(self.override_hyperparameters)
        object.__setattr__(self, "column_decisions", decisions)
        object.__setattr__(self, "decided_hyperparameters", decided_hyp)
        object.__setattr__(self, "override_hyperparameters", override_hyp)
        object.__setattr__(
            self,
            "numeric_sentinels",
            {k: list(v) for k, v in self.numeric_sentinels.items()},
        )
        object.__setattr__(
            self,
            "string_sentinels",
            {k: list(v) for k, v in self.string_sentinels.items()},
        )
        object.__setattr__(
            self, "units", _derive_units(decisions, decided_hyp, override_hyp)
        )
        object.__setattr__(
            self,
            "dropped_columns",
            tuple(c for c, d in decisions.items() if d.drop),
        )

    def with_strategy(
        self, column: str, strategy: "str | ImputationStrategy"
    ) -> "ImputationDecision":
        """Return a new plan with ``column`` routed to ``strategy``.

        The plan is immutable; this builds a fresh :class:`ImputationDecision`
        with the one column's strategy replaced and every unit re-derived, so the
        edit is visible in ``units`` while the original plan is untouched. Only
        edit-time *legality* is validated: the strategy must be declarable for the
        column's semantic type, and the output-only labels ``Dropped`` /
        ``Passthrough`` / ``Indicator`` / ``MNAR`` / ``Constant`` /
        ``ClusterConditional`` / ``GMMSampling`` are rejected with the same
        redirect messages ``NumericImputationConfig.set_per_column_strategy``
        surfaces. Data-size feasibility (size guards) is deferred to execution
        (ADR-0060). Because the new strategy may imply a different estimator
        family, ``model_choice`` is reset to ``None``; use
        :meth:`with_model_choice` to set it.

        Editing a column's strategy *is* forcing it, so the new decision carries
        ``forced=True`` (ADR-0066) and execution will raise rather than degrade
        should the strategy fail to train — even when the edit happens to name
        the strategy the router chose anyway.

        Parameters
        ----------
        column : str
            Column to re-route. Must already be present in the plan.
        strategy : str or ImputationStrategy
            The replacement strategy.

        Returns
        -------
        ImputationDecision
            A new plan with the edit applied.

        Raises
        ------
        KeyError
            If ``column`` is not part of the plan.
        ValueError
            If ``strategy`` is an output-only label, or is not declarable for the
            column's semantic type.
        """
        if column not in self.column_decisions:
            raise KeyError(f"Column '{column}' is not part of this plan.")
        strategy = ImputationStrategy(strategy)
        _validate_declarable_strategy(
            column, self.column_decisions[column].semantic_type, strategy
        )
        from dataclasses import replace

        new_decisions = dict(self.column_decisions)
        new_decisions[column] = replace(
            new_decisions[column],
            strategy=strategy,
            model_choice=None,
            forced=True,
            signals=new_decisions[column].signals
            + (f"per_column_strategy_override: user forced strategy={strategy}",),
        )
        return ImputationDecision(
            column_decisions=new_decisions,
            decided_for_shape=self.decided_for_shape,
            config_snapshot=self.config_snapshot,
            profile_provenance=self.profile_provenance,
            decided_hyperparameters=self.decided_hyperparameters,
            override_hyperparameters=self.override_hyperparameters,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def with_model_choice(
        self, column: str, model_choice: "Optional[str | ModelChoice]"
    ) -> "ImputationDecision":
        """Return a new plan overriding ``column``'s estimator family.

        The plan is immutable; this builds a fresh :class:`ImputationDecision`
        with the one column's ``model_choice`` replaced and every unit
        re-derived so the change is visible on the affected unit.

        Parameters
        ----------
        column : str
            Column whose estimator family to override. Must already be present
            in the plan.
        model_choice : str or ModelChoice or None
            The replacement estimator family, or ``None`` to clear it.

        Returns
        -------
        ImputationDecision
            A new plan with the edit applied.

        Raises
        ------
        KeyError
            If ``column`` is not part of the plan.
        """
        if column not in self.column_decisions:
            raise KeyError(f"Column '{column}' is not part of this plan.")
        resolved = ModelChoice(model_choice) if model_choice is not None else None
        from dataclasses import replace

        new_decisions = dict(self.column_decisions)
        new_decisions[column] = replace(new_decisions[column], model_choice=resolved)
        return ImputationDecision(
            column_decisions=new_decisions,
            decided_for_shape=self.decided_for_shape,
            config_snapshot=self.config_snapshot,
            profile_provenance=self.profile_provenance,
            decided_hyperparameters=self.decided_hyperparameters,
            override_hyperparameters=self.override_hyperparameters,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def with_hyperparameters(
        self, unit_id: str, hyperparameters: "Optional[dict[str, Any]]"
    ) -> "ImputationDecision":
        """Return a new plan overriding named hyperparameters on one unit.

        A per-key merge onto the unit's always-complete decided base (ADR-0073):
        the named keys are overridden and every other decided dial is kept, so
        the plan the user inspects is exactly the plan that fits. The override is
        stored as a sparse delta; ``units`` re-derives the merge at construction.
        Successive edits to the same unit accumulate per key. Passing ``None``
        resets the unit to its decided values by clearing its delta. Per-unit
        only: a joint unit shares one estimator, so per-column overrides on it
        would be meaningless.

        Every key must already exist in the unit's decided base — the base *is*
        the override schema (there is no separate allowlist), so a key the
        strategy did not resolve is an unknown key and is rejected here. Values
        are not type-checked: sklearn rejects an ill-typed value at fit time.

        Parameters
        ----------
        unit_id : str
            ID of the unit to override (e.g. ``"mice"``, ``"knn"``, or
            ``"median:age"``). Must already be present in the plan.
        hyperparameters : dict[str, Any] or None
            The keys to override and their replacement values, or ``None`` to
            reset the unit to its decided base.

        Returns
        -------
        ImputationDecision
            A new plan with the override applied.

        Raises
        ------
        KeyError
            If ``unit_id`` is not part of the plan's derived units.
        ValueError
            If ``hyperparameters`` names a key the unit's decided base does not
            carry, identifying the unit and the offending key.
        """
        if not any(u.unit_id == unit_id for u in self.units):
            raise KeyError(f"Unit '{unit_id}' is not part of this plan.")

        new_override_hyperparameters = dict(self.override_hyperparameters)
        if hyperparameters is None:
            new_override_hyperparameters.pop(unit_id, None)
        else:
            decided_keys = {k for k, _ in self.decided_hyperparameters.get(unit_id, ())}
            for key in hyperparameters:
                if key not in decided_keys:
                    raise ValueError(
                        f"Unit '{unit_id}' has no decided hyperparameter '{key}' "
                        f"to override."
                    )
            merged_delta = dict(new_override_hyperparameters.get(unit_id, ()))
            merged_delta.update(hyperparameters)
            new_override_hyperparameters[unit_id] = tuple(merged_delta.items())

        return ImputationDecision(
            column_decisions=self.column_decisions,
            decided_for_shape=self.decided_for_shape,
            config_snapshot=self.config_snapshot,
            profile_provenance=self.profile_provenance,
            decided_hyperparameters=self.decided_hyperparameters,
            override_hyperparameters=new_override_hyperparameters,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def to_dict(self) -> dict:
        """Serialise the plan to a plain, JSON-friendly dictionary.

        ``units`` and ``dropped_columns`` are emitted for readability but are
        re-derived — not consumed — on :meth:`from_dict`, so a hand-edited unit
        list can never desynchronise a reloaded plan.

        Returns
        -------
        dict
            The plan's fields with nested objects serialised to dicts.
        """
        return {
            "column_decisions": {
                col: d.to_dict() for col, d in self.column_decisions.items()
            },
            "decided_for_shape": [
                self.decided_for_shape[0],
                self.decided_for_shape[1],
                list(self.decided_for_shape[2]),
            ],
            "config_snapshot": self.config_snapshot,
            "profile_provenance": self.profile_provenance,
            "decided_hyperparameters": {
                unit_id: {k: v for k, v in hyp}
                for unit_id, hyp in self.decided_hyperparameters.items()
            },
            "override_hyperparameters": {
                unit_id: {k: v for k, v in hyp}
                for unit_id, hyp in self.override_hyperparameters.items()
            },
            "numeric_sentinels": {
                col: list(v) for col, v in self.numeric_sentinels.items()
            },
            "string_sentinels": {
                col: list(v) for col, v in self.string_sentinels.items()
            },
            "units": [u.to_dict() for u in self.units],
            "dropped_columns": list(self.dropped_columns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImputationDecision":
        """Reconstruct an ``ImputationDecision`` from a plain dictionary.

        The inverse of :meth:`to_dict`. ``column_decisions`` is the single
        source of truth: ``units`` and ``dropped_columns`` are re-derived, never
        read from ``data``, so ``from_dict(plan.to_dict())`` is structurally
        equal to ``plan`` even if the serialised unit list was tampered with.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationDecision
            Reconstructed plan instance.
        """
        raw_shape = data["decided_for_shape"]
        decided_hyperparameters = {
            unit_id: _hyperparameters_from_dict(hyp)
            for unit_id, hyp in data.get("decided_hyperparameters", {}).items()
        }
        override_hyperparameters = {
            unit_id: _hyperparameters_from_dict(hyp)
            for unit_id, hyp in data.get("override_hyperparameters", {}).items()
        }
        return cls(
            column_decisions={
                col: ColumnImputationDecision.from_dict(raw)
                for col, raw in data.get("column_decisions", {}).items()
            },
            decided_for_shape=(
                raw_shape[0],
                raw_shape[1],
                tuple(raw_shape[2]),
            ),
            config_snapshot=data.get("config_snapshot", {}),
            profile_provenance=data.get("profile_provenance", {}),
            decided_hyperparameters=decided_hyperparameters,
            override_hyperparameters=override_hyperparameters,
            numeric_sentinels={
                col: list(v) for col, v in data.get("numeric_sentinels", {}).items()
            },
            string_sentinels={
                col: list(v) for col, v in data.get("string_sentinels", {}).items()
            },
        )


@dataclass
class ImputationResult:
    """
    Output of FittedImputer.transform().

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame with imputed values (and any indicator columns appended).
    records : dict[str, ColumnImputationRecord]
        Per-column audit log keyed by column name.
    dropped_columns : list[str]
        Columns removed because they exceeded the drop threshold (>50% missing).
    """

    dataframe: pl.DataFrame
    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
