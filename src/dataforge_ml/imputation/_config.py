"""
Configuration and result dataclasses for the imputation phase — Phase 2.

ImputationConfig controls strategy thresholds and MNAR declarations.
Result dataclasses carry per-column audit records and the imputed DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
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


_STRATEGY_DIALS: dict[ImputationStrategy, dict[str, Any]] = {
    ImputationStrategy.MICE: {
        "max_iter": 10,
        "tol": 1e-3,
        "initial_strategy": "mean",
        "n_nearest_features": None,
    },
    ImputationStrategy.KNN: {
        "n_neighbors": 5,
        "weights": "uniform",
    },
    ImputationStrategy.MNAR: {
        "central_tendency": "median",
    },
    ImputationStrategy.ClusterConditional: {
        "central_tendency": "median",
    },
}
"""Every dial a strategy has, and the value it takes at neutral inputs.

The single definition of the decide-time hyperparameter base, read by both
authors: :func:`~dataforge_ml.imputation._decision_assembler.decide` starts from
a row and overwrites what it computed, and the manual door writes the row as-is.
Adding a fifth dial to a strategy is therefore the same act as giving it a
default, and the two authors' key sets cannot drift.

The values are not a second set of opinions: every one of ``decide``'s formulas
degenerates to exactly these numbers at neutral inputs, and each is also the
sklearn default of the estimator it reaches (``IterativeImputer.max_iter`` = 10,
``KNNImputer.n_neighbors`` = 5, and so on).

A strategy with no row has no dials at all — ``GMMSampling`` and ``Constant``
are driven entirely by profile facts and declared values, so
:meth:`ImputationDecision.with_hyperparameters` refuses every key on them. The
invariant is "every unit that has dials carries all of them", not "every unit id
appears in the map".
"""


def _dial_defaults(strategy: ImputationStrategy) -> dict[str, Any]:
    """Fresh copy of a strategy's dial row — empty when it has no dials."""
    return dict(_STRATEGY_DIALS.get(strategy, {}))


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

    ``Custom`` is the one member no branch of that factory builds: it marks a
    unit whose estimator the *user* supplied, through
    :func:`~dataforge_ml.imputation.author`'s ``estimators=`` channel (ADR-0083).
    The instance itself lives in
    :attr:`ImputationDecision.custom_estimators`, keyed by unit id, so this
    enum stays a label. It is a value rather than ``None`` because ``None``
    already carries the load-bearing "no estimator family, cannot train"
    meaning on the fit path; as a value, every existing reader of a
    ``ModelChoice`` stays correct without learning anything. Given up: the enum
    stops being a closed list of families the library can build — one member
    means "look elsewhere".
    """

    BayesianRidge = "bayesian_ridge"
    RandomForestRegressor = "random_forest_regressor"
    GradientBoostingRegressor = "gradient_boosting_regressor"
    Custom = "custom"


# ---------------------------------------------------------------------------
# Shared strategy-legality rules
#
# One source of truth for "which strategies may a user declare, and what does
# the redirect say when they cannot" — consumed by
# ``NumericImputationConfig.set_per_column_strategy``, the single declaration
# surface for a strategy (ADR-0082).
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

        Raises
        ------
        ValueError
            If ``data`` carries the retired ``mice_max_iter`` or
            ``knn_n_neighbors`` key (ADR-0083).
        """
        for retired in ("mice_max_iter", "knn_n_neighbors"):
            if retired in data:
                raise ValueError(
                    f"'{retired}' was removed from NumericImputationConfig "
                    "Set this dial on the plan instead, via "
                    "ImputationDecision.with_hyperparameters(unit_id, hyperparameters)."
                )

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


def _md_cell(value: "Any") -> str:
    """Render one value as a Markdown table cell.

    Absence is stated rather than left as a bare ``None``, enums render by
    their string form (matching ``to_dict``), and any pipe or newline in the
    value is neutralised so it cannot break the surrounding table.
    """
    if value is None:
        return "none"
    if isinstance(value, Enum):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_md_cell(v) for v in value) if value else "none"
    elif isinstance(value, dict):
        text = (
            ", ".join(f"{k}={_md_cell(v)}" for k, v in value.items())
            if value
            else "none"
        )
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _frame_lines(frame: pl.DataFrame) -> list[str]:
    """Render a Payload Frame's shape and dtypes, never its rows (ADR-0086)."""
    lines = [
        f"Shape: {frame.height:,} rows x {frame.width:,} columns\n",
        "| Column | Dtype |",
        "|---|---|",
    ]
    if frame.width:
        for name, dtype in frame.schema.items():
            lines.append(f"| `{name}` | {dtype} |")
    else:
        lines.append("| none | |")
    return lines


def _flatten_snapshot(value: "Any", prefix: str = "") -> "list[tuple[str, Any]]":
    """Flatten a nested config snapshot into dotted-key / value rows."""
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_snapshot(sub, path))
    else:
        rows.append((prefix, value))
    return rows


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
    center1 : float, optional
        First of the two bimodal cluster centres the GMM-Sampling and
        Cluster-Conditional strategies split on. Measured by Phase 1, not
        learned from training data. ``None`` for non-bimodal columns.
    center2 : float, optional
        Second bimodal cluster centre. See ``center1``.
    feature_cols : tuple[str, ...], optional
        Columns the Cluster-Conditional centroid branch measures its per-cluster
        centroids over — the features correlated with this column at decide-time.
        ``None`` for every other strategy.
    grouping_variable : str, optional
        Column whose groups the Cluster-Conditional group-wise branch aggregates
        within, when one was declared. ``None`` selects the centroid branch.
    constant_fill : float, optional
        The declared fill value for a ``Constant`` column. ``None`` for every
        other strategy. A declared value, never a learned one.
    indicator_flag : bool
        Whether a binary missingness indicator column will be appended.
    mnar : bool
        Whether the column is routed as Missing-Not-At-Random.
    drop : bool
        Whether the column will be dropped for exceeding the drop threshold.

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
    center1: Optional[float] = None
    center2: Optional[float] = None
    feature_cols: Optional[tuple[str, ...]] = None
    grouping_variable: Optional[str] = None
    constant_fill: Optional[float] = None
    indicator_flag: bool = False
    mnar: bool = False
    drop: bool = False

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
            "center1": self.center1,
            "center2": self.center2,
            "feature_cols": (
                list(self.feature_cols) if self.feature_cols is not None else None
            ),
            "grouping_variable": self.grouping_variable,
            "constant_fill": self.constant_fill,
            "indicator_flag": self.indicator_flag,
            "mnar": self.mnar,
            "drop": self.drop,
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
        raw_feature_cols = data.get("feature_cols")
        return cls(
            column=data["column"],
            semantic_type=SemanticType[data["semantic_type"]],
            strategy=ImputationStrategy[data["strategy"]],
            signals=tuple(data.get("signals", ())),
            model_choice=(
                ModelChoice[raw_model_choice] if raw_model_choice is not None else None
            ),
            domain_snap_bounds=(tuple(raw_bounds) if raw_bounds is not None else None),
            center1=data.get("center1"),
            center2=data.get("center2"),
            feature_cols=(
                tuple(raw_feature_cols) if raw_feature_cols is not None else None
            ),
            grouping_variable=data.get("grouping_variable"),
            constant_fill=data.get("constant_fill"),
            indicator_flag=bool(data.get("indicator_flag", False)),
            mnar=bool(data.get("mnar", False)),
            drop=bool(data.get("drop", False)),
        )

    def to_markdown(self) -> str:
        """Render the column decision as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning plan document composes it
        without a heading collision. Every field is covered; absent values are
        stated rather than left as a bare ``None``, and enums render by their
        string form.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<column>``` and a field table.
        """
        bounds = (
            f"{self.domain_snap_bounds[0]}, {self.domain_snap_bounds[1]}"
            if self.domain_snap_bounds is not None
            else "none"
        )
        lines = [
            f"### `{self.column}`\n",
            "| Field | Value |",
            "|---|---|",
            f"| semantic_type | {_md_cell(self.semantic_type)} |",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| model_choice | {_md_cell(self.model_choice)} |",
            f"| signals | {_md_cell(self.signals)} |",
            f"| domain_snap_bounds | {_md_cell(bounds)} |",
            f"| center1 | {_md_cell(self.center1)} |",
            f"| center2 | {_md_cell(self.center2)} |",
            f"| feature_cols | {_md_cell(self.feature_cols)} |",
            f"| grouping_variable | {_md_cell(self.grouping_variable)} |",
            f"| constant_fill | {_md_cell(self.constant_fill)} |",
            f"| indicator_flag | {_md_cell(self.indicator_flag)} |",
            f"| mnar | {_md_cell(self.mnar)} |",
            f"| drop | {_md_cell(self.drop)} |",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


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
    via the opt-in Evaluation phase (ADR-0058).
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

    def to_markdown(self) -> str:
        """Render the audit record as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning result document composes it
        without a heading collision. The composed
        :class:`ColumnImputationDecision` renders its own table — *what was
        decided* — and the learned ``fill_value`` and ``indicator_added``
        continue it, so decided and learned fields sit side by side in one
        table without being confusable.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<column>``` and a field table
            covering the decision's fields plus the learned ones.
        """
        lines = [
            self.decision.to_markdown(),
            f"| fill_value | {_md_cell(self.fill_value)} |",
            f"| indicator_added | {_md_cell(self.indicator_added)} |",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


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
    is_block : bool, default False
        Whether the unit's columns train together in one joint model call —
        ``True`` for the ``MICE`` and ``KNN`` blocks, ``False`` for every
        per-column unit. Structural only: it says nothing about how expensive
        the unit is, nor whether it can absorb inner parallelism (that follows
        from ``model_choice``, which stays on
        :class:`ColumnImputationDecision`). Derived from ``strategy``, so a
        caller never supplies it.

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
    is_block: bool = False

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
            "is_block": self.is_block,
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
            Reconstructed unit instance. ``is_block`` is re-derived from
            ``strategy``, so whatever the payload carries for it — a stale
            value, a wrong value, or nothing at all — is ignored.
        """
        strategy = ImputationStrategy[data["strategy"]]
        return cls(
            unit_id=data["unit_id"],
            strategy=strategy,
            columns=tuple(data.get("columns", ())),
            hyperparameters=(
                _hyperparameters_from_dict(data["hyperparameters"])
                if data.get("hyperparameters") is not None
                else None
            ),
            is_block=strategy in _JOINT_BLOCK_STRATEGIES,
        )

    def to_markdown(self) -> str:
        """Render the execution unit as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning plan document composes it
        without a heading collision. Every field is covered, including the
        merged hyperparameters stamped onto the unit at plan construction.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<unit_id>``` and a field table.
        """
        hyperparameters = (
            dict(self.hyperparameters) if self.hyperparameters is not None else None
        )
        lines = [
            f"### `{self.unit_id}`\n",
            "| Field | Value |",
            "|---|---|",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| columns | {_md_cell(self.columns)} |",
            f"| is_block | {_md_cell(self.is_block)} |",
            f"| hyperparameters | {_md_cell(hyperparameters)} |",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


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
                        is_block=True,
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
                        is_block=True,
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
                    is_block=False,
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

    The plan is immutable: :meth:`with_model_choice` and
    :meth:`with_hyperparameters` return a *new* ``ImputationDecision`` rather
    than mutating in place, and ``units`` is re-derived from
    ``column_decisions`` at every construction, so a stale unit list is
    structurally impossible and every plan that exists is valid by construction.

    Those edits are *dial* edits: they change how a unit trains, never which
    units exist, so they cannot invalidate the decided hyperparameter base.
    Changing which strategy a column takes is a *structural* edit and belongs to
    :func:`~dataforge_ml.imputation.decide` alone, declared through
    ``per_column_strategy`` (ADR-0082).

    Parameters
    ----------
    column_decisions : dict[str, ColumnImputationDecision]
        Per-column plan entries keyed by column name, in decision order. Held as
        an internal copy so the constructed plan is independent of the caller's
        mapping.
    config_snapshot : dict
        Serialised :class:`~dataforge_ml.PipelineConfig` (``config.to_dict()``)
        the plan was decided under.
    decided_hyperparameters : dict[str, tuple[tuple[str, Any], ...]]
        The decide-time hyperparameter base per unit id, written by the authoring
        function and never by an edit (ADR-0073). Complete for each unit's
        strategy: it carries every dial the strategy has in ``_STRATEGY_DIALS``,
        an invariant :meth:`from_dict` re-establishes on load, plus any profile
        facts the unit's fitter reads.
    override_hyperparameters : dict[str, tuple[tuple[str, Any], ...]]
        The sparse per-unit override delta, written only by
        :meth:`with_hyperparameters`. ``_derive_units`` stamps the per-key merge
        ``decided ⊕ delta`` onto each unit, so the two-map split is invisible
        below the plan surface (ADR-0073).
    custom_estimators : dict[str, Any]
        User-supplied estimator instances keyed by unit id, filled by
        :func:`~dataforge_ml.imputation.author`'s ``estimators=`` channel and
        paired with :attr:`ModelChoice.Custom` on the unit's columns (ADR-0083).
        The **caller's own object**, never a clone, here and on every derived
        copy: cloning would break identity, make ``get_params`` a
        construction-time requirement stricter than sklearn's own, and silently
        strip fitted state. Executing a plan cannot mutate it —
        ``IterativeImputer`` clones per column — so the only live hazard is
        deliberate post-authoring mutation, which is closed by this paragraph
        rather than by a guard. Unit-keyed and not per-column: MICE is the only
        strategy with an estimator slot, so a per-column spelling would let two
        estimators be named for one block. **Not serialised** —
        :meth:`to_dict` drops it and the plan reloads as ``Custom`` with the
        slot empty. It stays in ``==``, so a ``Custom`` plan compares unequal to
        its own round trip; that is the truth, since the restored plan cannot
        fit. Nothing learned is lost: the estimator is a line of the user's own
        code.
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

    A plan carrying ``custom_estimators`` may hold a *pre-fitted* estimator: the
    library tolerates it and never reads its state (sklearn refits from
    scratch), and no guard refuses it, because any check would recognise only
    sklearn's trailing-underscore spelling and would read as a guarantee it is
    not. ADR-0060's value-free claim therefore narrows from a structural
    guarantee to a statement about the library: **the library never writes
    learned state onto a plan.**
    """

    column_decisions: "dict[str, ColumnImputationDecision]"
    config_snapshot: dict
    decided_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]" = field(
        default_factory=dict
    )
    override_hyperparameters: "dict[str, tuple[tuple[str, Any], ...]]" = field(
        default_factory=dict
    )
    custom_estimators: "dict[str, Any]" = field(default_factory=dict)
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
        # A shallow copy: the mapping is the plan's own, the estimator instances
        # inside it are the caller's by identity (ADR-0083).
        object.__setattr__(self, "custom_estimators", dict(self.custom_estimators))
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

    def units_for(self, strategy: ImputationStrategy) -> tuple[ImputationUnit, ...]:
        """Return the plan's units that execute ``strategy``.

        The selection surface for a caller driving their own fit loop: it
        replaces scanning ``units`` against a hand-typed ``unit_id`` literal,
        so the id stays an internal name the caller never spells.

        Always returns a tuple — empty when the plan routed nothing to
        ``strategy``, which is an ordinary outcome rather than an error. A
        strategy that is structural, or that no column reached, therefore needs
        no guard at the call site: the loop simply runs zero times. ``MICE`` and
        ``KNN`` yield at most one unit each, being joint blocks.

        Parameters
        ----------
        strategy : ImputationStrategy
            The strategy to select on.

        Returns
        -------
        tuple[ImputationUnit, ...]
            The matching units, in plan order. Empty if there are none.
        """

        if strategy == _STRUCTURAL_STRATEGIES:
            raise ValueError(
                f"{strategy} is an incorrect strategy being provided, "
                f"Dropped, Passthrough and Indicator strategies are not allowed."
            )

        return tuple(u for u in self.units if u.strategy == strategy)

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
            config_snapshot=self.config_snapshot,
            decided_hyperparameters=self.decided_hyperparameters,
            override_hyperparameters=self.override_hyperparameters,
            custom_estimators=self.custom_estimators,
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

        Every key must be one of the unit's strategy's dials (``_STRATEGY_DIALS``)
        — the same table both authors build the decided base from, so what is
        dialable is one fact rather than a per-author allowlist. A strategy with
        no row has no dials, and every key is rejected for it. The decided base
        may carry more than the dials (profile facts a fitter reads, such as a
        bimodal unit's centres); those are not dialable. Values are not
        type-checked: sklearn rejects an ill-typed value at fit time.

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
            If ``hyperparameters`` names a key that is not one of the unit's
            strategy's dials, identifying the unit and the offending key.
        """
        unit = next((u for u in self.units if u.unit_id == unit_id), None)
        if unit is None:
            raise KeyError(f"Unit '{unit_id}' is not part of this plan.")

        new_override_hyperparameters = dict(self.override_hyperparameters)
        if hyperparameters is None:
            new_override_hyperparameters.pop(unit_id, None)
        else:
            dial_keys = set(_dial_defaults(unit.strategy))
            for key in hyperparameters:
                if key not in dial_keys:
                    raise ValueError(
                        f"Unit '{unit_id}' ({unit.strategy}) has no dial '{key}' "
                        f"to override."
                    )
            merged_delta = dict(new_override_hyperparameters.get(unit_id, ()))
            merged_delta.update(hyperparameters)
            new_override_hyperparameters[unit_id] = tuple(merged_delta.items())

        return ImputationDecision(
            column_decisions=self.column_decisions,
            config_snapshot=self.config_snapshot,
            decided_hyperparameters=self.decided_hyperparameters,
            override_hyperparameters=new_override_hyperparameters,
            custom_estimators=self.custom_estimators,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def to_dict(self) -> dict:
        """Serialise the plan to a plain, JSON-friendly dictionary.

        ``units`` and ``dropped_columns`` are emitted for readability but are
        re-derived — not consumed — on :meth:`from_dict`, so a hand-edited unit
        list can never desynchronise a reloaded plan.

        ``custom_estimators`` is **omitted**: a user-supplied estimator is a
        live object, not data, and the payload stays JSON-native (ADR-0083). The
        reloaded plan keeps :attr:`ModelChoice.Custom` with an empty slot, and
        fitting it raises rather than falling back to a library estimator.
        Rejected alternatives: refusing to serialize such a plan at all, and
        pickling the estimator into the envelope.

        Returns
        -------
        dict
            The plan's fields with nested objects serialised to dicts.
        """
        return {
            "column_decisions": {
                col: d.to_dict() for col, d in self.column_decisions.items()
            },
            "config_snapshot": self.config_snapshot,
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

        The decided base is gap-filled against ``_STRATEGY_DIALS`` on the way
        in — a dial the payload is missing takes the table's neutral value, and
        a dial the payload carries is never overwritten. "Complete for the
        unit's strategy" is thereby true of the type rather than of the two
        authoring functions, which is what lets a fitter subscript a dial
        outright. Cost taken knowingly: an artifact saved before a dial existed
        silently gains it rather than dying on a bare ``KeyError``.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationDecision
            Reconstructed plan instance.
        """
        decided_hyperparameters = {
            unit_id: _hyperparameters_from_dict(hyp)
            for unit_id, hyp in data.get("decided_hyperparameters", {}).items()
        }
        override_hyperparameters = {
            unit_id: _hyperparameters_from_dict(hyp)
            for unit_id, hyp in data.get("override_hyperparameters", {}).items()
        }
        column_decisions = {
            col: ColumnImputationDecision.from_dict(raw)
            for col, raw in data.get("column_decisions", {}).items()
        }
        for unit in _derive_units(column_decisions):
            base = dict(decided_hyperparameters.get(unit.unit_id, ()))
            missing = {
                key: value
                for key, value in _dial_defaults(unit.strategy).items()
                if key not in base
            }
            if missing:
                base.update(missing)
                decided_hyperparameters[unit.unit_id] = tuple(base.items())
        return cls(
            column_decisions=column_decisions,
            config_snapshot=data.get("config_snapshot", {}),
            decided_hyperparameters=decided_hyperparameters,
            override_hyperparameters=override_hyperparameters,
            numeric_sentinels={
                col: list(v) for col, v in data.get("numeric_sentinels", {}).items()
            },
            string_sentinels={
                col: list(v) for col, v in data.get("string_sentinels", {}).items()
            },
        )

    def to_markdown(self) -> str:
        """Render the whole plan as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels and delegates to the
        :class:`ColumnImputationDecision` and :class:`ImputationUnit` fragments
        beneath them. Every field of the plan is covered, so a plan from
        :func:`~dataforge_ml.imputation.decide` and one built by hand through
        :func:`~dataforge_ml.imputation.author` render identically in shape and
        are comparable by eye.

        ``custom_estimators`` holds live user objects rather than data, so it is
        reported by unit id and estimator class name only.

        Returns
        -------
        str
            Markdown document with a summary, the per-column decisions, the
            execution units, the declared sentinels, and the config snapshot the
            plan was decided under.
        """
        lines = ["# Imputation Plan\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| columns | {len(self.column_decisions)} |")
        lines.append(f"| units | {len(self.units)} |")
        lines.append(f"| dropped_columns | {_md_cell(self.dropped_columns)} |")
        lines.append(
            "| custom_estimators | "
            + _md_cell(
                {
                    unit_id: type(estimator).__name__
                    for unit_id, estimator in self.custom_estimators.items()
                }
            )
            + " |"
        )
        lines.append("")

        lines.append("## Column Decisions\n")
        lines.append(
            "| Column | Semantic type | Strategy | Model choice | Indicator "
            "| MNAR | Drop |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        if self.column_decisions:
            for decision in self.column_decisions.values():
                lines.append(
                    f"| `{decision.column}` | {_md_cell(decision.semantic_type)} "
                    f"| {_md_cell(decision.strategy)} "
                    f"| {_md_cell(decision.model_choice)} "
                    f"| {_md_cell(decision.indicator_flag)} "
                    f"| {_md_cell(decision.mnar)} | {_md_cell(decision.drop)} |"
                )
        else:
            lines.append("| none | | | | | | |")
        lines.append("")
        if self.column_decisions:
            for decision in self.column_decisions.values():
                lines.append(decision.to_markdown())
                lines.append("")

        lines.append("## Execution Units\n")
        if self.units:
            lines.append("| Unit | Strategy | Columns | Block |")
            lines.append("|---|---|---|---|")
            for unit in self.units:
                lines.append(
                    f"| `{unit.unit_id}` | {_md_cell(unit.strategy)} "
                    f"| {_md_cell(unit.columns)} | {_md_cell(unit.is_block)} |"
                )
            lines.append("")
            for unit in self.units:
                lines.append(unit.to_markdown())
                lines.append("")
        else:
            lines.append("none")
            lines.append("")

        lines.append("## Hyperparameters\n")
        lines.append("| Unit | Decided base | Override delta |")
        lines.append("|---|---|---|")
        unit_ids = list(
            dict.fromkeys(
                [u.unit_id for u in self.units]
                + list(self.decided_hyperparameters)
                + list(self.override_hyperparameters)
            )
        )
        if unit_ids:
            for unit_id in unit_ids:
                decided = dict(self.decided_hyperparameters.get(unit_id, ()))
                override = dict(self.override_hyperparameters.get(unit_id, ()))
                lines.append(
                    f"| `{unit_id}` | {_md_cell(decided)} | {_md_cell(override)} |"
                )
        else:
            lines.append("| none | | |")
        lines.append("")

        lines.append("## Declared Sentinels\n")
        lines.append("| Column | Numeric | String |")
        lines.append("|---|---|---|")
        sentinel_cols = list(
            dict.fromkeys(list(self.numeric_sentinels) + list(self.string_sentinels))
        )
        if sentinel_cols:
            for column in sentinel_cols:
                lines.append(
                    f"| `{column}` "
                    f"| {_md_cell(self.numeric_sentinels.get(column))} "
                    f"| {_md_cell(self.string_sentinels.get(column))} |"
                )
        else:
            lines.append("| none | | |")
        lines.append("")

        lines.append("## Config Snapshot\n")
        rows = _flatten_snapshot(self.config_snapshot)
        if rows:
            lines.append("| Setting | Value |")
            lines.append("|---|---|")
            for key, value in rows:
                lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")
        else:
            lines.append("not recorded")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Imputation Plan document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


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

    def to_markdown(self) -> str:
        """Render the transform result as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels and delegates to the
        :class:`ColumnImputationRecord` fragments beneath them. ``dataframe`` is
        a Payload Frame — its shape and dtypes are reported and its rows never
        are — so the output stays bounded in the number of rows imputed.

        Returns
        -------
        str
            Markdown document with a summary, the dropped columns, the imputed
            frame's shape and dtypes, and the per-column audit records.
        """
        lines = ["# Imputation Result\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| records | {len(self.records)} |")
        lines.append(f"| dropped_columns | {_md_cell(self.dropped_columns)} |")
        lines.append(
            "| indicators_added | "
            + _md_cell(sum(1 for r in self.records.values() if r.indicator_added))
            + " |"
        )
        lines.append("")

        lines.append("## Imputed Frame\n")
        lines.extend(_frame_lines(self.dataframe))
        lines.append("")

        lines.append("## Column Records\n")
        lines.append("| Column | Strategy | Fill value | Indicator added |")
        lines.append("|---|---|---|---|")
        if self.records:
            for record in self.records.values():
                lines.append(
                    f"| `{record.decision.column}` "
                    f"| {_md_cell(record.decision.strategy)} "
                    f"| {_md_cell(record.fill_value)} "
                    f"| {_md_cell(record.indicator_added)} |"
                )
        else:
            lines.append("| none | | | |")
        lines.append("")
        for record in self.records.values():
            lines.append(record.to_markdown())
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Imputation Result document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
