"""
Configuration and data-model dataclasses for the imputation phase — Phase 2.

``ImputationConfig`` controls strategy thresholds and MNAR declarations.
``ColumnRouting`` / ``ImputationRouting`` are the routing half of the layered
door (ADR-0088); ``ColumnImputationRecord`` / ``ImputationResult`` carry the
per-column audit trail and the imputed DataFrame produced by
``FittedImputer.transform``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any

import polars as pl

from ..config import SemanticType


class ImputationStrategy(StrEnum):
    """Imputation strategy assigned to a column after Phase 2 routing.

    Members fall into two categories:

    **Input strategies** — may be declared in ``per_column_strategy`` to
    override automatic routing: ``Mean``, ``Median``, ``Mode``, ``KNN``,
    ``MICE``, ``ClusterConditional``, ``GMMSampling``.

    **Output-only labels** — assigned by the engine after ``route()`` and
    recorded in ``ColumnRouting.strategy``; declaring them in
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

The single definition of the resolve-time hyperparameter base, read by
:func:`~dataforge_ml.imputation._recipe.resolve_recipe` for both a routed and a
hand-authored :class:`ImputationRouting`. A strategy with no row has no dials at
all — ``GMMSampling`` and ``Constant`` are driven entirely by profile facts and
declared values, so ``ImputationRecipe.with_hyperparameters`` refuses every key
on them.
"""


def _dial_defaults(strategy: ImputationStrategy) -> dict[str, Any]:
    """Fresh copy of a strategy's dial row — empty when it has no dials."""
    return dict(_STRATEGY_DIALS.get(strategy, {}))


class ModelChoice(StrEnum):
    """Concrete estimator family for a model-based imputation column.

    A value-free *name* for the sklearn estimator family a block trains with.
    It is a label, never a live or fitted estimator object. ``Custom`` marks a
    unit whose estimator the user supplied; the instance itself is held
    elsewhere, by identity, so this enum stays a label.

    :func:`~dataforge_ml.imputation.route` resolves the ``MICE`` block's
    choice off the Estimator Ladder (ADR-0094) — the most complex
    ``NonlinearityTag`` across the block's columns — whenever the block has
    any; :func:`~dataforge_ml.imputation.author` leaves it ``None`` unless the
    door's ``with_model_choice`` sets one explicitly (ADR-0090).
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
    }
)

# Human-readable signal recorded on the Passthrough routing of an
# Imputation-soft-excluded column. Rationale only — code reads
# ``ColumnRouting.excluded``; the one exception is ``ColumnRouting.from_dict``
# inferring the flag for a payload saved before the flag existed.
_EXCLUSION_SIGNAL = "soft-excluded for Imputation phase"

# Shipped NumericImputationConfig keys the escalation point deleted outright,
# mapped to what replaced each one. ``from_dict`` rejects them rather than
# ignoring them: each was a user-set threshold, so dropping it silently would
# change routing without notice.
_DELETED_NUMERIC_KEYS: dict[str, str] = {
    "mice_min_rows": (
        "The Feasibility Floor replaced it; set mice_min_rows_per_predictor "
    ),
    "knn_max_features": (
        "A KNN block now measures distance over every active numeric column "
        "knn_max_rows is the only KNN Resource Ceiling."
    ),
    "gradient_boost_min_rows": (
        "The booster is off the Estimator Ladder (ADR-0097); choose it "
        "explicitly with ImputationRouting.with_model_choice."
    ),
    "mcar_feature_predictability_threshold": (
        "The Signal Score replaced it; tune the signal_score_* dials instead "
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
        KNN's Resource Ceiling: the raw row count above which KNN is refused
        in favour of MICE. Reads ``row_count`` directly, unlike the
        Feasibility Floor's Usable Rows, because memory scales with the whole
        matrix (ADR-0091).
    mice_min_rows_per_predictor : int
        The Feasibility Floor's sole bound (ADR-0091, ADR-0097): the minimum
        Rows per Predictor (Usable Rows ÷ predictor count) a target column
        must clear for the MICE candidate to be feasible. KNN carries no
        floor bound — only ``knn_max_rows``.
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
    signal_score_breadth_weight : float
        The Signal Score's ``w_breadth`` weight (ADR-0092): how much Signal
        Breadth (the saturating count of predictors above
        ``mice_correlation_threshold``) may add to the base Explainable
        Variance reading. Argued, not measured (#519 measures only the
        shape's weights, not the shape itself).
    signal_score_latent_weight : float
        The Signal Score's ``w_latent`` weight (ADR-0092): the floor the
        Latent Structure component (mapped mutual information) sets under the
        score — evidence the R²/correlation probe was too weak for the
        structure present. Argued, not measured.
    signal_score_middle_tier_min : float
        The Signal Score value at or above which a column enters the middle
        Signal Tier (the lowest feasible candidate on the Capability Ladder)
        rather than the bottom tier (a scalar fill, regardless of
        feasibility). Argued, not measured (ADR-0092).
    signal_score_top_tier_min : float
        The Signal Score value at or above which a column enters the top
        Signal Tier (the richest feasible candidate on the Capability
        Ladder) rather than the middle tier. Argued, not measured (ADR-0092).
    per_column_strategy : dict[str, ImputationStrategy]
        Explicit per-column strategy overrides that fire at Priority 1.5 in the
        routing chain — after ``DropCandidate`` but before MNAR routing.  A
        column listed here bypasses all routing priorities 2–7.  Defaults to
        empty dict (no overrides).  Allowed values: ``Mean``, ``Median``,
        ``Mode``, ``KNN``, ``MICE``, ``ClusterConditional``, ``GMMSampling``.
        To route a column to a
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
        qualify the Bimodal Imputation Framework for branch 2 (model-based);
        columns with fewer correlated features fall to branch 3 (Cluster-Conditional).
    bimodal_correlation_threshold : float
        Minimum absolute Pearson correlation ``|r|`` a feature must have against
        a bimodal column for it to count toward the branch 2/3 feature tally in
        the Bimodal Imputation Framework.
    max_workers : int, optional
        Degree of thread parallelism for the numeric fit (ADR-0056).  The
        mutually-independent strategy blocks and the independent columns
        within a per-column strategy are fitted concurrently on threads,
        capped at this many workers. ``None`` (the default) auto-sizes to the
        available CPU count; ``1`` forces a fully sequential fit.  Concurrency
        never changes a result: the same ``random_seed`` yields a
        byte-identical ``FittedImputer`` regardless of this value.

    Raises
    ------
    ValueError
        If any column in ``per_column_strategy`` is mapped to
        ``Passthrough``, ``Indicator``, ``Dropped``, or ``MNAR``.  ``Constant``
        columns should use ``per_column_constant_fill``; ``Dropped`` columns
        should use ``PipelineConfig.exclude_columns``; ``MNAR`` columns should
        use ``mnar_columns``; ``Passthrough`` and ``Indicator`` are
        internal-only.
        If a column in ``bimodal_grouping_variables`` is forced to a strategy
        other than ``ClusterConditional``.
    """

    knn_max_rows: int = 50_000
    mice_min_rows_per_predictor: int = 2
    base_max_iter: int = 10
    knn_min_neighbors: int = 5
    knn_max_neighbors: int = 25
    knn_distance_weight_max_null_ratio: float = 0.15
    knn_distance_weight_max_features: int = 30
    mice_n_nearest_features_min_cols: int = 10
    mice_max_nearest_features: int = 20
    mice_correlation_threshold: float = 0.1
    signal_score_breadth_weight: float = 0.1
    signal_score_latent_weight: float = 0.3
    signal_score_middle_tier_min: float = 0.3
    signal_score_top_tier_min: float = 0.6
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
            If a strategy other than 'ClusterConditional' is set for a column
            that has an entry in ``bimodal_grouping_variables``.
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

        if strategy != ImputationStrategy.ClusterConditional:
            conflicts = sorted(
                set(column) & set(self._bimodal_grouping_variables.keys())
            )
            if conflicts:
                names = ", ".join(f"'{c}'" for c in conflicts)
                raise ValueError(
                    f"Columns have a bimodal grouping variable declared but are forced to "
                    f"strategy '{strategy}' (not 'ClusterConditional'): {names}. "
                    f"A grouping variable contradicts any forced strategy other than ClusterConditional."
                )

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
            If any column is already forced to a strategy other than
            'ClusterConditional' in ``per_column_strategy``.
        """
        if not grouping_variable or not grouping_variable.strip():
            raise ValueError("Grouping variable cannot be empty or purely whitespace.")

        if isinstance(column, str):
            column = [column]

        conflicts = sorted(
            col
            for col in column
            if col in self._per_column_strategy
            and self._per_column_strategy[col] != ImputationStrategy.ClusterConditional
        )
        if conflicts:
            details = ", ".join(
                f"'{c}' ({self._per_column_strategy[c]})" for c in conflicts
            )
            raise ValueError(
                f"Columns are forced to a strategy other than ClusterConditional: {details}. "
                f"A grouping variable contradicts any forced strategy other than ClusterConditional."
            )

        for col in column:
            self._bimodal_grouping_variables[col] = grouping_variable

    def validate(self) -> None:
        """
        Validate numeric imputation configuration for cross-field conflicts.

        Raises
        ------
        ValueError
            If any column has a bimodal grouping variable declared while being
            forced to a strategy other than ``ClusterConditional``.
        """
        conflicts = sorted(
            col
            for col in self._bimodal_grouping_variables
            if col in self._per_column_strategy
            and self._per_column_strategy[col] != ImputationStrategy.ClusterConditional
        )
        if conflicts:
            details = ", ".join(
                f"'{col}' (strategy={self._per_column_strategy[col]}, "
                f"grouping_variable='{self._bimodal_grouping_variables[col]}')"
                for col in conflicts
            )
            raise ValueError(
                f"Columns have both a bimodal grouping variable and a forced strategy "
                f"other than ClusterConditional: {details}. "
                f"A grouping variable contradicts any forced strategy other than ClusterConditional."
            )

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
        self.validate()

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
            "mice_min_rows_per_predictor": self.mice_min_rows_per_predictor,
            "base_max_iter": self.base_max_iter,
            "knn_min_neighbors": self.knn_min_neighbors,
            "knn_max_neighbors": self.knn_max_neighbors,
            "knn_distance_weight_max_null_ratio": self.knn_distance_weight_max_null_ratio,
            "knn_distance_weight_max_features": self.knn_distance_weight_max_features,
            "mice_n_nearest_features_min_cols": self.mice_n_nearest_features_min_cols,
            "mice_max_nearest_features": self.mice_max_nearest_features,
            "mice_correlation_threshold": self.mice_correlation_threshold,
            "signal_score_breadth_weight": self.signal_score_breadth_weight,
            "signal_score_latent_weight": self.signal_score_latent_weight,
            "signal_score_middle_tier_min": self.signal_score_middle_tier_min,
            "signal_score_top_tier_min": self.signal_score_top_tier_min,
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
            If ``data`` carries a retired key. ``mice_max_iter`` and
            ``knn_n_neighbors`` moved to the recipe (ADR-0083).
            ``mice_min_rows``, ``knn_max_features``,
            ``gradient_boost_min_rows`` and
            ``mcar_feature_predictability_threshold`` were deleted by the
            escalation point (ADR-0091, ADR-0092, ADR-0093, ADR-0097). The
            message names what replaced each one.
        """
        for retired in ("mice_max_iter", "knn_n_neighbors"):
            if retired in data:
                raise ValueError(
                    f"'{retired}' was removed from NumericImputationConfig. "
                    "Set this dial on the recipe instead, via "
                    "ImputationRecipe.with_hyperparameters(unit_id, hyperparameters)."
                )
        for retired, replacement in _DELETED_NUMERIC_KEYS.items():
            if retired in data:
                raise ValueError(
                    f"'{retired}' was removed from NumericImputationConfig. "
                    f"{replacement}"
                )

        config = cls(
            knn_max_rows=int(data.get("knn_max_rows", 50_000)),
            mice_min_rows_per_predictor=int(
                data.get("mice_min_rows_per_predictor", 2)
            ),
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
            signal_score_breadth_weight=float(
                data.get("signal_score_breadth_weight", 0.1)
            ),
            signal_score_latent_weight=float(
                data.get("signal_score_latent_weight", 0.3)
            ),
            signal_score_middle_tier_min=float(
                data.get("signal_score_middle_tier_min", 0.3)
            ),
            signal_score_top_tier_min=float(
                data.get("signal_score_top_tier_min", 0.6)
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
        These receive a binary missingness indicator and keep their nulls,
        regardless of Phase 1 signals. A data-derived fill (observed mean or
        median, skew-driven) is computed and exposed on
        ``ColumnImputationRecord.fill_value`` but not applied (ADR-0098).

    Raises
    ------
    ValueError
        If any column appears in both ``mnar_columns`` and
        ``numeric.per_column_strategy``.  These declarations are mutually
        exclusive: ``mnar_columns`` adds an indicator and exposes a
        data-derived fill without applying it; ``per_column_strategy`` directs the routing engine to a
        user-specified strategy.  Declaring the same column in both is
        contradictory and is caught at construction time before any data is
        touched.
    """

    numeric: NumericImputationConfig = field(default_factory=NumericImputationConfig)
    _mnar_columns: list[str] = field(default_factory=list, init=False)

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
                f"Use mnar_columns for MNAR semantics (indicator + exposed fill) "
                f"or per_column_strategy for a user-specified strategy, not both."
            )

        for c in column:
            if c not in self._mnar_columns:
                self._mnar_columns.append(c)

    def validate(self) -> None:
        """
        Validate the configuration for cross-field conflicts.

        Raises
        ------
        ValueError
            If any column appears in both ``mnar_columns`` and
            ``numeric.per_column_strategy``.
            If any column in ``numeric.bimodal_grouping_variables`` is forced to
            a strategy other than ``ClusterConditional``.
        """
        self.numeric.validate()
        conflicts = sorted(
            set(self._mnar_columns) & set(self.numeric.per_column_strategy.keys())
        )
        if conflicts:
            names = ", ".join(f"'{c}'" for c in conflicts)
            raise ValueError(
                f"Columns appear in both mnar_columns and numeric.per_column_strategy, "
                f"which are mutually exclusive: {names}. "
                f"Use mnar_columns for MNAR semantics (indicator + exposed fill) "
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
        return config


def _md_cell(value: Any) -> str:
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


def _flatten_snapshot(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
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
class ColumnRouting:
    """One column's routed imputation strategy, and nothing computed from it.

    The routing half of the layered imputation door (ADR-0088): a complete,
    inspectable description of *which approach* a column takes, and the two
    config declarations that complete that approach — ``constant_fill`` and
    ``grouping_variable``. Everything computed *from* that choice (dials,
    profile-derived estimates such as bimodal centres, execution units) belongs
    to later layers: :class:`~dataforge_ml.imputation.ImputationRecipe` and
    :func:`~dataforge_ml.imputation.derive_units`.

    The dataclass is frozen and every field holds an immutable value, so a
    routing is safe to share.

    Parameters
    ----------
    column : str
        Column name this routing applies to.
    semantic_type : SemanticType
        Detected semantic type of the column, carried from the profile.
    strategy : ImputationStrategy
        Strategy routed for this column.
    signals : tuple[str, ...], optional
        Human-readable routing rationale — the reasons that drove ``strategy``.
    constant_fill : float, optional
        The declared fill value for a ``Constant`` column. ``None`` for every
        other strategy. A declared value, never a learned one.
    grouping_variable : str, optional
        Column whose groups a ``ClusterConditional`` group-wise unit aggregates
        within, when one was declared. ``None`` selects the centroid branch.
    indicator_flag : bool
        Whether a binary missingness indicator column will be appended.
    mnar : bool
        Whether the column is routed as Missing-Not-At-Random.
    drop : bool
        Whether the column will be dropped for exceeding the drop threshold.
    excluded : bool
        Whether the column is Imputation-soft-excluded: carried as
        ``Passthrough``, its missing values ride through untouched, and it is
        never counted as an MICE or KNN predictor.
    """

    column: str
    semantic_type: SemanticType
    strategy: ImputationStrategy
    signals: tuple[str, ...] = ()
    constant_fill: float | None = None
    grouping_variable: str | None = None
    indicator_flag: bool = False
    mnar: bool = False
    drop: bool = False
    excluded: bool = False

    def to_dict(self) -> dict:
        """Serialise the routing to a plain dictionary.

        Enums are rendered by member *name* and tuples as lists, so the result
        is JSON-friendly and structurally round-trippable.

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
            "constant_fill": self.constant_fill,
            "grouping_variable": self.grouping_variable,
            "indicator_flag": self.indicator_flag,
            "mnar": self.mnar,
            "drop": self.drop,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ColumnRouting:
        """Reconstruct a ``ColumnRouting`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        A payload saved before ``excluded`` existed recorded the exclusion only
        as a signal, so the flag falls back to that signal when absent.

        Returns
        -------
        ColumnRouting
            Reconstructed routing instance.
        """
        signals = tuple(data.get("signals", ()))
        return cls(
            column=data["column"],
            semantic_type=SemanticType[data["semantic_type"]],
            strategy=ImputationStrategy[data["strategy"]],
            signals=signals,
            constant_fill=data.get("constant_fill"),
            grouping_variable=data.get("grouping_variable"),
            indicator_flag=bool(data.get("indicator_flag", False)),
            mnar=bool(data.get("mnar", False)),
            drop=bool(data.get("drop", False)),
            excluded=bool(data.get("excluded", _EXCLUSION_SIGNAL in signals)),
        )

    def to_markdown(self) -> str:
        """Render the column routing as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning document composes it without
        a heading collision.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<column>``` and a field table.
        """
        lines = [
            f"### `{self.column}`\n",
            "| Field | Value |",
            "|---|---|",
            f"| semantic_type | {_md_cell(self.semantic_type)} |",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| signals | {_md_cell(self.signals)} |",
            f"| constant_fill | {_md_cell(self.constant_fill)} |",
            f"| grouping_variable | {_md_cell(self.grouping_variable)} |",
            f"| indicator_flag | {_md_cell(self.indicator_flag)} |",
            f"| mnar | {_md_cell(self.mnar)} |",
            f"| drop | {_md_cell(self.drop)} |",
            f"| excluded | {_md_cell(self.excluded)} |",
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


@dataclass(frozen=True)
class ImputationRouting:
    """The routed, immutable choice-of-approach for every column (ADR-0088).

    Produced by :func:`~dataforge_ml.imputation.route` or
    :func:`~dataforge_ml.imputation.author`: a complete, inspectable mapping of
    *which strategy* every column takes, plus the block-level MICE estimator
    choice. Everything computed *from* that choice — dials, profile-derived
    estimates, execution units — belongs to
    :func:`~dataforge_ml.imputation.resolve_recipe` and
    :func:`~dataforge_ml.imputation.derive_units`, not here.

    Parameters
    ----------
    column_routings : dict[str, ColumnRouting]
        Per-column routing entries keyed by column name, in routing order.
        Held as an internal copy so the constructed routing is independent of
        the caller's mapping.
    mice_model_choice : ModelChoice, optional
        The estimator family the MICE block trains with. ``None`` when no
        column is routed to MICE. :func:`~dataforge_ml.imputation.route`
        resolves it off the Estimator Ladder (ADR-0094) whenever the block is
        non-empty; :func:`~dataforge_ml.imputation.author` leaves it ``None``
        unless ``with_model_choice`` sets one.
    mice_estimator : Any, optional
        A user-supplied estimator instance for the MICE block, held by
        identity and never cloned. ``None`` unless the block trains with a
        foreign estimator.
    """

    column_routings: dict[str, ColumnRouting]
    mice_model_choice: ModelChoice | None = None
    mice_estimator: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "column_routings", dict(self.column_routings)
        )

    def with_model_choice(self, choice: ModelChoice | Any) -> ImputationRouting:
        """Return a new routing with the MICE block's estimator set (ADR-0090).

        The one door for the MICE block's estimator, whether it is a library
        family or a foreign one: pass a :class:`ModelChoice` member to pick a
        family the factory builds, or a live sklearn-compatible estimator
        instance to use as-is. An instance sets ``mice_model_choice`` to
        :attr:`~dataforge_ml.ModelChoice.Custom` and holds the instance by
        identity, never cloned — this is the only way
        ``GradientBoostingRegressor`` is reachable, since it is off the
        Estimator Ladder (ADR-0094).

        Parameters
        ----------
        choice : ModelChoice or estimator instance
            A :class:`ModelChoice` member naming a library estimator family
            (every member except :attr:`~dataforge_ml.ModelChoice.Custom`),
            or a live, unfitted sklearn-compatible estimator instance.

        Returns
        -------
        ImputationRouting
            A new routing with ``mice_model_choice`` (and, for an instance,
            ``mice_estimator``) set. Every other field is carried over
            unchanged.

        Raises
        ------
        ValueError
            If ``choice`` is the bare :attr:`~dataforge_ml.ModelChoice.Custom`
            label with no estimator instance, or if this routing has no MICE
            block to set an estimator on.
        """
        import dataclasses

        mice_present = any(
            r.strategy == ImputationStrategy.MICE
            for r in self.column_routings.values()
        )
        if not mice_present:
            raise ValueError(
                "with_model_choice() requires a MICE block: this routing "
                "carries no column routed to ImputationStrategy.MICE."
            )
        if isinstance(choice, ModelChoice):
            if choice == ModelChoice.Custom:
                raise ValueError(
                    "ModelChoice.Custom is a label, not an estimator: pass "
                    "the estimator instance itself to with_model_choice() "
                    "instead of the bare Custom member."
                )
            return dataclasses.replace(
                self, mice_model_choice=choice, mice_estimator=None
            )
        return dataclasses.replace(
            self, mice_model_choice=ModelChoice.Custom, mice_estimator=choice
        )

    def to_dict(self) -> dict:
        """Serialise the routing to a plain, JSON-friendly dictionary.

        ``mice_estimator`` is **omitted**: a user-supplied estimator is a live
        object, not data. The reloaded routing keeps ``mice_model_choice`` as
        recorded and an empty ``mice_estimator`` slot.

        Returns
        -------
        dict
            The routing's fields with nested objects serialised to dicts.
        """
        return {
            "column_routings": {
                col: r.to_dict() for col, r in self.column_routings.items()
            },
            "mice_model_choice": (
                self.mice_model_choice.name
                if self.mice_model_choice is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImputationRouting:
        """Reconstruct an ``ImputationRouting`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationRouting
            Reconstructed routing instance. ``mice_estimator`` is never
            carried by the wire format and comes back ``None``.
        """
        raw_choice = data.get("mice_model_choice")
        return cls(
            column_routings={
                col: ColumnRouting.from_dict(raw)
                for col, raw in data.get("column_routings", {}).items()
            },
            mice_model_choice=(
                ModelChoice[raw_choice] if raw_choice is not None else None
            ),
        )

    def to_markdown(self) -> str:
        """Render the whole routing as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns
        the ``#`` and ``##`` heading levels and delegates to the
        :class:`ColumnRouting` fragments beneath them.

        Returns
        -------
        str
            Markdown document with a summary and the per-column routings.
        """
        lines = ["# Imputation Routing\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| columns | {len(self.column_routings)} |")
        lines.append(f"| mice_model_choice | {_md_cell(self.mice_model_choice)} |")
        lines.append(
            f"| mice_estimator | "
            f"{_md_cell(type(self.mice_estimator).__name__ if self.mice_estimator is not None else None)} |"
        )
        lines.append("")

        lines.append("## Column Routings\n")
        lines.append("| Column | Semantic type | Strategy | Indicator | MNAR | Drop |")
        lines.append("|---|---|---|---|---|---|")
        if self.column_routings:
            for routing in self.column_routings.values():
                lines.append(
                    f"| `{routing.column}` | {_md_cell(routing.semantic_type)} "
                    f"| {_md_cell(routing.strategy)} "
                    f"| {_md_cell(routing.indicator_flag)} "
                    f"| {_md_cell(routing.mnar)} | {_md_cell(routing.drop)} |"
                )
        else:
            lines.append("| none | | | | | |")
        lines.append("")
        if self.column_routings:
            for routing in self.column_routings.values():
                lines.append(routing.to_markdown())
                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Imputation Routing document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


# ---------------------------------------------------------------------------
# ImputationUnit — the plan's execution units (ADR-0084)
# ---------------------------------------------------------------------------

# Strategies whose execution is one joint block over several columns; each
# collapses to a single unit keyed by the block id rather than one unit per
# column.
_JOINT_BLOCK_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {ImputationStrategy.MICE, ImputationStrategy.KNN}
)

# Structural strategies carry no learned value and train nothing, so the
# routing projects their records straight through without materialising a unit.
_STRUCTURAL_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.Dropped,
        ImputationStrategy.Passthrough,
        ImputationStrategy.Indicator,
    }
)


@dataclass(frozen=True)
class ImputationUnit:
    """One executable unit of an imputation routing (ADR-0084).

    A value-free projection of *what execution trains together*: the joint
    ``MICE`` block, the joint ``KNN`` block, and one unit per independent
    column (GMM-Sampling / Cluster-Conditional / scalar / Constant / MNAR).
    Structural strategies (Dropped / Passthrough / Indicator) train nothing
    and produce no unit. Built by
    :func:`~dataforge_ml.imputation.derive_units`, never by hand.

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
        Whether the unit's columns train together in one joint model call.
        Derived from ``strategy``, so a caller never supplies it.
    """

    unit_id: str
    strategy: ImputationStrategy
    columns: tuple[str, ...]
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
            "is_block": self.is_block,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImputationUnit:
        """Reconstruct an ``ImputationUnit`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationUnit
            Reconstructed unit instance. ``is_block`` is re-derived from
            ``strategy``, so whatever the payload carries for it is ignored.
        """
        strategy = ImputationStrategy[data["strategy"]]
        return cls(
            unit_id=data["unit_id"],
            strategy=strategy,
            columns=tuple(data.get("columns", ())),
            is_block=strategy in _JOINT_BLOCK_STRATEGIES,
        )

    def to_markdown(self) -> str:
        """Render the execution unit as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning document composes it without
        a heading collision.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<unit_id>``` and a field table.
        """
        lines = [
            f"### `{self.unit_id}`\n",
            "| Field | Value |",
            "|---|---|",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| columns | {_md_cell(self.columns)} |",
            f"| is_block | {_md_cell(self.is_block)} |",
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
    Per-column audit entry produced after ``FittedImputer.transform``.

    Composes the pure, value-free :class:`ColumnRouting` (*what was routed*,
    reachable via ``record.decision``) with the values fitting learned (*what
    was learned*): the scalar ``fill_value`` and the ``indicator_added``
    fit-metadata flag.

    Parameters
    ----------
    decision : ColumnRouting
        The routed, value-free per-column entry.
    fill_value : Any, optional
        Scalar fill value learned from training data (None for model-based
        strategies).
    indicator_added : bool
        Whether a binary missingness indicator column was appended — a
        fit-time fact distinct from the routed ``decision.indicator_flag``.

    Notes
    -----
    Fit-quality metrics are not carried here. Fitting only learns fill values
    and models; quality measurement is a deliberate second step via the
    opt-in Evaluation phase.
    """

    decision: ColumnRouting
    fill_value: Any | None = None
    indicator_added: bool = False

    def to_dict(self) -> dict:
        """
        Serialise the audit record to a plain dictionary.

        Flattens the routing's serialised fields alongside the learned
        ``fill_value`` and ``indicator_added`` so the whole record round-trips
        through :meth:`from_dict`.

        Returns
        -------
        dict
            The routing's fields plus ``fill_value`` and ``indicator_added``.
        """
        return {
            **self.decision.to_dict(),
            "fill_value": self.fill_value,
            "indicator_added": self.indicator_added,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ColumnImputationRecord:
        """
        Reconstruct a ``ColumnImputationRecord`` from a plain dictionary.

        The inverse of :meth:`to_dict`: the routing is rebuilt from the same
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
            decision=ColumnRouting.from_dict(data),
            fill_value=data.get("fill_value"),
            indicator_added=bool(data.get("indicator_added", False)),
        )

    def to_markdown(self) -> str:
        """Render the audit record as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning result document composes it
        without a heading collision. The composed :class:`ColumnRouting`
        renders its own table — *what was routed* — and the learned
        ``fill_value`` and ``indicator_added`` continue it.

        Returns
        -------
        str
            Markdown subsection headed by ``### `<column>``` and a field table
            covering the routing's fields plus the learned ones.
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
