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
    ``Regression``, ``MICE``.

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
    Regression = "regression"
    MICE = "mice"
    MNAR = "mnar"
    Constant = "constant"
    Dropped = "dropped"
    Passthrough = "passthrough"  # output-only: assigned to columns with no missing values in training; cannot be declared in per_column_strategy
    ClusterConditional = "cluster_conditional"
    GMMSampling = "gmm_sampling"
    Indicator = "indicator"  # output-only: assigned to {col}_missing columns appended by the MNAR mechanism; cannot be declared in per_column_strategy


@dataclass
class NumericImputationConfig:
    """
    Operational thresholds for the numeric imputation sub-processor.

    Parameters
    ----------
    knn_max_rows : int
        Maximum number of rows before KNN is skipped in favour of Regression.
    knn_max_features : int
        Maximum number of features before KNN is skipped in favour of Regression.
    regression_min_rows : int
        Minimum number of rows required to fit a stable Regression model.
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
        Regression are not attempted because the feature set contains no useful
        predictive signal. Applies only to MCAR paths; MAR paths are not
        affected. Default of ``0.2`` preserves existing behaviour (no check
        applied today).
    per_column_strategy : dict[str, ImputationStrategy]
        Explicit per-column strategy overrides that fire at Priority 1.5 in the
        routing chain — after ``DropCandidate`` but before MNAR routing.  A
        column listed here bypasses all routing priorities 2–7.  Defaults to
        empty dict (no overrides).  Allowed values: ``Mean``, ``Median``,
        ``Mode``, ``KNN``, ``Regression``, ``MICE``.  To route a column to a
        constant fill, use ``per_column_constant_fill``
    per_column_constant_fill : dict[str, float]
        Self-sufficient constant fill declarations.  Each column listed here
        is routed to ``ImputationStrategy.Constant`` at Priority 1.5,
        bypassing all routing priorities 2–7.  No companion entry in
        ``per_column_strategy`` is required or allowed.  Keyed by column name.
        Defaults to empty dict.
    per_column_max_iter : dict[str, int]
        Overrides the dynamically-computed ``max_iter`` for named Regression
        columns only.  Keys are column names; values replace whatever
        ``_compute_max_iter`` would produce.
        Set manually.  Defaults to empty dict (no overrides).
    knn_n_neighbors : int, optional
        Overrides the dynamically-computed ``n_neighbors`` for the entire KNN
        block. A single value governs all KNN columns.
    mice_max_iter : int, optional
        Overrides the dynamically-computed ``max_iter`` for the entire MICE
        block. A single value governs all MICE columns.
    refit_r2_min_complete_rows : int
        Minimum number of complete rows required to attempt R² computation
        during fit.  When fewer complete rows are available, ``r2_train`` on
        ``ImputationFitDiagnostic`` is set to ``None``.  With k-fold CV
        (``refit_r2_cv_folds=5``), each validation fold contains 1/k of the
        complete rows; the floor of 50 ensures at least 10 rows per fold.
        Default ``50``.
    refit_r2_cv_folds : int
        Number of folds for cross-validated R² computation.  Applied
        uniformly across Regression, KNN, and MICE diagnostics.  Default ``5``.
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
    regression_min_rows: int = 500
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
    _per_column_max_iter: dict[str, int] = field(default_factory=dict)
    knn_n_neighbors: Optional[int] = None
    mice_max_iter: Optional[int] = None
    refit_r2_min_complete_rows: int = 50
    refit_r2_cv_folds: int = 5
    _bimodal_grouping_variables: dict[str, str] = field(default_factory=dict)
    bimodal_min_correlated_features: int = 3
    bimodal_correlation_threshold: float = 0.2

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
    def per_column_max_iter(self) -> MappingProxyType[str, int]:
        """
        Overrides the dynamically-computed ``max_iter`` for named Regression
        columns only.

        Returns
        -------
        MappingProxyType[str, int]
            Read-only view of per-column max iteration overrides.
        """
        return MappingProxyType(self._per_column_max_iter)

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

    def set_per_column_strategy(self, column: str | list[str], strategy: str | ImputationStrategy) -> None:
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
        
        _BLOCKED = {
            ImputationStrategy.Passthrough,
            ImputationStrategy.Indicator,
            ImputationStrategy.Dropped,
            ImputationStrategy.MNAR,
            ImputationStrategy.ClusterConditional,
            ImputationStrategy.GMMSampling,
        }
        
        if strategy in _BLOCKED:
            for col in column:
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

        if strategy == ImputationStrategy.Constant:
            for col in column:
                if col not in self._per_column_constant_fill:
                    raise ValueError(
                        f"Column '{col}': strategy is 'Constant' but no fill value was provided. "
                        f"Add an entry to per_column_constant_fill."
                    )
                    
        for col in column:
            self._per_column_strategy[col] = strategy

    def set_per_column_constant_fill(self, column: str | list[str], value: float) -> None:
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

    def set_per_column_max_iter(self, column: str | list[str], value: int) -> None:
        """
        Set the maximum iterations for regression models on one or more columns.

        Parameters
        ----------
        column : str | list[str]
            A single column name or list of column names.
        value : int
            The maximum iterations count.

        Raises
        ------
        ValueError
            If the value is less than or equal to 0.
        """
        if value <= 0:
            raise ValueError(f"Max iterations must be > 0, got {value}.")
            
        if isinstance(column, str):
            column = [column]
            
        for col in column:
            self._per_column_max_iter[col] = value

    def set_bimodal_grouping_variable(self, column: str | list[str], grouping_variable: str) -> None:
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
            if strategy == ImputationStrategy.Constant:
                if col not in self._per_column_constant_fill:
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
            "regression_min_rows": self.regression_min_rows,
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
            "per_column_strategy": {k: str(v) for k, v in self._per_column_strategy.items()},
            "per_column_constant_fill": dict(self._per_column_constant_fill),
            "per_column_max_iter": dict(self._per_column_max_iter),
            "knn_n_neighbors": self.knn_n_neighbors,
            "mice_max_iter": self.mice_max_iter,
            "refit_r2_min_complete_rows": self.refit_r2_min_complete_rows,
            "refit_r2_cv_folds": self.refit_r2_cv_folds,
            "bimodal_grouping_variables": dict(self._bimodal_grouping_variables),
            "bimodal_min_correlated_features": self.bimodal_min_correlated_features,
            "bimodal_correlation_threshold": self.bimodal_correlation_threshold,
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
            regression_min_rows=int(data.get("regression_min_rows", 500)),
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
            _per_column_max_iter={},
            knn_n_neighbors=(
                int(data["knn_n_neighbors"]) if data.get("knn_n_neighbors") is not None else None
            ),
            mice_max_iter=(
                int(data["mice_max_iter"]) if data.get("mice_max_iter") is not None else None
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
        )
        
        for col, val in data.get("per_column_constant_fill", {}).items():
            config.set_per_column_constant_fill(col, float(val))
        for col, val in data.get("per_column_strategy", {}).items():
            config.set_per_column_strategy(col, ImputationStrategy(val))
        for col, val in data.get("per_column_max_iter", {}).items():
            config.set_per_column_max_iter(col, int(val))
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
            
        conflicts = sorted(
            set(column) & set(self.numeric.per_column_strategy.keys())
        )
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
            numeric=NumericImputationConfig.from_dict(data.get("numeric", {}))
        )
        if "mnar_columns" in data:
            config.add_mnar_column(data["mnar_columns"])
        if "add_indicator_columns" in data:
            config.add_indicator_column(data["add_indicator_columns"])
        return config


@dataclass
class ImputationFitDiagnostic:
    """Fit quality diagnostic attached to each model-based column after fit().

    Present on ``ColumnImputationRecord.diagnostic`` for KNN, Regression, and
    MICE columns.  ``None`` for Passthrough, Dropped, Constant, and all scalar
    strategies (Mean, Median, Mode).

    Parameters
    ----------
    r2_train : float, optional
        Mean R² across k cross-validation folds on complete rows (k =
        ``refit_r2_cv_folds``).  ``None`` when fewer than
        ``refit_r2_min_complete_rows`` complete rows are available or when all
        folds are skipped due to zero variance in ``y_true``.
    converged : bool, optional
        Whether ``IterativeImputer`` halted before reaching ``max_iter``.
        ``None`` for KNN columns (convergence is not applicable).
    n_iter : int, optional
        Actual iteration count of ``IterativeImputer``.  ``None`` for KNN
        columns.
    imputed_mean : float
        Mean of the values imputed for null rows during fit.
    imputed_std : float
        Standard deviation of the imputed values.
    observed_mean : float
        Mean of non-null training values for this column.
    observed_std : float
        Standard deviation of non-null training values for this column.
    variance_ratio : float
        ``imputed_std / observed_std``.  A value near zero indicates
        distribution collapse — the model is predicting near-constant values.
    n_neighbors_used : int, optional
        Actual ``n_neighbors`` used by the KNN block.  ``None`` for Regression
        and MICE columns.
    k_capped : bool, optional
        ``True`` when the adaptive k formula's raw output exceeded ``n_rows − 1``
        and was forced to that bound (the model is averaging nearly all rows).
        ``False`` when the formula ran within bounds.  ``None`` when
        ``knn_n_neighbors`` override is active (adaptive formula was bypassed)
        or strategy is not KNN.
    """

    r2_train: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    converged: Optional[bool]
    n_iter: Optional[int]
    imputed_mean: float
    imputed_std: float
    observed_mean: float
    observed_std: float
    variance_ratio: float
    n_neighbors_used: Optional[int] = None
    k_capped: Optional[bool] = None

    def to_dict(self) -> dict:
        """Serialise the diagnostic to a plain dictionary.

        Returns
        -------
        dict
            All ten field values keyed by field name.
        """
        return {
            "r2_train": self.r2_train,
            "rmse": self.rmse,
            "mae": self.mae,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "imputed_mean": self.imputed_mean,
            "imputed_std": self.imputed_std,
            "observed_mean": self.observed_mean,
            "observed_std": self.observed_std,
            "variance_ratio": self.variance_ratio,
            "n_neighbors_used": self.n_neighbors_used,
            "k_capped": self.k_capped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImputationFitDiagnostic":
        """Reconstruct an ``ImputationFitDiagnostic`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``.

        Returns
        -------
        ImputationFitDiagnostic
            Reconstructed diagnostic instance.
        """
        return cls(
            r2_train=data.get("r2_train"),
            rmse=data.get("rmse"),
            mae=data.get("mae"),
            converged=data.get("converged"),
            n_iter=data.get("n_iter"),
            imputed_mean=float(data["imputed_mean"]),
            imputed_std=float(data["imputed_std"]),
            observed_mean=float(data["observed_mean"]),
            observed_std=float(data["observed_std"]),
            variance_ratio=float(data["variance_ratio"]),
            n_neighbors_used=data.get("n_neighbors_used"),
            k_capped=data.get("k_capped"),
        )


@dataclass
class ColumnImputationRecord:
    """
    Per-column audit entry produced after fit().

    Parameters
    ----------
    column : str
        Column name.
    semantic_type : SemanticType
        Detected semantic type of the column.
    strategy : ImputationStrategy
        Strategy applied to this column.
    fill_value : Any, optional
        Scalar fill value learned from training data (None for model-based strategies).
    indicator_added : bool
        Whether a binary missingness indicator column was appended.
    signals : list[str]
        Human-readable reasons that drove the strategy decision.
    domain_snap_bounds : tuple[float, float], optional
        ``(min, max)`` bounds applied to snap model-based predictions for
        BoundedDiscrete columns.  ``None`` for all other columns.
    diagnostic : ImputationFitDiagnostic, optional
        Fit quality metrics for KNN, Regression, and MICE columns.  ``None``
        for Passthrough, Dropped, Constant, and all scalar strategies.
    """

    column: str
    semantic_type: SemanticType
    strategy: ImputationStrategy
    fill_value: Optional[Any] = None
    indicator_added: bool = False
    signals: list[str] = field(default_factory=list)
    domain_snap_bounds: Optional[tuple[float, float]] = None
    diagnostic: Optional[ImputationFitDiagnostic] = None

    def to_dict(self) -> dict:
        """
        Serialise the audit record to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "column": self.column,
            "semantic_type": str(self.semantic_type),
            "strategy": str(self.strategy),
            "fill_value": self.fill_value,
            "indicator_added": self.indicator_added,
            "signals": list(self.signals),
            "domain_snap_bounds": (
                list(self.domain_snap_bounds)
                if self.domain_snap_bounds is not None
                else None
            ),
            "diagnostic": (
                self.diagnostic.to_dict() if self.diagnostic is not None else None
            ),
        }


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
    exclusions_applied : bool
        ``True`` when ``FittedImputer.apply_exclusions`` was called before
        ``transform()``, meaning dropped columns have been propagated into the
        pipeline config's hard exclusion set. A future Phase 3 orchestrator
        will raise if this is ``False`` when it receives an
        ``ImputationResult``.
    """

    dataframe: pl.DataFrame
    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    exclusions_applied: bool = False
