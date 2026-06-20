"""
Configuration and result dataclasses for the imputation phase — Phase 2.

ImputationConfig controls strategy thresholds and MNAR declarations.
Result dataclasses carry per-column audit records and the imputed DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

import polars as pl

from ..config import SemanticType


class ImputationStrategy(StrEnum):
    """Imputation strategy assigned to a column after Phase 2 fitting."""

    Mean = "mean"
    Median = "median"
    Mode = "mode"
    KNN = "knn"
    Regression = "regression"
    MICE = "mice"
    MNAR = "mnar"
    Constant = "constant"  # deprecated — kept only for from_dict() migration shim
    Dropped = "dropped"
    Passthrough = "passthrough"
    Indicator = "indicator"


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
        return cls(
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
        )


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
    """

    numeric: NumericImputationConfig = field(default_factory=NumericImputationConfig)
    mnar_columns: list[str] = field(default_factory=list)
    add_indicator_columns: list[str] = field(default_factory=list)

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
            "mnar_columns": list(self.mnar_columns),
            "add_indicator_columns": list(self.add_indicator_columns),
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
        return cls(
            numeric=NumericImputationConfig.from_dict(data.get("numeric", {})),
            mnar_columns=list(data.get("mnar_columns", [])),
            add_indicator_columns=list(data.get("add_indicator_columns", [])),
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
    """

    column: str
    semantic_type: SemanticType
    strategy: ImputationStrategy
    fill_value: Optional[Any] = None
    indicator_added: bool = False
    signals: list[str] = field(default_factory=list)
    domain_snap_bounds: Optional[tuple[float, float]] = None

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
