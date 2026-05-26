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
    Mean = "mean"
    Median = "median"
    Mode = "mode"
    KNN = "knn"
    Regression = "regression"
    MICE = "mice"
    Constant = "constant"
    Dropped = "dropped"
    Passthrough = "passthrough"


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
    mnar_constant_fill : float
        Constant value used to fill MNAR-declared numeric columns.
    """

    knn_max_rows: int = 50_000
    knn_max_features: int = 50
    regression_min_rows: int = 500
    mnar_constant_fill: float = -1

    def to_dict(self) -> dict:
        return {
            "knn_max_rows": self.knn_max_rows,
            "knn_max_features": self.knn_max_features,
            "regression_min_rows": self.regression_min_rows,
            "mnar_constant_fill": self.mnar_constant_fill,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NumericImputationConfig:
        return cls(
            knn_max_rows=int(data.get("knn_max_rows", 50_000)),
            knn_max_features=int(data.get("knn_max_features", 50)),
            regression_min_rows=int(data.get("regression_min_rows", 500)),
            mnar_constant_fill=float(data.get("mnar_constant_fill", -1)),
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
        These receive Constant fill + a missingness indicator regardless
        of the signals detected in Phase 1.
    add_indicator_columns : list[str]
        Columns for which a binary missingness indicator should be added
        even when they are not MNAR.
    """

    numeric: NumericImputationConfig = field(default_factory=NumericImputationConfig)
    mnar_columns: list[str] = field(default_factory=list)
    add_indicator_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "numeric": self.numeric.to_dict(),
            "mnar_columns": list(self.mnar_columns),
            "add_indicator_columns": list(self.add_indicator_columns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImputationConfig:
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
    """

    column: str
    semantic_type: SemanticType
    strategy: ImputationStrategy
    fill_value: Optional[Any] = None
    indicator_added: bool = False
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "semantic_type": str(self.semantic_type),
            "strategy": str(self.strategy),
            "fill_value": self.fill_value,
            "indicator_added": self.indicator_added,
            "signals": list(self.signals),
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
    """

    dataframe: pl.DataFrame
    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
