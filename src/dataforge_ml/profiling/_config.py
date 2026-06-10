"""
Configuration and result dataclasses for the profiling phase — Phase 1 redesign.

ProfileConfig controls the structural profiler's behaviour.
Stats dataclasses hold per-column and dataset-level profiling results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union

from ..config import SemanticType, Modality
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessProfileConfig,
)
from ._correlation_config import (
    CorrelationProfileResult,
    CorrelationProfileConfig,
)
from ._categorical_config import (
    CategoricalStats,
    CategoricalProfileConfig,
)
from ._numeric_config import (
    NumericStats,
    NumericProfileConfig,
)
from ._datetime_config import (
    DatetimeStats,
    DatetimeProfileConfig,
)
from ._type_detection_config import TypeDetectionConfig
from ._boolean_config import BooleanStats
from ._text_config import TextStats
from ._target_config import TargetProfileResult

# ---------------------------------------------------------------------------
# Type-detection enums — kept for TypeDetector compatibility
# ---------------------------------------------------------------------------


class NumericKind(StrEnum):
    Continuous = "continuous"
    BoundedDiscrete = "bounded_discrete"


class TypeFlag(StrEnum):
    NumericCoerced = "numeric_coerced"
    DatetimeCoerced = "datetime_coerced"
    BooleanCandidate = "boolean_candidate"
    EncodedCategory = "encoded_category"
    IdentifierColumn = "identifier_column"
    SequentialIndex = "sequential_index"
    FloatSequentialIndex = "float_sequential_index"
    FreeTextCandidate = "free_text_candidate"
    UserOverride = "user_override"
    NumericKindOverride = "numeric_kind_override"


# ---------------------------------------------------------------------------
# Column and dataset result containers
# ---------------------------------------------------------------------------

AnyStats = Union[NumericStats, CategoricalStats, DatetimeStats, BooleanStats, TextStats]


@dataclass
class ColumnProfile:
    """
    Per-column result produced by the structural profiler.

    Carries the type classification, missingness summary, and computed
    statistics for a single column after Phase 1 profiling completes.
    """

    name: str = ""
    semantic_type: Optional[SemanticType] = None
    numeric_kind: Optional[NumericKind] = None
    type_flags: list[TypeFlag] = field(default_factory=list)
    original_dtype: str = ""
    inferred_dtype: str = ""
    missingness: Optional[ColumnMissingnessProfile] = None
    is_target: bool = False
    stats: Optional[AnyStats] = None

    def to_dict(self) -> dict:
        """
        Serialise the column profile to a plain dictionary.

        Returns
        -------
        dict
            All field values with nested objects serialised to their string or
            dict representations.
        """
        return {
            "name": self.name,
            "semantic_type": str(self.semantic_type) if self.semantic_type else None,
            "numeric_kind": str(self.numeric_kind) if self.numeric_kind else None,
            "type_flags": [str(f) for f in self.type_flags],
            "original_dtype": self.original_dtype,
            "inferred_dtype": self.inferred_dtype,
            "missingness": self.missingness.to_dict() if self.missingness else None,
            "is_target": self.is_target,
            "stats": self.stats.to_dict() if self.stats else None,
        }


@dataclass
class RowMissingnessDistribution:
    """
    Dataset-level summary of per-row missing-value counts.
    Computed by StructuralProfiler over the full active column set.
    """

    pct_zero_missing: float = 0.0
    pct_one_to_two: float = 0.0
    pct_three_to_five: float = 0.0
    pct_over_five: float = 0.0
    pct_over_half_missing: float = 0.0
    drop_candidate_row_count: int = 0

    def to_dict(self) -> dict:
        return {
            "pct_zero_missing": self.pct_zero_missing,
            "pct_one_to_two": self.pct_one_to_two,
            "pct_three_to_five": self.pct_three_to_five,
            "pct_over_five": self.pct_over_five,
            "pct_over_half_missing": self.pct_over_half_missing,
            "drop_candidate_row_count": self.drop_candidate_row_count,
        }


@dataclass
class MemoryBreakdown:
    column_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def sorted_by_usage(self) -> list[tuple[str, int]]:
        return sorted(self.column_bytes.items(), key=lambda x: x[1], reverse=True)

    def top_consumers(self, n: int = 10) -> list[tuple[str, int]]:
        return self.sorted_by_usage[:n]

    def to_dict(self) -> dict:
        return {"column_bytes": dict(self.column_bytes)}


@dataclass
class DatasetStats:
    """
    Dataset-level statistics produced by the structural profiler.

    Aggregates row and memory counts, duplicate and sparsity ratios, the
    missingness matrix, row-level missingness distribution, and correlation
    results for the full profiled DataFrame.
    """

    modality: Modality = Modality.Tabular
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    memory_breakdown: Optional[MemoryBreakdown] = None
    duplicate_count: int = 0
    duplicate_ratio: float = 0.0
    overall_sparsity: float = 0.0
    was_chunked: bool = False
    missingness_matrix: Optional[dict[str, dict[str, float]]] = None
    row_distribution: RowMissingnessDistribution = field(
        default_factory=RowMissingnessDistribution
    )

    feature_correlation: Optional[CorrelationProfileResult] = None

    target_correlations: dict[str, CorrelationProfileResult] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict:
        """
        Serialise the dataset stats to a plain dictionary.

        Returns
        -------
        dict
            All field values with nested objects serialised to their dict
            representations.
        """
        return {
            "modality": str(self.modality),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "memory_breakdown": self.memory_breakdown.to_dict() if self.memory_breakdown else None,
            "duplicate_count": self.duplicate_count,
            "duplicate_ratio": self.duplicate_ratio,
            "overall_sparsity": self.overall_sparsity,
            "was_chunked": self.was_chunked,
            "missingness_matrix": self.missingness_matrix,
            "row_distribution": self.row_distribution.to_dict(),
            "feature_correlation": self.feature_correlation.to_dict() if self.feature_correlation else None,
            "target_correlations": {k: v.to_dict() for k, v in self.target_correlations.items()},
        }


@dataclass
class StructuralProfileResult:
    """
    Top-level result returned by ``StructuralProfiler.profile()``.

    Contains per-column profiles, dataset-level statistics, and any target
    variable analyses requested via ``ProfileConfig.target_columns``.
    """

    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    dataset: DatasetStats = field(default_factory=DatasetStats)
    targets: dict[str, TargetProfileResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Serialise the full profiling result to a plain dictionary.

        Returns
        -------
        dict
            Nested dictionary with ``columns``, ``dataset``, and ``targets``
            keys, each recursively serialised.
        """
        return {
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "dataset": self.dataset.to_dict(),
            "targets": {k: v.to_dict() for k, v in self.targets.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serialise the full profiling result to a JSON string.

        Parameters
        ----------
        indent : int
            Number of spaces used for JSON indentation.

        Returns
        -------
        str
            JSON representation of ``to_dict()``.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# ProfileConfig — clean break from per-profiler column lists
# ---------------------------------------------------------------------------


def _default_missingness_config() -> MissingnessProfileConfig:
    return MissingnessProfileConfig()


def _default_numeric_config() -> NumericProfileConfig:
    return NumericProfileConfig()


def _default_type_detection_config() -> TypeDetectionConfig:
    return TypeDetectionConfig()


def _default_categorical_config() -> CategoricalProfileConfig:
    return CategoricalProfileConfig()


def _default_correlation_config() -> CorrelationProfileConfig:
    return CorrelationProfileConfig()


def _default_datetime_config() -> DatetimeProfileConfig:
    return DatetimeProfileConfig()


@dataclass
class ProfileConfig:
    """
    Controls the structural profiler's behaviour.

    Parameters
    ----------
    modality : Modality
        Data modality. Currently only Tabular is implemented.
    target_columns : list[str]
        Names of label/target columns, if any.
    compute_correlation : bool
        Whether to compute the feature-feature correlation matrix.
    correlation_target_column : Optional[str]
        Column used for feature-target correlation metrics.
    memory_threshold_mb : float
        Memory (MB) above which chunked processing activates.
    chunk_size : int
        Rows per chunk when chunked processing is active.
    row_drop_threshold : float
        Fraction of columns that must be missing in a row for that row to be
        counted as a drop candidate in ``RowMissingnessDistribution``.
    missingness : MissingnessProfileConfig
        Threshold configuration for the missingness sub-processor.
    numeric : NumericProfileConfig
        Threshold configuration for the numeric distribution sub-processor.
    type_detection : TypeDetectionConfig
        Threshold configuration for the type-detection sub-processor.
    categorical : CategoricalProfileConfig
        Threshold configuration for the categorical sub-processor.
    correlation : CorrelationProfileConfig
        Threshold configuration for the correlation sub-processor.
    datetime_ : DatetimeProfileConfig
        Threshold configuration for the datetime sub-processor.
    """

    modality: Modality = Modality.Tabular
    target_columns: list[str] = field(default_factory=list)
    compute_correlation: bool = False
    correlation_target_column: Optional[str] = None
    memory_threshold_mb: float = 500.0
    chunk_size: int = 100_000
    row_drop_threshold: float = 0.50
    missingness: MissingnessProfileConfig = field(
        default_factory=_default_missingness_config
    )
    numeric: NumericProfileConfig = field(default_factory=_default_numeric_config)
    type_detection: TypeDetectionConfig = field(
        default_factory=_default_type_detection_config
    )
    categorical: CategoricalProfileConfig = field(
        default_factory=_default_categorical_config
    )
    correlation: CorrelationProfileConfig = field(
        default_factory=_default_correlation_config
    )
    datetime_: DatetimeProfileConfig = field(default_factory=_default_datetime_config)

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values, with nested sub-configs serialised recursively.
        """
        return {
            "modality": str(self.modality),
            "target_columns": list(self.target_columns),
            "compute_correlation": self.compute_correlation,
            "correlation_target_column": self.correlation_target_column,
            "memory_threshold_mb": self.memory_threshold_mb,
            "chunk_size": self.chunk_size,
            "row_drop_threshold": self.row_drop_threshold,
            "missingness": self.missingness.to_dict(),
            "numeric": self.numeric.to_dict(),
            "type_detection": self.type_detection.to_dict(),
            "categorical": self.categorical.to_dict(),
            "correlation": self.correlation.to_dict(),
            "datetime_": self.datetime_.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProfileConfig:
        """
        Construct a ``ProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        ProfileConfig
            Reconstructed config instance with all sub-configs deserialised.
        """
        return cls(
            modality=Modality(data.get("modality", Modality.Tabular)),
            target_columns=list(data.get("target_columns", [])),
            compute_correlation=bool(data.get("compute_correlation", False)),
            correlation_target_column=data.get("correlation_target_column"),
            memory_threshold_mb=float(data.get("memory_threshold_mb", 500.0)),
            chunk_size=int(data.get("chunk_size", 100_000)),
            row_drop_threshold=float(data.get("row_drop_threshold", 0.50)),
            missingness=MissingnessProfileConfig.from_dict(
                data.get("missingness", {})
            ),
            numeric=NumericProfileConfig.from_dict(data.get("numeric", {})),
            type_detection=TypeDetectionConfig.from_dict(
                data.get("type_detection", {})
            ),
            categorical=CategoricalProfileConfig.from_dict(
                data.get("categorical", {})
            ),
            correlation=CorrelationProfileConfig.from_dict(
                data.get("correlation", {})
            ),
            datetime_=DatetimeProfileConfig.from_dict(data.get("datetime_", {})),
        )

    def to_json(self) -> str:
        """
        Serialise the config to a JSON string.

        Returns
        -------
        str
            JSON representation of ``to_dict()``.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> ProfileConfig:
        """
        Construct a ``ProfileConfig`` from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON produced by ``to_json()``.

        Returns
        -------
        ProfileConfig
            Reconstructed config instance.
        """
        return cls.from_dict(json.loads(json_str))


@dataclass
class ColumnTypeInfo:
    column: str
    original_dtype: str
    inferred_dtype: str
    numeric_kind: Optional[NumericKind] = None
    flags: list[TypeFlag] = field(default_factory=list)
    semantic_type: Optional[SemanticType] = None

    def has_flag(self, flag: TypeFlag) -> bool:
        return flag in self.flags
