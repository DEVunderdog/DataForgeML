"""
Configuration and result dataclasses for the profiling phase — Phase 1 redesign.

ProfileConfig controls the structural profiler's behaviour.
Stats dataclasses hold per-column and dataset-level profiling results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, InitVar
from enum import StrEnum
from types import MappingProxyType
from typing import Optional, Union

from ..config import SemanticType, Modality
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingnessProfileConfig,
    MissingSeverity,
    RowMissingnessDistribution,
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
    NonlinearityTag,
    NumericFlag,
    NumericStats,
    NumericProfileConfig,
    NonlinearityProfileConfig,
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


class EpochUnit(StrEnum):
    s = "s"
    ms = "ms"
    us = "us"
    ns = "ns"
    d = "d"


# ---------------------------------------------------------------------------
# Column and dataset result containers
# ---------------------------------------------------------------------------

AnyStats = Union[NumericStats, CategoricalStats, DatetimeStats, BooleanStats, TextStats]


def _format_dict_lines(d: dict, indent: int = 0) -> list[str]:
    out = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            if not v:
                out.append(f"{prefix}- **{k}**: (empty)")
            else:
                out.append(f"{prefix}- **{k}**:")
                out.extend(_format_dict_lines(v, indent + 1))
        elif isinstance(v, list):
            if not v:
                out.append(f"{prefix}- **{k}**: (empty)")
            else:
                out.append(f"{prefix}- **{k}**:")
                for item in v:
                    out.append(f"{prefix}  - {item}")
        else:
            out.append(f"{prefix}- **{k}**: {v}")
    return out


def _top_n_abs_correlations(
    matrix: dict[str, dict[str, float]], col_name: str, n: int = 5
) -> list[tuple[str, float]]:
    row = matrix.get(col_name, {})
    pairs = [(other, value) for other, value in row.items() if other != col_name]
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs[:n]


def _is_clean_column(col: "ColumnProfile") -> bool:
    missingness = col.missingness
    if missingness is not None and missingness.flags:
        return False
    severity = missingness.severity if missingness is not None else None
    if severity not in (None, MissingSeverity.Minor):
        return False
    if isinstance(col.stats, NumericStats):
        if col.stats.flags:
            return False
        if col.stats.nonlinearity_tag not in (None, NonlinearityTag.Linear):
            return False
    return True


def _compact_column_detail(
    col: "ColumnProfile",
    feature_correlation: Optional[CorrelationProfileResult],
) -> dict:
    """
    Build the per-column dict used in the Flagged Columns detail section.

    Applies the compact-view field rules (ADR-0040) on top of
    ``ColumnProfile.to_dict()``: drops ``total_rows`` from the missingness
    subsection and ``histogram`` from the stats subsection, and caps
    ``top_values`` (present on both ``NumericStats`` and ``CategoricalStats``)
    to 3 entries. All other fields — including redundant scalar pairs,
    percentiles, bimodal stats, and ``correlated_with`` — pass through
    unchanged. When ``feature_correlation`` carries a row for this column,
    a ``correlations`` entry is added with the top-5 highest absolute
    Pearson and top-5 highest absolute Spearman correlations (descending
    by absolute value) in place of the full N×N matrices.

    Parameters
    ----------
    col : ColumnProfile
        The column profile to render.
    feature_correlation : CorrelationProfileResult or None
        Dataset-level feature-feature correlation result, if computed.

    Returns
    -------
    dict
        Trimmed dictionary suitable for ``_format_dict_lines``.
    """
    data = col.to_dict()
    missingness = data.get("missingness")
    if missingness is not None:
        missingness.pop("total_rows", None)
    stats = data.get("stats")
    if stats is not None:
        stats.pop("histogram", None)
        if "top_values" in stats:
            stats["top_values"] = stats["top_values"][:3]

    if feature_correlation is not None:
        top_pearson = _top_n_abs_correlations(feature_correlation.pearson_matrix, col.name)
        top_spearman = _top_n_abs_correlations(feature_correlation.spearman_matrix, col.name)
        if top_pearson or top_spearman:
            data["correlations"] = {
                "top_pearson": [f"{name}: {value}" for name, value in top_pearson],
                "top_spearman": [f"{name}: {value}" for name, value in top_spearman],
            }
    return data


def _flagged_column_tier(col: "ColumnProfile") -> int:
    missingness = col.missingness
    flags = missingness.flags if missingness is not None else []
    severity = missingness.severity if missingness is not None else None

    if MissingnessFlag.DropCandidate in flags or MissingnessFlag.FullyNull in flags:
        return 0
    if severity == MissingSeverity.Severe:
        return 1
    if severity == MissingSeverity.High:
        return 2
    if severity == MissingSeverity.Moderate:
        return 3
    if flags:
        return 4
    return 5


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

    Contains per-column profiles, dataset-level statistics, any target
    variable analyses requested via ``ProfileConfig.target_columns``, and
    the declared sentinel mappings copied from ``ProfileConfig`` so Phase 2
    can consume them without holding a config reference.
    """

    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    dataset: DatasetStats = field(default_factory=DatasetStats)
    targets: dict[str, TargetProfileResult] = field(default_factory=dict)
    numeric_sentinels: dict[str, list[float]] = field(default_factory=dict)
    string_sentinels: dict[str, list[str]] = field(default_factory=dict)

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
            "numeric_sentinels": dict(self.numeric_sentinels),
            "string_sentinels": {k: list(v) for k, v in self.string_sentinels.items()},
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

    def to_markdown(self) -> str:
        """
        Produce a compact, human-oriented Markdown view of the profiling result.

        The document contains a Dataset Overview section (scalar
        dataset-level fields only — ``memory_breakdown`` and
        ``missingness_matrix`` are omitted), a Column Summary table with one
        row per column, and a Flagged Columns section with a full detail
        subsection for every column that exceeds the clean threshold (see
        ``_is_clean_column``), ordered severity-first then alphabetically
        (see ``_flagged_column_tier``). Within each flagged column's detail
        section, ``histogram`` bins and the missingness ``total_rows`` field
        are dropped, ``top_values`` is capped to 3 entries, and the full
        Pearson/Spearman correlation matrices are replaced by the top-5
        highest absolute correlations for that column (see
        ``_compact_column_detail``); all other fields — including redundant
        scalar pairs, percentiles, bimodal stats, and ``correlated_with`` —
        are kept in full. A Target Analysis section follows with the top-5
        absolute Pearson and Spearman correlations per feature column for
        each declared target, and a Sentinels section renders
        ``numeric_sentinels`` / ``string_sentinels`` unchanged. Use
        ``to_full_markdown()`` for the complete lossless serialization.

        Returns
        -------
        str
            Markdown string containing the Dataset Overview, Column Summary,
            Flagged Columns, Target Analysis, and Sentinels sections.
        """
        lines = ["# Structural Profile Report (Compact)\n"]

        ds = self.dataset
        lines.append("## Dataset Overview\n")
        lines.append(f"- **modality**: {ds.modality}")
        lines.append(f"- **row_count**: {ds.row_count}")
        lines.append(f"- **column_count**: {ds.column_count}")
        lines.append(f"- **memory_bytes**: {ds.memory_bytes}")
        lines.append(f"- **duplicate_count**: {ds.duplicate_count}")
        lines.append(f"- **duplicate_ratio**: {ds.duplicate_ratio}")
        lines.append(f"- **overall_sparsity**: {ds.overall_sparsity}")
        lines.append(f"- **was_chunked**: {ds.was_chunked}")
        lines.append("- **row_distribution**:")
        for key, value in ds.row_distribution.to_dict().items():
            lines.append(f"  - **{key}**: {value}")
        lines.append("")

        lines.append("## Column Summary\n")
        lines.append(
            "| Column | Semantic Type | Missing % | Severity | "
            "Missingness Flags | Numeric Flags |"
        )
        lines.append("|---|---|---|---|---|---|")
        for col_name, col in self.columns.items():
            sem_type = col.semantic_type if col.semantic_type else "None"
            missingness = col.missingness
            if missingness is not None:
                missing_str = f"{missingness.effective_null_ratio * 100:.2f}%"
                severity = missingness.severity if missingness.severity else "None"
                missingness_flags = (
                    ", ".join(str(f) for f in missingness.flags)
                    if missingness.flags
                    else "None"
                )
            else:
                missing_str = "0.00%"
                severity = "None"
                missingness_flags = "None"
            numeric_flags = "None"
            if isinstance(col.stats, NumericStats) and col.stats.flags:
                numeric_flags = ", ".join(str(f) for f in col.stats.flags)
            lines.append(
                f"| `{col_name}` | {sem_type} | {missing_str} | {severity} | "
                f"{missingness_flags} | {numeric_flags} |"
            )
        lines.append("")

        lines.append("## Flagged Columns\n")
        flagged_columns = [
            col for col in self.columns.values() if not _is_clean_column(col)
        ]
        flagged_columns.sort(key=lambda col: (_flagged_column_tier(col), col.name))
        for col in flagged_columns:
            lines.append(f"### `{col.name}`\n")
            lines.extend(
                _format_dict_lines(
                    _compact_column_detail(col, self.dataset.feature_correlation)
                )
            )
            lines.append("")

        if self.dataset.target_correlations:
            lines.append("## Target Analysis\n")
            for target_name, corr in self.dataset.target_correlations.items():
                lines.append(f"### Target: `{target_name}`\n")
                feature_cols = [
                    c for c in corr.analysed_numeric_columns if c != target_name
                ]
                for feat in feature_cols:
                    top_pearson = _top_n_abs_correlations(corr.pearson_matrix, feat)
                    top_spearman = _top_n_abs_correlations(corr.spearman_matrix, feat)
                    lines.append(f"#### `{feat}`\n")
                    lines.extend(
                        _format_dict_lines(
                            {
                                "top_pearson": [
                                    f"{name}: {value}" for name, value in top_pearson
                                ],
                                "top_spearman": [
                                    f"{name}: {value}" for name, value in top_spearman
                                ],
                            }
                        )
                    )
                    lines.append("")

        lines.append("## Sentinels\n")
        lines.extend(
            _format_dict_lines(
                {
                    "numeric_sentinels": dict(self.numeric_sentinels),
                    "string_sentinels": {
                        k: list(v) for k, v in self.string_sentinels.items()
                    },
                }
            )
        )
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def to_full_markdown(self) -> str:
        """
        Produce a complete, lossless Markdown serialization for debugging and archival use.

        Every field present in ``to_dict()`` — including histogram bins, full
        correlation matrices, memory breakdown, and all per-column fields —
        is rendered as Markdown. For an 82-column dataset this produces
        roughly 1 MB of text; for human inspection of large datasets prefer
        ``to_markdown()`` once the compact view lands (ADR-0040).

        Returns
        -------
        str
            Markdown string containing a summary table followed by per-column detail
            sections and dataset-level statistics.
        """
        data = self.to_dict()
        lines = ["# Structural Profile Report\n"]

        # 1. Summary navigation table
        lines.append("## Summary\n")
        lines.append("| Column | Semantic Type | Missing % | Severity | Key Flags |")
        lines.append("|---|---|---|---|---|")
        
        for col_name, col_data in data.get("columns", {}).items():
            sem_type = col_data.get("semantic_type") or "None"
            missingness = col_data.get("missingness") or {}
            missing_pct = missingness.get("effective_null_ratio", 0.0) * 100
            missing_str = f"{missing_pct:.2f}%"
            severity = missingness.get("severity") or "None"
            flags = ", ".join(col_data.get("type_flags", [])) or "None"
            lines.append(f"| {col_name} | {sem_type} | {missing_str} | {severity} | {flags} |")
        
        lines.append("\n## Column Details\n")

        for col_name, col_data in data.get("columns", {}).items():
            lines.append(f"### `{col_name}`\n")
            lines.extend(_format_dict_lines(col_data))
            lines.append("")

        lines.append("## Dataset\n")
        lines.extend(_format_dict_lines(data.get("dataset", {})))
        lines.append("")

        lines.append("## Targets\n")
        lines.extend(_format_dict_lines(data.get("targets", {})))
        lines.append("")

        lines.append("## Numeric Sentinels\n")
        lines.extend(_format_dict_lines(data.get("numeric_sentinels", {})))
        lines.append("")

        lines.append("## String Sentinels\n")
        lines.extend(_format_dict_lines(data.get("string_sentinels", {})))
        lines.append("")

        return "\n".join(lines).strip() + "\n"


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


def _default_nonlinearity_config() -> NonlinearityProfileConfig:
    return NonlinearityProfileConfig()


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
    compute_nonlinearity : bool
        Whether to run ``NonlinearityProfiler`` and populate the four
        nonlinearity signal fields on ``NumericStats``.  When ``True`` and
        ``compute_correlation`` is also ``True``, Pearson and Spearman matrices
        are reused from the correlation step rather than recomputed.
        Default ``False``.
    nonlinearity : NonlinearityProfileConfig
        Threshold configuration for the nonlinearity sub-processor.
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
    compute_nonlinearity: bool = False
    nonlinearity: NonlinearityProfileConfig = field(
        default_factory=_default_nonlinearity_config
    )
    numeric_sentinels: InitVar[Optional[dict[str, list[float]]]] = None
    string_sentinels: InitVar[Optional[dict[str, list[str]]]] = None
    datetime_epoch_units: InitVar[Optional[dict[str, Union[str, EpochUnit]]]] = None
    datetime_formats: InitVar[Optional[dict[str, str]]] = None
    _numeric_sentinels: dict[str, list[float]] = field(default_factory=dict, init=False)
    _string_sentinels: dict[str, list[str]] = field(default_factory=dict, init=False)
    _datetime_epoch_units: dict[str, EpochUnit] = field(default_factory=dict, init=False)
    _datetime_formats: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(
        self,
        numeric_sentinels: Optional[dict[str, list[float]]],
        string_sentinels: Optional[dict[str, list[str]]],
        datetime_epoch_units: Optional[dict[str, Union[str, EpochUnit]]] = None,
        datetime_formats: Optional[dict[str, str]] = None,
    ) -> None:
        if numeric_sentinels is not None and not isinstance(numeric_sentinels, property):
            for k, vals in numeric_sentinels.items():
                self.set_numeric_sentinel(k, vals)
        if string_sentinels is not None and not isinstance(string_sentinels, property):
            for k, vals in string_sentinels.items():
                self.set_string_sentinel(k, vals)
        if datetime_epoch_units is not None and not isinstance(datetime_epoch_units, property):
            for k, val in datetime_epoch_units.items():
                self.set_datetime_epoch_unit(k, val)
        if datetime_formats is not None and not isinstance(datetime_formats, property):
            for k, fmt in datetime_formats.items():
                self.set_datetime_format(k, fmt)

    @property
    def numeric_sentinels(self) -> MappingProxyType[str, list[float]]:
        """
        Get the per-column numeric sentinel declarations.

        Keys are column names; values are lists of float-compatible sentinel
        values that should be treated as effective nulls (e.g.
        ``{"age": [-999.0, 9999.0]}``). Applies to any column whose dtype
        passes ``_numeric_sentinel_eligible`` (all integer and float Polars
        dtypes). Defaults to an empty dict — columns with no declaration are
        completely unaffected.

        Returns
        -------
        MappingProxyType[str, list[float]]
            Read-only mapping of column names to numeric sentinel values.
        """
        return MappingProxyType(self._numeric_sentinels)

    @property
    def string_sentinels(self) -> MappingProxyType[str, list[str]]:
        """
        Get the per-column user-declared string sentinel declarations.

        Keys are column names; values are lists of string values that should
        be treated as effective nulls for that column (e.g.
        ``{"status": ["N/A", "missing"]}``). Uses **replace semantics**: when
        a declaration exists for a column, only the declared values are
        matched (case-insensitive); the hardcoded defaults (``"NA"``,
        ``"NAN"``, ``"NULL"``, ``"NONE"``, ``"?"``) are not applied for that
        column. Empty/whitespace-only strings are always effective null
        regardless of any declaration. Defaults to an empty dict — columns
        with no declaration continue to use the hardcoded defaults unchanged.

        Returns
        -------
        MappingProxyType[str, list[str]]
            Read-only mapping of column names to string sentinel values.
        """
        return MappingProxyType(self._string_sentinels)

    @property
    def datetime_epoch_units(self) -> MappingProxyType[str, EpochUnit]:
        """
        Get the per-column datetime epoch units.

        Returns
        -------
        MappingProxyType[str, EpochUnit]
            Read-only mapping of column names to epoch units.
        """
        return MappingProxyType(self._datetime_epoch_units)

    @property
    def datetime_formats(self) -> MappingProxyType[str, str]:
        """
        Get the per-column declared datetime format strings.

        Keys are column names; values are strftime-style format strings (e.g.
        ``{"Year": "%Y"}``) applied by ``DatetimeProfiler`` with
        ``strict=False`` when coercing that column to Datetime. A declaration
        applies to any column profiled as Datetime, whether overridden or
        auto-detected. Format strings are not validated against strftime
        grammar at declaration time — a bad format surfaces at profiling time.
        Defaults to an empty dict — columns with no declaration fall back to
        Polars format inference.

        Returns
        -------
        MappingProxyType[str, str]
            Read-only mapping of column names to declared datetime formats.
        """
        return MappingProxyType(self._datetime_formats)

    def set_numeric_sentinel(self, column: str | list[str], values: list[float]) -> None:
        """
        Set numeric sentinel values for one or more columns.

        Parameters
        ----------
        column : str or list of str
            Column name or list of column names to apply the sentinels to.
        values : list of float
            List of sentinel values. Must be numeric and non-empty.

        Raises
        ------
        ValueError
            If the `values` list is empty, or if any element is not numeric.
        """
        if not values:
            raise ValueError("values list cannot be empty.")
        for v in values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"Value {v!r} is not numeric.")
        
        columns = [column] if isinstance(column, str) else column
        for c in columns:
            self._numeric_sentinels[c] = [float(v) for v in values]

    def set_string_sentinel(self, column: str | list[str], values: list[str]) -> None:
        """
        Set string sentinel values for one or more columns.

        Parameters
        ----------
        column : str or list of str
            Column name or list of column names to apply the sentinels to.
        values : list of str
            List of string sentinel values. Must be strings and non-empty.

        Raises
        ------
        ValueError
            If the `values` list is empty, or if any element is not a string.
        """
        if not values:
            raise ValueError("values list cannot be empty.")
        for v in values:
            if not isinstance(v, str):
                raise ValueError(f"Value {v!r} is not a string.")
        
        columns = [column] if isinstance(column, str) else column
        for c in columns:
            self._string_sentinels[c] = list(values)

    def set_datetime_epoch_unit(self, column: str | list[str], unit: str | EpochUnit) -> None:
        """
        Set epoch unit for one or more columns.

        Parameters
        ----------
        column : str or list of str
            Column name or list of column names to apply the epoch unit to.
        unit : str or EpochUnit
            Epoch unit (e.g. "s", "ms").

        Raises
        ------
        ValueError
            If the unit is not a valid EpochUnit.
        """
        try:
            enum_unit = EpochUnit(unit)
        except ValueError:
            valid_units = ", ".join([u.value for u in EpochUnit])
            raise ValueError(f"Unknown epoch unit {unit!r}. Valid options are: {valid_units}")
        
        columns = [column] if isinstance(column, str) else column
        for c in columns:
            self._datetime_epoch_units[c] = enum_unit

    def set_datetime_format(self, column: str | list[str], format: str) -> None:
        """
        Declare a datetime format string for one or more columns.

        The format is applied by ``DatetimeProfiler`` with ``strict=False``
        when coercing the column to Datetime, and is not validated against
        strftime grammar or the data at declaration time — a bad format
        surfaces at profiling time, consistent with ``set_column_type`` and
        ``set_datetime_epoch_unit``.

        Parameters
        ----------
        column : str or list of str
            Column name or list of column names to apply the format to.
        format : str
            A non-empty strftime-style format string (e.g. ``"%Y"``).

        Raises
        ------
        ValueError
            If any column name is empty, or if `format` is not a non-empty
            string.
        """
        if not isinstance(format, str) or not format:
            raise ValueError("format must be a non-empty string.")

        columns = [column] if isinstance(column, str) else column
        for c in columns:
            if not isinstance(c, str) or not c:
                raise ValueError("column name must be a non-empty string.")
            self._datetime_formats[c] = format

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
            "compute_nonlinearity": self.compute_nonlinearity,
            "nonlinearity": self.nonlinearity.to_dict(),
            "numeric_sentinels": {k: list(v) for k, v in self.numeric_sentinels.items()},
            "string_sentinels": {k: list(v) for k, v in self.string_sentinels.items()},
            "datetime_epoch_units": {k: v.value for k, v in self.datetime_epoch_units.items()},
            "datetime_formats": {k: v for k, v in self.datetime_formats.items()},
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
        config = cls(
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
            compute_nonlinearity=bool(data.get("compute_nonlinearity", False)),
            nonlinearity=NonlinearityProfileConfig.from_dict(
                data.get("nonlinearity", {})
            ),
            numeric_sentinels=data.get("numeric_sentinels", {}),
            string_sentinels=data.get("string_sentinels", {}),
            datetime_epoch_units=data.get("datetime_epoch_units", {}),
            datetime_formats=data.get("datetime_formats", {}),
        )

        return config

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
