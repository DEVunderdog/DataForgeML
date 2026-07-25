"""
Result dataclasses for missingness profiling.

Populated by MissingnessProfiler, which is always run as part of
StructuralProfiler (non-optional Phase 1 component).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Row-level distribution summary
# ---------------------------------------------------------------------------


@dataclass
class RowMissingnessDistribution:
    """
    Dataset-level summary of per-row missing-value counts.

    Computed by ``MissingnessProfiler`` using effective-null counts across all
    profiled columns, and by ``StructuralProfiler`` over the full active column
    set for the dataset-level ``DatasetStats.row_distribution``.

    Attributes
    ----------
    row_missingness_p90 : int
        90th-percentile count of missing columns per row across all profiled
        columns.  Zero means fewer than 10 % of rows have any missing values,
        so globally-sparse rows are not a concern.  Used by
        ``build_label_matrix`` to emit the compound row missingness signal.
    pct_zero_missing : float
        Proportion of rows with zero effective-null columns.
    pct_one_to_two : float
        Proportion of rows missing in exactly one or two columns.
    pct_three_to_five : float
        Proportion of rows missing in three to five columns.
    pct_over_five : float
        Proportion of rows missing in more than five columns.
    pct_over_half_missing : float
        Proportion of rows where more than half the profiled columns are
        effective null.
    drop_candidate_row_count : int
        Number of rows exceeding the ``row_drop_threshold`` fraction.
    complete_row_fraction : float
        Dataset-level fraction of analysed rows with zero missing values,
        equal to ``(row_missing == 0).mean()`` over the analysed columns.
        Harvested from the same per-row missing-count vector as
        ``row_missingness_p90`` and consumed by the imputation decision layer
        as a joint-convergence signal.
    """

    row_missingness_p90: int = 0
    pct_zero_missing: float = 0.0
    pct_one_to_two: float = 0.0
    pct_three_to_five: float = 0.0
    pct_over_five: float = 0.0
    pct_over_half_missing: float = 0.0
    drop_candidate_row_count: int = 0
    complete_row_fraction: float = 0.0

    def to_dict(self) -> dict:
        """
        Serialise the distribution to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "row_missingness_p90": self.row_missingness_p90,
            "pct_zero_missing": self.pct_zero_missing,
            "pct_one_to_two": self.pct_one_to_two,
            "pct_three_to_five": self.pct_three_to_five,
            "pct_over_five": self.pct_over_five,
            "pct_over_half_missing": self.pct_over_half_missing,
            "drop_candidate_row_count": self.drop_candidate_row_count,
            "complete_row_fraction": self.complete_row_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RowMissingnessDistribution":
        """
        Reconstruct a distribution from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        RowMissingnessDistribution
            Reconstructed distribution instance.
        """
        return cls(
            row_missingness_p90=data.get("row_missingness_p90", 0),
            pct_zero_missing=data.get("pct_zero_missing", 0.0),
            pct_one_to_two=data.get("pct_one_to_two", 0.0),
            pct_three_to_five=data.get("pct_three_to_five", 0.0),
            pct_over_five=data.get("pct_over_five", 0.0),
            pct_over_half_missing=data.get("pct_over_half_missing", 0.0),
            drop_candidate_row_count=data.get("drop_candidate_row_count", 0),
            complete_row_fraction=data.get("complete_row_fraction", 0.0),
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MissingSeverity(StrEnum):
    Minor = "minor"  # < 1%   missing
    Moderate = "moderate"  # 1–5%   missing
    High = "high"  # 5–20%  missing
    Severe = "severe"  # > 20%  missing


class MissingnessFlag(StrEnum):
    FullyNull = "fully_null"  # missing ratio == 1.0 → must drop
    MARSuspect = "mar_suspect"  # correlated missingness with ≥1 other col
    DropCandidate = "drop_candidate"  # >50% of rows missing across the column


# ---------------------------------------------------------------------------
# Per-column result
# ---------------------------------------------------------------------------


@dataclass
class ColumnMissingnessProfile:
    """
    Full missingness profile for a single column.

    Attributes
    ----------
    column : str
        Column name.
    total_rows : int
        Total rows in the DataFrame.
    standard_null_count : int
        Polars-level nulls (None / NaN for floats).
    effective_null_count : int
        Standard nulls + whitespace-only strings + sentinel strings
        ("NA", "NAN", "NULL", "NONE", "?") — i.e. the count used for
        imputation decisions.
    standard_null_ratio : float
        standard_null_count / total_rows.
    effective_null_ratio : float
        effective_null_count / total_rows.
    severity : MissingSeverity
        Derived from effective_null_ratio.
    flags : list[MissingnessFlag]
        Zero or more non-exclusive behavioural flags.
    correlated_with : list[str]
        Columns whose missingness indicator correlates > 0.6 with this
        column's indicator (populated after the correlation matrix pass).
    """

    column: str
    total_rows: int

    standard_null_count: int = 0
    effective_null_count: int = 0
    standard_null_ratio: float = 0.0
    effective_null_ratio: float = 0.0

    severity: Optional[MissingSeverity] = None

    flags: list[MissingnessFlag] = field(default_factory=list)
    correlated_with: list[str] = field(default_factory=list)

    def has_flag(self, flag: MissingnessFlag) -> bool:
        """Return whether this column carries the given ``MissingnessFlag``.

        Parameters
        ----------
        flag : MissingnessFlag
            The flag to test for.

        Returns
        -------
        bool
            ``True`` if *flag* is present in ``self.flags``, ``False`` otherwise.
        """
        return flag in self.flags

    def to_dict(self) -> dict:
        """Serialise the column profile to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name. ``severity`` is converted to
            its string value or ``None``; ``flags`` are converted to their
            string values.
        """
        return {
            "column": self.column,
            "total_rows": self.total_rows,
            "standard_null_count": self.standard_null_count,
            "effective_null_count": self.effective_null_count,
            "standard_null_ratio": self.standard_null_ratio,
            "effective_null_ratio": self.effective_null_ratio,
            "severity": str(self.severity) if self.severity else None,
            "flags": [str(f) for f in self.flags],
            "correlated_with": list(self.correlated_with),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnMissingnessProfile":
        """Reconstruct a column missingness profile from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ColumnMissingnessProfile
            Reconstructed profile instance.
        """
        raw_severity = data.get("severity")
        return cls(
            column=data["column"],
            total_rows=data["total_rows"],
            standard_null_count=data.get("standard_null_count", 0),
            effective_null_count=data.get("effective_null_count", 0),
            standard_null_ratio=data.get("standard_null_ratio", 0.0),
            effective_null_ratio=data.get("effective_null_ratio", 0.0),
            severity=MissingSeverity(raw_severity) if raw_severity else None,
            flags=[MissingnessFlag(f) for f in data.get("flags", [])],
            correlated_with=list(data.get("correlated_with", [])),
        )

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"  Column : {self.column}",
            f"    Standard nulls     : {self.standard_null_count:,}"
            f"  ({self.standard_null_ratio:.2%})",
            f"    Effective nulls    : {self.effective_null_count:,}"
            f"  ({self.effective_null_ratio:.2%})",
            f"    Severity           : {self.severity or 'N/A'}",
        ]
        if self.correlated_with:
            lines.append(f"    MAR correlates with: {', '.join(self.correlated_with)}")
        if self.flags:
            lines.append(f"    Flags              : {', '.join(self.flags)}")
        return "\n".join(lines)


@dataclass
class MissingnessProfileResult:
    """
    Missingness profile for all analysed columns.

    Attributes
    ----------
    columns : dict[str, ColumnMissingnessProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled.
    fully_null_columns : list[str]
        Columns where effective_null_ratio == 1.0.  Must be dropped.
    correlation_matrix : dict[str, dict[str, float]]
        Pairwise Pearson correlations between binary missingness indicators.
        Only populated when ≥ 2 columns have at least one missing value.
        Stored as a nested dict: matrix[col_a][col_b] = correlation.
    row_distribution : RowMissingnessDistribution
        Row-wise missingness summary including ``row_missingness_p90`` — the
        90th-percentile count of missing columns per row.  Used by
        ``build_label_matrix`` to emit the compound row signal.
    """

    columns: dict[str, ColumnMissingnessProfile] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)
    fully_null_columns: list[str] = field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    row_distribution: RowMissingnessDistribution = field(
        default_factory=RowMissingnessDistribution
    )

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Missingness Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        if self.fully_null_columns:
            lines.append(
                f"\n  Fully-null columns (must drop): "
                f"{', '.join(self.fully_null_columns)}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sub-config
# ---------------------------------------------------------------------------


@dataclass
class MissingnessProfileConfig:
    """
    Threshold configuration for the missingness sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``MissingnessProfileConfig()`` produces identical behaviour to
    the pre-config implementation.

    Parameters
    ----------
    severity_minor : float
        Effective null ratio upper bound (exclusive) for ``MissingSeverity.Minor``.
        Columns with a ratio below this value are classified as Minor.
    severity_moderate : float
        Effective null ratio upper bound (exclusive) for ``MissingSeverity.Moderate``.
    severity_high : float
        Effective null ratio upper bound (exclusive) for ``MissingSeverity.High``.
        Columns at or above this value are classified as ``MissingSeverity.Severe``.
    mar_correlation_threshold : float
        Minimum absolute Pearson correlation between binary missingness indicators
        for a column pair to receive the ``MissingnessFlag.MARSuspect`` flag.
    col_drop_threshold : float
        Effective null ratio above which a non-fully-null column receives the
        ``MissingnessFlag.DropCandidate`` flag.
    """

    severity_minor: float = 0.01
    severity_moderate: float = 0.05
    severity_high: float = 0.20
    mar_correlation_threshold: float = 0.60
    col_drop_threshold: float = 0.50

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "severity_minor": self.severity_minor,
            "severity_moderate": self.severity_moderate,
            "severity_high": self.severity_high,
            "mar_correlation_threshold": self.mar_correlation_threshold,
            "col_drop_threshold": self.col_drop_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissingnessProfileConfig:
        """
        Construct a ``MissingnessProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        MissingnessProfileConfig
            Reconstructed config instance.
        """
        return cls(
            severity_minor=float(data.get("severity_minor", 0.01)),
            severity_moderate=float(data.get("severity_moderate", 0.05)),
            severity_high=float(data.get("severity_high", 0.20)),
            mar_correlation_threshold=float(data.get("mar_correlation_threshold", 0.60)),
            col_drop_threshold=float(data.get("col_drop_threshold", 0.50)),
        )
