"""
Result dataclasses for numeric distribution profiling.

Populated by NumericProfiler, which is opt-in via
ProfileConfig.numeric_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, List


@dataclass
class PercentileSnapshot:
    p1: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    @property
    def iqr(self) -> Optional[float]:
        if self.p25 is not None and self.p75 is not None:
            return self.p75 - self.p25
        return None

    def to_dict(self) -> dict:
        return {
            "p1": self.p1, "p5": self.p5, "p25": self.p25, "p50": self.p50,
            "p75": self.p75, "p95": self.p95, "p99": self.p99,
        }


class SkewSeverity(StrEnum):
    Normal = "normal"
    Moderate = "moderate"
    High = "high"
    Severe = "severe"


class KurtosisTag(StrEnum):
    Platykurtic = "platykurtic"
    Mesokurtic = "mesokurtic"
    Leptokurtic = "leptokurtic"


class NumericFlag(StrEnum):
    ScaleAnomaly = "scale_anomaly"
    NearConstant = "near_constant"


@dataclass
class NumericTopValueEntry:
    value: float
    count: int
    percentage: float

    def to_dict(self) -> dict:
        return {"value": self.value, "count": self.count, "percentage": self.percentage}


@dataclass
class HistogramBin:
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float

    def to_dict(self) -> dict:
        return {
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "count": self.count, "percentage": self.percentage,
        }


@dataclass
class NumericStats:
    mean: Optional[float] = None
    median: Optional[float] = None
    mean_median_ratio: Optional[float] = None
    mode: Optional[float] = None
    mode_frequency: float = 0.0
    top_values: list[NumericTopValueEntry] = field(default_factory=list)
    histogram: list[HistogramBin] = field(default_factory=list)
    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    percentiles: PercentileSnapshot = field(default_factory=PercentileSnapshot)
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness_severity: Optional[SkewSeverity] = None
    kurtosis_tag: Optional[KurtosisTag] = None
    flags: List[NumericFlag] = field(default_factory=list)

    @property
    def iqr(self) -> Optional[float]:
        return self.percentiles.iqr

    def has_flag(self, flag: NumericFlag) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "median": self.median,
            "mean_median_ratio": self.mean_median_ratio,
            "mode": self.mode,
            "mode_frequency": self.mode_frequency,
            "top_values": [v.to_dict() for v in self.top_values],
            "histogram": [b.to_dict() for b in self.histogram],
            "std": self.std,
            "variance": self.variance,
            "min": self.min,
            "max": self.max,
            "percentiles": self.percentiles.to_dict(),
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "skewness_severity": str(self.skewness_severity) if self.skewness_severity else None,
            "kurtosis_tag": str(self.kurtosis_tag) if self.kurtosis_tag else None,
            "flags": [str(f) for f in self.flags],
        }


ColumnNumericProfile = NumericStats


# ---------------------------------------------------------------------------
# Sub-config
# ---------------------------------------------------------------------------


@dataclass
class NumericProfileConfig:
    """
    Threshold configuration for the numeric distribution sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``NumericProfileConfig()`` produces identical behaviour to the
    pre-config implementation.

    Parameters
    ----------
    skew_normal : float
        Absolute skewness upper bound (inclusive) for ``SkewSeverity.Normal``.
    skew_moderate : float
        Absolute skewness upper bound (inclusive) for ``SkewSeverity.Moderate``.
        Columns with ``|skew| > skew_normal`` and ``|skew| <= skew_moderate``
        are classified as Moderate.
    skew_high : float
        Absolute skewness upper bound (inclusive) for ``SkewSeverity.High``.
        Columns with ``|skew| > skew_moderate`` and ``|skew| <= skew_high``
        are classified as High. Columns above this bound are Severe.
    kurt_platykurtic_upper : float
        Excess kurtosis upper bound (exclusive) for ``KurtosisTag.Platykurtic``.
        Columns with ``kurtosis < kurt_platykurtic_upper`` are Platykurtic.
    kurt_leptokurtic_lower : float
        Excess kurtosis lower bound (exclusive) for ``KurtosisTag.Leptokurtic``.
        Columns with ``kurtosis > kurt_leptokurtic_lower`` are Leptokurtic.
        All others are Mesokurtic.
    near_constant_threshold : float
        Mode frequency above which a column receives ``NumericFlag.NearConstant``.
        Expressed as a fraction of total rows (e.g. 0.90 = 90%).
    scale_orders_of_magnitude : int
        Number of orders of magnitude the absolute value range must span for a
        column to receive ``NumericFlag.ScaleAnomaly`` (i.e. ratio >= 10^n).
    discrete_max_unique : int
        Maximum number of unique values for a non-integer column to be treated
        as discrete (top-value counts) rather than continuous (histogram).
    """

    skew_normal: float = 0.5
    skew_moderate: float = 1.0
    skew_high: float = 2.0
    kurt_platykurtic_upper: float = -1.0
    kurt_leptokurtic_lower: float = 3.0
    near_constant_threshold: float = 0.90
    scale_orders_of_magnitude: int = 3
    discrete_max_unique: int = 20

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "skew_normal": self.skew_normal,
            "skew_moderate": self.skew_moderate,
            "skew_high": self.skew_high,
            "kurt_platykurtic_upper": self.kurt_platykurtic_upper,
            "kurt_leptokurtic_lower": self.kurt_leptokurtic_lower,
            "near_constant_threshold": self.near_constant_threshold,
            "scale_orders_of_magnitude": self.scale_orders_of_magnitude,
            "discrete_max_unique": self.discrete_max_unique,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NumericProfileConfig:
        """
        Construct a ``NumericProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        NumericProfileConfig
            Reconstructed config instance.
        """
        return cls(
            skew_normal=float(data.get("skew_normal", 0.5)),
            skew_moderate=float(data.get("skew_moderate", 1.0)),
            skew_high=float(data.get("skew_high", 2.0)),
            kurt_platykurtic_upper=float(data.get("kurt_platykurtic_upper", -1.0)),
            kurt_leptokurtic_lower=float(data.get("kurt_leptokurtic_lower", 3.0)),
            near_constant_threshold=float(data.get("near_constant_threshold", 0.90)),
            scale_orders_of_magnitude=int(data.get("scale_orders_of_magnitude", 3)),
            discrete_max_unique=int(data.get("discrete_max_unique", 20)),
        )


@dataclass
class NumericProfileResult:
    """
    Numeric distribution profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, ColumnNumericProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, NumericStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Numeric Distribution Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)
