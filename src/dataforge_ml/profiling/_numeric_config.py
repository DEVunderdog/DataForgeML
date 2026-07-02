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
class BimodalStats:
    """Statistics and significance from Hartigan's Dip Test for bimodality.

    Parameters
    ----------
    dip_statistic : float
        The calculated dip statistic.
    dip_p_value : float
        The p-value of the dip test indicating significance.
    center1 : float
        The estimated location of the first mode.
    center2 : float
        The estimated location of the second mode.
    cluster_separation : float
        Ashman's D cluster separation metric (distance between cluster centers relative to dispersion).
    minority_weight : float
        Mixing weight of the smaller Gaussian component.
    """

    dip_statistic: float
    dip_p_value: float
    center1: float
    center2: float
    cluster_separation: float
    minority_weight: float

    def to_dict(self) -> dict:
        """Serialise the bimodal statistics to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "dip_statistic": self.dip_statistic,
            "dip_p_value": self.dip_p_value,
            "center1": self.center1,
            "center2": self.center2,
            "cluster_separation": self.cluster_separation,
            "minority_weight": self.minority_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BimodalStats":
        """Reconstruct a ``BimodalStats`` instance from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``.

        Returns
        -------
        BimodalStats
            Reconstructed instance.
        """
        return cls(
            dip_statistic=float(data["dip_statistic"]),
            dip_p_value=float(data["dip_p_value"]),
            center1=float(data["center1"]),
            center2=float(data["center2"]),
            cluster_separation=float(data["cluster_separation"]),
            minority_weight=float(data["minority_weight"]),
        )


class TailAsymmetryTag(StrEnum):
    """Characterises the asymmetry between the left and right tails."""

    Symmetric = "symmetric"
    RightHeavy = "right_heavy"
    LeftHeavy = "left_heavy"

@dataclass
class PercentileSnapshot:
    """Snapshot of a numeric column's distribution at key percentile positions.

    Attributes
    ----------
    p1 : float, optional
        1st percentile value.
    p5 : float, optional
        5th percentile value.
    p25 : float, optional
        25th percentile value.
    p50 : float, optional
        50th percentile value (median).
    p75 : float, optional
        75th percentile value.
    p95 : float, optional
        95th percentile value.
    p99 : float, optional
        99th percentile value.
    """

    p1: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    @property
    def iqr(self) -> Optional[float]:
        """Compute the interquartile range from this snapshot.

        Returns
        -------
        float or None
            Difference between ``p75`` and ``p25``, or ``None`` if either is unset.
        """
        if self.p25 is not None and self.p75 is not None:
            return self.p75 - self.p25
        return None

    def to_dict(self) -> dict:
        """Serialise the percentile snapshot to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
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


class NonlinearityTag(StrEnum):
    """Characterises the dominant relationship between a numeric column and its predictors."""

    Linear = "linear"
    MonotonicNonlinear = "monotonic_nonlinear"
    ComplexNonlinear = "complex_nonlinear"
    Unpredictable = "unpredictable"


class NumericFlag(StrEnum):
    ScaleAnomaly = "scale_anomaly"
    NearConstant = "near_constant"
    Bimodal = "bimodal"
    HighOutlierDensity = "high_outlier_density"
    FormatMismatch = "format_mismatch"


@dataclass
class NumericTopValueEntry:
    """A single entry in the top-value frequency list for a numeric column.

    Attributes
    ----------
    value : float
        The numeric value.
    count : int
        Number of rows containing this value.
    percentage : float
        Fraction of total rows containing this value (range [0, 1]).
    """

    value: float
    count: int
    percentage: float

    def to_dict(self) -> dict:
        """Serialise this entry to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {"value": self.value, "count": self.count, "percentage": self.percentage}


@dataclass
class HistogramBin:
    """A single bin in an equi-width histogram for a numeric column.

    Attributes
    ----------
    lower_bound : float
        Inclusive lower boundary of this bin.
    upper_bound : float
        Exclusive upper boundary of this bin.
    count : int
        Number of rows whose value falls within this bin.
    percentage : float
        Fraction of total rows within this bin (range [0, 1]).
    """

    lower_bound: float
    upper_bound: float
    count: int
    percentage: float

    def to_dict(self) -> dict:
        """Serialise this bin to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "count": self.count, "percentage": self.percentage,
        }


@dataclass
class NumericStats:
    """Computed statistics and flags for a numeric column.

    Parameters
    ----------
    mean : float, optional
        Arithmetic mean of the non-null values.
    median : float, optional
        Median value of the non-null values.
    mean_median_ratio : float, optional
        Ratio of the mean to the median, indicating skew or asymmetry.
    mode : float, optional
        Most frequent value in the column.
    mode_frequency : float, default 0.0
        Proportion of rows matching the mode value (range [0, 1]).
    top_values : list of NumericTopValueEntry, default factory=list
        Frequent value entries sorted by frequency.
    histogram : list of HistogramBin, default factory=list
        Equi-width histogram bins.
    std : float, optional
        Standard deviation of the non-null values.
    variance : float, optional
        Variance of the non-null values.
    min : float, optional
        Minimum value in the column.
    max : float, optional
        Maximum value in the column.
    percentiles : PercentileSnapshot, default factory=PercentileSnapshot
        Snapshot of values at key percentiles.
    skewness : float, optional
        Fisher-Pearson standardized skewness coefficient.
    kurtosis : float, optional
        Excess kurtosis of the non-null values.
    skewness_severity : SkewSeverity, optional
        Categorized classification of the skewness severity.
    kurtosis_tag : KurtosisTag, optional
        Categorized tag for the kurtosis shape.
    flags : list of NumericFlag, default factory=list
        Flags identifying anomalies like NearConstant or ScaleAnomaly.
    nonlinearity_tag : NonlinearityTag, optional
        Detected relationship tag between this column and its predictors.
    spearman_pearson_discrepancy : float, optional
        Max Spearman-Pearson correlation difference across predictors.
    mean_mutual_information : float, optional
        Mean mutual information across all predictor columns.
    r2_gap : float, optional
        Difference between RandomForest and Linear Regression cross-validated R2.
    heteroscedasticity_p_value : float, optional
        P-value for Breusch-Pagan heteroscedasticity test on linear residuals.
    bimodal_stats : BimodalStats, optional
        Statistics related to bimodality detected by Hartigan's Dip Test.
    tail_asymmetry_tag : TailAsymmetryTag, optional
        Categorized tag for the tail asymmetry of the distribution.
    tail_asymmetry_share : float, optional
        Bounded share of the right tail spread against total extreme tail spread, range [0, 1].
    outlier_density : float, optional
        Proportion of values considered as extreme outliers.
    """

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
    nonlinearity_tag: Optional[NonlinearityTag] = None
    spearman_pearson_discrepancy: Optional[float] = None
    mean_mutual_information: Optional[float] = None
    r2_gap: Optional[float] = None
    heteroscedasticity_p_value: Optional[float] = None
    bimodal_stats: Optional[BimodalStats] = None
    tail_asymmetry_tag: Optional[TailAsymmetryTag] = None
    tail_asymmetry_share: Optional[float] = None
    outlier_density: Optional[float] = None

    @property
    def iqr(self) -> Optional[float]:
        """Compute the interquartile range (IQR) for the column.

        Returns
        -------
        float or None
            Difference between the 75th and 25th percentiles, or None if unset.
        """
        return self.percentiles.iqr

    def has_flag(self, flag: NumericFlag) -> bool:
        """Check if a specific numeric anomaly flag is present.

        Parameters
        ----------
        flag : NumericFlag
            The anomaly flag to check.

        Returns
        -------
        bool
            True if the flag is present, False otherwise.
        """
        return flag in self.flags

    def to_dict(self) -> dict:
        """Serialise the numeric statistics to a plain dictionary.

        Returns
        -------
        dict
            All field values serialized to plain types or nested dicts.
        """
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
            "nonlinearity_tag": str(self.nonlinearity_tag) if self.nonlinearity_tag else None,
            "spearman_pearson_discrepancy": self.spearman_pearson_discrepancy,
            "mean_mutual_information": self.mean_mutual_information,
            "r2_gap": self.r2_gap,
            "heteroscedasticity_p_value": self.heteroscedasticity_p_value,
            "bimodal_stats": self.bimodal_stats.to_dict() if self.bimodal_stats else None,
            "tail_asymmetry_tag": str(self.tail_asymmetry_tag) if self.tail_asymmetry_tag else None,
            "tail_asymmetry_share": self.tail_asymmetry_share,
            "outlier_density": self.outlier_density,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumericStats":
        """Reconstruct the numeric statistics from a dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``.

        Returns
        -------
        NumericStats
            Reconstructed instance.
        """
        return cls(
            mean=data.get("mean"),
            median=data.get("median"),
            mean_median_ratio=data.get("mean_median_ratio"),
            mode=data.get("mode"),
            mode_frequency=data.get("mode_frequency", 0.0),
            top_values=[NumericTopValueEntry(**v) for v in data.get("top_values", [])],
            histogram=[HistogramBin(**b) for b in data.get("histogram", [])],
            std=data.get("std"),
            variance=data.get("variance"),
            min=data.get("min"),
            max=data.get("max"),
            percentiles=PercentileSnapshot(**data.get("percentiles", {})),
            skewness=data.get("skewness"),
            kurtosis=data.get("kurtosis"),
            skewness_severity=SkewSeverity(data["skewness_severity"]) if data.get("skewness_severity") else None,
            kurtosis_tag=KurtosisTag(data["kurtosis_tag"]) if data.get("kurtosis_tag") else None,
            flags=[NumericFlag(f) for f in data.get("flags", [])],
            nonlinearity_tag=NonlinearityTag(data["nonlinearity_tag"]) if data.get("nonlinearity_tag") else None,
            spearman_pearson_discrepancy=data.get("spearman_pearson_discrepancy"),
            mean_mutual_information=data.get("mean_mutual_information"),
            r2_gap=data.get("r2_gap"),
            heteroscedasticity_p_value=data.get("heteroscedasticity_p_value"),
            bimodal_stats=BimodalStats.from_dict(data["bimodal_stats"]) if data.get("bimodal_stats") else None,
            tail_asymmetry_tag=TailAsymmetryTag(data["tail_asymmetry_tag"]) if data.get("tail_asymmetry_tag") else None,
            tail_asymmetry_share=data.get("tail_asymmetry_share"),
            outlier_density=data.get("outlier_density"),
        )


ColumnNumericProfile = NumericStats


# ---------------------------------------------------------------------------
# Nonlinearity sub-config and result types
# ---------------------------------------------------------------------------


@dataclass
class NonlinearityProfileConfig:
    """
    Threshold configuration for the NonlinearityProfiler sub-processor.

    Defaults are calibrated for typical tabular datasets (hundreds to thousands
    of rows):

    - A Spearman/Pearson discrepancy ≥ 0.10 reliably separates monotonic
      non-linear relationships from linear ones across common real-world data.
    - p < 0.05 is the standard significance level for the Breusch-Pagan test.
    - An R² gap ≥ 0.10 represents a meaningful improvement of a non-linear model
      over a linear baseline (calibrated against held-out evaluation).
    - R²_RF < 0.05 (cross-validated on a bootstrap sample) indicates the column
      is not meaningfully predictable from its numeric co-variates.

    Parameters
    ----------
    spearman_pearson_discrepancy_threshold : float
        Minimum max-over-predictors ``|Spearman_r − Pearson_r|`` that triggers
        the monotonic-nonlinear signal.  Values in [0, 1]; default 0.10.
    mutual_information_threshold : float
        Minimum mean mutual information (nats) across predictors that contributes
        to a ``ComplexNonlinear`` classification.  Default 0.05.
    r2_gap_threshold : float
        Minimum ``R²_RF − R²_linear`` (both cross-validated on a bootstrap
        sample) to classify a column as ``ComplexNonlinear`` rather than
        ``MonotonicNonlinear``.  Default 0.10.
    heteroscedasticity_p_value_threshold : float
        Breusch-Pagan p-value below which heteroscedasticity is detected,
        contributing to a ``MonotonicNonlinear`` classification.  Default 0.05.
    r2_rf_unpredictable_threshold : float
        Cross-validated ``R²_RF`` below which a column is tagged
        ``Unpredictable``.  Default 0.05.
    min_rows : int
        Minimum number of complete (non-null) rows required across the target
        column and all its predictors to run the profiler for that column.
        Default 20.
    bootstrap_sample_size : int
        Maximum number of rows sampled for the R² gap and Breusch-Pagan
        computations.  Larger datasets are down-sampled uniformly at random.
        Default 500.
    random_state : int
        Random seed used for row sampling and ``RandomForestRegressor``.
        Default 42.
    """

    spearman_pearson_discrepancy_threshold: float = 0.10
    mutual_information_threshold: float = 0.05
    r2_gap_threshold: float = 0.10
    heteroscedasticity_p_value_threshold: float = 0.05
    r2_rf_unpredictable_threshold: float = 0.05
    min_rows: int = 20
    bootstrap_sample_size: int = 500
    random_state: int = 42

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "spearman_pearson_discrepancy_threshold": self.spearman_pearson_discrepancy_threshold,
            "mutual_information_threshold": self.mutual_information_threshold,
            "r2_gap_threshold": self.r2_gap_threshold,
            "heteroscedasticity_p_value_threshold": self.heteroscedasticity_p_value_threshold,
            "r2_rf_unpredictable_threshold": self.r2_rf_unpredictable_threshold,
            "min_rows": self.min_rows,
            "bootstrap_sample_size": self.bootstrap_sample_size,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NonlinearityProfileConfig":
        """
        Construct a ``NonlinearityProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        NonlinearityProfileConfig
            Reconstructed config instance.
        """
        return cls(
            spearman_pearson_discrepancy_threshold=float(
                data.get("spearman_pearson_discrepancy_threshold", 0.10)
            ),
            mutual_information_threshold=float(
                data.get("mutual_information_threshold", 0.05)
            ),
            r2_gap_threshold=float(data.get("r2_gap_threshold", 0.10)),
            heteroscedasticity_p_value_threshold=float(
                data.get("heteroscedasticity_p_value_threshold", 0.05)
            ),
            r2_rf_unpredictable_threshold=float(
                data.get("r2_rf_unpredictable_threshold", 0.05)
            ),
            min_rows=int(data.get("min_rows", 20)),
            bootstrap_sample_size=int(data.get("bootstrap_sample_size", 500)),
            random_state=int(data.get("random_state", 42)),
        )


@dataclass
class NonlinearitySignals:
    """
    Raw nonlinearity signal values and assigned tag for one numeric column.

    Returned inside ``NonlinearityProfileResult`` and consumed by
    ``StructuralProfiler`` to populate ``NumericStats`` nonlinearity fields.

    Attributes
    ----------
    tag : NonlinearityTag
        Assigned nonlinearity classification.
    spearman_pearson_discrepancy : float
        Max ``|Spearman_r − Pearson_r|`` over all predictors.
    mean_mutual_information : float
        Mean ``mutual_info_regression`` score across all predictors.
    r2_gap : float
        Cross-validated ``R²_RF − R²_linear`` on a bootstrap sample.
    heteroscedasticity_p_value : float
        Breusch-Pagan p-value from the linear model residuals.
    """

    tag: NonlinearityTag
    spearman_pearson_discrepancy: float
    mean_mutual_information: float
    r2_gap: float
    heteroscedasticity_p_value: float


@dataclass
class NonlinearityProfileResult:
    """
    Per-column nonlinearity signals returned by ``NonlinearityProfiler``.

    Attributes
    ----------
    columns : dict[str, NonlinearitySignals]
        Signal values and tag for each profiled column.
    analysed_columns : list[str]
        Columns for which signals were successfully computed.
    """

    columns: dict[str, NonlinearitySignals] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)


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
    bimodal_dip_p_value_threshold : float
        P-value threshold for Hartigan's Dip Test to classify as bimodal. Default 0.05.
    bimodal_min_separation_threshold : float
        Ashman's D separation threshold above which bimodality is confirmed. Default 2.0.
    bimodal_min_component_weight : float
        Minimum mixing weight for the smaller component to confirm bimodality. Default 0.05.
    tail_asymmetry_right_share_threshold : float
        Threshold for tail asymmetry share above which the distribution is RightHeavy. Default 2/3.
    tail_asymmetry_left_share_threshold : float
        Threshold for tail asymmetry share below which the distribution is LeftHeavy. Default 1/3.
    outlier_sigma_threshold : float
        Sigma threshold for outlier detection. Default 3.0.
    high_outlier_density_threshold : float
        Threshold for outlier density above which HighOutlierDensity flag is raised. Default 0.05.
    """

    skew_normal: float = 0.5
    skew_moderate: float = 1.0
    skew_high: float = 2.0
    kurt_platykurtic_upper: float = -1.0
    kurt_leptokurtic_lower: float = 3.0
    near_constant_threshold: float = 0.90
    scale_orders_of_magnitude: int = 3
    bimodal_dip_p_value_threshold: float = 0.05
    bimodal_min_separation_threshold: float = 2.0
    bimodal_min_component_weight: float = 0.05
    tail_asymmetry_right_share_threshold: float = 2.0 / 3.0
    tail_asymmetry_left_share_threshold: float = 1.0 / 3.0
    outlier_sigma_threshold: float = 3.0
    high_outlier_density_threshold: float = 0.05

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
            "bimodal_dip_p_value_threshold": self.bimodal_dip_p_value_threshold,
            "bimodal_min_separation_threshold": self.bimodal_min_separation_threshold,
            "bimodal_min_component_weight": self.bimodal_min_component_weight,
            "tail_asymmetry_right_share_threshold": self.tail_asymmetry_right_share_threshold,
            "tail_asymmetry_left_share_threshold": self.tail_asymmetry_left_share_threshold,
            "outlier_sigma_threshold": self.outlier_sigma_threshold,
            "high_outlier_density_threshold": self.high_outlier_density_threshold,
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
            bimodal_dip_p_value_threshold=float(data.get("bimodal_dip_p_value_threshold", 0.05)),
            bimodal_min_separation_threshold=float(data.get("bimodal_min_separation_threshold", 2.0)),
            bimodal_min_component_weight=float(data.get("bimodal_min_component_weight", 0.05)),
            tail_asymmetry_right_share_threshold=float(data.get("tail_asymmetry_right_share_threshold", 2.0 / 3.0)),
            tail_asymmetry_left_share_threshold=float(data.get("tail_asymmetry_left_share_threshold", 1.0 / 3.0)),
            outlier_sigma_threshold=float(data.get("outlier_sigma_threshold", 3.0)),
            high_outlier_density_threshold=float(data.get("high_outlier_density_threshold", 0.05)),
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
