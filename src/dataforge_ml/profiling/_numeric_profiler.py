"""
NumericProfiler  –  Phase 1 extension: Numeric Distribution Profiling.

Per-column metrics (opt-in via ProfileConfig.numeric_columns):
  1. Central tendency     – mean, median, mean/median ratio
  2. Spread               – std, variance, IQR (Q3 – Q1)
  3. Skewness & kurtosis  – with severity/tag labels
  4. Range                – min, max
  5. Percentile profile   – p1, p5, p25, p50, p75, p95, p99
  6. Scale-anomaly flag   – values spanning 3+ orders of magnitude

Only numeric Polars dtypes are profiled; string columns in the list are
silently skipped (a warning is produced if the caller passes non-numeric
column names).

Integration
-----------
Add ``numeric_columns: list[str] | None`` to ProfileConfig, then call::

    from profiling.numeric_profiler import NumericProfiler

    num_profiler = NumericProfiler(
        columns=["age", "income", "temperature"],
        config=cfg,
    )
    num_result = num_profiler.profile(df)

Attach ``num_result`` to ``TabularProfileResult`` as
``result.numeric_profile``.
"""

from __future__ import annotations


import polars as pl

from ._base import ColumnBatchProfiler
from ._correlation_profiler import _INT_DTYPES
from ._numeric_config import (
    NumericProfileConfig,
    NumericProfileResult,
    NumericStats,
    PercentileSnapshot,
    KurtosisTag,
    TailAsymmetryTag,
    NumericFlag,
    SkewSeverity,
    NumericTopValueEntry,
    HistogramBin,
    BimodalStats,
)

# Percentile quantile levels — not a user-configurable threshold
_QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class NumericProfiler(ColumnBatchProfiler[NumericProfileResult]):
    """
    Numeric distribution profiler for Polars DataFrames.

    Profiles every column passed to profile(df, columns) — no internal
    eligibility gate.
    """

    def __init__(self, config: NumericProfileConfig | None = None) -> None:
        self._config = config if config is not None else NumericProfileConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str],
    ) -> NumericProfileResult:
        """
        Profile the specified numeric columns in a DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            The input Polars DataFrame containing the columns to profile.
        columns : list[str]
            A list of column names to profile. Non-numeric columns in this list
            are skipped.

        Returns
        -------
        NumericProfileResult
            A result object containing distribution statistics for the profiled columns.
        """
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(
        self,
        df: pl.DataFrame,
        columns: list[str],
    ) -> NumericProfileResult:
        result = NumericProfileResult()
        n_rows = df.height

        available = self._resolve_columns(df.columns, columns)
        result.analysed_columns = available

        if not available:
            return result

        # One df.select([...]) for all scalar stats across all columns so
        # Polars can parallelise expression evaluation rather than running
        # independent query plans per column.
        exprs: list[pl.Expr] = []
        for col in available:
            c = pl.col(col).cast(pl.Float64, strict=False)
            exprs.append(c.mean().alias(f"{col}__mean"))
            exprs.append(c.median().alias(f"{col}__median"))
            exprs.append(c.min().alias(f"{col}__min"))
            exprs.append(c.max().alias(f"{col}__max"))
            exprs.append(c.std(ddof=1).alias(f"{col}__std"))
            for q in _QUANTILE_LEVELS:
                exprs.append(
                    c.quantile(q, interpolation="linear").alias(f"{col}__q{q}")
                )

        batch = df.select(exprs).row(0, named=True)

        for col in available:
            series = df[col]
            f64 = series.cast(pl.Float64, strict=False)
            clean = f64.drop_nulls()
            profile = NumericStats()

            if clean.len() == 0:
                result.columns[col] = profile
                continue

            # Central tendency
            mean = float(batch[f"{col}__mean"])
            median = float(batch[f"{col}__median"])
            profile.mean = mean
            profile.median = median
            if median == 0.0:
                profile.mean_median_ratio = None if mean != 0.0 else 1.0
            else:
                profile.mean_median_ratio = mean / median

            # Range
            profile.min = float(batch[f"{col}__min"])
            profile.max = float(batch[f"{col}__max"])

            # Spread — Polars returns null for std with ddof=1 on a single row
            std_val = batch[f"{col}__std"]
            profile.std = float(std_val) if std_val is not None else 0.0
            profile.variance = profile.std**2

            # Percentiles
            q_vals = [batch[f"{col}__q{q}"] for q in _QUANTILE_LEVELS]
            profile.percentiles = PercentileSnapshot(
                p1=q_vals[0],
                p5=q_vals[1],
                p25=q_vals[2],
                p50=q_vals[3],
                p75=q_vals[4],
                p95=q_vals[5],
                p99=q_vals[6],
            )

            # Frequency / distribution stays per-column (returns a frame, not a scalar)
            self._compute_frequency_and_distribution(
                series, clean, profile, n_rows, self._config
            )

            # Shape stays per-column (delegates to scipy on a numpy array)
            self._compute_shape(clean, profile, self._config)

            self._check_scale_anomaly(profile, self._config)
            
            self._compute_tail_asymmetry(profile, self._config)

            self._compute_outlier_density(clean, profile, self._config)

            self._compute_bimodality(clean, profile, self._config)

            result.columns[col] = profile

        return result

    # ------------------------------------------------------------------
    # Per-column helpers (frequency/distribution and shape only —
    # scalar stats are now batched in _run above)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_frequency_and_distribution(
        original_series: pl.Series,
        clean_f64: pl.Series,
        profile: NumericStats,
        n_rows: int,
        config: NumericProfileConfig,
    ) -> None:
        """
        Compute Mode, and depending on whether the feature is continuous or discrete,
        calculate a 20-bin histogram OR Top-10 value counts.
        """
        if clean_f64.len() == 0:
            return

        vc = clean_f64.value_counts(sort=True)
        col_name = clean_f64.name

        # --- Absolute Mode Frequency ---
        mode_val = float(vc[col_name][0])
        mode_count = int(vc["count"][0])
        mode_freq = mode_count / n_rows if n_rows > 0 else 0.0

        profile.mode = mode_val
        profile.mode_frequency = mode_freq

        if mode_freq > config.near_constant_threshold:
            profile.flags.append(NumericFlag.NearConstant)

        n_unique = vc.height
        is_discrete = original_series.dtype in _INT_DTYPES

        if is_discrete:
            # --- Top-10 Distribution (Discrete) ---
            top_rows = min(10, n_unique)
            profile.top_values = [
                NumericTopValueEntry(
                    value=float(vc[col_name][i]),
                    count=int(vc["count"][i]),
                    percentage=int(vc["count"][i]) / n_rows if n_rows > 0 else 0.0,
                )
                for i in range(top_rows)
            ]
        else:
            # --- Histogram Distribution (Continuous) ---
            import numpy as np

            counts, bin_edges = np.histogram(clean_f64.to_numpy(), bins="auto")
            n_clean = clean_f64.len()
            profile.histogram = [
                HistogramBin(
                    lower_bound=float(bin_edges[i]),
                    upper_bound=float(bin_edges[i + 1]),
                    count=int(counts[i]),
                    percentage=int(counts[i]) / n_clean if n_clean > 0 else 0.0,
                )
                for i in range(len(counts))
            ]

    # ------------------------------------------------------------------
    # Step 2: Shape — skewness and kurtosis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_shape(
        clean: pl.Series,
        profile: NumericStats,
        config: NumericProfileConfig,
    ) -> None:
        from scipy.stats import skew, kurtosis as scipy_kurtosis

        if clean.len() < 3:
            return

        if profile.std is None or profile.std == 0.0:
            profile.skewness = 0.0
            profile.kurtosis = 0.0
            profile.skewness_severity = SkewSeverity.Normal
            profile.kurtosis_tag = KurtosisTag.Mesokurtic
            return

        arr = clean.to_numpy()
        profile.skewness = float(skew(arr, bias=False))
        profile.kurtosis = float(scipy_kurtosis(arr, bias=False))

        abs_skew = abs(profile.skewness)
        if abs_skew <= config.skew_normal:
            profile.skewness_severity = SkewSeverity.Normal
        elif abs_skew <= config.skew_moderate:
            profile.skewness_severity = SkewSeverity.Moderate
        elif abs_skew <= config.skew_high:
            profile.skewness_severity = SkewSeverity.High
        else:
            profile.skewness_severity = SkewSeverity.Severe

        if profile.kurtosis < config.kurt_platykurtic_upper:
            profile.kurtosis_tag = KurtosisTag.Platykurtic
        elif profile.kurtosis > config.kurt_leptokurtic_lower:
            profile.kurtosis_tag = KurtosisTag.Leptokurtic
        else:
            profile.kurtosis_tag = KurtosisTag.Mesokurtic

    # ------------------------------------------------------------------
    # Step 3: Scale-anomaly flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_scale_anomaly(
        profile: NumericStats,
        config: NumericProfileConfig,
    ) -> None:
        """
        Flag when values span ≥ N orders of magnitude *on the positive side*.

        Rationale: a column with values like [0.002, 15000] almost certainly
        mixes units or scales, which will mislead distance-based models.

        We use the absolute-value range to handle columns that cross zero
        (e.g. log-returns that go from -0.05 to 500).  Columns whose
        entire range is within [-1, 1] are exempt (percentages, probabilities).
        """
        col_min = profile.min
        col_max = profile.max

        if col_min is None or col_max is None:
            return

        abs_min = abs(col_min)
        abs_max = abs(col_max)

        # Skip all-zero or all-same-sign tiny ranges
        if abs_max == 0.0:
            return

        # Exempt probability / ratio columns
        if abs_max <= 1.0 and abs_min <= 1.0:
            return

        # Compute orders of magnitude
        if abs_min == 0.0:
            # Any non-zero max with a zero minimum → infinite ratio →
            # conservatively flag if max is large enough to be suspicious.
            if abs_max >= 10**config.scale_orders_of_magnitude:
                profile.flags.append(NumericFlag.ScaleAnomaly)
            return

        ratio = abs_max / abs_min
        if ratio >= 10**config.scale_orders_of_magnitude:
            profile.flags.append(NumericFlag.ScaleAnomaly)

    # ------------------------------------------------------------------
    # Step 4: Tail Asymmetry
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tail_asymmetry(
        profile: NumericStats,
        config: NumericProfileConfig,
    ) -> None:
        p1 = profile.percentiles.p1
        p5 = profile.percentiles.p5
        p95 = profile.percentiles.p95
        p99 = profile.percentiles.p99

        if p1 is None or p5 is None or p95 is None or p99 is None:
            return

        denominator = p5 - p1
        if denominator == 0.0:
            profile.tail_asymmetry_ratio = None
            profile.tail_asymmetry_tag = None
            return

        ratio = (p99 - p95) / denominator
        profile.tail_asymmetry_ratio = ratio

        if ratio > config.tail_asymmetry_right_threshold:
            profile.tail_asymmetry_tag = TailAsymmetryTag.RightHeavy
        elif ratio < config.tail_asymmetry_left_threshold:
            profile.tail_asymmetry_tag = TailAsymmetryTag.LeftHeavy
        else:
            profile.tail_asymmetry_tag = TailAsymmetryTag.Symmetric

    # ------------------------------------------------------------------
    # Step 5: Outlier Density
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_outlier_density(
        clean: pl.Series,
        profile: NumericStats,
        config: NumericProfileConfig,
    ) -> None:
        if profile.std is None or profile.std == 0.0 or profile.mean is None:
            profile.outlier_density = None
            return

        n_non_null = clean.len()
        if n_non_null == 0:
            return

        threshold = config.outlier_sigma_threshold * profile.std
        outliers = (clean - profile.mean).abs() > threshold
        outlier_count = outliers.sum()
        
        if outlier_count is None:
            outlier_count = 0
            
        density = outlier_count / n_non_null
        profile.outlier_density = float(density)
        
        if density > config.high_outlier_density_threshold:
            profile.flags.append(NumericFlag.HighOutlierDensity)

    # ------------------------------------------------------------------
    # Step 6: Bimodality Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bimodality(
        clean: pl.Series,
        profile: NumericStats,
        config: NumericProfileConfig,
    ) -> None:
        if profile.has_flag(NumericFlag.NearConstant):
            return

        n_non_null = clean.len()
        if n_non_null < 3:
            return

        import diptest
        from sklearn.mixture import GaussianMixture

        arr = clean.to_numpy()
        
        # Hartigan's Dip Test
        dip_stat, dip_p_value = diptest.diptest(arr)

        if dip_p_value < config.bimodal_dip_p_value_threshold:
            # Fit 2-component GMM
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(arr.reshape(-1, 1))
            
            centers = gmm.means_.flatten()
            centers.sort()
            
            profile.bimodal_stats = BimodalStats(
                dip_statistic=float(dip_stat),
                dip_p_value=float(dip_p_value),
                center1=float(centers[0]),
                center2=float(centers[1]),
            )
            profile.flags.append(NumericFlag.Bimodal)
