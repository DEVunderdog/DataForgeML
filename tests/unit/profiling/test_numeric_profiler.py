import polars as pl
import pytest

from dataforge_ml.profiling._numeric_profiler import NumericProfiler
from dataforge_ml.profiling._numeric_config import (
    KurtosisTag,
    NumericFlag,
    NumericProfileConfig,
    NumericProfileResult,
    NumericStats,
    PercentileSnapshot,
    SkewSeverity,
    TailAsymmetryTag,
)


# ---------------------------------------------------------------------------
# Result type & column eligibility
# ---------------------------------------------------------------------------


def test_result_type(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score"])
    assert isinstance(result, NumericProfileResult)


def test_analysed_columns_only_eligible(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score", "salary"])
    assert "score" in result.analysed_columns
    assert "salary" in result.analysed_columns


def test_analysed_columns_matches_columns_dict(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score", "salary"])
    assert set(result.analysed_columns) == set(result.columns.keys())


# ---------------------------------------------------------------------------
# Core stats present for a normal float column
# ---------------------------------------------------------------------------


def test_core_stats_non_null_for_float(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    assert stats.mean is not None
    assert stats.median is not None
    assert stats.std is not None
    assert stats.min is not None
    assert stats.max is not None
    assert stats.mean_median_ratio is not None


def test_min_lte_max(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    assert stats.min <= stats.max


# ---------------------------------------------------------------------------
# All-null column
# ---------------------------------------------------------------------------


def test_all_null_column_no_crash(all_null_df):
    result = NumericProfiler().profile(all_null_df, ["float_col"])
    assert "float_col" in result.analysed_columns
    stats = result.columns["float_col"]
    assert isinstance(stats, NumericStats)
    assert stats.mean is None
    assert stats.std is None
    assert stats.min is None
    assert stats.max is None


# ---------------------------------------------------------------------------
# Single-value column
# ---------------------------------------------------------------------------


def test_single_value_std_and_skewness_zero(single_value_df):
    stats = NumericProfiler().profile(single_value_df, ["score"]).columns["score"]
    assert stats.std == 0.0
    assert stats.skewness == 0.0


# ---------------------------------------------------------------------------
# ScaleAnomaly flag
# ---------------------------------------------------------------------------


def test_scale_anomaly_flag_set():
    # 0.5 to 5000 → ratio = 10 000 ≥ 10^3 → flag
    df = pl.DataFrame({"v": pl.Series([0.5, 1.0, 1.5, 2.0, 5000.0] * 12, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.ScaleAnomaly in stats.flags


def test_scale_anomaly_flag_absent_normal_range():
    df = pl.DataFrame({"v": pl.Series([10.0, 20.0, 30.0, 40.0, 50.0] * 12, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.ScaleAnomaly not in stats.flags


# ---------------------------------------------------------------------------
# NearConstant flag
# ---------------------------------------------------------------------------


def test_near_constant_flag_set():
    # 55/60 = 0.917 > 0.90 → flag
    data = [5.0] * 55 + [1.0, 2.0, 3.0, 4.0, 6.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant in stats.flags


def test_near_constant_flag_absent():
    # 30/60 = 0.50 ≤ 0.90 → no flag
    data = [5.0] * 30 + [6.0] * 30
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant not in stats.flags


# ---------------------------------------------------------------------------
# Skewness severity bands
# ---------------------------------------------------------------------------


def test_skewness_severity_normal():
    # Symmetric uniform 1–60 → |skew| ≈ 0 → Normal
    data = [float(i) for i in range(1, 61)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.skewness_severity == SkewSeverity.Normal


def test_skewness_severity_severe():
    # 57 near-zero values + 3 extreme values → |skew| >> 2.0 → Severe
    data = [0.1] * 57 + [100.0, 200.0, 300.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.skewness_severity == SkewSeverity.Severe


# ---------------------------------------------------------------------------
# Kurtosis tag bands
# ---------------------------------------------------------------------------


def test_kurtosis_tag_leptokurtic():
    # Mass concentrated at 5.0 with symmetric outliers → excess kurtosis >> 3.0
    data = [5.0] * 54 + [0.1, 0.1, 0.1, 9.9, 9.9, 9.9]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Leptokurtic


def test_kurtosis_tag_platykurtic():
    # Uniform over 4 equally-spaced values → excess kurtosis < -1.0
    data = [1.0] * 15 + [4.0] * 15 + [7.0] * 15 + [10.0] * 15
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Platykurtic


def test_kurtosis_tag_mesokurtic():
    # Bell-curve approximation (discrete triangular) → excess kurtosis in (-1, 3)
    data = [1.0]*3 + [2.0]*7 + [3.0]*12 + [4.0]*16 + [5.0]*12 + [6.0]*7 + [7.0]*3
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Mesokurtic


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentiles_type_and_all_fields_present(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    p = stats.percentiles
    assert isinstance(p, PercentileSnapshot)
    for val in (p.p1, p.p5, p.p25, p.p50, p.p75, p.p95, p.p99):
        assert val is not None


def test_percentiles_monotonically_non_decreasing(normal_mixed_df):
    p = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"].percentiles
    vals = [p.p1, p.p5, p.p25, p.p50, p.p75, p.p95, p.p99]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# Discrete vs continuous distribution representation
# ---------------------------------------------------------------------------


def test_integer_column_produces_top_values():
    # Int64 dtype always triggers the discrete path
    data = [i % 5 for i in range(60)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Int64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert len(stats.top_values) > 0
    assert len(stats.histogram) == 0


def test_continuous_float_produces_histogram():
    # 60 distinct floats → n_unique > _DISCRETE_MAX_UNIQUE (20) → continuous path
    data = [round(i * 0.37, 4) for i in range(60)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert len(stats.histogram) > 0
    assert len(stats.top_values) == 0


def test_histogram_bin_count_adapts_to_data():
    # bins='auto' adapts to data size and spread rather than being fixed at 20.
    # Small dataset → fewer bins (Sturges gives ~6 for n=50).
    # Large dataset → more bins than the small one.
    import numpy as np
    rng = np.random.default_rng(42)

    small = list(rng.normal(0, 1, size=50))
    df_small = pl.DataFrame({"v": pl.Series(small, dtype=pl.Float64)})
    stats_small = NumericProfiler().profile(df_small, ["v"]).columns["v"]
    assert len(stats_small.histogram) < 20

    large = list(rng.normal(0, 1, size=500))
    df_large = pl.DataFrame({"v": pl.Series(large, dtype=pl.Float64)})
    stats_large = NumericProfiler().profile(df_large, ["v"]).columns["v"]
    assert len(stats_large.histogram) > len(stats_small.histogram)


# ---------------------------------------------------------------------------
# Batched scalar stat correctness (issue #53)
# ---------------------------------------------------------------------------


def test_batched_stats_match_single_column_baseline():
    # Profile 10 columns at once; each stat must equal the value produced when
    # profiling that same column in isolation.
    import numpy as np
    rng = np.random.default_rng(0)
    n_cols = 10
    col_names = [f"col_{i}" for i in range(n_cols)]
    data = {c: rng.normal(loc=i, scale=1.0 + i * 0.1, size=200).tolist() for i, c in enumerate(col_names)}
    df = pl.DataFrame({c: pl.Series(v, dtype=pl.Float64) for c, v in data.items()})

    batched = NumericProfiler().profile(df, col_names)

    for col in col_names:
        single = NumericProfiler().profile(df, [col]).columns[col]
        batch_col = batched.columns[col]

        assert pytest.approx(batch_col.mean, rel=1e-9) == single.mean
        assert pytest.approx(batch_col.median, rel=1e-9) == single.median
        assert pytest.approx(batch_col.std, rel=1e-9) == single.std
        assert pytest.approx(batch_col.min, rel=1e-9) == single.min
        assert pytest.approx(batch_col.max, rel=1e-9) == single.max

        bp, sp = batch_col.percentiles, single.percentiles
        for attr in ("p1", "p5", "p25", "p50", "p75", "p95", "p99"):
            assert pytest.approx(getattr(bp, attr), rel=1e-9) == getattr(sp, attr)


def test_batched_50_column_profiling_speed():
    # Regression guard: 50-column profiling should complete well under 10s on
    # any reasonable machine, confirming the single-select path is active.
    import time
    import numpy as np
    rng = np.random.default_rng(1)
    n_cols = 50
    col_names = [f"feat_{i}" for i in range(n_cols)]
    df = pl.DataFrame(
        {c: pl.Series(rng.normal(size=1_000).tolist(), dtype=pl.Float64) for c in col_names}
    )

    t0 = time.perf_counter()
    result = NumericProfiler().profile(df, col_names)
    elapsed = time.perf_counter() - t0

    assert len(result.analysed_columns) == n_cols
    assert elapsed < 10.0, f"50-column profiling took {elapsed:.2f}s — batching may be broken"


# ---------------------------------------------------------------------------
# NumericProfileConfig — construction and defaults
# ---------------------------------------------------------------------------


def test_config_instantiates_with_defaults():
    cfg = NumericProfileConfig()
    assert cfg.skew_normal == 0.5
    assert cfg.skew_moderate == 1.0
    assert cfg.skew_high == 2.0
    assert cfg.kurt_platykurtic_upper == -1.0
    assert cfg.kurt_leptokurtic_lower == 3.0
    assert cfg.near_constant_threshold == 0.90
    assert cfg.scale_orders_of_magnitude == 3


def test_profiler_accepts_config_parameter():
    cfg = NumericProfileConfig(skew_high=3.0)
    profiler = NumericProfiler(config=cfg)
    assert profiler is not None


def test_profiler_constructed_without_config_uses_defaults():
    profiler = NumericProfiler()
    assert profiler._config.skew_high == 2.0
    assert profiler._config.near_constant_threshold == 0.90


# ---------------------------------------------------------------------------
# skew_high override
# ---------------------------------------------------------------------------


def test_skew_high_override_relabels_severe_to_high():
    # Data with |skew| >> 2.0 → Severe by default.
    # Raising skew_high to a very large value → same data becomes High
    # (|skew| > skew_moderate=1.0 and |skew| <= new skew_high).
    data = [0.1] * 57 + [100.0, 200.0, 300.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})

    default_stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert default_stats.skewness_severity == SkewSeverity.Severe

    cfg = NumericProfileConfig(skew_high=1_000.0)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert overridden_stats.skewness_severity == SkewSeverity.High


def test_skew_moderate_override_relabels_high_to_moderate():
    # Data with |skew| >> 1.0 but close to 1.0; default → High.
    # Raising skew_moderate to a very large value → Moderate.
    data = [0.1] * 57 + [100.0, 200.0, 300.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})

    cfg = NumericProfileConfig(skew_moderate=1_000.0, skew_high=10_000.0)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert overridden_stats.skewness_severity == SkewSeverity.Moderate


# ---------------------------------------------------------------------------
# kurt_leptokurtic_lower override
# ---------------------------------------------------------------------------


def test_kurt_leptokurtic_lower_override_suppresses_leptokurtic():
    # Data known to be Leptokurtic by default (kurtosis >> 3.0).
    # Raising the lower bound to an impossibly large value → Mesokurtic.
    data = [5.0] * 54 + [0.1, 0.1, 0.1, 9.9, 9.9, 9.9]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})

    default_stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert default_stats.kurtosis_tag == KurtosisTag.Leptokurtic

    cfg = NumericProfileConfig(kurt_leptokurtic_lower=1_000.0)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert overridden_stats.kurtosis_tag == KurtosisTag.Mesokurtic


def test_kurt_platykurtic_upper_override_suppresses_platykurtic():
    # Uniform 4-value distribution → Platykurtic by default.
    # Lowering upper bound to -1000.0 (impossible to go below) → Mesokurtic.
    data = [1.0] * 15 + [4.0] * 15 + [7.0] * 15 + [10.0] * 15
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})

    default_stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert default_stats.kurtosis_tag == KurtosisTag.Platykurtic

    cfg = NumericProfileConfig(kurt_platykurtic_upper=-1_000.0)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert overridden_stats.kurtosis_tag == KurtosisTag.Mesokurtic


# ---------------------------------------------------------------------------
# near_constant_threshold override
# ---------------------------------------------------------------------------


def test_near_constant_threshold_override_triggers_flag():
    # 85/100 = 0.85 → default 0.90: NOT NearConstant; override 0.80: NearConstant
    data = [5.0] * 85 + list(range(15))
    df = pl.DataFrame({"v": pl.Series([float(x) for x in data], dtype=pl.Float64)})

    default_stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant not in default_stats.flags

    cfg = NumericProfileConfig(near_constant_threshold=0.80)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant in overridden_stats.flags


def test_near_constant_threshold_override_suppresses_flag():
    # 55/60 = 0.917 → default 0.90: NearConstant; override 0.95: NOT NearConstant
    data = [5.0] * 55 + [1.0, 2.0, 3.0, 4.0, 6.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})

    default_stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant in default_stats.flags

    cfg = NumericProfileConfig(near_constant_threshold=0.95)
    overridden_stats = NumericProfiler(config=cfg).profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant not in overridden_stats.flags


# ---------------------------------------------------------------------------
# NumericProfileConfig serialisation round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip_preserves_all_fields():
    cfg = NumericProfileConfig(
        skew_normal=0.3,
        skew_moderate=0.8,
        skew_high=1.5,
        kurt_platykurtic_upper=-2.0,
        kurt_leptokurtic_lower=5.0,
        bimodal_dip_p_value_threshold=0.01,
        bimodal_min_separation_threshold=3.0,
        bimodal_min_component_weight=0.1,
        near_constant_threshold=0.85,
        scale_orders_of_magnitude=4,
    )
    restored = NumericProfileConfig.from_dict(cfg.to_dict())
    assert restored.skew_normal == cfg.skew_normal
    assert restored.skew_moderate == cfg.skew_moderate
    assert restored.skew_high == cfg.skew_high
    assert restored.kurt_platykurtic_upper == cfg.kurt_platykurtic_upper
    assert restored.kurt_leptokurtic_lower == cfg.kurt_leptokurtic_lower
    assert restored.near_constant_threshold == cfg.near_constant_threshold
    assert restored.scale_orders_of_magnitude == cfg.scale_orders_of_magnitude
    assert restored.bimodal_min_separation_threshold == cfg.bimodal_min_separation_threshold
    assert restored.bimodal_min_component_weight == cfg.bimodal_min_component_weight


def test_config_from_dict_uses_defaults_for_missing_keys():
    restored = NumericProfileConfig.from_dict({})
    default = NumericProfileConfig()
    assert restored.skew_normal == default.skew_normal
    assert restored.skew_high == default.skew_high
    assert restored.kurt_leptokurtic_lower == default.kurt_leptokurtic_lower
    assert restored.near_constant_threshold == default.near_constant_threshold
    assert restored.scale_orders_of_magnitude == default.scale_orders_of_magnitude
    assert restored.bimodal_min_separation_threshold == default.bimodal_min_separation_threshold
    assert restored.bimodal_min_component_weight == default.bimodal_min_component_weight


def test_config_to_dict_contains_all_keys():
    cfg = NumericProfileConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "skew_normal",
        "skew_moderate",
        "skew_high",
        "kurt_platykurtic_upper",
        "kurt_leptokurtic_lower",
        "near_constant_threshold",
        "scale_orders_of_magnitude",
        "bimodal_dip_p_value_threshold",
        "bimodal_min_separation_threshold",
        "bimodal_min_component_weight",
        "tail_asymmetry_right_share_threshold",
        "tail_asymmetry_left_share_threshold",
        "outlier_sigma_threshold",
        "high_outlier_density_threshold",
    }


# ---------------------------------------------------------------------------
# Tail Asymmetry
# ---------------------------------------------------------------------------


def test_tail_asymmetry_symmetric():
    data = list(range(1, 101))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.tail_asymmetry_tag == TailAsymmetryTag.Symmetric
    assert stats.tail_asymmetry_share is not None
    assert 1.0 / 3.0 <= stats.tail_asymmetry_share <= 2.0 / 3.0


def test_tail_asymmetry_right_heavy():
    # numerator = 1000 - 95 = 905, left_band = 5 - 1 = 4
    # denominator_sum = 909
    # share = 905 / 909 = 0.995... > 2/3
    data = list(range(1, 96)) + [100, 200, 500, 800, 1000]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.tail_asymmetry_tag == TailAsymmetryTag.RightHeavy
    assert stats.tail_asymmetry_share > 2.0 / 3.0


def test_tail_asymmetry_left_heavy():
    # Left tail spreads much more than right tail
    data = [-1000, -500, -200, -100, -50] + list(range(-40, 55))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.tail_asymmetry_tag == TailAsymmetryTag.LeftHeavy
    assert stats.tail_asymmetry_share < 1.0 / 3.0


def test_tail_asymmetry_flat_left_tail():
    # p5 == p1 => left band is 0, right band is non-zero
    # Therefore denominator_sum = numerator
    # share = numerator / numerator = 1.0
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + list(range(1, 91))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.percentiles.p5 == stats.percentiles.p1
    assert stats.tail_asymmetry_share == 1.0
    assert stats.tail_asymmetry_tag == TailAsymmetryTag.RightHeavy

def test_tail_asymmetry_true_zero():
    # p5 == p1 and p99 == p95 => both bands flat
    data = [0] * 10 + list(range(10, 90)) + [100] * 10
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.percentiles.p5 == stats.percentiles.p1
    assert stats.percentiles.p99 == stats.percentiles.p95
    assert stats.tail_asymmetry_share is None
    assert stats.tail_asymmetry_tag is None


def test_tail_asymmetry_configurable_thresholds():
    data_right = list(range(1, 96)) + [100, 200, 500, 800, 1000]
    df_right = pl.DataFrame({"v": pl.Series(data_right, dtype=pl.Float64)})
    cfg1 = NumericProfileConfig(tail_asymmetry_right_share_threshold=1.0)
    stats_right = NumericProfiler(config=cfg1).profile(df_right, ["v"]).columns["v"]
    assert stats_right.tail_asymmetry_tag == TailAsymmetryTag.Symmetric

    data_left = [-1000, -500, -200, -100, -50] + list(range(-40, 55))
    df_left = pl.DataFrame({"v": pl.Series(data_left, dtype=pl.Float64)})
    cfg2 = NumericProfileConfig(tail_asymmetry_left_share_threshold=-0.1)
    stats_left = NumericProfiler(config=cfg2).profile(df_left, ["v"]).columns["v"]
    assert stats_left.tail_asymmetry_tag == TailAsymmetryTag.Symmetric

# ---------------------------------------------------------------------------
# Bimodality Detection
# ---------------------------------------------------------------------------


def test_bimodality_detected():
    import numpy as np
    rng = np.random.default_rng(42)
    # Two clearly separated peaks
    data = list(rng.normal(0, 1, 100)) + list(rng.normal(10, 1, 100))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert NumericFlag.Bimodal in stats.flags
    assert stats.bimodal_stats is not None
    assert stats.bimodal_stats.center1 is not None
    assert stats.bimodal_stats.center2 is not None
    assert stats.bimodal_stats.center1 < stats.bimodal_stats.center2
    assert stats.bimodal_stats.cluster_separation >= 2.0
    assert stats.bimodal_stats.minority_weight >= 0.05


def test_unimodal_not_bimodal():
    import numpy as np
    rng = np.random.default_rng(42)
    data = list(rng.normal(0, 1, 200))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert NumericFlag.Bimodal not in stats.flags
    assert stats.bimodal_stats is None


def test_near_constant_skips_bimodality():
    # 55/60 > 0.90
    data = [5.0] * 55 + [1.0, 2.0, 3.0, 4.0, 6.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert NumericFlag.NearConstant in stats.flags
    assert NumericFlag.Bimodal not in stats.flags
    assert stats.bimodal_stats is None


def test_bimodality_gate_fails_separation():
    import numpy as np
    rng = np.random.default_rng(42)
    # Dip test often fires on highly skewed/boundary-pileup distributions,
    # but the separation should be low (e.g. Ashman's D < 2.0)
    data = [0.0] * 100 + list(rng.normal(0.5, 2.0, 100))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert NumericFlag.Bimodal not in stats.flags
    assert stats.bimodal_stats is None


def test_bimodality_gate_fails_minority_weight():
    import numpy as np
    rng = np.random.default_rng(42)
    # Two well separated clusters, but one is very small (e.g. 2% of data)
    data = list(rng.normal(0, 1, 980)) + list(rng.normal(20, 1, 20))
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert NumericFlag.Bimodal not in stats.flags
    assert stats.bimodal_stats is None


def test_bimodal_stats_round_trip():
    from dataforge_ml.profiling._numeric_config import BimodalStats
    stats = BimodalStats(dip_statistic=0.1, dip_p_value=0.01, center1=1.0, center2=5.0, cluster_separation=3.5, minority_weight=0.25)
    restored = BimodalStats.from_dict(stats.to_dict())
    assert restored.dip_statistic == 0.1
    assert restored.dip_p_value == 0.01
    assert restored.center1 == 1.0
    assert restored.center2 == 5.0
    assert restored.cluster_separation == 3.5
    assert restored.minority_weight == 0.25

# ---------------------------------------------------------------------------
# mean_median_ratio behavior
# ---------------------------------------------------------------------------

def test_mean_median_ratio_zero_inflated():
    # zero-inflated column: mode frequency > 50%, median = 0, mean > 0
    # mean_median_ratio should be None
    data = [0.0] * 60 + [100.0] * 40
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert stats.median == 0.0
    assert stats.mean > 0.0
    assert stats.mean_median_ratio is None
    
    # Check serialization
    import json
    # Just asserting it's serializable without Infinity
    serialized = json.dumps(stats.to_dict())
    assert "Infinity" not in serialized
    assert json.loads(serialized)


def test_mean_median_ratio_all_zero():
    # both mean and median are 0
    # mean_median_ratio should be 1.0
    data = [0.0] * 100
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    
    assert stats.median == 0.0
    assert stats.mean == 0.0
    assert stats.mean_median_ratio == 1.0

