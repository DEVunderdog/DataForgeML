"""
Unit tests for NumericImputer strategy routing.

Tests exercise the external contract: given a profile with specific signals,
assert the correct ImputationStrategy and fill_value type in the returned record.
All tests construct minimal StructuralProfileResult stubs — no real profiler run.
"""

import numpy as np
import polars as pl
import pytest

from dataforge_ml.config import SemanticType
from dataforge_ml.imputation._config import (ColumnImputationRecord,
                                             ImputationStrategy,
                                             NumericImputationConfig)
from dataforge_ml.imputation._numeric_imputer import (FittedRegression,
                                                      NumericImputer,
                                                      _fallback_to_median,
                                                      _fallback_to_mode)
from dataforge_ml.profiling._config import (ColumnProfile, NumericKind,
                                            StructuralProfileResult)
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile, MissingnessFlag, MissingSeverity)
from dataforge_ml.profiling._numeric_config import (KurtosisTag, NonlinearityTag,
                                                    NumericFlag, NumericStats,
                                                    SkewSeverity)

# ---------------------------------------------------------------------------
# Helpers — build minimal stubs
# ---------------------------------------------------------------------------


def _make_profile(col: str, cp: ColumnProfile) -> StructuralProfileResult:
    result = StructuralProfileResult()
    result.columns[col] = cp
    return result


def _numeric_cp(
    *,
    null_count: int = 0,
    total_rows: int = 100,
    severity: MissingSeverity | None = None,
    flags: list[MissingnessFlag] | None = None,
    correlated_with: list[str] | None = None,
    numeric_kind: NumericKind = NumericKind.Continuous,
    skewness_severity: SkewSeverity | None = None,
    kurtosis_tag: KurtosisTag | None = None,
    numeric_flags: list[NumericFlag] | None = None,
) -> ColumnProfile:
    missingness = None
    if null_count > 0 or flags:
        missingness = ColumnMissingnessProfile(
            column="col",
            total_rows=total_rows,
            effective_null_count=null_count,
            effective_null_ratio=null_count / total_rows,
            severity=severity,
            flags=flags or [],
            correlated_with=correlated_with or [],
        )
    stats = NumericStats(
        skewness_severity=skewness_severity,
        kurtosis_tag=kurtosis_tag,
        flags=numeric_flags or [],
    )
    return ColumnProfile(
        name="col",
        semantic_type=SemanticType.Numeric,
        numeric_kind=numeric_kind,
        missingness=missingness,
        stats=stats,
    )


_COL = "col"
_DEFAULT_CONFIG = NumericImputationConfig()
_NO_MNAR: set[str] = set()


def _fit_one(df: pl.DataFrame, cp: ColumnProfile, mnar: set[str] | None = None):
    profile = _make_profile(_COL, cp)
    bundle = NumericImputer().fit(
        train_df=df,
        columns=[_COL],
        profile=profile,
        config=_DEFAULT_CONFIG,
        mnar_columns=mnar or _NO_MNAR,
    )
    assert len(bundle.records) == 1
    return bundle.records[0]


# ---------------------------------------------------------------------------
# Strategy: Dropped
# ---------------------------------------------------------------------------


def test_drop_candidate_produces_dropped_strategy():
    df = pl.DataFrame({_COL: pl.Series([None] * 60 + [1.0] * 40, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=60,
        total_rows=100,
        severity=MissingSeverity.Severe,
        flags=[MissingnessFlag.DropCandidate],
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Dropped
    assert rec.fill_value is None
    assert rec.indicator_added is False


def test_drop_candidate_signal_describes_missingness():
    df = pl.DataFrame({_COL: pl.Series([None] * 60 + [1.0] * 40, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=60, total_rows=100, flags=[MissingnessFlag.DropCandidate]
    )
    rec = _fit_one(df, cp)
    assert any("drop_candidate" in s for s in rec.signals)


# ---------------------------------------------------------------------------
# Strategy: MNAR
# ---------------------------------------------------------------------------


def test_mnar_declared_column_gets_mnar_strategy():
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, None], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=2, total_rows=4, severity=MissingSeverity.Moderate)
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.indicator_added is True


def test_mnar_takes_priority_over_mar_flag():
    """MNAR declared by user overrides any MARSuspect flag."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=3, severity=MissingSeverity.Minor,
        flags=[MissingnessFlag.MARSuspect], correlated_with=["other_col"],
    )
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR


def test_mnar_normal_skew_fill_equals_observed_mean():
    # Non-missing: [10, 20, 40, 50] → mean = 30.0
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=5, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Normal)
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.fill_value == pytest.approx(30.0)
    assert rec.indicator_added is True


def test_mnar_moderate_skew_fill_equals_observed_median():
    # Non-missing: [10, 20, 40, 50] → median = 30.0
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=5, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Moderate)
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.fill_value == pytest.approx(30.0)
    assert rec.indicator_added is True


def test_mnar_absent_stats_falls_back_to_median():
    # Non-missing: [10, 20, 40, 50] → median = 30.0
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=5, severity=MissingSeverity.Minor)
    cp.stats = None
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.fill_value == pytest.approx(30.0)
    assert rec.indicator_added is True


def test_mnar_signals_contain_declaration_and_fill_entries():
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Normal)
    rec = _fit_one(df, cp, mnar={_COL})
    assert any("declared MNAR" in s for s in rec.signals)
    assert any("mnar_fill:" in s for s in rec.signals)


def test_mnar_integer_column_fill_value_is_rounded():
    # Non-missing: [10, 30, 40] → mean = 26.666... → rounded to 27.0
    df = pl.DataFrame({_COL: pl.Series([10, None, 30, 40], dtype=pl.Int64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Normal)
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.fill_value == float(round(rec.fill_value))


def test_mnar_bounded_discrete_uses_mode_fill():
    """BoundedDiscrete MNAR column: fill == mode, strategy == MNAR, indicator added, signal contains 'mnar_fill: mode'."""
    # Non-missing: [1, 2, 2, 3, 4] → mode = 2; median would be 2.0 but {1,2,3,4} → 2.5 — mode is the correct domain member
    df = pl.DataFrame({_COL: pl.Series([1, 2, 2, None, 3, 4], dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1,
        total_rows=6,
        severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.indicator_added is True
    assert rec.fill_value == pytest.approx(2.0)
    assert any("mnar_fill: mode" in s for s in rec.signals)


def test_mnar_non_bounded_discrete_skew_driven_fill_unchanged():
    """Non-BoundedDiscrete MNAR still routes to skew-driven mean/median (no regression)."""
    # Non-missing: [10, 20, 40, 50] → mean = 30.0 (Normal skew)
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1,
        total_rows=5,
        severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.Continuous,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp, mnar={_COL})
    assert rec.strategy == ImputationStrategy.MNAR
    assert rec.fill_value == pytest.approx(30.0)
    assert any("mnar_fill: mean" in s for s in rec.signals)


# ---------------------------------------------------------------------------
# Strategy: Passthrough
# ---------------------------------------------------------------------------


def test_no_missingness_produces_passthrough():
    df = pl.DataFrame({_COL: pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=0)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Passthrough
    assert rec.fill_value is None
    assert rec.indicator_added is False


def test_passthrough_signal_is_informative():
    df = pl.DataFrame({_COL: pl.Series([1.0, 2.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=0)
    rec = _fit_one(df, cp)
    assert len(rec.signals) >= 1


# ---------------------------------------------------------------------------
# Strategy: Median (MAR-Suspect fallback)
# ---------------------------------------------------------------------------


def test_mar_suspect_minor_severity_falls_back_to_median():
    # Minor MAR with no multi-MAR → Median (no correlations path applies at Minor)
    values = [1.0, 2.0, None, 4.0, 5.0]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        flags=[MissingnessFlag.MARSuspect], correlated_with=["x"],
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median
    assert rec.fill_value == pytest.approx(3.0)


def test_mar_suspect_signal_mentions_mar():
    values = [1.0, 2.0, None, 4.0]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=4, severity=MissingSeverity.Minor,
        flags=[MissingnessFlag.MARSuspect], correlated_with=["y"],
    )
    rec = _fit_one(df, cp)
    assert any("mar_suspect" in s.lower() for s in rec.signals)


# ---------------------------------------------------------------------------
# Strategy: Mode (Discrete)
# ---------------------------------------------------------------------------


def test_discrete_column_minor_normal_gets_mode_strategy():
    """BoundedDiscrete + Minor + Normal skew → Mode (BoundedDiscrete scalar-fill rule)."""
    # non-null: [1, 2, 1, 1, 3, 1] → mode = 1
    values = [1, 2, 1, 1, None, 3, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert rec.fill_value == pytest.approx(1.0)


def test_discrete_mode_computed_on_training_data():
    """Mode fill value is computed from non-null training values for BoundedDiscrete."""
    # non-null: [5, 5, 5, 1, 1] → mode = 5
    values = [5, 5, 5, 1, 1, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert rec.fill_value == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Strategy: Mean (MCAR minor + normal skew)
# ---------------------------------------------------------------------------


def test_mcar_minor_normal_skew_gets_mean():
    values = [1.0, 2.0, 3.0, 4.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean
    assert rec.fill_value == pytest.approx(2.5)


def test_mean_computed_excluding_nulls():
    values = [10.0, 20.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=3, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.fill_value == pytest.approx(15.0)


def test_mcar_minor_no_skew_info_gets_mean():
    """When skewness_severity is None, treat as normal skew → Mean."""
    values = [1.0, 2.0, 3.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=4, severity=MissingSeverity.Minor,
        skewness_severity=None,
    )
    # stats has no skewness_severity set
    cp.stats = NumericStats(skewness_severity=None)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean


# ---------------------------------------------------------------------------
# Strategy: Median (MCAR with skew >= Moderate or Moderate severity)
# ---------------------------------------------------------------------------


def test_mcar_minor_moderate_skew_gets_median():
    values = [1.0, 2.0, 3.0, 4.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Moderate,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median


def test_mcar_moderate_severity_gets_median():
    values = [1.0, 2.0, 3.0, 4.0, None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=3, total_rows=7, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median


def test_mcar_high_severity_gets_knn_when_guards_pass():
    # 1 feature, 100 rows — both KNN guards pass with defaults
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=20, total_rows=100, severity=MissingSeverity.High,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mcar_high_severity_falls_back_to_median_when_all_guards_fail():
    # Force both guards to fail: tiny dataset (< regression_min_rows) + large features
    config = NumericImputationConfig(
        knn_max_rows=10,      # row guard: fail for 100-row df
        knn_max_features=0,   # feature guard: always fail
        regression_min_rows=10_000,  # regression guard: fail for 100-row df
    )
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    profile = _make_profile(_COL, cp)
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL], profile=profile,
        config=config, mnar_columns=set(),
    )
    assert bundle.records[0].strategy == ImputationStrategy.Median


def test_mcar_severe_severity_gets_mice():
    values = [float(i) for i in range(70)] + [None] * 30
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=30, total_rows=100, severity=MissingSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.MICE


def test_median_computed_excluding_nulls():
    values = [1.0, 3.0, 5.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=4, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Moderate,
    )
    rec = _fit_one(df, cp)
    assert rec.fill_value == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Record metadata
# ---------------------------------------------------------------------------


def test_record_column_name_matches():
    df = pl.DataFrame({_COL: pl.Series([1.0, None], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=2, severity=MissingSeverity.Minor)
    rec = _fit_one(df, cp)
    assert rec.column == _COL


def test_record_semantic_type_is_numeric():
    df = pl.DataFrame({_COL: pl.Series([1.0, None], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=2, severity=MissingSeverity.Minor)
    rec = _fit_one(df, cp)
    assert rec.semantic_type == SemanticType.Numeric


# ---------------------------------------------------------------------------
# BoundedDiscrete routing — Priority 3, model-aware sub-chain with domain-snap
# ---------------------------------------------------------------------------


def test_bounded_discrete_minor_normal_skew_gets_mode():
    """BoundedDiscrete + Minor + Normal skew → Mode (BoundedDiscrete scalar-fill rule)."""
    values = [1, 2, 1, 1, None, 3, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert rec.fill_value == pytest.approx(1.0)  # mode of [1,2,1,1,3,1]


def test_bounded_discrete_high_severity_gets_knn():
    """BoundedDiscrete + High severity → KNN via domain-snapped MCAR sub-chain."""
    values = [1, 2, 3, 4, 5, None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=3, total_rows=8, severity=MissingSeverity.High,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_bounded_discrete_mar_suspect_low_severity_no_corrs_gets_mode():
    """BoundedDiscrete + MARSuspect + Minor + no correlations → Mode (terminal replaces Median)."""
    values = [1, 2, 3, 4, 5, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=2, total_rows=7, severity=MissingSeverity.Minor,
        flags=[MissingnessFlag.MARSuspect],
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode


# BoundedDiscrete sub-chain coverage
# ---------------------------------------------------------------------------


def test_bounded_discrete_unpredictable_routes_to_mode_with_signal():
    """BoundedDiscrete + Unpredictable → Mode; unpredictable_guard signal recorded."""
    from dataforge_ml.profiling._numeric_config import NumericFlag
    values = [1, 2, 3, 1, None, 2, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.High,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(nonlinearity_tag=NonlinearityTag.Unpredictable)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert any("unpredictable_guard" in s for s in rec.signals)


def test_bounded_discrete_near_constant_routes_to_mode_with_signal():
    """BoundedDiscrete + NearConstant → Mode; near_constant signal recorded."""
    from dataforge_ml.profiling._numeric_config import NumericFlag
    values = [1, 1, 1, 1, None, 1, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(flags=[NumericFlag.NearConstant])
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert any("near_constant" in s for s in rec.signals)


def test_bounded_discrete_mar_suspect_severe_routes_to_mice_with_snap():
    """BoundedDiscrete + MARSuspect + Severe → MICE with domain_snap_bounds set."""
    values = [1, 2, 3, 4, 5, None, None, None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=5, total_rows=10, severity=MissingSeverity.Severe,
        flags=[MissingnessFlag.MARSuspect], correlated_with=["other"],
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(min=1.0, max=5.0)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.MICE
    assert rec.domain_snap_bounds == (1.0, 5.0)
    assert any("domain_snap_bounds" in s for s in rec.signals)


def test_bounded_discrete_mar_suspect_high_with_corrs_routes_to_regression():
    """BoundedDiscrete + MARSuspect + High + large enough dataset → Regression with snap."""
    rng = np.random.default_rng(0)
    n = 600
    values = [int(v % 5) + 1 if i % 10 != 0 else None for i, v in enumerate(range(n))]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        _COL: pl.Series(values, dtype=pl.Int64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _make_profile(_COL, _numeric_cp(
        null_count=60, total_rows=n, severity=MissingSeverity.High,
        flags=[MissingnessFlag.MARSuspect], correlated_with=["feat"],
        numeric_kind=NumericKind.BoundedDiscrete,
    ))
    profile.columns[_COL].stats = NumericStats(min=1.0, max=5.0)
    profile.columns["feat"] = ColumnProfile(
        name="feat", semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous, missingness=None, stats=NumericStats(),
    )
    config = NumericImputationConfig(regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    assert rec.strategy == ImputationStrategy.Regression
    assert rec.domain_snap_bounds == (1.0, 5.0)


def test_bounded_discrete_mcar_high_routes_to_knn_with_snap():
    """BoundedDiscrete + MCAR High + size guards met → KNN with domain_snap_bounds."""
    values = [1, 2, 3, 4, 5, None, None, 3]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=2, total_rows=8, severity=MissingSeverity.High,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(min=1.0, max=5.0)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN
    assert rec.domain_snap_bounds == (1.0, 5.0)
    assert any("domain_snap_bounds" in s for s in rec.signals)


def test_bounded_discrete_mcar_minor_normal_skew_mode_fill_value():
    """BoundedDiscrete + MCAR Minor + Normal skew → Mode fill equals observed mode."""
    # [1, 5, 1, 5, None, 3] → non-null: [1, 5, 1, 5, 3], mode = 1 (or 5; both appear twice — sorted → 1)
    values = [1, 5, 1, 5, None, 3]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(min=1.0, max=5.0, skewness_severity=None)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert rec.fill_value == pytest.approx(1.0)


def test_non_bounded_discrete_mcar_minor_normal_skew_still_gets_mean():
    """Non-BoundedDiscrete MCAR Minor + Normal skew still produces strategy == Mean (no regression)."""
    # Non-missing: [10, 20, 40, 50] → mean = 30.0
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.Continuous,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean
    assert rec.fill_value == pytest.approx(30.0)


def test_bounded_discrete_all_size_guards_fail_routes_to_mode():
    """BoundedDiscrete + Minor + Moderate skew → Mode (terminal, not Median)."""
    values = [1, 2, 1, 1, None, 1, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
        skewness_severity=SkewSeverity.Moderate,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode
    assert any("terminal" in s for s in rec.signals)


def test_continuous_integer_with_gaps_falls_to_mcar_chain():
    """Column failing signal 1 (gaps) → classified Continuous → strategy is not Mode."""
    values = [18.0, 22.0, 35.0, None, 22.0]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.Continuous,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy != ImputationStrategy.Mode


def test_continuous_integer_non_zero_origin_falls_to_mcar_chain():
    """Column failing signal 4 (min≠0/1) → classified Continuous → strategy is not Mode."""
    values = [18.0, 19.0, 20.0, 21.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.Continuous,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy != ImputationStrategy.Mode


def test_record_signals_are_non_empty_for_all_strategies():
    """Every routing decision should produce at least one signal string."""
    cases = [
        _numeric_cp(null_count=60, total_rows=100, flags=[MissingnessFlag.DropCandidate]),
        _numeric_cp(null_count=0),
        _numeric_cp(null_count=1, total_rows=10, severity=MissingSeverity.Minor,
                    skewness_severity=SkewSeverity.Normal),
        _numeric_cp(null_count=1, total_rows=10, severity=MissingSeverity.Minor,
                    skewness_severity=SkewSeverity.Moderate),
        _numeric_cp(null_count=1, total_rows=10, severity=MissingSeverity.Minor,
                    numeric_kind=NumericKind.BoundedDiscrete),
    ]
    for cp in cases:
        df = pl.DataFrame({_COL: pl.Series([1.0, 2.0, 3.0, None], dtype=pl.Float64)})
        rec = _fit_one(df, cp)
        assert len(rec.signals) >= 1, f"No signals for strategy {rec.strategy}"


# ---------------------------------------------------------------------------
# Phase 2 routing: numeric_kind_overrides flow-through
# ---------------------------------------------------------------------------


def test_override_forced_bounded_discrete_gets_mode_on_minor_normal():
    """A column forced to BoundedDiscrete via override + Minor + Normal skew → Mode."""
    # non-null: [25, 32, 45, 28, 39] → mode = 25 (sorted first of all unique values)
    values = [25, 32, 45, 28, None, 39]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode


def test_override_forced_continuous_falls_to_mcar_chain():
    """A column forced to Continuous via override does not get Mode."""
    values = [1, 2, 3, 4, 5, None, 1, 2, 3, 4]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=10, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.Continuous,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy != ImputationStrategy.Mode


# ---------------------------------------------------------------------------
# FittedRegression dataclass — structural tests (Issue #138)
# ---------------------------------------------------------------------------


def test_fitted_regression_has_model_field():
    fr = FittedRegression(model=object(), target_idx=0, all_cols=["y", "x1", "x2"])
    assert fr.model is not None


def test_fitted_regression_stores_target_idx():
    fr = FittedRegression(model=None, target_idx=0, all_cols=["y", "x"])
    assert fr.target_idx == 0


def test_fitted_regression_stores_all_cols():
    cols = ["y", "x1", "x2"]
    fr = FittedRegression(model=None, target_idx=0, all_cols=cols)
    assert fr.all_cols == ["y", "x1", "x2"]


def test_fitted_regression_all_cols_includes_target_at_index_zero():
    cols = ["target_col", "feat_a", "feat_b"]
    fr = FittedRegression(model=None, target_idx=0, all_cols=cols)
    assert fr.all_cols[fr.target_idx] == "target_col"


def test_fitted_regression_has_signals_field():
    fr = FittedRegression(model=None, target_idx=0, all_cols=["y", "x"])
    assert isinstance(fr.signals, list)


def test_fitted_regression_signals_default_empty():
    fr = FittedRegression(model=None, target_idx=0, all_cols=["y", "x"])
    assert fr.signals == []


# ---------------------------------------------------------------------------
# Regression overhaul — Issue #141 (Scope 0 + 12 combined)
# ---------------------------------------------------------------------------


def _regression_profile(col: str, n: int, feat_cols: list[str]) -> StructuralProfileResult:
    """Build a minimal StructuralProfileResult that routes col to Regression."""
    cp = ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column=col,
            total_rows=n,
            effective_null_count=n // 10,
            effective_null_ratio=0.1,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=feat_cols,
        ),
        stats=NumericStats(),
    )
    result = StructuralProfileResult()
    result.columns[col] = cp
    for fc in feat_cols:
        result.columns[fc] = ColumnProfile(
            name=fc,
            semantic_type=SemanticType.Numeric,
            numeric_kind=NumericKind.Continuous,
            missingness=None,
            stats=NumericStats(),
        )
    return result


def test_rows_with_missing_features_are_imputed_not_dropped():
    """IterativeImputer does not drop rows where feature columns have NaN."""
    rng = np.random.default_rng(42)
    n = 600
    # target missing on even rows, feature missing on odd rows → zero complete rows
    col_vals = [None if i % 2 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = [None if i % 2 == 1 else rng.standard_normal() for i in range(n)]
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Regression


def test_unpredictable_tag_routes_to_median_with_signal():
    """NonlinearityTag.Unpredictable forces Median fallback with a recorded signal."""
    rng = np.random.default_rng(0)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    profile.columns["target"].stats = NumericStats(
        nonlinearity_tag=NonlinearityTag.Unpredictable
    )
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Median
    assert any("Unpredictable" in s for s in rec.signals)


def test_unpredictable_guard_fires_before_mar_routing():
    """Priority 4 Unpredictable guard routes to Median before MARSuspect routing is reached."""
    rng = np.random.default_rng(0)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    # MARSuspect + Unpredictable: Priority 4 must fire before Priority 5 (MARSuspect)
    profile = _regression_profile("target", n, ["feat"])
    profile.columns["target"].stats = NumericStats(
        nonlinearity_tag=NonlinearityTag.Unpredictable
    )
    config = NumericImputationConfig(regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Median
    assert any("unpredictable_guard" in s for s in rec.signals)
    # MAR routing (Priority 5) must not have fired — its signal starts with "mar_suspect: correlated"
    assert not any(s.startswith("mar_suspect: correlated") for s in rec.signals)
    assert not any(s in ("MICE", "Regression", "KNN") for s in
                   [r.strategy for r in bundle.records if r.column == "target"])


def test_estimator_choice_recorded_in_signals():
    """A signal naming the estimator and tag is appended after successful regression fit."""
    rng = np.random.default_rng(1)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    profile.columns["target"].stats = NumericStats(
        nonlinearity_tag=NonlinearityTag.Linear
    )
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Regression
    assert any("regression_estimator" in s for s in rec.signals)


def test_convergence_warning_appears_when_max_iter_reached():
    """convergence_warning signal is recorded when n_iter_ == max_iter."""
    rng = np.random.default_rng(2)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    profile.columns["target"].stats = NumericStats(
        nonlinearity_tag=NonlinearityTag.Linear
    )
    # base_max_iter=1 means IterativeImputer stops after 1 iteration — almost never converged
    config = NumericImputationConfig(
        knn_max_rows=100, regression_min_rows=500, base_max_iter=1,
    )
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Regression
    assert any("convergence_warning" in s for s in rec.signals)


def test_duplicate_regression_column_raises_runtime_error():
    """Passing the same column twice in the column list raises RuntimeError naming it."""
    rng = np.random.default_rng(3)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    with pytest.raises(RuntimeError, match="target"):
        NumericImputer().fit(
            train_df=df,
            columns=["target", "feat", "target"],  # duplicate triggers guard
            profile=profile,
            config=config,
            mnar_columns=set(),
        )


def test_fallback_to_median_no_feature_columns():
    """Regression column with no features falls back to Median with signal."""
    rng = np.random.default_rng(4)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    df = pl.DataFrame({"target": pl.Series(col_vals, dtype=pl.Float64)})
    # Profile with no feature columns
    result = StructuralProfileResult()
    result.columns["target"] = ColumnProfile(
        name="target",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column="target",
            total_rows=n,
            effective_null_count=60,
            effective_null_ratio=0.1,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=["other"],
        ),
        stats=NumericStats(),
    )
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target"],
        profile=result, config=config, mnar_columns=set(),
    )
    rec = bundle.records[0]
    assert rec.strategy == ImputationStrategy.Median
    assert any("no feature" in s.lower() or "fallback" in s.lower() for s in rec.signals)


def test_fallback_to_median_insufficient_target_observations():
    """Target column with fewer than 2 non-null values falls back to Median."""
    # target has only 1 non-null value; feat is complete
    df = pl.DataFrame({
        "target": pl.Series([1.0, None, None, None, None], dtype=pl.Float64),
        "feat": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
    })
    n = 5
    result = StructuralProfileResult()
    result.columns["target"] = ColumnProfile(
        name="target",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column="target",
            total_rows=n,
            effective_null_count=4,
            effective_null_ratio=0.8,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=["feat"],
        ),
        stats=NumericStats(),
    )
    result.columns["feat"] = ColumnProfile(
        name="feat",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=None,
        stats=NumericStats(),
    )
    config = NumericImputationConfig(knn_max_rows=1, regression_min_rows=1)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=result, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Median
    assert any("fallback" in s.lower() for s in rec.signals)


def test_bounded_discrete_regression_fallback_produces_mode():
    """BoundedDiscrete → Regression where _fit_regression returns None → fallback to Mode."""
    # target has only 1 non-null value → _fit_regression returns None
    df = pl.DataFrame({
        "target": pl.Series([3, None, None, None, None], dtype=pl.Int64),
        "feat": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
    })
    n = 5
    result = StructuralProfileResult()
    result.columns["target"] = ColumnProfile(
        name="target",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.BoundedDiscrete,
        missingness=ColumnMissingnessProfile(
            column="target",
            total_rows=n,
            effective_null_count=4,
            effective_null_ratio=0.8,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=["feat"],
        ),
        stats=NumericStats(min=1.0, max=5.0),
    )
    result.columns["feat"] = ColumnProfile(
        name="feat",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=None,
        stats=NumericStats(),
    )
    config = NumericImputationConfig(knn_max_rows=1, regression_min_rows=1)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=result, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Mode
    assert rec.fill_value == pytest.approx(3.0)
    assert any("fallback_to_mode" in s for s in rec.signals)
    assert rec.domain_snap_bounds is None


def test_non_bounded_discrete_regression_fallback_still_produces_median():
    """Non-BoundedDiscrete Regression failure still falls back to Median (regression guard)."""
    df = pl.DataFrame({
        "target": pl.Series([10.0, None, None, None, None], dtype=pl.Float64),
        "feat": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
    })
    n = 5
    result = StructuralProfileResult()
    result.columns["target"] = ColumnProfile(
        name="target",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column="target",
            total_rows=n,
            effective_null_count=4,
            effective_null_ratio=0.8,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=["feat"],
        ),
        stats=NumericStats(),
    )
    result.columns["feat"] = ColumnProfile(
        name="feat",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=None,
        stats=NumericStats(),
    )
    config = NumericImputationConfig(knn_max_rows=1, regression_min_rows=1)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=result, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Median
    assert any("fallback_to_median" in s for s in rec.signals)


# ---------------------------------------------------------------------------
# _fallback_to_median pure function unit tests
# ---------------------------------------------------------------------------


def test_fallback_to_median_pure_returns_updated_record():
    """_fallback_to_median returns a new record with Median strategy."""
    df = pl.DataFrame({"col": pl.Series([1.0, 3.0, 5.0], dtype=pl.Float64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["existing_signal"],
    )
    updated = _fallback_to_median(df, "col", record, "test reason")
    assert updated.strategy == ImputationStrategy.Median
    assert updated.fill_value == pytest.approx(3.0)
    assert any("fallback_to_median" in s for s in updated.signals)
    assert any("test reason" in s for s in updated.signals)


def test_fallback_to_median_pure_does_not_mutate_input():
    """_fallback_to_median must not modify the original record."""
    df = pl.DataFrame({"col": pl.Series([1.0, 3.0, 5.0], dtype=pl.Float64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["original"],
    )
    original_strategy = record.strategy
    original_signals = list(record.signals)
    _fallback_to_median(df, "col", record, "some reason")
    assert record.strategy == original_strategy
    assert record.signals == original_signals


def test_fallback_to_median_pure_preserves_existing_signals():
    """Existing signals on the record are preserved in the returned record."""
    df = pl.DataFrame({"col": pl.Series([2.0, 4.0, 6.0], dtype=pl.Float64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["first_signal", "second_signal"],
    )
    updated = _fallback_to_median(df, "col", record, "a reason")
    assert "first_signal" in updated.signals
    assert "second_signal" in updated.signals


def test_fallback_to_median_pure_fill_value_from_training_data():
    """Fill value is the median of the non-null training data values."""
    df = pl.DataFrame({"col": pl.Series([10.0, 20.0, 30.0, None], dtype=pl.Float64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=[],
    )
    updated = _fallback_to_median(df, "col", record, "reason")
    assert updated.fill_value == pytest.approx(20.0)
    updated = _fallback_to_median(df, "col", record, "reason")
    assert updated.fill_value == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# _fallback_to_mode pure function unit tests
# ---------------------------------------------------------------------------


def test_fallback_to_mode_pure_returns_updated_record():
    """_fallback_to_mode returns a new record with Mode strategy and mode fill value."""
    # non-null: [1, 3, 1, 5] → mode = 1
    df = pl.DataFrame({"col": pl.Series([1, 3, 1, 5], dtype=pl.Int64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["existing_signal"],
        domain_snap_bounds=(1.0, 5.0),
    )
    updated = _fallback_to_mode(df, "col", record, "test reason")
    assert updated.strategy == ImputationStrategy.Mode
    assert updated.fill_value == pytest.approx(1.0)
    assert any("fallback_to_mode" in s for s in updated.signals)
    assert any("test reason" in s for s in updated.signals)


def test_fallback_to_mode_pure_clears_domain_snap_bounds():
    """_fallback_to_mode sets domain_snap_bounds to None on the returned record."""
    df = pl.DataFrame({"col": pl.Series([2, 2, 4], dtype=pl.Int64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=[],
        domain_snap_bounds=(1.0, 5.0),
    )
    updated = _fallback_to_mode(df, "col", record, "reason")
    assert updated.domain_snap_bounds is None


def test_fallback_to_mode_pure_does_not_mutate_input():
    """_fallback_to_mode must not modify the original record."""
    df = pl.DataFrame({"col": pl.Series([1, 2, 3], dtype=pl.Int64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["original"],
        domain_snap_bounds=(1.0, 5.0),
    )
    original_strategy = record.strategy
    original_signals = list(record.signals)
    original_bounds = record.domain_snap_bounds
    _fallback_to_mode(df, "col", record, "some reason")
    assert record.strategy == original_strategy
    assert record.signals == original_signals
    assert record.domain_snap_bounds == original_bounds


def test_fallback_to_mode_pure_preserves_existing_signals():
    """Existing signals on the record are preserved in the returned record."""
    df = pl.DataFrame({"col": pl.Series([3, 3, 7], dtype=pl.Int64)})
    record = ColumnImputationRecord(
        column="col",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        fill_value=None,
        indicator_added=False,
        signals=["first_signal", "second_signal"],
    )
    updated = _fallback_to_mode(df, "col", record, "a reason")
    assert "first_signal" in updated.signals
    assert "second_signal" in updated.signals


# ---------------------------------------------------------------------------
# MAR High + empty correlations → MCAR High fallback chain (Issue #95)
# ---------------------------------------------------------------------------


def _mar_high_empty_corrs_profile(col: str, n: int, feat_cols: list[str]) -> StructuralProfileResult:
    """Build a profile with MARSuspect + High severity + no missingness correlations."""
    cp = ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column=col,
            total_rows=n,
            effective_null_count=n // 5,
            effective_null_ratio=0.2,
            severity=MissingSeverity.High,
            flags=[MissingnessFlag.MARSuspect],
            correlated_with=[],
        ),
        stats=NumericStats(),
    )
    result = StructuralProfileResult()
    result.columns[col] = cp
    for fc in feat_cols:
        result.columns[fc] = ColumnProfile(
            name=fc,
            semantic_type=SemanticType.Numeric,
            numeric_kind=NumericKind.Continuous,
            missingness=None,
            stats=NumericStats(),
        )
    return result


def test_mar_high_empty_corrs_large_dataset_routes_to_knn():
    """MAR High + no missingness correlations + KNN guards pass → KNN via MCAR fallback chain."""
    rng = np.random.default_rng(0)
    n = 1000
    col_vals = [None if i % 5 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        _COL: pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _mar_high_empty_corrs_profile(_COL, n, ["feat"])
    config = NumericImputationConfig(knn_max_rows=50_000, knn_max_features=50)
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    assert rec.strategy == ImputationStrategy.KNN
    assert any("MAR high + no missingness correlations" in s for s in rec.signals)


def test_mar_high_empty_corrs_small_dataset_falls_to_median():
    """MAR High + no missingness correlations + all size guards fail → Median."""
    n = 100
    col_vals = [None if i % 5 == 0 else float(i) for i in range(n)]
    feat_vals = [float(i) for i in range(n)]
    df = pl.DataFrame({
        _COL: pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _mar_high_empty_corrs_profile(_COL, n, ["feat"])
    config = NumericImputationConfig(
        knn_max_rows=10,
        knn_max_features=0,
        regression_min_rows=10_000,
    )
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    assert rec.strategy == ImputationStrategy.Median
    assert any("MAR high + no missingness correlations" in s for s in rec.signals)


def test_mar_high_with_corrs_still_routes_to_regression():
    """MAR High + non-empty missingness correlations → Regression (existing behaviour unchanged)."""
    rng = np.random.default_rng(0)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        _COL: pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile(_COL, n, ["feat"])
    config = NumericImputationConfig(regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    assert rec.strategy == ImputationStrategy.Regression


def test_unpredictable_with_mar_suspect_signal_includes_mar_suspect_true():
    """Priority 4 Unpredictable guard signal records mar_suspect=True when column is also MARSuspect."""
    rng = np.random.default_rng(0)
    n = 600
    col_vals = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        "target": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    profile = _regression_profile("target", n, ["feat"])
    profile.columns["target"].stats = NumericStats(
        nonlinearity_tag=NonlinearityTag.Unpredictable
    )
    config = NumericImputationConfig(regression_min_rows=500)
    bundle = NumericImputer().fit(
        train_df=df, columns=["target", "feat"],
        profile=profile, config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == "target")
    assert rec.strategy == ImputationStrategy.Median
    assert any("unpredictable_guard" in s for s in rec.signals)
    assert any("mar_suspect=True" in s for s in rec.signals)


# ---------------------------------------------------------------------------
# MCAR distribution shape escalation — NearConstant cap + Leptokurtic/Severe
# ---------------------------------------------------------------------------


def test_mcar_minor_leptokurtic_escalates_to_knn():
    """MCAR Minor + Leptokurtic escalates to KNN instead of Mean."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN
    assert any("leptokurtic" in s for s in rec.signals)


def test_mcar_minor_normal_skew_platykurtic_routes_to_mean():
    """MCAR Minor + Normal skew + Platykurtic routes to Mean (Platykurtic does not escalate)."""
    values = [1.0, 2.0, 3.0, 4.0, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Normal,
        kurtosis_tag=KurtosisTag.Platykurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean


def test_mcar_moderate_leptokurtic_escalates_to_knn():
    """MCAR Moderate + Leptokurtic escalates to KNN instead of Median."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN
    assert any("leptokurtic" in s for s in rec.signals)


def test_mcar_moderate_severe_skew_escalates_to_knn():
    """MCAR Moderate + SkewSeverity.Severe escalates to KNN instead of Median."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN
    assert any("skew=severe" in s for s in rec.signals)


def test_mcar_moderate_mesokurtic_normal_skew_routes_to_median():
    """MCAR Moderate + Mesokurtic + Normal skew routes to Median (unchanged)."""
    values = [1.0, 2.0, 3.0, 4.0, None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=3, total_rows=7, severity=MissingSeverity.Moderate,
        kurtosis_tag=KurtosisTag.Mesokurtic,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median


def test_near_constant_leptokurtic_routes_to_median():
    """NearConstant + Leptokurtic routes to Median (NearConstant cap overrides escalation)."""
    values = [1.0] * 95 + [2.0] * 3 + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
        numeric_flags=[NumericFlag.NearConstant],
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median


def test_near_constant_cap_signal_recorded():
    """NearConstant cap records signal 'near_constant: model-based escalation suppressed'."""
    values = [1.0] * 95 + [2.0] * 3 + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
        numeric_flags=[NumericFlag.NearConstant],
    )
    rec = _fit_one(df, cp)
    assert any("near_constant: model-based escalation suppressed" in s for s in rec.signals)


def test_distribution_shape_escalation_signal_recorded_for_minor_leptokurtic():
    """Distribution shape escalation signal is recorded when Minor + Leptokurtic escalates."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert any("leptokurtic" in s.lower() for s in rec.signals)


def test_distribution_shape_escalation_signal_recorded_for_moderate_severe_skew():
    """Distribution shape escalation signal is recorded when Moderate + Severe skew escalates."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert any("skew=severe" in s for s in rec.signals)


def test_mcar_high_unchanged_by_distribution_shape():
    """MCAR High is not affected by distribution shape signals; still routes to KNN."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=20, total_rows=100, severity=MissingSeverity.High,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mcar_severe_unchanged_by_distribution_shape():
    """MCAR Severe is not affected by distribution shape signals; still routes to MICE."""
    values = [float(i) for i in range(70)] + [None] * 30
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=30, total_rows=100, severity=MissingSeverity.Severe,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.MICE


# ---------------------------------------------------------------------------
# MAR Minor/Moderate distribution shape escalation (Issue #95)
# ---------------------------------------------------------------------------


def _mar_cp(
    *,
    null_count: int,
    total_rows: int,
    severity: MissingSeverity,
    kurtosis_tag: KurtosisTag | None = None,
    skewness_severity: SkewSeverity | None = None,
    correlated_with: list[str] | None = None,
) -> ColumnProfile:
    """Build a MARSuspect ColumnProfile with the given distribution shape tags."""
    return _numeric_cp(
        null_count=null_count,
        total_rows=total_rows,
        severity=severity,
        flags=[MissingnessFlag.MARSuspect],
        correlated_with=correlated_with or [],
        kurtosis_tag=kurtosis_tag,
        skewness_severity=skewness_severity,
    )


def test_mar_minor_leptokurtic_escalates_to_knn():
    """MAR Minor + Leptokurtic escalates to KNN instead of Median."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mar_moderate_severe_skew_escalates_to_knn():
    """MAR Moderate + SkewSeverity.Severe escalates to KNN instead of Median."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mar_minor_leptokurtic_small_dataset_routes_to_median():
    """MAR Minor + Leptokurtic on a small dataset (size guards fail) routes to Median."""
    values = [float(i) for i in range(8)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=2, total_rows=10, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    profile = _make_profile(_COL, cp)
    config = NumericImputationConfig(
        knn_max_rows=5,
        knn_max_features=0,
        regression_min_rows=10_000,
    )
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = bundle.records[0]
    assert rec.strategy == ImputationStrategy.Median


def test_mar_minor_mesokurtic_routes_to_median():
    """MAR Minor + Mesokurtic routes to Median (Mesokurtic does not escalate)."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Mesokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Median


def test_mar_minor_severe_skew_alone_escalates_to_knn():
    """MAR Minor + SkewSeverity.Severe alone (without Leptokurtic) escalates to KNN."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Severe,
        kurtosis_tag=KurtosisTag.Mesokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mar_moderate_leptokurtic_alone_escalates_to_knn():
    """MAR Moderate + Leptokurtic alone (without Severe skew) escalates to KNN."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        kurtosis_tag=KurtosisTag.Leptokurtic,
        skewness_severity=SkewSeverity.Normal,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.KNN


def test_mar_severe_not_affected_by_distribution_shape():
    """MAR Severe routes to MICE regardless of distribution shape."""
    values = [float(i) for i in range(70)] + [None] * 30
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=30, total_rows=100, severity=MissingSeverity.Severe,
        kurtosis_tag=KurtosisTag.Leptokurtic,
        skewness_severity=SkewSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.MICE


def test_mar_multi_mar_not_affected_by_distribution_shape():
    """multi-MAR routes to MICE regardless of distribution shape on individual columns."""
    rng = np.random.default_rng(0)
    n = 100
    col_a = [None if i % 10 == 0 else float(i) for i in range(n)]
    col_b = [None if i % 10 == 1 else float(i) for i in range(n)]
    df = pl.DataFrame({
        "a": pl.Series(col_a, dtype=pl.Float64),
        "b": pl.Series(col_b, dtype=pl.Float64),
    })

    def _mar_profile(col: str) -> ColumnProfile:
        return ColumnProfile(
            name=col,
            semantic_type=SemanticType.Numeric,
            numeric_kind=NumericKind.Continuous,
            missingness=ColumnMissingnessProfile(
                column=col,
                total_rows=n,
                effective_null_count=10,
                effective_null_ratio=0.1,
                severity=MissingSeverity.Minor,
                flags=[MissingnessFlag.MARSuspect],
                correlated_with=[],
            ),
            stats=NumericStats(kurtosis_tag=KurtosisTag.Leptokurtic),
        )

    profile = StructuralProfileResult()
    profile.columns["a"] = _mar_profile("a")
    profile.columns["b"] = _mar_profile("b")

    bundle = NumericImputer().fit(
        train_df=df, columns=["a", "b"], profile=profile,
        config=_DEFAULT_CONFIG, mnar_columns=set(),
    )
    for rec in bundle.records:
        assert rec.strategy == ImputationStrategy.MICE


def test_mar_distribution_shape_escalation_signal_recorded():
    """Distribution shape escalation signal is recorded in ColumnImputationRecord.signals."""
    values = [float(i) for i in range(98)] + [None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=2, total_rows=100, severity=MissingSeverity.Minor,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert any("distribution shape escalation" in s for s in rec.signals)


def test_mar_distribution_shape_moderate_severe_skew_signal_recorded():
    """Escalation signal mentions 'skew=severe' when SkewSeverity.Severe triggers escalation."""
    values = [float(i) for i in range(97)] + [None, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _mar_cp(
        null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Severe,
    )
    rec = _fit_one(df, cp)
    assert any("skew=severe" in s for s in rec.signals)


def test_bounded_discrete_mar_leptokurtic_not_escalated():
    """BoundedDiscrete + MARSuspect Minor + Leptokurtic → Mode (shape escalation not wired for BoundedDiscrete)."""
    values = [1, 2, 3, 4, 5, 1, None, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=2, total_rows=8, severity=MissingSeverity.Minor,
        flags=[MissingnessFlag.MARSuspect],
        numeric_kind=NumericKind.BoundedDiscrete,
        kurtosis_tag=KurtosisTag.Leptokurtic,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mode


# ---------------------------------------------------------------------------
# MCAR feature-predictability check (Issue #95)
# ---------------------------------------------------------------------------


def _make_profile_with_pearson(
    col: str,
    cp: ColumnProfile,
    pearson_matrix: dict,
) -> StructuralProfileResult:
    from dataforge_ml.profiling._correlation_config import CorrelationProfileResult

    profile = StructuralProfileResult()
    profile.columns[col] = cp
    profile.dataset.feature_correlation = CorrelationProfileResult(
        pearson_matrix=pearson_matrix
    )
    return profile


def _fit_one_with_profile(
    df: pl.DataFrame,
    profile: StructuralProfileResult,
    col: str = _COL,
    config: NumericImputationConfig | None = None,
) -> ColumnImputationRecord:
    bundle = NumericImputer().fit(
        train_df=df,
        columns=[col],
        profile=profile,
        config=config or _DEFAULT_CONFIG,
        mnar_columns=set(),
    )
    assert len(bundle.records) == 1
    return bundle.records[0]


def test_mcar_high_below_predictability_threshold_routes_to_median():
    """MCAR High + max |r| < threshold (0.2) → Median (feature-predictability check fires)."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    # pearson_matrix: only correlation is 0.1, below default threshold 0.2
    pearson_matrix = {_COL: {"feat": 0.1}, "feat": {_COL: 0.1}}
    profile = _make_profile_with_pearson(_COL, cp, pearson_matrix)
    rec = _fit_one_with_profile(df, profile)
    assert rec.strategy == ImputationStrategy.Median


def test_mcar_high_below_threshold_signal_recorded():
    """Signal records max |r| and threshold when feature-predictability check fires."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    pearson_matrix = {_COL: {"feat": 0.1}, "feat": {_COL: 0.1}}
    profile = _make_profile_with_pearson(_COL, cp, pearson_matrix)
    rec = _fit_one_with_profile(df, profile)
    assert any("feature-predictability check failed" in s for s in rec.signals)
    assert any("max |r|=0.10" in s for s in rec.signals)


def test_mcar_high_above_predictability_threshold_routes_to_knn():
    """MCAR High + max |r| >= threshold (0.2) → KNN (feature-predictability check passes)."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    # pearson_matrix: correlation is 0.5, above default threshold 0.2
    pearson_matrix = {_COL: {"feat": 0.5}, "feat": {_COL: 0.5}}
    profile = _make_profile_with_pearson(_COL, cp, pearson_matrix)
    rec = _fit_one_with_profile(df, profile)
    assert rec.strategy == ImputationStrategy.KNN


def test_mcar_high_custom_threshold_respected():
    """Custom mcar_feature_predictability_threshold=0.35 is respected."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    # correlation 0.3 is below 0.35 threshold but above default 0.2
    pearson_matrix = {_COL: {"feat": 0.3}, "feat": {_COL: 0.3}}
    profile = _make_profile_with_pearson(_COL, cp, pearson_matrix)
    config = NumericImputationConfig(mcar_feature_predictability_threshold=0.35)
    rec = _fit_one_with_profile(df, profile, config=config)
    assert rec.strategy == ImputationStrategy.Median
    assert any("threshold=0.35" in s for s in rec.signals)


def test_mcar_high_no_pearson_matrix_skips_check():
    """When feature_correlation is None, predictability check is skipped → KNN as before."""
    values = [float(i) for i in range(80)] + [None] * 20
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    # No feature_correlation set → profile.dataset.feature_correlation is None
    profile = _make_profile(_COL, cp)
    rec = _fit_one_with_profile(df, profile)
    assert rec.strategy == ImputationStrategy.KNN


def test_mcar_severe_not_affected_by_predictability_check():
    """MCAR Severe routes to MICE regardless of feature-predictability check."""
    values = [float(i) for i in range(70)] + [None] * 30
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Float64)})
    cp = _numeric_cp(null_count=30, total_rows=100, severity=MissingSeverity.Severe)
    # Low correlation that would block KNN/Regression on MCAR High
    pearson_matrix = {_COL: {"feat": 0.05}, "feat": {_COL: 0.05}}
    profile = _make_profile_with_pearson(_COL, cp, pearson_matrix)
    rec = _fit_one_with_profile(df, profile)
    assert rec.strategy == ImputationStrategy.MICE


def test_mar_high_below_threshold_still_routes_to_model_based():
    """MAR High + max |r| < 0.2 → still model-based (feature-predictability check does not apply to MAR)."""
    rng = np.random.default_rng(0)
    n = 1000
    col_vals = [None if i % 5 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        _COL: pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    # MARSuspect + empty missingness correlations → MCAR High fallback chain
    profile = _mar_high_empty_corrs_profile(_COL, n, ["feat"])
    # Attach a low-correlation pearson matrix — should NOT block MAR routing
    from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
    profile.dataset.feature_correlation = CorrelationProfileResult(
        pearson_matrix={_COL: {"feat": 0.05}, "feat": {_COL: 0.05}}
    )
    config = NumericImputationConfig(knn_max_rows=50_000, knn_max_features=50)
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    # MAR High with empty missingness corrs delegates to _mcar_model_strategy,
    # but the feature-predictability check is not passed (col/feature_correlation
    # not supplied via _mar_strategy) — so should still route to KNN
    assert rec.strategy in (ImputationStrategy.KNN, ImputationStrategy.Regression, ImputationStrategy.MICE)


# ---------------------------------------------------------------------------
# _StrategyRouter direct unit tests — no DataFrame construction needed
# ---------------------------------------------------------------------------


from dataforge_ml.imputation._strategy_router import _StrategyRouter  # noqa: E402


def _router_route(
    cp: ColumnProfile,
    *,
    n_rows: int = 1000,
    n_features: int = 5,
    multi_mar: bool = False,
    mnar_columns: set | None = None,
    config: NumericImputationConfig | None = None,
    feature_correlation=None,
) -> tuple:
    """Call _StrategyRouter.route() with sensible defaults."""
    return _StrategyRouter().route(
        col=_COL,
        cp=cp,
        config=config or _DEFAULT_CONFIG,
        n_rows=n_rows,
        n_features=n_features,
        multi_mar=multi_mar,
        mnar_columns=mnar_columns or set(),
        feature_correlation=feature_correlation,
    )


def test_router_drop_candidate_returns_dropped():
    """DropCandidate → Dropped strategy with drop_candidate signal."""
    cp = _numeric_cp(null_count=60, total_rows=100, flags=[MissingnessFlag.DropCandidate])
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Dropped
    assert any("drop_candidate" in s for s in signals)


def test_router_mnar_declared_returns_mnar():
    """Column in mnar_columns → MNAR strategy with mnar_fill signal."""
    cp = _numeric_cp(null_count=5, total_rows=100, severity=MissingSeverity.Minor)
    strategy, signals = _router_route(cp, mnar_columns={_COL})
    assert strategy == ImputationStrategy.MNAR
    assert any("mnar_fill:" in s for s in signals)


def test_router_no_missingness_returns_passthrough():
    """No effective missingness → Passthrough."""
    cp = _numeric_cp(null_count=0)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Passthrough


def test_router_unpredictable_guard_returns_median():
    """NonlinearityTag.Unpredictable → Median unconditionally."""
    cp = _numeric_cp(null_count=10, total_rows=100, severity=MissingSeverity.High,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=["x"])
    cp.stats = NumericStats(nonlinearity_tag=NonlinearityTag.Unpredictable)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.Median
    assert any("unpredictable_guard" in s for s in signals)
    assert not any(s.startswith("mar_suspect: correlated") for s in signals)


def test_router_unpredictable_signal_records_mar_suspect_true():
    """Unpredictable guard signal includes mar_suspect=True when MARSuspect flag is also set."""
    cp = _numeric_cp(null_count=10, total_rows=100, severity=MissingSeverity.High,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=["x"])
    cp.stats = NumericStats(nonlinearity_tag=NonlinearityTag.Unpredictable)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert any("mar_suspect=True" in s for s in signals)


def test_router_mar_high_empty_corrs_routes_to_knn():
    """MAR High + empty correlations → KNN via MCAR High fallback."""
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=[])
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("MAR high + no missingness correlations" in s for s in signals)


def test_router_mar_high_with_corrs_routes_to_regression():
    """MAR High + non-empty correlations + large dataset → Regression."""
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=["x"])
    config = NumericImputationConfig(regression_min_rows=500)
    strategy, signals = _router_route(cp, n_rows=600, config=config)
    assert strategy == ImputationStrategy.Regression


def test_router_mcar_minor_leptokurtic_routes_to_knn():
    """MCAR Minor + Leptokurtic → KNN instead of Mean."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     kurtosis_tag=KurtosisTag.Leptokurtic)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("leptokurtic" in s for s in signals)


def test_router_mcar_minor_normal_skew_routes_to_mean():
    """MCAR Minor + Normal skew + Mesokurtic → Mean."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Normal, kurtosis_tag=KurtosisTag.Mesokurtic)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Mean


def test_router_mcar_minor_platykurtic_normal_skew_routes_to_mean():
    """MCAR Minor + Platykurtic + Normal skew → Mean (Platykurtic does not escalate)."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     skewness_severity=SkewSeverity.Normal, kurtosis_tag=KurtosisTag.Platykurtic)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Mean
    assert any("platykurtic" in s for s in signals)


def test_router_mcar_moderate_leptokurtic_routes_to_knn():
    """MCAR Moderate + Leptokurtic → KNN instead of Median."""
    cp = _numeric_cp(null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
                     kurtosis_tag=KurtosisTag.Leptokurtic)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("leptokurtic" in s for s in signals)


def test_router_mcar_moderate_severe_skew_routes_to_knn():
    """MCAR Moderate + Severe skew → KNN."""
    cp = _numeric_cp(null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
                     skewness_severity=SkewSeverity.Severe)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("skew=severe" in s for s in signals)


def test_router_mcar_moderate_mesokurtic_routes_to_median():
    """MCAR Moderate + Mesokurtic + Normal skew → Median (unchanged)."""
    cp = _numeric_cp(null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
                     kurtosis_tag=KurtosisTag.Mesokurtic, skewness_severity=SkewSeverity.Normal)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Median


def test_router_near_constant_leptokurtic_caps_to_median():
    """NearConstant + Leptokurtic → Median (NearConstant cap overrides escalation)."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     kurtosis_tag=KurtosisTag.Leptokurtic,
                     numeric_flags=[NumericFlag.NearConstant])
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Median
    assert any("near_constant" in s for s in signals)


def test_router_mar_minor_leptokurtic_routes_to_knn():
    """MAR Minor + Leptokurtic → KNN via distribution shape escalation."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=[],
                     kurtosis_tag=KurtosisTag.Leptokurtic)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("distribution shape escalation" in s for s in signals)


def test_router_mar_moderate_severe_skew_routes_to_knn():
    """MAR Moderate + Severe skew → KNN via distribution shape escalation."""
    cp = _numeric_cp(null_count=3, total_rows=100, severity=MissingSeverity.Moderate,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=[],
                     skewness_severity=SkewSeverity.Severe)
    strategy, signals = _router_route(cp, n_rows=1000)
    assert strategy == ImputationStrategy.KNN
    assert any("distribution shape escalation" in s for s in signals)


def test_router_mcar_high_feature_predictability_check_blocks_knn():
    """MCAR High + max |r| < threshold → Median (feature-predictability check)."""
    from dataforge_ml.profiling._correlation_config import CorrelationProfileResult

    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    pearson_matrix = {_COL: {"feat": 0.05}, "feat": {_COL: 0.05}}
    fc = CorrelationProfileResult(pearson_matrix=pearson_matrix)
    strategy, signals = _router_route(cp, n_rows=1000, feature_correlation=fc)
    assert strategy == ImputationStrategy.Median
    assert any("feature-predictability check failed" in s for s in signals)


def test_router_mcar_high_feature_predictability_check_passes():
    """MCAR High + max |r| >= threshold → KNN (feature-predictability check passes)."""
    from dataforge_ml.profiling._correlation_config import CorrelationProfileResult

    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    pearson_matrix = {_COL: {"feat": 0.5}, "feat": {_COL: 0.5}}
    fc = CorrelationProfileResult(pearson_matrix=pearson_matrix)
    strategy, signals = _router_route(cp, n_rows=1000, feature_correlation=fc)
    assert strategy == ImputationStrategy.KNN


def test_router_mcar_severe_routes_to_mice():
    """MCAR Severe → MICE regardless of distribution shape."""
    cp = _numeric_cp(null_count=30, total_rows=100, severity=MissingSeverity.Severe,
                     kurtosis_tag=KurtosisTag.Leptokurtic)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.MICE


def test_router_bounded_discrete_unpredictable_routes_to_mode():
    """BoundedDiscrete + Unpredictable → Mode."""
    cp = _numeric_cp(null_count=5, total_rows=100, severity=MissingSeverity.High,
                     numeric_kind=NumericKind.BoundedDiscrete)
    cp.stats = NumericStats(nonlinearity_tag=NonlinearityTag.Unpredictable)
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Mode
    assert any("unpredictable_guard" in s for s in signals)


def test_router_bounded_discrete_near_constant_routes_to_mode():
    """BoundedDiscrete + NearConstant → Mode."""
    cp = _numeric_cp(null_count=2, total_rows=100, severity=MissingSeverity.Minor,
                     numeric_kind=NumericKind.BoundedDiscrete,
                     numeric_flags=[NumericFlag.NearConstant])
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.Mode
    assert any("near_constant" in s for s in signals)


def test_router_multi_mar_routes_to_mice():
    """Multi-MAR (≥2 MARSuspect columns) → MICE."""
    cp = _numeric_cp(null_count=5, total_rows=100, severity=MissingSeverity.Minor,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=[])
    strategy, signals = _router_route(cp, multi_mar=True)
    assert strategy == ImputationStrategy.MICE
    assert any("multi-MAR" in s for s in signals)


def test_router_mar_severe_routes_to_mice():
    """MAR Severe → MICE."""
    cp = _numeric_cp(null_count=30, total_rows=100, severity=MissingSeverity.Severe,
                     flags=[MissingnessFlag.MARSuspect], correlated_with=["x"])
    strategy, signals = _router_route(cp)
    assert strategy == ImputationStrategy.MICE


def test_router_size_guards_fail_falls_to_median():
    """MCAR High + all size guards fail → Median."""
    config = NumericImputationConfig(
        knn_max_rows=10, knn_max_features=0, regression_min_rows=10_000
    )
    cp = _numeric_cp(null_count=20, total_rows=100, severity=MissingSeverity.High)
    strategy, signals = _router_route(cp, n_rows=100, config=config)
    assert strategy == ImputationStrategy.Median
    assert any("all size guards failed" in s for s in signals)


# ---------------------------------------------------------------------------
# per_column_strategy — Priority 1.5 override (Scope 14)
# ---------------------------------------------------------------------------


def _fit_one_with_config(
    df: pl.DataFrame,
    cp: ColumnProfile,
    config: NumericImputationConfig,
    mnar: set[str] | None = None,
) -> ColumnImputationRecord:
    profile = _make_profile(_COL, cp)
    bundle = NumericImputer().fit(
        train_df=df,
        columns=[_COL],
        profile=profile,
        config=config,
        mnar_columns=mnar or _NO_MNAR,
    )
    assert len(bundle.records) == 1
    return bundle.records[0]


def test_per_column_strategy_overrides_mcar_minor_to_regression():
    """MCAR Minor profile with per_column_strategy=Regression produces strategy=Regression."""
    rng = np.random.default_rng(0)
    n = 600
    values = [None if i % 10 == 0 else rng.standard_normal() for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()
    df = pl.DataFrame({
        _COL: pl.Series(values, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })
    cp = _numeric_cp(
        null_count=60, total_rows=n, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Normal,
    )
    feat_cp = ColumnProfile(
        name="feat",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=None,
        stats=NumericStats(),
    )
    profile = StructuralProfileResult()
    profile.columns[_COL] = cp
    profile.columns["feat"] = feat_cp
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Regression},
        regression_min_rows=500,
    )
    bundle = NumericImputer().fit(
        train_df=df, columns=[_COL, "feat"], profile=profile,
        config=config, mnar_columns=set(),
    )
    rec = next(r for r in bundle.records if r.column == _COL)
    assert rec.strategy == ImputationStrategy.Regression


def test_per_column_strategy_fires_before_mnar():
    """A column with per_column_strategy override is NOT treated as MNAR."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, None], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=2, total_rows=4, severity=MissingSeverity.Moderate)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Median},
    )
    # mnar_columns does NOT include _COL — so no MNAR conflict; just checking priority
    rec = _fit_one_with_config(df, cp, config)
    assert rec.strategy == ImputationStrategy.Median
    assert not any("declared MNAR" in s for s in rec.signals)


def test_per_column_strategy_drop_candidate_still_dropped():
    """DropCandidate columns are dropped even when they appear in per_column_strategy."""
    df = pl.DataFrame({_COL: pl.Series([None] * 60 + [1.0] * 40, dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=60, total_rows=100, severity=MissingSeverity.Severe,
        flags=[MissingnessFlag.DropCandidate],
    )
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Median},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert rec.strategy == ImputationStrategy.Dropped


def test_per_column_strategy_signal_is_recorded():
    """per_column_strategy_override signal is appended to record.signals."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Median},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert any("per_column_strategy_override" in s for s in rec.signals)
    assert any("user forced strategy=median" in s for s in rec.signals)


def test_per_column_constant_fill_alone_routes_to_constant():
    """per_column_constant_fill alone (no Constant in per_column_strategy) routes to Constant."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_constant_fill={_COL: 0.0},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert rec.strategy == ImputationStrategy.Constant
    assert rec.fill_value == pytest.approx(0.0)


def test_per_column_constant_fill_nonzero_value_stored_correctly():
    """per_column_constant_fill with non-zero value is stored in fill_value."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_constant_fill={_COL: -99.0},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert rec.strategy == ImputationStrategy.Constant
    assert rec.fill_value == pytest.approx(-99.0)


def test_per_column_constant_fill_signal_is_recorded():
    """per_column_constant_fill_override signal is appended to record.signals."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=4, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_constant_fill={_COL: 0.0},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert any("per_column_constant_fill_override" in s for s in rec.signals)


def test_per_column_strategy_override_forced_median_is_mean_and_not_mnar():
    """Column with per_column_strategy=Mean gets Mean strategy, not the auto-routed strategy."""
    df = pl.DataFrame({_COL: pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Moderate,
        skewness_severity=SkewSeverity.Severe,
    )
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Mean},
    )
    rec = _fit_one_with_config(df, cp, config)
    assert rec.strategy == ImputationStrategy.Mean


def test_per_column_strategy_non_overridden_column_unchanged():
    """Columns not in per_column_strategy still route normally."""
    df = pl.DataFrame({_COL: pl.Series([1.0, 2.0, 3.0, 4.0, None], dtype=pl.Float64)})
    cp = _numeric_cp(
        null_count=1, total_rows=5, severity=MissingSeverity.Minor,
        skewness_severity=SkewSeverity.Normal,
    )
    config = NumericImputationConfig(
        per_column_strategy={"other_col": ImputationStrategy.Median},
    )
    rec = _fit_one_with_config(df, cp, config)
    # MCAR Minor + Normal skew auto-routes to Mean
    assert rec.strategy == ImputationStrategy.Mean


# ---------------------------------------------------------------------------
# per_column_strategy — fit-time size-guard validation (Scope 14)
# ---------------------------------------------------------------------------


def test_per_column_strategy_regression_raises_when_too_few_rows():
    """ValueError raised before fit when per_column_strategy=Regression and n_rows < regression_min_rows."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=3, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Regression},
        regression_min_rows=500,
    )
    with pytest.raises(ValueError, match="regression_min_rows"):
        _fit_one_with_config(df, cp, config)


def test_per_column_strategy_knn_raises_when_too_many_rows():
    """ValueError raised before fit when per_column_strategy=KNN and n_rows > knn_max_rows."""
    df = pl.DataFrame({_COL: pl.Series([float(i) for i in range(19)] + [None], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=20, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.KNN},
        knn_max_rows=10,
    )
    with pytest.raises(ValueError, match="knn_max_rows"):
        _fit_one_with_config(df, cp, config)


def test_per_column_strategy_knn_raises_when_too_many_features():
    """ValueError raised before fit when per_column_strategy=KNN and n_features > knn_max_features."""
    df = pl.DataFrame({
        _COL: pl.Series([1.0, None, 3.0], dtype=pl.Float64),
        "feat1": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        "feat2": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float64),
    })
    profile = StructuralProfileResult()
    profile.columns[_COL] = _numeric_cp(null_count=1, total_rows=3, severity=MissingSeverity.Minor)
    profile.columns["feat1"] = ColumnProfile(
        name="feat1", semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous, missingness=None, stats=NumericStats(),
    )
    profile.columns["feat2"] = ColumnProfile(
        name="feat2", semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous, missingness=None, stats=NumericStats(),
    )
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.KNN},
        knn_max_features=2,  # 3 columns > 2 → guard fires
    )
    with pytest.raises(ValueError, match="knn_max_features"):
        NumericImputer().fit(
            train_df=df,
            columns=[_COL, "feat1", "feat2"],
            profile=profile,
            config=config,
            mnar_columns=set(),
        )


def test_per_column_strategy_size_guard_error_names_column_strategy_guard_and_values():
    """Error message names the column, strategy, guard name, actual value, and threshold."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=3, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.Regression},
        regression_min_rows=500,
    )
    with pytest.raises(ValueError) as exc_info:
        _fit_one_with_config(df, cp, config)
    err = str(exc_info.value)
    assert _COL in err
    assert "Regression" in err or "regression" in err
    assert "regression_min_rows" in err
    assert "500" in err
    assert "3" in err


def test_per_column_strategy_size_guards_pass_does_not_raise():
    """No error raised when size guards pass for all model-based overrides."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0, 4.0, 5.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=5, severity=MissingSeverity.Minor)
    config = NumericImputationConfig(
        per_column_strategy={_COL: ImputationStrategy.KNN},
        knn_max_rows=50_000,
        knn_max_features=50,
    )
    _fit_one_with_config(df, cp, config)


def test_per_column_strategy_scalar_overrides_not_checked_by_size_guard():
    """Scalar-strategy overrides (Mean, Median, Mode) skip size-guard validation."""
    df = pl.DataFrame({_COL: pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    cp = _numeric_cp(null_count=1, total_rows=3, severity=MissingSeverity.Minor)
    for strategy in (ImputationStrategy.Mean, ImputationStrategy.Median, ImputationStrategy.Mode):
        config = NumericImputationConfig(
            per_column_strategy={_COL: strategy},
            knn_max_rows=0,
            knn_max_features=0,
            regression_min_rows=999_999,
        )
        _fit_one_with_config(df, cp, config)

    # Constant via per_column_constant_fill also skips size-guard validation
    config_const = NumericImputationConfig(
        per_column_constant_fill={_COL: 0.0},
        knn_max_rows=0,
        knn_max_features=0,
        regression_min_rows=999_999,
    )
    _fit_one_with_config(df, cp, config_const)
