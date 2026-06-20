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
                                                      _fallback_to_median)
from dataforge_ml.profiling._config import (ColumnProfile, NumericKind,
                                            StructuralProfileResult)
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile, MissingnessFlag, MissingSeverity)
from dataforge_ml.profiling._numeric_config import (NonlinearityTag,
                                                    NumericStats, SkewSeverity)

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
    stats = NumericStats(skewness_severity=skewness_severity)
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


def test_discrete_column_minor_normal_gets_mean_strategy():
    """BoundedDiscrete + Minor + Normal skew → Mean (sub-chain step 5)."""
    values = [1, 2, 1, 1, None, 3, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean
    assert rec.fill_value == pytest.approx(9 / 6)


def test_discrete_mean_computed_on_training_data():
    """Mean fill value is computed from non-null training values for BoundedDiscrete."""
    values = [5, 5, 5, 1, 1, None]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean
    assert rec.fill_value == pytest.approx(17 / 5)


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


def test_bounded_discrete_minor_normal_skew_gets_mean():
    """BoundedDiscrete + Minor + Normal skew → Mean (sub-chain step 5)."""
    values = [1, 2, 1, 1, None, 3, 1]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=7, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean


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


def test_bounded_discrete_mcar_minor_normal_skew_mean_is_snapped():
    """BoundedDiscrete + MCAR Minor + Normal skew → Mean snapped to [min, max]."""
    values = [1, 5, 1, 5, None, 3]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    cp.stats = NumericStats(min=1.0, max=5.0, skewness_severity=None)
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean
    assert rec.fill_value is not None
    assert 1.0 <= rec.fill_value <= 5.0
    assert any("snapped" in s for s in rec.signals)


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


def test_override_forced_bounded_discrete_gets_mean_on_minor_normal():
    """A column forced to BoundedDiscrete via override + Minor + Normal skew → Mean."""
    values = [25, 32, 45, 28, None, 39]
    df = pl.DataFrame({_COL: pl.Series(values, dtype=pl.Int64)})
    cp = _numeric_cp(
        null_count=1, total_rows=6, severity=MissingSeverity.Minor,
        numeric_kind=NumericKind.BoundedDiscrete,
    )
    rec = _fit_one(df, cp)
    assert rec.strategy == ImputationStrategy.Mean


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
    # MAR routing must not have fired
    assert not any("mar_suspect" in s for s in rec.signals)
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
