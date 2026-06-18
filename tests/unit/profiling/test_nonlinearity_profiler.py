"""
Unit tests for NonlinearityProfiler and NonlinearityProfileConfig (Issue #139).

Four synthetic tag outcomes are covered:
- Linear         : tight linear relationship, homoscedastic
- MonotonicNonlinear : linear conditional mean, heteroscedastic residuals
- ComplexNonlinear   : interaction term — linear model blind, RF succeeds
- Unpredictable  : target is independent white noise
"""

import numpy as np
import polars as pl
import pytest

from dataforge_ml.profiling._nonlinearity_profiler import NonlinearityProfiler
from dataforge_ml.profiling._numeric_config import (
    NonlinearityProfileConfig,
    NonlinearityProfileResult,
    NonlinearitySignals,
    NonlinearityTag,
    NumericStats,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

_N = 500
_RNG = np.random.default_rng(42)

_X1 = _RNG.standard_normal(_N)
_X2 = _RNG.standard_normal(_N)
_NOISE = _RNG.standard_normal(_N)


@pytest.fixture(scope="module")
def linear_df() -> pl.DataFrame:
    """y is a tight linear combination of x1, x2."""
    y = 2.0 * _X1 + 3.0 * _X2 + 0.05 * _NOISE
    return pl.DataFrame({"x1": _X1, "x2": _X2, "y": y})




@pytest.fixture(scope="module")
def complex_nonlinear_df() -> pl.DataFrame:
    """
    y = x1 * x2 + small noise.

    Linear model cannot capture the interaction → R²_linear ≈ 0.
    RF captures it easily → large r2_gap.
    """
    y = _X1 * _X2 + 0.1 * _NOISE
    return pl.DataFrame({"x1": _X1, "x2": _X2, "y": y})


@pytest.fixture(scope="module")
def unpredictable_df() -> pl.DataFrame:
    """y is white noise uncorrelated with x1 and x2."""
    rng2 = np.random.default_rng(99)
    y = rng2.standard_normal(_N)
    return pl.DataFrame({"x1": _X1, "x2": _X2, "y": y})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _profile_col(df: pl.DataFrame, target: str) -> NonlinearitySignals | None:
    """Run NonlinearityProfiler and return signals for target column."""
    cols = list(df.columns)
    result = NonlinearityProfiler(numeric_columns=cols).profile(df)
    return result.columns.get(target)


# ---------------------------------------------------------------------------
# Tag outcomes — four branches
# ---------------------------------------------------------------------------


def test_linear_tag(linear_df):
    signals = _profile_col(linear_df, "y")
    assert signals is not None
    assert signals.tag == NonlinearityTag.Linear


def test_monotonic_nonlinear_tag():
    """Curvature signal (discrepancy ≥ threshold) with small r2_gap → MonotonicNonlinear."""
    profiler = NonlinearityProfiler(numeric_columns=[])
    tag = profiler._assign_tag(
        r2_rf=0.50,
        discrepancy=0.15,
        bp_pvalue=0.40,
        r2_gap=0.04,
        mi=0.01,
    )
    assert tag == NonlinearityTag.MonotonicNonlinear


def test_complex_nonlinear_tag(complex_nonlinear_df):
    signals = _profile_col(complex_nonlinear_df, "y")
    assert signals is not None
    assert signals.tag == NonlinearityTag.ComplexNonlinear


def test_unpredictable_tag(unpredictable_df):
    signals = _profile_col(unpredictable_df, "y")
    assert signals is not None
    assert signals.tag == NonlinearityTag.Unpredictable


# ---------------------------------------------------------------------------
# All four raw signal fields are populated
# ---------------------------------------------------------------------------


def test_all_signal_fields_populated_linear(linear_df):
    signals = _profile_col(linear_df, "y")
    assert signals is not None
    assert signals.spearman_pearson_discrepancy is not None
    assert signals.mean_mutual_information is not None
    assert signals.r2_gap is not None
    assert signals.heteroscedasticity_p_value is not None


def test_all_signal_fields_populated_complex(complex_nonlinear_df):
    signals = _profile_col(complex_nonlinear_df, "y")
    assert signals is not None
    assert signals.spearman_pearson_discrepancy is not None
    assert signals.mean_mutual_information is not None
    assert signals.r2_gap is not None
    assert signals.heteroscedasticity_p_value is not None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_result_type(linear_df):
    result = NonlinearityProfiler(numeric_columns=list(linear_df.columns)).profile(
        linear_df
    )
    assert isinstance(result, NonlinearityProfileResult)


def test_analysed_columns_populated(linear_df):
    result = NonlinearityProfiler(numeric_columns=list(linear_df.columns)).profile(
        linear_df
    )
    assert "y" in result.analysed_columns


def test_columns_dict_matches_analysed(linear_df):
    result = NonlinearityProfiler(numeric_columns=list(linear_df.columns)).profile(
        linear_df
    )
    assert set(result.analysed_columns) == set(result.columns.keys())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_column_produces_empty_result():
    """Only one numeric column — no predictors — nothing should be profiled."""
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = NonlinearityProfiler(numeric_columns=["y"]).profile(df)
    assert result.analysed_columns == []


def test_too_few_rows_skips_column():
    """DataFrame has fewer complete rows than min_rows — column is skipped."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        "x": rng.standard_normal(10).tolist(),
        "y": rng.standard_normal(10).tolist(),
    })
    cfg = NonlinearityProfileConfig(min_rows=20)
    result = NonlinearityProfiler(numeric_columns=["x", "y"], config=cfg).profile(df)
    assert result.analysed_columns == []


def test_precomputed_matrices_are_reused(linear_df):
    """Passing pre-computed matrices should produce the same tag as computing internally."""
    cols = list(linear_df.columns)
    from dataforge_ml.profiling._correlation_profiler import CorrelationProfiler

    p_mat, s_mat = CorrelationProfiler._compute_matrices(linear_df, cols)
    result_with = NonlinearityProfiler(numeric_columns=cols).profile(
        linear_df, pearson_matrix=p_mat, spearman_matrix=s_mat
    )
    result_without = NonlinearityProfiler(numeric_columns=cols).profile(linear_df)
    assert result_with.columns["y"].tag == result_without.columns["y"].tag


# ---------------------------------------------------------------------------
# NonlinearityProfileConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = NonlinearityProfileConfig()
    assert cfg.spearman_pearson_discrepancy_threshold == 0.10
    assert cfg.mutual_information_threshold == 0.05
    assert cfg.r2_gap_threshold == 0.10
    assert cfg.heteroscedasticity_p_value_threshold == 0.05
    assert cfg.r2_rf_unpredictable_threshold == 0.05
    assert cfg.min_rows == 20
    assert cfg.bootstrap_sample_size == 500
    assert cfg.random_state == 42


def test_config_to_dict_roundtrip():
    cfg = NonlinearityProfileConfig(r2_gap_threshold=0.20, min_rows=30)
    restored = NonlinearityProfileConfig.from_dict(cfg.to_dict())
    assert restored.r2_gap_threshold == 0.20
    assert restored.min_rows == 30


def test_config_from_dict_uses_defaults_for_missing_keys():
    cfg = NonlinearityProfileConfig.from_dict({})
    assert cfg.min_rows == 20


# ---------------------------------------------------------------------------
# Wiring through StructuralProfiler
# ---------------------------------------------------------------------------


def test_structural_profiler_populates_numericstats(complex_nonlinear_df):
    """
    After profiling with compute_nonlinearity=True, NumericStats for the
    target column must have nonlinearity_tag and all four signal fields set.
    """
    from dataforge_ml import PipelineConfig, StructuralProfiler
    from dataforge_ml.profiling._config import ProfileConfig

    pc = ProfileConfig(compute_nonlinearity=True)
    pipeline_cfg = PipelineConfig(profiling=pc)
    profiler = StructuralProfiler(config=pipeline_cfg)
    result = profiler.profile(complex_nonlinear_df)

    y_profile = result.columns.get("y")
    assert y_profile is not None
    stats = y_profile.stats
    assert isinstance(stats, NumericStats)
    assert stats.nonlinearity_tag is not None
    assert stats.spearman_pearson_discrepancy is not None
    assert stats.mean_mutual_information is not None
    assert stats.r2_gap is not None
    assert stats.heteroscedasticity_p_value is not None
