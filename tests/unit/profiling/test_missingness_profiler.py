import polars as pl
import pytest

from dataforge_ml.profiling._missingness_profiler import MissingnessProfiler
from dataforge_ml.profiling._missingness_config import MissingnessFlag, MissingnessProfileConfig, MissingSeverity


# ---------------------------------------------------------------------------
# Instantiation — no config required
# ---------------------------------------------------------------------------


def test_instantiates_with_no_arguments():
    profiler = MissingnessProfiler()
    assert profiler is not None


# ---------------------------------------------------------------------------
# null_count equals actual null count in the series
# ---------------------------------------------------------------------------


def test_null_count_equals_actual_null_count():
    values = [1, None, 3, None, None, 6, 7, None]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.standard_null_count == df["x"].null_count()


# ---------------------------------------------------------------------------
# null_ratio equals null_count / n_rows
# ---------------------------------------------------------------------------


def test_null_ratio_equals_null_count_over_n_rows():
    values = [10, None, 30, None, 50]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    expected_ratio = profile.standard_null_count / df.height
    assert abs(profile.standard_null_ratio - expected_ratio) < 1e-10


# ---------------------------------------------------------------------------
# All-null column produces null_ratio == 1.0 without crashing
# ---------------------------------------------------------------------------


def test_all_null_column_ratio_is_one():
    df = pl.DataFrame({"x": pl.Series([None, None, None, None], dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.effective_null_ratio == 1.0


# ---------------------------------------------------------------------------
# Fully populated column has null_count == 0 and null_ratio == 0.0
# ---------------------------------------------------------------------------


def test_fully_populated_column_has_zero_nulls():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.standard_null_count == 0
    assert profile.standard_null_ratio == 0.0


# ---------------------------------------------------------------------------
# Sentinel strings in a String column are counted as effectively null
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", ["NA", "NULL", "NONE", "NAN", "?"])
def test_sentinel_string_counted_as_effective_null(sentinel):
    df = pl.DataFrame({"col": pl.Series(["valid", sentinel, "also_valid"], dtype=pl.String)})
    profile = MissingnessProfiler().profile(df, ["col"]).columns["col"]
    assert profile.effective_null_count == 1
    assert profile.standard_null_count == 0


def test_all_sentinel_variants_counted():
    df = pl.DataFrame({"col": pl.Series(["NA", "NULL", "NONE", "NAN", "?", "real"], dtype=pl.String)})
    profile = MissingnessProfiler().profile(df, ["col"]).columns["col"]
    assert profile.effective_null_count == 5


# ---------------------------------------------------------------------------
# Sentinel detection is unconditional for String columns — no override suppresses it
# ---------------------------------------------------------------------------


def test_sentinel_detection_unconditional_for_string_columns():
    # A String column with values that look numeric still gets sentinel detection.
    # Previously a Numeric SemanticType override would have suppressed this — it no longer does.
    df = pl.DataFrame({"score": pl.Series(["1.0", "2.5", "NA", "3.1"], dtype=pl.String)})
    profile = MissingnessProfiler().profile(df, ["score"]).columns["score"]
    assert profile.effective_null_count == 1


# ---------------------------------------------------------------------------
# Float columns count NaN and Inf as effectively null
# ---------------------------------------------------------------------------


def test_float_nan_counted_as_effective_null():
    df = pl.DataFrame({"x": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.effective_null_count == 1
    assert profile.standard_null_count == 0


def test_float_inf_counted_as_effective_null():
    df = pl.DataFrame({"x": pl.Series([1.0, float("inf"), -float("inf"), 4.0], dtype=pl.Float64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.effective_null_count == 2


def test_float32_nan_and_inf_counted():
    df = pl.DataFrame({"x": pl.Series([1.0, float("nan"), float("inf")], dtype=pl.Float32)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.effective_null_count == 2


# ---------------------------------------------------------------------------
# MissingnessProfileConfig — construction and defaults
# ---------------------------------------------------------------------------


def test_config_instantiates_with_defaults():
    cfg = MissingnessProfileConfig()
    assert cfg.severity_minor == 0.01
    assert cfg.severity_moderate == 0.05
    assert cfg.severity_high == 0.20
    assert cfg.mar_correlation_threshold == 0.60
    assert cfg.col_drop_threshold == 0.50


def test_profiler_accepts_config_parameter():
    cfg = MissingnessProfileConfig(severity_high=0.30)
    profiler = MissingnessProfiler(config=cfg)
    assert profiler is not None


def test_profiler_constructed_without_config_uses_defaults():
    profiler = MissingnessProfiler()
    assert profiler._config.severity_high == 0.20
    assert profiler._config.col_drop_threshold == 0.50


# ---------------------------------------------------------------------------
# Severity boundary override
# ---------------------------------------------------------------------------


def test_severity_high_override_reclassifies_column_from_severe_to_high():
    # 22 nulls out of 100 rows → ratio 0.22
    # Default severity_high=0.20: 0.22 >= 0.20 → Severe
    # Override severity_high=0.25: 0.22 < 0.25 → High
    values = [None] * 22 + list(range(78))
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.Int64)})

    default_result = MissingnessProfiler().profile(df, ["col"])
    assert default_result.columns["col"].severity == MissingSeverity.Severe

    cfg = MissingnessProfileConfig(severity_high=0.25)
    overridden_result = MissingnessProfiler(config=cfg).profile(df, ["col"])
    assert overridden_result.columns["col"].severity == MissingSeverity.High


def test_severity_moderate_override_reclassifies_column():
    # 3 nulls out of 100 rows → ratio 0.03
    # Default severity_moderate=0.05: 0.03 < 0.05 → Moderate
    # Override severity_moderate=0.02: 0.03 >= 0.02 → High (< default severity_high=0.20)
    values = [None] * 3 + list(range(97))
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.Int64)})

    default_result = MissingnessProfiler().profile(df, ["col"])
    assert default_result.columns["col"].severity == MissingSeverity.Moderate

    cfg = MissingnessProfileConfig(severity_moderate=0.02)
    overridden_result = MissingnessProfiler(config=cfg).profile(df, ["col"])
    assert overridden_result.columns["col"].severity == MissingSeverity.High


# ---------------------------------------------------------------------------
# DropCandidate flag override
# ---------------------------------------------------------------------------


def test_col_drop_threshold_override_triggers_drop_candidate():
    # 40 nulls out of 100 rows → ratio 0.40
    # Default col_drop_threshold=0.50: 0.40 > 0.50 is False → no DropCandidate
    # Override col_drop_threshold=0.30: 0.40 > 0.30 is True → DropCandidate
    values = [None] * 40 + list(range(60))
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.Int64)})

    default_result = MissingnessProfiler().profile(df, ["col"])
    assert MissingnessFlag.DropCandidate not in default_result.columns["col"].flags

    cfg = MissingnessProfileConfig(col_drop_threshold=0.30)
    overridden_result = MissingnessProfiler(config=cfg).profile(df, ["col"])
    assert MissingnessFlag.DropCandidate in overridden_result.columns["col"].flags


def test_col_drop_threshold_override_suppresses_drop_candidate():
    # 60 nulls out of 100 rows → ratio 0.60
    # Default col_drop_threshold=0.50: 0.60 > 0.50 → DropCandidate
    # Override col_drop_threshold=0.70: 0.60 > 0.70 is False → no DropCandidate
    values = [None] * 60 + list(range(40))
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.Int64)})

    default_result = MissingnessProfiler().profile(df, ["col"])
    assert MissingnessFlag.DropCandidate in default_result.columns["col"].flags

    cfg = MissingnessProfileConfig(col_drop_threshold=0.70)
    overridden_result = MissingnessProfiler(config=cfg).profile(df, ["col"])
    assert MissingnessFlag.DropCandidate not in overridden_result.columns["col"].flags


# ---------------------------------------------------------------------------
# MARSuspect flag override
# ---------------------------------------------------------------------------


def test_mar_threshold_default_flags_correlated_columns():
    # Identical null patterns → correlation = 1.0 > default 0.60 → MARSuspect
    null_rows: list = [None, None, None, 4, 5, 6, 7, 8, 9, 10]
    df = pl.DataFrame({
        "a": pl.Series(null_rows, dtype=pl.Int64),
        "b": pl.Series(null_rows, dtype=pl.Int64),
    })
    result = MissingnessProfiler().profile(df, ["a", "b"])
    assert MissingnessFlag.MARSuspect in result.columns["a"].flags
    assert MissingnessFlag.MARSuspect in result.columns["b"].flags


def test_mar_threshold_override_suppresses_mar_suspect():
    # Same identical null patterns (corr = 1.0) but threshold raised to 2.0
    # (correlation can never exceed 1.0) → MARSuspect never fires
    null_rows: list = [None, None, None, 4, 5, 6, 7, 8, 9, 10]
    df = pl.DataFrame({
        "a": pl.Series(null_rows, dtype=pl.Int64),
        "b": pl.Series(null_rows, dtype=pl.Int64),
    })
    cfg = MissingnessProfileConfig(mar_correlation_threshold=2.0)
    result = MissingnessProfiler(config=cfg).profile(df, ["a", "b"])
    assert MissingnessFlag.MARSuspect not in result.columns["a"].flags
    assert MissingnessFlag.MARSuspect not in result.columns["b"].flags


# ---------------------------------------------------------------------------
# MissingnessProfileConfig serialisation round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip_preserves_all_fields():
    cfg = MissingnessProfileConfig(
        severity_minor=0.02,
        severity_moderate=0.08,
        severity_high=0.25,
        mar_correlation_threshold=0.75,
        col_drop_threshold=0.60,
    )
    restored = MissingnessProfileConfig.from_dict(cfg.to_dict())
    assert restored.severity_minor == cfg.severity_minor
    assert restored.severity_moderate == cfg.severity_moderate
    assert restored.severity_high == cfg.severity_high
    assert restored.mar_correlation_threshold == cfg.mar_correlation_threshold
    assert restored.col_drop_threshold == cfg.col_drop_threshold


def test_config_from_dict_uses_defaults_for_missing_keys():
    restored = MissingnessProfileConfig.from_dict({})
    default = MissingnessProfileConfig()
    assert restored.severity_minor == default.severity_minor
    assert restored.severity_moderate == default.severity_moderate
    assert restored.severity_high == default.severity_high
    assert restored.mar_correlation_threshold == default.mar_correlation_threshold
    assert restored.col_drop_threshold == default.col_drop_threshold


def test_config_to_dict_contains_all_keys():
    cfg = MissingnessProfileConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "severity_minor",
        "severity_moderate",
        "severity_high",
        "mar_correlation_threshold",
        "col_drop_threshold",
    }
