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


# ---------------------------------------------------------------------------
# numeric_sentinels — sentinel rows counted as effective nulls (Int64)
# ---------------------------------------------------------------------------


def test_sentinel_on_int64_counted_in_effective_null_count():
    df = pl.DataFrame({"age": pl.Series([25, -999, 30, -999, 40], dtype=pl.Int64)})
    profiler = MissingnessProfiler(numeric_sentinels={"age": [-999.0]})
    profile = profiler.profile(df, ["age"]).columns["age"]

    assert profile.effective_null_count == 2
    assert profile.standard_null_count == 0


def test_sentinel_on_int64_effective_null_ratio_reflects_sentinel_rows():
    n = 10
    values = [-999, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profiler = MissingnessProfiler(numeric_sentinels={"x": [-999.0]})
    profile = profiler.profile(df, ["x"]).columns["x"]

    assert abs(profile.effective_null_ratio - 1 / n) < 1e-10


def test_sentinel_on_float64_counted_alongside_inf_and_nan():
    df = pl.DataFrame({
        "val": pl.Series([-999.0, float("nan"), float("inf"), 1.0], dtype=pl.Float64)
    })
    profiler = MissingnessProfiler(numeric_sentinels={"val": [-999.0]})
    profile = profiler.profile(df, ["val"]).columns["val"]

    assert profile.effective_null_count == 3


def test_sentinel_and_native_null_counts_are_additive():
    df = pl.DataFrame({"x": pl.Series([-999, None, 1, 2], dtype=pl.Int64)})
    profiler = MissingnessProfiler(numeric_sentinels={"x": [-999.0]})
    profile = profiler.profile(df, ["x"]).columns["x"]

    assert profile.effective_null_count == 2
    assert profile.standard_null_count == 1


def test_sentinel_severity_classification_uses_sentinel_inclusive_ratio():
    # 6 sentinel rows in 10 → ratio 0.60 → Severe (>= severity_high default 0.20)
    values = [-999] * 6 + [1, 2, 3, 4]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profiler = MissingnessProfiler(numeric_sentinels={"x": [-999.0]})
    profile = profiler.profile(df, ["x"]).columns["x"]

    assert profile.severity == MissingSeverity.Severe


def test_sentinel_column_over_drop_threshold_receives_drop_candidate_flag():
    # col_drop_threshold default = 0.50 — 6/10 sentinel rows exceeds it
    values = [-999] * 6 + [1, 2, 3, 4]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profiler = MissingnessProfiler(numeric_sentinels={"x": [-999.0]})
    result = profiler.profile(df, ["x"])

    assert MissingnessFlag.DropCandidate in result.columns["x"].flags


def test_sentinel_rows_participate_in_mar_correlation_detection():
    # Identical sentinel pattern in two columns → correlation = 1.0 → MARSuspect
    values = [-999, -999, -999, 4, 5, 6, 7, 8, 9, 10]
    df = pl.DataFrame({
        "a": pl.Series(values, dtype=pl.Int64),
        "b": pl.Series(values, dtype=pl.Int64),
    })
    profiler = MissingnessProfiler(numeric_sentinels={"a": [-999.0], "b": [-999.0]})
    result = profiler.profile(df, ["a", "b"])

    assert MissingnessFlag.MARSuspect in result.columns["a"].flags
    assert MissingnessFlag.MARSuspect in result.columns["b"].flags


def test_sentinels_none_output_identical_to_default_behaviour():
    df = pl.DataFrame({
        "a": pl.Series([1, None, 3], dtype=pl.Int64),
        "b": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64),
    })
    result_default = MissingnessProfiler().profile(df, ["a", "b"])
    result_none = MissingnessProfiler(numeric_sentinels=None).profile(df, ["a", "b"])

    for col in ["a", "b"]:
        assert (
            result_default.columns[col].effective_null_count
            == result_none.columns[col].effective_null_count
        )


def test_column_without_declared_sentinel_is_unaffected():
    df = pl.DataFrame({
        "a": pl.Series([-999, 1, 2], dtype=pl.Int64),
        "b": pl.Series([-999, 1, 2], dtype=pl.Int64),
    })
    # Sentinel declared only for "a"
    profiler = MissingnessProfiler(numeric_sentinels={"a": [-999.0]})
    result = profiler.profile(df, ["a", "b"])

    assert result.columns["a"].effective_null_count == 1
    assert result.columns["b"].effective_null_count == 0


def test_multiple_sentinels_per_column_all_counted():
    df = pl.DataFrame({
        "x": pl.Series([-999, 9999, 0, 42], dtype=pl.Int32)
    })
    profiler = MissingnessProfiler(numeric_sentinels={"x": [-999.0, 9999.0]})
    profile = profiler.profile(df, ["x"]).columns["x"]

    assert profile.effective_null_count == 2


# ---------------------------------------------------------------------------
# string_sentinels — user-declared replace semantics
# ---------------------------------------------------------------------------


def test_string_sentinel_counted_in_effective_null_count():
    df = pl.DataFrame({"status": pl.Series(["N/A", "active", "N/A"], dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"status": ["N/A"]})
    profile = profiler.profile(df, ["status"]).columns["status"]

    assert profile.effective_null_count == 2
    assert profile.standard_null_count == 0


def test_string_sentinel_hardcoded_defaults_suppressed_for_declared_column():
    # "?" is in _SENTINEL_STRINGS; with a declaration it must NOT be counted.
    df = pl.DataFrame({"col": pl.Series(["N/A", "?", "valid"], dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"col": ["N/A"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert profile.effective_null_count == 1  # only "N/A"; "?" suppressed


def test_string_sentinel_empty_and_whitespace_always_counted():
    df = pl.DataFrame({"col": pl.Series(["", "  ", "declared", "real"], dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"col": ["declared"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert profile.effective_null_count == 3  # "", "  ", "declared"


def test_string_sentinel_effective_null_ratio_reflects_declaration():
    df = pl.DataFrame({"col": pl.Series(["N/A", "ok", "N/A", "ok", "N/A"], dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"col": ["N/A"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert abs(profile.effective_null_ratio - 3 / 5) < 1e-10


def test_string_sentinel_severity_uses_declaration_inclusive_ratio():
    # 6 declared-sentinel rows in 10 → ratio 0.60 → Severe (>= severity_high 0.20)
    values = ["N/A"] * 6 + ["real"] * 4
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"col": ["N/A"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert profile.severity == MissingSeverity.Severe


def test_string_sentinel_over_drop_threshold_receives_drop_candidate_flag():
    # 6/10 declared-sentinel rows → ratio 0.60 > default col_drop_threshold 0.50
    values = ["N/A"] * 6 + ["real"] * 4
    df = pl.DataFrame({"col": pl.Series(values, dtype=pl.String)})
    profiler = MissingnessProfiler(string_sentinels={"col": ["N/A"]})
    result = profiler.profile(df, ["col"])

    assert MissingnessFlag.DropCandidate in result.columns["col"].flags


def test_string_sentinel_rows_participate_in_mar_correlation_detection():
    # Identical declared-sentinel pattern in two columns → correlation = 1.0 → MARSuspect
    values = ["N/A", "N/A", "N/A", "real", "real", "real", "real", "real", "real", "real"]
    df = pl.DataFrame({
        "a": pl.Series(values, dtype=pl.String),
        "b": pl.Series(values, dtype=pl.String),
    })
    profiler = MissingnessProfiler(string_sentinels={"a": ["N/A"], "b": ["N/A"]})
    result = profiler.profile(df, ["a", "b"])

    assert MissingnessFlag.MARSuspect in result.columns["a"].flags
    assert MissingnessFlag.MARSuspect in result.columns["b"].flags


def test_string_sentinels_none_output_identical_to_default_behaviour():
    df = pl.DataFrame({
        "col": pl.Series(["NA", "?", "ok", None], dtype=pl.String),
    })
    result_default = MissingnessProfiler().profile(df, ["col"])
    result_none = MissingnessProfiler(string_sentinels=None).profile(df, ["col"])

    assert (
        result_default.columns["col"].effective_null_count
        == result_none.columns["col"].effective_null_count
    )


def test_string_sentinel_column_without_declaration_is_unaffected():
    # "a" has a declaration, "b" does not — "b" uses hardcoded defaults.
    df = pl.DataFrame({
        "a": pl.Series(["N/A", "real"], dtype=pl.String),
        "b": pl.Series(["NA", "real"], dtype=pl.String),  # "NA" is a hardcoded default
    })
    profiler = MissingnessProfiler(string_sentinels={"a": ["N/A"]})
    result = profiler.profile(df, ["a", "b"])

    assert result.columns["a"].effective_null_count == 1   # declared "N/A"
    assert result.columns["b"].effective_null_count == 1   # hardcoded "NA" still fires


def test_string_sentinel_declared_matching_is_case_insensitive():
    df = pl.DataFrame({
        "col": pl.Series(["missing", "MISSING", "Missing", "real"], dtype=pl.String)
    })
    profiler = MissingnessProfiler(string_sentinels={"col": ["missing"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert profile.effective_null_count == 3


def test_string_sentinel_multiple_declared_values_all_counted():
    df = pl.DataFrame({
        "col": pl.Series(["N/A", "unknown", "real", "missing"], dtype=pl.String)
    })
    profiler = MissingnessProfiler(string_sentinels={"col": ["N/A", "unknown", "missing"]})
    profile = profiler.profile(df, ["col"]).columns["col"]

    assert profile.effective_null_count == 3


# ---------------------------------------------------------------------------
# RowMissingnessDistribution — row_missingness_p90
# ---------------------------------------------------------------------------


def test_row_missingness_p90_is_zero_for_fully_populated_dataset():
    """A dataset with no effective nulls produces p90 == 0."""
    df = pl.DataFrame({
        "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
        "b": pl.Series([10, 20, 30, 40, 50], dtype=pl.Int64),
    })
    result = MissingnessProfiler().profile(df, ["a", "b"])
    assert result.row_distribution.row_missingness_p90 == 0


def test_row_missingness_p90_single_column_all_missing_is_one():
    """A single fully-null column produces p90 == 1 (every row has 1 missing column)."""
    df = pl.DataFrame({"x": pl.Series([None, None, None, None, None], dtype=pl.Int64)})
    result = MissingnessProfiler().profile(df, ["x"])
    assert result.row_distribution.row_missingness_p90 == 1


def test_row_missingness_p90_reflects_actual_90th_percentile():
    """p90 matches numpy's 90th percentile of per-row effective-null counts."""
    import numpy as np

    # 10 rows, 3 columns. Rows 0-5: no nulls. Rows 6-9: all 3 columns null.
    # Per-row null count: [0, 0, 0, 0, 0, 0, 3, 3, 3, 3]
    # 90th percentile: position 9 * 0.9 = 8.1 → between sorted[8]=3, sorted[9]=3 → 3
    vals = [None] * 4
    ok = [1, 2, 3, 4]
    df = pl.DataFrame({
        "a": pl.Series([1, 2, 3, 4, 5, 6, None, None, None, None], dtype=pl.Int64),
        "b": pl.Series([1, 2, 3, 4, 5, 6, None, None, None, None], dtype=pl.Int64),
        "c": pl.Series([1, 2, 3, 4, 5, 6, None, None, None, None], dtype=pl.Int64),
    })
    result = MissingnessProfiler().profile(df, ["a", "b", "c"])
    per_row = np.array([0, 0, 0, 0, 0, 0, 3, 3, 3, 3])
    expected_p90 = int(np.percentile(per_row, 90))
    assert result.row_distribution.row_missingness_p90 == expected_p90


def test_row_distribution_accessible_on_missingness_profile_result():
    """row_distribution is present on MissingnessProfileResult after profile()."""
    from dataforge_ml.profiling._missingness_config import RowMissingnessDistribution

    df = pl.DataFrame({"x": pl.Series([1, None, 3], dtype=pl.Int64)})
    result = MissingnessProfiler().profile(df, ["x"])
    assert hasattr(result, "row_distribution")
    assert isinstance(result.row_distribution, RowMissingnessDistribution)
