import polars as pl

from dataforge_ml.profiling._boolean_profiler import BooleanProfiler
from dataforge_ml.profiling._boolean_config import BooleanProfileResult, BooleanStats


# ---------------------------------------------------------------------------
# Result type & analysed_columns
# ---------------------------------------------------------------------------


def test_result_type_and_analysed_columns():
    df = pl.DataFrame(
        {
            "flag": pl.Series([True, False, True], dtype=pl.Boolean),
            "score": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        }
    )
    result = BooleanProfiler().profile(df, ["flag", "score"])
    assert isinstance(result, BooleanProfileResult)
    assert "flag" in result.analysed_columns


def test_utf8_boolean_strings_are_profiled():
    # TypeDetector flags Utf8 columns containing "true"/"false" as SemanticType.Boolean.
    # BooleanProfiler must handle them — _to_bool_series maps known strings.
    df = pl.DataFrame({"active": pl.Series(["true", "false", "true", "false", "true"], dtype=pl.Utf8)})
    result = BooleanProfiler().profile(df, ["active"])
    assert "active" in result.analysed_columns
    stats = result.columns["active"]
    assert stats.true_count == 3
    assert stats.false_count == 2


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_true_false_count_sum_equals_non_null_count():
    values = [True, False, True, None, True, False, None]
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    non_null_count = df["flag"].drop_nulls().len()
    assert stats.true_count + stats.false_count == non_null_count


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------


def test_true_ratio_plus_false_ratio_equals_one():
    values = [True, True, False, True, False, True]
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert abs(stats.true_ratio + stats.false_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


def test_tied_column_mode_is_none():
    values = [True] * 5 + [False] * 5
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert stats.mode is None


# ---------------------------------------------------------------------------
# Integer {0, 1} columns
# ---------------------------------------------------------------------------


def test_integer_01_eligible_with_correct_counts_and_ratios():
    values = [1, 0, 1, 1, 0, None]
    df = pl.DataFrame({"bin": pl.Series(values, dtype=pl.Int64)})
    result = BooleanProfiler().profile(df, ["bin"])
    assert "bin" in result.analysed_columns
    stats = result.columns["bin"]
    non_null = [v for v in values if v is not None]
    expected_true = sum(non_null)
    expected_false = len(non_null) - expected_true
    assert stats.true_count == expected_true
    assert stats.false_count == expected_false
    assert abs(stats.true_ratio + stats.false_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# All-null boolean column
# ---------------------------------------------------------------------------


def test_all_null_boolean_returns_default_stats_without_crashing():
    df = pl.DataFrame({"flag": pl.Series([None, None, None], dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert isinstance(stats, BooleanStats)
    assert stats.true_count == 0
    assert stats.false_count == 0


def test_override_coercion_error_raised_for_total_failure():
    import pytest
    from dataforge_ml.profiling import OverrideCoercionError

    df = pl.DataFrame({"bool_col": pl.Series(["apple", "banana", "cherry"])})
    with pytest.raises(OverrideCoercionError, match="completely failed coercion"):
        BooleanProfiler().profile(df, ["bool_col"], user_overrides={"bool_col"})


def test_override_coercion_error_not_raised_for_partial_failure():
    from dataforge_ml.profiling import OverrideCoercionError
    # 1 valid true string, 2 invalid
    df = pl.DataFrame({"bool_col": pl.Series(["true", "banana", "cherry"])})
    result = BooleanProfiler().profile(df, ["bool_col"], user_overrides={"bool_col"})
    assert "bool_col" in result.analysed_columns


def test_override_coercion_error_not_raised_for_auto_detected_total_failure():
    df = pl.DataFrame({"bool_col": pl.Series(["apple", "banana", "cherry"])})
    result = BooleanProfiler().profile(df, ["bool_col"])
    # bool column will be profiled, but length will be 0 -> it just returns empty stats
    stats = result.columns["bool_col"]
    assert stats.true_count == 0



# ---------------------------------------------------------------------------
# FormatMismatch flag + BooleanFlag / has_flag machinery
# ---------------------------------------------------------------------------


from dataforge_ml.profiling._boolean_config import BooleanFlag


def test_boolean_stats_flags_defaults_empty_and_has_flag():
    stats = BooleanStats()
    assert stats.flags == []
    assert not stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_stats_to_dict_includes_flags_as_strings():
    stats = BooleanStats(flags=[BooleanFlag.FormatMismatch])
    d = stats.to_dict()
    assert d["flags"] == ["format_mismatch"]
    assert stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_format_mismatch_flag_fires_on_dirty_column():
    df = pl.DataFrame({"b": pl.Series(["yes", "no", "maybe", "yes"], dtype=pl.Utf8)})
    result = BooleanProfiler().profile(df, ["b"])
    stats = result.columns["b"]
    assert stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_format_mismatch_flag_absent_on_clean_column():
    df = pl.DataFrame({"b": pl.Series(["yes", "no", "yes", "no"], dtype=pl.Utf8)})
    result = BooleanProfiler().profile(df, ["b"])
    stats = result.columns["b"]
    assert not stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_format_mismatch_flag_absent_on_native_bool_column():
    df = pl.DataFrame({"b": pl.Series([True, False, True], dtype=pl.Boolean)})
    result = BooleanProfiler().profile(df, ["b"])
    stats = result.columns["b"]
    assert not stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_format_mismatch_does_not_alter_existing_stats():
    clean = pl.DataFrame({"b": pl.Series(["yes", "no", "yes"], dtype=pl.Utf8)})
    dirty = pl.DataFrame({"b": pl.Series(["yes", "no", "yes", "maybe"], dtype=pl.Utf8)})
    clean_stats = BooleanProfiler().profile(clean, ["b"]).columns["b"]
    dirty_stats = BooleanProfiler().profile(dirty, ["b"]).columns["b"]
    # The unrecognized value is dropped exactly as before the flag existed.
    assert dirty_stats.true_count == clean_stats.true_count
    assert dirty_stats.false_count == clean_stats.false_count
    assert dirty_stats.true_ratio == clean_stats.true_ratio
    assert dirty_stats.mode == clean_stats.mode
