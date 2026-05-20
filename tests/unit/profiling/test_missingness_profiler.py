import polars as pl
import pytest

from dataforge_ml.profiling._missingness_profiler import MissingnessProfiler


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
    import math
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
