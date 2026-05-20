"""
Tests for issue #66: type profilers accept no constructor arguments and
profile every column in the list they receive without internal filtering.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from dataforge_ml.profiling._numeric_profiler import NumericProfiler
from dataforge_ml.profiling._categorical import CategoricalProfiler
from dataforge_ml.profiling._datetime_profiler import DatetimeProfiler
from dataforge_ml.profiling._boolean_profiler import BooleanProfiler
from dataforge_ml.profiling._text_profiler import TextProfiler


# ---------------------------------------------------------------------------
# No-argument instantiation
# ---------------------------------------------------------------------------


def test_numeric_profiler_no_args():
    NumericProfiler()


def test_categorical_profiler_no_args():
    CategoricalProfiler()


def test_datetime_profiler_no_args():
    DatetimeProfiler()


def test_boolean_profiler_no_args():
    BooleanProfiler()


def test_text_profiler_no_args():
    TextProfiler()


# ---------------------------------------------------------------------------
# All columns in the list are profiled (no internal eligibility gate)
# ---------------------------------------------------------------------------


def test_numeric_profiler_profiles_all_columns():
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            "b": pl.Series([10, 20, 30], dtype=pl.Int64),
        }
    )
    result = NumericProfiler().profile(df, ["a", "b"])
    assert set(result.analysed_columns) == {"a", "b"}
    assert "a" in result.columns
    assert "b" in result.columns


def test_categorical_profiler_profiles_all_columns():
    df = pl.DataFrame(
        {
            "x": pl.Series(["cat", "dog", "cat"], dtype=pl.Utf8),
            "y": pl.Series(["red", "blue", "red"], dtype=pl.Utf8),
        }
    )
    result = CategoricalProfiler().profile(df, ["x", "y"])
    assert set(result.analysed_columns) == {"x", "y"}
    assert "x" in result.columns
    assert "y" in result.columns


def test_datetime_profiler_profiles_all_columns():
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(5)]
    df = pl.DataFrame(
        {
            "d1": pl.Series(dates, dtype=pl.Date),
            "d2": pl.Series(dates, dtype=pl.Date),
        }
    )
    result = DatetimeProfiler().profile(df, ["d1", "d2"])
    assert set(result.analysed_columns) == {"d1", "d2"}
    assert "d1" in result.columns
    assert "d2" in result.columns


def test_boolean_profiler_profiles_all_columns():
    df = pl.DataFrame(
        {
            "flag1": pl.Series([True, False, True], dtype=pl.Boolean),
            "flag2": pl.Series([False, True, False], dtype=pl.Boolean),
        }
    )
    result = BooleanProfiler().profile(df, ["flag1", "flag2"])
    assert set(result.analysed_columns) == {"flag1", "flag2"}
    assert "flag1" in result.columns
    assert "flag2" in result.columns


def test_text_profiler_profiles_all_columns():
    df = pl.DataFrame(
        {
            "desc": pl.Series(["hello world", "foo bar baz"], dtype=pl.Utf8),
            "notes": pl.Series(["a quick brown fox", "lazy dog"], dtype=pl.Utf8),
        }
    )
    result = TextProfiler().profile(df, ["desc", "notes"])
    assert set(result.analysed_columns) == {"desc", "notes"}
    assert "desc" in result.columns
    assert "notes" in result.columns


# ---------------------------------------------------------------------------
# No _eligible method on any type profiler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profiler_cls",
    [NumericProfiler, CategoricalProfiler, DatetimeProfiler, BooleanProfiler, TextProfiler],
)
def test_no_eligible_method(profiler_cls):
    assert not hasattr(profiler_cls, "_eligible"), (
        f"{profiler_cls.__name__} must not have an _eligible method"
    )


# ---------------------------------------------------------------------------
# No config attribute on any type profiler instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profiler_cls",
    [NumericProfiler, CategoricalProfiler, DatetimeProfiler, BooleanProfiler, TextProfiler],
)
def test_no_config_attribute(profiler_cls):
    instance = profiler_cls()
    assert not hasattr(instance, "config"), (
        f"{profiler_cls.__name__} instance must not have a config attribute"
    )
