"""
Unit tests for _resolve_effective_nulls() in isolation.

No orchestrator, no profile, no sub-processors — just the function and DataFrames.
"""

import math

import polars as pl
import pytest

from dataforge_ml.utils._null_normalization import _resolve_effective_nulls


# ---------------------------------------------------------------------------
# String sentinels — each sentinel converts to null
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", ["NA", "NAN", "NULL", "NONE", "?"])
def test_uppercase_sentinel_converted_to_null(sentinel):
    df = pl.DataFrame({"s": pl.Series([sentinel, "valid"], dtype=pl.String)})
    out = _resolve_effective_nulls(df)
    assert out["s"][0] is None
    assert out["s"][1] == "valid"


# ---------------------------------------------------------------------------
# String sentinels — case-insensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", ["na", "Na", "nAn", "null", "Null", "none", "None"])
def test_lowercase_mixed_case_sentinel_converted_to_null(sentinel):
    df = pl.DataFrame({"s": pl.Series([sentinel, "ok"], dtype=pl.String)})
    out = _resolve_effective_nulls(df)
    assert out["s"][0] is None


# ---------------------------------------------------------------------------
# String — empty and whitespace-only strings
# ---------------------------------------------------------------------------


def test_empty_string_converted_to_null():
    df = pl.DataFrame({"s": pl.Series(["", "hello"], dtype=pl.String)})
    out = _resolve_effective_nulls(df)
    assert out["s"][0] is None
    assert out["s"][1] == "hello"


@pytest.mark.parametrize("ws", [" ", "  ", "\t", "\n", "  \t  "])
def test_whitespace_only_string_converted_to_null(ws):
    df = pl.DataFrame({"s": pl.Series([ws, "real"], dtype=pl.String)})
    out = _resolve_effective_nulls(df)
    assert out["s"][0] is None
    assert out["s"][1] == "real"


# ---------------------------------------------------------------------------
# String — valid non-sentinel strings are left unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["hello", "N/A", "0", "NAs", "NULLY", "?!"])
def test_valid_string_left_unchanged(value):
    df = pl.DataFrame({"s": pl.Series([value], dtype=pl.String)})
    out = _resolve_effective_nulls(df)
    assert out["s"][0] == value


# ---------------------------------------------------------------------------
# Float — NaN converted to null
# ---------------------------------------------------------------------------


def test_nan_in_float64_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64)})
    out = _resolve_effective_nulls(df)
    assert out["f"][1] is None
    assert out["f"][0] == pytest.approx(1.0)
    assert out["f"][2] == pytest.approx(3.0)


def test_nan_in_float32_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([1.0, float("nan")], dtype=pl.Float32)})
    out = _resolve_effective_nulls(df)
    assert out["f"][1] is None


# ---------------------------------------------------------------------------
# Float — Inf (positive and negative) converted to null
# ---------------------------------------------------------------------------


def test_positive_inf_in_float64_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([1.0, float("inf"), 3.0], dtype=pl.Float64)})
    out = _resolve_effective_nulls(df)
    assert out["f"][1] is None


def test_negative_inf_in_float64_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([1.0, float("-inf"), 3.0], dtype=pl.Float64)})
    out = _resolve_effective_nulls(df)
    assert out["f"][1] is None


def test_positive_inf_in_float32_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([float("inf"), 2.0], dtype=pl.Float32)})
    out = _resolve_effective_nulls(df)
    assert out["f"][0] is None


def test_negative_inf_in_float32_converted_to_null():
    df = pl.DataFrame({"f": pl.Series([float("-inf"), 2.0], dtype=pl.Float32)})
    out = _resolve_effective_nulls(df)
    assert out["f"][0] is None


# ---------------------------------------------------------------------------
# Non-eligible dtypes — pass through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,value", [
    (pl.Int32, 42),
    (pl.Int64, -1),
    (pl.UInt8, 0),
])
def test_integer_column_unchanged(dtype, value):
    df = pl.DataFrame({"x": pl.Series([value, None], dtype=dtype)})
    out = _resolve_effective_nulls(df)
    assert out["x"][0] == value
    assert out["x"][1] is None


def test_boolean_column_unchanged():
    df = pl.DataFrame({"b": pl.Series([True, False, None], dtype=pl.Boolean)})
    out = _resolve_effective_nulls(df)
    assert out["b"][0] is True
    assert out["b"][1] is False
    assert out["b"][2] is None


def test_datetime_column_unchanged():
    import datetime
    dt = datetime.datetime(2024, 1, 1)
    df = pl.DataFrame({"d": pl.Series([dt, None], dtype=pl.Datetime)})
    out = _resolve_effective_nulls(df)
    assert out["d"][0] == dt
    assert out["d"][1] is None


# ---------------------------------------------------------------------------
# No effective nulls — same object returned
# ---------------------------------------------------------------------------


def test_no_eligible_columns_returns_same_object():
    df = pl.DataFrame({"i": pl.Series([1, 2, 3], dtype=pl.Int64)})
    out = _resolve_effective_nulls(df)
    assert out is df


def test_eligible_columns_but_no_effective_nulls_returns_equal_frame():
    df = pl.DataFrame({
        "s": pl.Series(["hello", "world"], dtype=pl.String),
        "f": pl.Series([1.0, 2.0], dtype=pl.Float64),
    })
    out = _resolve_effective_nulls(df)
    assert out.equals(df)


# ---------------------------------------------------------------------------
# Mixed-dtype DataFrame — each column processed by its own rule
# ---------------------------------------------------------------------------


def test_mixed_dtype_frame_applies_correct_rule_per_column():
    df = pl.DataFrame({
        "str_col": pl.Series(["NA", "hello", ""], dtype=pl.String),
        "float_col": pl.Series([1.0, float("nan"), float("inf")], dtype=pl.Float64),
        "int_col": pl.Series([10, 20, 30], dtype=pl.Int32),
    })
    out = _resolve_effective_nulls(df)

    # string column: "NA" → null, "hello" stays, "" → null
    assert out["str_col"][0] is None
    assert out["str_col"][1] == "hello"
    assert out["str_col"][2] is None

    # float column: NaN → null, Inf → null
    assert out["float_col"][0] == pytest.approx(1.0)
    assert out["float_col"][1] is None
    assert out["float_col"][2] is None

    # int column: untouched
    assert out["int_col"][0] == 10
    assert out["int_col"][1] == 20
    assert out["int_col"][2] == 30


def test_existing_polars_nulls_remain_null_after_normalization():
    df = pl.DataFrame({
        "s": pl.Series(["hello", None], dtype=pl.String),
        "f": pl.Series([None, 2.0], dtype=pl.Float64),
    })
    out = _resolve_effective_nulls(df)
    assert out["s"][1] is None
    assert out["f"][0] is None
