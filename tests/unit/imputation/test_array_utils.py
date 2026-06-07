"""Unit tests for imputation/_utils.py — _df_to_numpy and _numpy_to_df."""

import math

import numpy as np
import polars as pl
import pytest

from dataforge_ml.imputation._utils import _df_to_numpy, _numpy_to_df


# ---------------------------------------------------------------------------
# _df_to_numpy
# ---------------------------------------------------------------------------


def test_polars_nulls_become_nan():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0])})
    arr = _df_to_numpy(df, ["a"])
    assert arr.shape == (3, 1)
    assert not math.isnan(arr[0, 0])
    assert math.isnan(arr[1, 0])
    assert not math.isnan(arr[2, 0])


def test_non_null_values_pass_through():
    df = pl.DataFrame({"x": pl.Series([42.0, -7.5, 0.0])})
    arr = _df_to_numpy(df, ["x"])
    assert arr[0, 0] == pytest.approx(42.0)
    assert arr[1, 0] == pytest.approx(-7.5)
    assert arr[2, 0] == pytest.approx(0.0)


def test_multiple_columns_correct_shape():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, None], "c": [5.0, 6.0]})
    arr = _df_to_numpy(df, ["a", "b", "c"])
    assert arr.shape == (2, 3)
    assert arr[0, 0] == pytest.approx(1.0)
    assert math.isnan(arr[1, 1])
    assert arr[1, 2] == pytest.approx(6.0)


def test_column_order_matches_cols_argument():
    df = pl.DataFrame({"x": [10.0, 20.0], "y": [30.0, 40.0]})
    arr = _df_to_numpy(df, ["y", "x"])
    assert arr[0, 0] == pytest.approx(30.0)
    assert arr[0, 1] == pytest.approx(10.0)


def test_output_dtype_is_float64():
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)})
    arr = _df_to_numpy(df, ["a"])
    assert arr.dtype == np.float64


def test_sentinel_value_passes_through_unchanged():
    # Sentinel precondition (ADR-0028): numeric sentinels stored as real floats
    # are NOT treated as nulls — they pass through as their float value.
    df = pl.DataFrame({"a": pl.Series([-999.0, 1.0, 2.0])})
    arr = _df_to_numpy(df, ["a"])
    assert arr[0, 0] == pytest.approx(-999.0)
    assert not math.isnan(arr[0, 0])


# ---------------------------------------------------------------------------
# _numpy_to_df
# ---------------------------------------------------------------------------


def test_numpy_to_df_float64_roundtrip():
    df = pl.DataFrame({"f": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)})
    arr = np.array([[1.5], [2.5], [3.5]], dtype=np.float64)
    result = _numpy_to_df(df, ["f"], arr)
    assert result["f"].dtype == pl.Float64
    assert result["f"].to_list() == pytest.approx([1.5, 2.5, 3.5])


def test_numpy_to_df_float32_roundtrip():
    df = pl.DataFrame({"f": pl.Series([1.0, 2.0], dtype=pl.Float32)})
    arr = np.array([[1.5], [2.5]], dtype=np.float64)
    result = _numpy_to_df(df, ["f"], arr)
    assert result["f"].dtype == pl.Float32
    assert result["f"].to_list() == pytest.approx([1.5, 2.5], abs=1e-5)


def test_numpy_to_df_int32_roundtrip():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})
    arr = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    result = _numpy_to_df(df, ["x"], arr)
    assert result["x"].dtype == pl.Int32
    assert result["x"].to_list() == [10, 20, 30]


def test_numpy_to_df_int64_at_2_pow_53_survives():
    val = 2**53  # 9007199254740992 — largest int exactly representable in float64
    df = pl.DataFrame({"x": pl.Series([0], dtype=pl.Int64)})
    arr = np.array([[float(val)]], dtype=np.float64)
    result = _numpy_to_df(df, ["x"], arr)
    assert result["x"][0] == val


def test_numpy_to_df_int64_above_2_pow_53_consistent():
    # 2^53+1 cannot be represented exactly in float64; it rounds to 2^53.
    # _numpy_to_df must produce a consistent integer value, not crash.
    val_float = float(2**53 + 1)  # rounds to float(2**53) in float64
    df = pl.DataFrame({"x": pl.Series([0], dtype=pl.Int64)})
    arr = np.array([[val_float]], dtype=np.float64)
    result = _numpy_to_df(df, ["x"], arr)
    assert result["x"].dtype == pl.Int64
    assert result["x"][0] == int(val_float)  # consistent with float64 representation


def test_numpy_to_df_int64_near_max_survives():
    val = 2**62  # exactly representable in float64 (power of 2), near Int64 max
    df = pl.DataFrame({"x": pl.Series([0], dtype=pl.Int64)})
    arr = np.array([[float(val)]], dtype=np.float64)
    result = _numpy_to_df(df, ["x"], arr)
    assert result["x"][0] == val


def test_numpy_to_df_nan_in_int_column_raises_assertion():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})
    arr = np.array([[1.0], [float("nan")], [3.0]], dtype=np.float64)
    with pytest.raises(AssertionError):
        _numpy_to_df(df, ["x"], arr)


def test_numpy_to_df_utf8_dtype_raises_value_error():
    df = pl.DataFrame({"s": pl.Series(["a", "b"], dtype=pl.Utf8)})
    arr = np.array([[1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="s"):
        _numpy_to_df(df, ["s"], arr)


def test_numpy_to_df_boolean_dtype_raises_value_error():
    df = pl.DataFrame({"b": pl.Series([True, False], dtype=pl.Boolean)})
    arr = np.array([[1.0], [0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="b"):
        _numpy_to_df(df, ["b"], arr)
