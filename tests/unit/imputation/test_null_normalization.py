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


# ---------------------------------------------------------------------------
# numeric_sentinels — user-declared sentinel values
# ---------------------------------------------------------------------------


def test_numeric_sentinel_int64_converted_to_null():
    df = pl.DataFrame({"age": pl.Series([-999, 25, 30], dtype=pl.Int64)})
    out = _resolve_effective_nulls(df, numeric_sentinels={"age": [-999.0]})
    assert out["age"][0] is None
    assert out["age"][1] == 25
    assert out["age"][2] == 30


def test_numeric_sentinel_float64_converted_and_inf_nan_rules_still_apply():
    df = pl.DataFrame({
        "x": pl.Series([-999.0, float("nan"), float("inf"), 1.0], dtype=pl.Float64)
    })
    out = _resolve_effective_nulls(df, numeric_sentinels={"x": [-999.0]})
    assert out["x"][0] is None   # declared sentinel
    assert out["x"][1] is None   # NaN rule still fires
    assert out["x"][2] is None   # Inf rule still fires
    assert out["x"][3] == pytest.approx(1.0)


def test_numeric_sentinel_multiple_values_all_converted():
    df = pl.DataFrame({"score": pl.Series([-999, 9999, 0, 42], dtype=pl.Int32)})
    out = _resolve_effective_nulls(df, numeric_sentinels={"score": [-999.0, 9999.0]})
    assert out["score"][0] is None
    assert out["score"][1] is None
    assert out["score"][2] == 0
    assert out["score"][3] == 42


def test_numeric_sentinel_column_absent_from_dataframe_no_error():
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
    out = _resolve_effective_nulls(df, numeric_sentinels={"missing_col": [-999.0]})
    assert out["a"].to_list() == [1, 2, 3]


def test_numeric_sentinel_no_matching_rows_is_noop():
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
    out = _resolve_effective_nulls(df, numeric_sentinels={"a": [-999.0]})
    assert out["a"].to_list() == [1, 2, 3]


def test_numeric_sentinel_does_not_affect_other_columns():
    df = pl.DataFrame({
        "a": pl.Series([-999, 1], dtype=pl.Int64),
        "b": pl.Series([-999, 1], dtype=pl.Int64),
    })
    out = _resolve_effective_nulls(df, numeric_sentinels={"a": [-999.0]})
    assert out["a"][0] is None
    assert out["b"][0] == -999   # sentinel declared only for "a", not "b"


def test_numeric_sentinels_none_preserves_existing_behaviour():
    df = pl.DataFrame({
        "s": pl.Series(["NA", "ok"], dtype=pl.String),
        "f": pl.Series([float("nan"), 1.0], dtype=pl.Float64),
        "i": pl.Series([-999, 1], dtype=pl.Int64),
    })
    out_default = _resolve_effective_nulls(df)
    out_none = _resolve_effective_nulls(df, numeric_sentinels=None)
    assert out_default.equals(out_none)
    # integer column untouched (no sentinel declared)
    assert out_none["i"][0] == -999


# ---------------------------------------------------------------------------
# string_sentinels — user-declared replace semantics
# ---------------------------------------------------------------------------


def test_string_sentinel_no_declaration_hardcoded_defaults_apply():
    # Regression: column with no declaration continues to use _SENTINEL_STRINGS.
    df = pl.DataFrame({"s": pl.Series(["NA", "?", "hello"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={})
    assert out["s"][0] is None
    assert out["s"][1] is None
    assert out["s"][2] == "hello"


def test_string_sentinel_declared_replaces_hardcoded_defaults():
    # "?" is a hardcoded default but not declared — must NOT be converted.
    # "N/A" is declared — must be converted.
    df = pl.DataFrame({"s": pl.Series(["N/A", "?", "valid"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"s": ["N/A"]})
    assert out["s"][0] is None    # declared sentinel
    assert out["s"][1] == "?"     # hardcoded default suppressed
    assert out["s"][2] == "valid"


def test_string_sentinel_declared_matching_is_case_insensitive():
    df = pl.DataFrame({"s": pl.Series(["missing", "MISSING", "Missing", "ok"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"s": ["missing"]})
    assert out["s"][0] is None
    assert out["s"][1] is None
    assert out["s"][2] is None
    assert out["s"][3] == "ok"


def test_string_sentinel_multiple_declared_values_all_converted():
    df = pl.DataFrame({"s": pl.Series(["N/A", "unknown", "real", "missing"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"s": ["N/A", "unknown", "missing"]})
    assert out["s"][0] is None
    assert out["s"][1] is None
    assert out["s"][2] == "real"
    assert out["s"][3] is None


def test_string_sentinel_empty_and_whitespace_always_converted():
    # Even with a declaration, empty/whitespace detection is unconditional.
    df = pl.DataFrame({"s": pl.Series(["", "  ", "declared", "other"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"s": ["declared"]})
    assert out["s"][0] is None   # empty string
    assert out["s"][1] is None   # whitespace-only
    assert out["s"][2] is None   # declared sentinel
    assert out["s"][3] == "other"


def test_string_sentinel_column_absent_from_dataframe_no_error():
    df = pl.DataFrame({"a": pl.Series(["hello", "world"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"nonexistent": ["N/A"]})
    assert out["a"].to_list() == ["hello", "world"]


def test_string_sentinels_none_output_identical_to_no_arg():
    df = pl.DataFrame({
        "s": pl.Series(["NA", "?", "ok", ""], dtype=pl.String),
        "f": pl.Series([1.0, float("nan"), 2.0, 3.0], dtype=pl.Float64),
    })
    out_default = _resolve_effective_nulls(df)
    out_none = _resolve_effective_nulls(df, string_sentinels=None)
    assert out_default.equals(out_none)


def test_string_sentinel_declaration_does_not_affect_other_columns():
    # "N/A" is declared for "a" but not "b"; "N/A" is not in _SENTINEL_STRINGS,
    # so column "b" (no declaration, uses hardcoded defaults) leaves it unchanged.
    df = pl.DataFrame({
        "a": pl.Series(["N/A", "real"], dtype=pl.String),
        "b": pl.Series(["N/A", "real"], dtype=pl.String),
    })
    out = _resolve_effective_nulls(df, string_sentinels={"a": ["N/A"]})
    assert out["a"][0] is None     # declared sentinel → null
    assert out["b"][0] == "N/A"   # no declaration; "N/A" not in hardcoded defaults
    assert out["a"][1] == "real"
    assert out["b"][1] == "real"


def test_string_sentinel_declared_value_not_in_hardcoded_defaults_converted():
    # "custom_missing" is not in _SENTINEL_STRINGS; with a declaration it must convert.
    df = pl.DataFrame({"s": pl.Series(["custom_missing", "real"], dtype=pl.String)})
    out = _resolve_effective_nulls(df, string_sentinels={"s": ["custom_missing"]})
    assert out["s"][0] is None
    assert out["s"][1] == "real"
