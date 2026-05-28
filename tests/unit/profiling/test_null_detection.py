import inspect

import polars as pl
import pytest

from dataforge_ml.utils._null_detection import (
    _SENTINEL_STRINGS,
    _inf_eligible,
    _sentinel_eligible,
)


# ---------------------------------------------------------------------------
# _sentinel_eligible — String/Utf8 dtypes
# ---------------------------------------------------------------------------


def test_sentinel_eligible_string():
    assert _sentinel_eligible(pl.String) is True


def test_sentinel_eligible_utf8():
    assert _sentinel_eligible(pl.Utf8) is True


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
        pl.Boolean,
        pl.Date, pl.Datetime, pl.Duration, pl.Time,
    ],
)
def test_sentinel_eligible_false_for_non_string(dtype):
    assert _sentinel_eligible(dtype) is False


def test_sentinel_eligible_accepts_no_override_parameter():
    sig = inspect.signature(_sentinel_eligible)
    assert list(sig.parameters.keys()) == ["dtype"]


# ---------------------------------------------------------------------------
# _inf_eligible — Float32/Float64 only
# ---------------------------------------------------------------------------


def test_inf_eligible_float32():
    assert _inf_eligible(pl.Float32) is True


def test_inf_eligible_float64():
    assert _inf_eligible(pl.Float64) is True


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.String, pl.Utf8,
        pl.Boolean,
        pl.Date, pl.Datetime, pl.Duration, pl.Time,
    ],
)
def test_inf_eligible_false_for_non_float(dtype):
    assert _inf_eligible(dtype) is False


# ---------------------------------------------------------------------------
# _SENTINEL_STRINGS contents
# ---------------------------------------------------------------------------


def test_sentinel_strings_contains_expected_values():
    assert _SENTINEL_STRINGS == frozenset({"NA", "NAN", "NULL", "NONE", "?"})


def test_sentinel_strings_is_frozenset():
    assert isinstance(_SENTINEL_STRINGS, frozenset)
