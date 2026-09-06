"""The Dtype Floor normaliser in isolation (#508 / ADR-0085).

No active column reaches a consumer as ``pl.String`` unless its semantic type
is ``Text`` or ``Identifier``. These tests pin the rule itself: what is cast,
what deliberately is not, what a failed coercion becomes, and that the caller's
frame is never touched. The phase-entry wiring is asserted in
``test_dtype_floor_phase_entry.py``.
"""

from __future__ import annotations

import polars as pl

from dataforge_ml.config import SemanticType
from dataforge_ml.utils._dtype_floor import _apply_dtype_floor

# ---------------------------------------------------------------------------
# The floor: each non-string semantic type names the dtype it materialises as
# ---------------------------------------------------------------------------


def test_string_column_profiled_categorical_becomes_categorical():
    df = pl.DataFrame({"c": pl.Series(["a", "b", "a"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"c": SemanticType.Categorical})
    assert out["c"].dtype == pl.Categorical
    assert out["c"].to_list() == ["a", "b", "a"]


def test_string_column_profiled_boolean_becomes_boolean():
    df = pl.DataFrame({"b": pl.Series(["yes", "no", "TRUE"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"b": SemanticType.Boolean})
    assert out["b"].dtype == pl.Boolean
    assert out["b"].to_list() == [True, False, True]


def test_string_column_profiled_datetime_becomes_datetime():
    df = pl.DataFrame(
        {"d": pl.Series(["2024-01-01", "2024-06-30"], dtype=pl.String)}
    )
    out = _apply_dtype_floor(df, {"d": SemanticType.Datetime})
    assert out["d"].dtype == pl.Datetime
    assert out["d"][0].year == 2024


# ---------------------------------------------------------------------------
# The exemptions: the rule is a floor keyed off the dtype, not a biconditional
# ---------------------------------------------------------------------------


def test_text_column_stays_string():
    df = pl.DataFrame(
        {"t": pl.Series(["a full sentence here", "another one"], dtype=pl.String)}
    )
    out = _apply_dtype_floor(df, {"t": SemanticType.Text})
    assert out["t"].dtype == pl.String


def test_identifier_column_stays_string():
    df = pl.DataFrame({"i": pl.Series(["a1", "b2", "c3"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"i": SemanticType.Identifier})
    assert out["i"].dtype == pl.String


def test_int_coded_categorical_is_not_cast():
    # TypeFlag.EncodedCategory columns carry SemanticType.Categorical on an
    # integer dtype and stay pl.Int* — the floor only ever touches pl.String.
    df = pl.DataFrame({"e": pl.Series([1, 2, 1, 3], dtype=pl.Int64)})
    out = _apply_dtype_floor(df, {"e": SemanticType.Categorical})
    assert out["e"].dtype == pl.Int64


def test_numeric_and_unprofiled_columns_are_untouched():
    df = pl.DataFrame(
        {
            "n": pl.Series([1.5, 2.5], dtype=pl.Float64),
            "unknown": pl.Series(["x", "y"], dtype=pl.String),
        }
    )
    out = _apply_dtype_floor(df, {"n": SemanticType.Numeric})
    assert out["n"].dtype == pl.Float64
    assert out["unknown"].dtype == pl.String


def test_none_semantic_type_leaves_the_column_alone():
    df = pl.DataFrame({"c": pl.Series(["a", "b"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"c": None})
    assert out["c"].dtype == pl.String


# ---------------------------------------------------------------------------
# A failed coercion is an Effective Null
# ---------------------------------------------------------------------------


def test_unrecognised_boolean_token_becomes_null():
    df = pl.DataFrame({"b": pl.Series(["yes", "maybe", "no"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"b": SemanticType.Boolean})
    assert out["b"].to_list() == [True, None, False]


def test_unparseable_datetime_becomes_null():
    df = pl.DataFrame(
        {"d": pl.Series(["2024-01-01", "banana"], dtype=pl.String)}
    )
    out = _apply_dtype_floor(df, {"d": SemanticType.Datetime})
    assert out["d"].null_count() == 1


def test_wholly_unparseable_datetime_column_resolves_to_all_null():
    # Polars cannot even infer a format here; the column is still nulled
    # silently rather than raising.
    df = pl.DataFrame({"d": pl.Series(["banana", "apple"], dtype=pl.String)})
    out = _apply_dtype_floor(df, {"d": SemanticType.Datetime})
    assert out["d"].dtype == pl.Datetime
    assert out["d"].null_count() == 2


def test_null_count_may_exceed_the_profiles_recorded_missingness():
    # The stated consequence of ADR-0085: a genuine typo is absorbed as
    # missing, so the phase-entry null count can exceed the profile's.
    df = pl.DataFrame(
        {"b": pl.Series(["true", "ture", "false", None], dtype=pl.String)}
    )
    assert df["b"].null_count() == 1
    assert _apply_dtype_floor(df, {"b": SemanticType.Boolean})["b"].null_count() == 2


# ---------------------------------------------------------------------------
# Constructive, idempotent, and non-mutating
# ---------------------------------------------------------------------------


def test_the_input_frame_is_not_mutated():
    df = pl.DataFrame(
        {
            "c": pl.Series(["a", "b"], dtype=pl.String),
            "b": pl.Series(["yes", "no"], dtype=pl.String),
        }
    )
    before = dict(df.schema)
    _apply_dtype_floor(df, {"c": SemanticType.Categorical, "b": SemanticType.Boolean})
    assert dict(df.schema) == before


def test_applying_the_floor_twice_is_a_no_op():
    types = {
        "c": SemanticType.Categorical,
        "b": SemanticType.Boolean,
        "d": SemanticType.Datetime,
    }
    df = pl.DataFrame(
        {
            "c": pl.Series(["a", "b"], dtype=pl.String),
            "b": pl.Series(["yes", "no"], dtype=pl.String),
            "d": pl.Series(["2024-01-01", "2024-01-02"], dtype=pl.String),
        }
    )
    once = _apply_dtype_floor(df, types)
    twice = _apply_dtype_floor(once, types)
    assert twice.schema == once.schema
    assert twice.equals(once)


def test_a_frame_needing_no_cast_is_returned_as_is():
    df = pl.DataFrame({"n": pl.Series([1, 2], dtype=pl.Int64)})
    assert _apply_dtype_floor(df, {"n": SemanticType.Numeric}) is df
