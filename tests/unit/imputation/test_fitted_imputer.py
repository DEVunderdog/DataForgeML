"""
Unit tests for FittedImputer.transform() and to_dict()/from_dict().

FittedImputer is constructed directly with known records — no profiler run.
"""

import polars as pl
import pytest

from dataforge_ml.config import SemanticType
from dataforge_ml.imputation._config import (
    ColumnImputationRecord,
    ImputationResult,
    ImputationStrategy,
)
from dataforge_ml.imputation._fitted_imputer import FittedImputer, UnfittedColumnError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(col: str, strategy: ImputationStrategy, fill_value=None,
            indicator_added: bool = False, signals: list | None = None) -> ColumnImputationRecord:
    return ColumnImputationRecord(
        column=col,
        semantic_type=SemanticType.Numeric,
        strategy=strategy,
        fill_value=fill_value,
        indicator_added=indicator_added,
        signals=signals or [],
    )


# ---------------------------------------------------------------------------
# transform() — basic structure
# ---------------------------------------------------------------------------


def test_transform_returns_imputation_result():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert isinstance(result, ImputationResult)


def test_transform_output_has_no_nulls_in_imputed_column():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None, None], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["a"].null_count() == 0


def test_transform_uses_stored_fill_value_not_recomputed():
    """Fill value must come from the record, not recomputed from the transform df."""
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=999.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    filled = result.dataframe["a"].filter(result.dataframe["a"] == 999.0)
    assert len(filled) == 1


def test_transform_preserves_non_null_values():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Median, fill_value=10.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["a"][0] == pytest.approx(1.0)
    assert result.dataframe["a"][2] == pytest.approx(3.0)


def test_transform_int_column_fills_with_rounded_value():
    imputer = FittedImputer(records={
        "x": _record("x", ImputationStrategy.Mode, fill_value=3.0),
    })
    df = pl.DataFrame({"x": pl.Series([1, None, 5], dtype=pl.Int64)})
    result = imputer.transform(df)
    assert result.dataframe["x"].null_count() == 0
    assert result.dataframe["x"].dtype == pl.Int64


def test_transform_float_column_fills_nan_values():
    imputer = FittedImputer(records={
        "f": _record("f", ImputationStrategy.Median, fill_value=7.0),
    })
    df = pl.DataFrame({"f": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    nan_count = result.dataframe["f"].is_nan().sum()
    assert nan_count == 0


# ---------------------------------------------------------------------------
# transform() — dropped columns
# ---------------------------------------------------------------------------


def test_dropped_column_absent_from_output():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=1.0),
    })
    n = 10
    df = pl.DataFrame({
        "drop_me": pl.Series([None] * 7 + [1.0] * 3, dtype=pl.Float64),
        "keep_me": pl.Series([1.0, None, 3.0] + [1.0] * 7, dtype=pl.Float64),
    })
    result = imputer.transform(df)
    assert "drop_me" not in result.dataframe.columns
    assert "keep_me" in result.dataframe.columns


def test_dropped_columns_listed_in_result():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    df = pl.DataFrame({"drop_me": pl.Series([None, 1.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert "drop_me" in result.dropped_columns


def test_transform_with_no_dropped_columns_gives_empty_list():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dropped_columns == []


# ---------------------------------------------------------------------------
# transform() — MNAR constant fill + indicator
# ---------------------------------------------------------------------------


def test_mnar_constant_fill_applied():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.Constant,
                          fill_value=-1.0, indicator_added=True),
    })
    df = pl.DataFrame({"income": pl.Series([50000.0, None, 75000.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["income"].null_count() == 0
    assert result.dataframe["income"][1] == pytest.approx(-1.0)


def test_mnar_indicator_column_appended():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.Constant,
                          fill_value=-1.0, indicator_added=True),
    })
    df = pl.DataFrame({"income": pl.Series([50000.0, None, 75000.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert "income_missing" in result.dataframe.columns


def test_mnar_indicator_marks_originally_null_rows():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.Constant,
                          fill_value=-1.0, indicator_added=True),
    })
    df = pl.DataFrame({"income": pl.Series([50000.0, None, 75000.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    indicator = result.dataframe["income_missing"]
    assert indicator[0] == 0
    assert indicator[1] == 1
    assert indicator[2] == 0


def test_indicator_reflects_pre_fill_nullness():
    """Indicator must be based on original nullness, not on post-fill values."""
    imputer = FittedImputer(records={
        "v": _record("v", ImputationStrategy.Constant,
                     fill_value=-1.0, indicator_added=True),
    })
    df = pl.DataFrame({"v": pl.Series([-1.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    # Only the original None should be marked, not the pre-existing -1.0
    assert result.dataframe["v_missing"][0] == 0
    assert result.dataframe["v_missing"][1] == 1


# ---------------------------------------------------------------------------
# transform() — Passthrough
# ---------------------------------------------------------------------------


def test_passthrough_column_passes_unchanged_when_no_nulls():
    imputer = FittedImputer(records={
        "clean": _record("clean", ImputationStrategy.Passthrough),
    })
    df = pl.DataFrame({"clean": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["clean"].to_list() == [1.0, 2.0, 3.0]


def test_passthrough_with_nulls_raises_unfitted_column_error():
    imputer = FittedImputer(records={
        "clean": _record("clean", ImputationStrategy.Passthrough),
    })
    df = pl.DataFrame({"clean": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    with pytest.raises(UnfittedColumnError, match="clean"):
        imputer.transform(df)


def test_unfitted_column_error_message_is_informative():
    imputer = FittedImputer(records={
        "my_col": _record("my_col", ImputationStrategy.Passthrough),
    })
    df = pl.DataFrame({"my_col": pl.Series([None, 2.0], dtype=pl.Float64)})
    with pytest.raises(UnfittedColumnError) as exc_info:
        imputer.transform(df)
    assert "my_col" in str(exc_info.value)


def test_unfitted_column_error_names_all_offending_columns():
    """All passthrough columns with missing values must be named in the single raised error."""
    imputer = FittedImputer(records={
        "col_a": _record("col_a", ImputationStrategy.Passthrough),
        "col_b": _record("col_b", ImputationStrategy.Passthrough),
    })
    df = pl.DataFrame({
        "col_a": pl.Series([None, 2.0], dtype=pl.Float64),
        "col_b": pl.Series([1.0, None], dtype=pl.Float64),
    })
    with pytest.raises(UnfittedColumnError) as exc_info:
        imputer.transform(df)
    msg = str(exc_info.value)
    assert "col_a" in msg
    assert "col_b" in msg


def test_columns_not_in_records_do_not_trigger_unfitted_column_error():
    """Columns outside the active set (not in records) must not raise UnfittedColumnError."""
    imputer = FittedImputer(records={
        "num": _record("num", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "txt": pl.Series([None, "hello"], dtype=pl.Utf8),
    })
    result = imputer.transform(df)
    assert "txt" in result.dataframe.columns


# ---------------------------------------------------------------------------
# transform() — columns not in records pass through unchanged
# ---------------------------------------------------------------------------


def test_columns_not_in_records_pass_through():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "b": pl.Series(["x", "y"], dtype=pl.Utf8),
    })
    result = imputer.transform(df)
    assert "b" in result.dataframe.columns
    assert result.dataframe["b"].to_list() == ["x", "y"]


# ---------------------------------------------------------------------------
# transform() — effective null normalisation (Inf, NaN, string sentinels)
# ---------------------------------------------------------------------------


def test_inf_in_float_column_filled_by_mean_record():
    imputer = FittedImputer(records={
        "f": _record("f", ImputationStrategy.Mean, fill_value=9.0),
    })
    df = pl.DataFrame({"f": pl.Series([1.0, float("inf"), 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["f"].null_count() == 0
    assert not result.dataframe["f"].is_infinite().any()
    assert result.dataframe["f"][1] == pytest.approx(9.0)


def test_neg_inf_in_float_column_filled_by_median_record():
    imputer = FittedImputer(records={
        "f": _record("f", ImputationStrategy.Median, fill_value=5.0),
    })
    df = pl.DataFrame({"f": pl.Series([1.0, float("-inf"), 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["f"].null_count() == 0
    assert not result.dataframe["f"].is_infinite().any()
    assert result.dataframe["f"][1] == pytest.approx(5.0)


def test_nan_in_float_column_filled_by_mean_record():
    imputer = FittedImputer(records={
        "f": _record("f", ImputationStrategy.Mean, fill_value=4.0),
    })
    df = pl.DataFrame({"f": pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["f"].null_count() == 0
    assert result.dataframe["f"].is_nan().sum() == 0
    assert result.dataframe["f"][1] == pytest.approx(4.0)


def test_string_sentinel_na_filled_by_constant_record():
    imputer = FittedImputer(records={
        "cat": ColumnImputationRecord(
            column="cat", semantic_type=SemanticType.Categorical,
            strategy=ImputationStrategy.Constant, fill_value="unknown",
            indicator_added=False, signals=[],
        ),
    })
    df = pl.DataFrame({"cat": pl.Series(["NA", "hello", "world"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["cat"].null_count() == 0
    assert result.dataframe["cat"][0] == "unknown"


def test_string_sentinel_question_mark_filled_by_constant_record():
    imputer = FittedImputer(records={
        "cat": ColumnImputationRecord(
            column="cat", semantic_type=SemanticType.Categorical,
            strategy=ImputationStrategy.Constant, fill_value="missing",
            indicator_added=False, signals=[],
        ),
    })
    df = pl.DataFrame({"cat": pl.Series(["?", "hello", "?"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["cat"].null_count() == 0
    assert result.dataframe["cat"][0] == "missing"
    assert result.dataframe["cat"][2] == "missing"


def test_indicator_set_to_one_for_inf_rows():
    imputer = FittedImputer(records={
        "f": _record("f", ImputationStrategy.Constant, fill_value=-1.0, indicator_added=True),
    })
    df = pl.DataFrame({"f": pl.Series([1.0, float("inf"), 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["f_missing"][0] == 0
    assert result.dataframe["f_missing"][1] == 1
    assert result.dataframe["f_missing"][2] == 0


def test_indicator_set_to_one_for_string_sentinel_rows():
    imputer = FittedImputer(records={
        "cat": ColumnImputationRecord(
            column="cat", semantic_type=SemanticType.Categorical,
            strategy=ImputationStrategy.Constant, fill_value="unknown",
            indicator_added=True, signals=[],
        ),
    })
    df = pl.DataFrame({"cat": pl.Series(["?", "hello", "world"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["cat_missing"][0] == 1
    assert result.dataframe["cat_missing"][1] == 0
    assert result.dataframe["cat_missing"][2] == 0


def test_unfitted_column_error_raised_for_inf_in_passthrough_float_column():
    imputer = FittedImputer(records={
        "clean": _record("clean", ImputationStrategy.Passthrough),
    })
    df = pl.DataFrame({"clean": pl.Series([1.0, float("inf"), 3.0], dtype=pl.Float64)})
    with pytest.raises(UnfittedColumnError, match="clean"):
        imputer.transform(df)


def test_unfitted_column_error_raised_for_string_sentinel_in_passthrough_column():
    imputer = FittedImputer(records={
        "cat": ColumnImputationRecord(
            column="cat", semantic_type=SemanticType.Categorical,
            strategy=ImputationStrategy.Passthrough, fill_value=None,
            indicator_added=False, signals=[],
        ),
    })
    df = pl.DataFrame({"cat": pl.Series(["NA", "hello"], dtype=pl.String)})
    with pytest.raises(UnfittedColumnError, match="cat"):
        imputer.transform(df)


# ---------------------------------------------------------------------------
# to_dict() / from_dict()
# ---------------------------------------------------------------------------


def test_to_dict_contains_records_key():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=3.14),
    })
    d = imputer.to_dict()
    assert "records" in d
    assert "a" in d["records"]


def test_round_trip_preserves_strategy():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Median, fill_value=2.5),
        "b": _record("b", ImputationStrategy.Dropped),
        "c": _record("c", ImputationStrategy.Passthrough),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["a"].strategy == ImputationStrategy.Median
    assert restored.records["b"].strategy == ImputationStrategy.Dropped
    assert restored.records["c"].strategy == ImputationStrategy.Passthrough


def test_round_trip_preserves_fill_value():
    imputer = FittedImputer(records={
        "x": _record("x", ImputationStrategy.Mean, fill_value=42.5),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["x"].fill_value == pytest.approx(42.5)


def test_round_trip_preserves_indicator_added():
    imputer = FittedImputer(records={
        "mnar_col": _record("mnar_col", ImputationStrategy.Constant,
                            fill_value=-1.0, indicator_added=True),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["mnar_col"].indicator_added is True


def test_round_trip_preserves_signals():
    imputer = FittedImputer(records={
        "z": _record("z", ImputationStrategy.Mode, fill_value=3.0,
                     signals=["discrete numeric"]),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["z"].signals == ["discrete numeric"]


def test_deserialised_imputer_produces_identical_output():
    """Serialised and deserialised FittedImputer must produce identical transform output."""
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "b": _record("b", ImputationStrategy.Constant, fill_value=-1.0, indicator_added=True),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())

    df = pl.DataFrame({
        "a": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
        "b": pl.Series([10.0, None, 30.0], dtype=pl.Float64),
    })
    r1 = imputer.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)
    assert r1.dropped_columns == r2.dropped_columns
