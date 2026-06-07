"""
Unit tests for FittedImputer.transform() and to_dict()/from_dict().

FittedImputer is constructed directly with known records — no profiler run.
"""

import warnings

import polars as pl
import pytest

from dataforge_ml.config import SemanticType
from dataforge_ml.imputation._config import (
    ColumnImputationRecord,
    ImputationResult,
    ImputationStrategy,
)
from dataforge_ml.imputation._fitted_imputer import (
    DroppedColumnAbsentWarning,
    FittedColumnAbsentError,
    FittedImputer,
    UnfittedColumnError,
    UnseenColumnError,
)


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


def test_transform_int64_fill_value_above_2_pow_53_survives():
    # 2^53 is the largest integer exactly representable in float64.
    # Verifies the integer round-trip does not corrupt the value through a
    # Float64 Polars intermediate (regression guard for the Scope 13b fix).
    fill = float(2**53)  # exactly representable
    imputer = FittedImputer(records={
        "x": _record("x", ImputationStrategy.Mean, fill_value=fill),
    })
    df = pl.DataFrame({"x": pl.Series([2**53, None], dtype=pl.Int64)})
    result = imputer.transform(df)
    assert result.dataframe["x"].dtype == pl.Int64
    assert result.dataframe["x"][1] == 2**53


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


def test_unfitted_column_error_not_raised_for_passthrough_column_absent_from_df():
    """FittedColumnAbsentError fires (not UnfittedColumnError) when a Passthrough column is absent."""
    imputer = FittedImputer(records={
        "clean": _record("clean", ImputationStrategy.Passthrough),
        "other": _record("other", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({"other": pl.Series([1.0, 2.0], dtype=pl.Float64)})
    with pytest.raises(FittedColumnAbsentError):
        imputer.transform(df)


# ---------------------------------------------------------------------------
# transform() — UnseenColumnError schema enforcement
# ---------------------------------------------------------------------------


def test_unseen_column_error_fires_for_column_not_in_records():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "unknown": pl.Series(["x", "y"], dtype=pl.Utf8),
    })
    with pytest.raises(UnseenColumnError, match="unknown"):
        imputer.transform(df)


def test_unseen_column_error_fires_even_when_unknown_column_has_no_nulls():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({
        "a": pl.Series([1.0, 2.0], dtype=pl.Float64),
        "extra": pl.Series(["x", "y"], dtype=pl.Utf8),
    })
    with pytest.raises(UnseenColumnError):
        imputer.transform(df)


def test_unseen_column_error_names_all_unknown_columns_in_single_raise():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "ghost1": pl.Series(["x", "y"], dtype=pl.Utf8),
        "ghost2": pl.Series([1, 2], dtype=pl.Int64),
    })
    with pytest.raises(UnseenColumnError) as exc_info:
        imputer.transform(df)
    msg = str(exc_info.value)
    assert "ghost1" in msg
    assert "ghost2" in msg


def test_unseen_column_error_fires_before_resolve_effective_nulls():
    """UnseenColumnError must fire before any DataFrame mutation (incl. null normalisation)."""
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    df = pl.DataFrame({
        "a": pl.Series([float("inf"), None], dtype=pl.Float64),
        "unseen": pl.Series([1.0, 2.0], dtype=pl.Float64),
    })
    with pytest.raises(UnseenColumnError):
        imputer.transform(df)


# ---------------------------------------------------------------------------
# transform() — FittedColumnAbsentError schema enforcement
# ---------------------------------------------------------------------------


def test_fitted_column_absent_error_fires_for_active_strategy_column_absent():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "b": _record("b", ImputationStrategy.Median, fill_value=3.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    with pytest.raises(FittedColumnAbsentError, match="b"):
        imputer.transform(df)


def test_fitted_column_absent_error_names_all_absent_columns():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "b": _record("b", ImputationStrategy.Median, fill_value=3.0),
        "c": _record("c", ImputationStrategy.Mode, fill_value=1.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    with pytest.raises(FittedColumnAbsentError) as exc_info:
        imputer.transform(df)
    msg = str(exc_info.value)
    assert "b" in msg
    assert "c" in msg


def test_fitted_column_absent_error_does_not_fire_for_dropped_column_absent():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    with pytest.warns(DroppedColumnAbsentWarning):
        result = imputer.transform(df)
    assert isinstance(result, ImputationResult)


def test_fitted_column_absent_error_does_not_fire_for_indicator_column_absent():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "a_missing": ColumnImputationRecord(
            column="a_missing",
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
            fill_value=None,
            indicator_added=False,
            signals=[],
        ),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert isinstance(result, ImputationResult)


def test_unseen_column_error_fires_before_fitted_column_absent_error():
    """When both conditions apply, UnseenColumnError must be the one raised."""
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "b": _record("b", ImputationStrategy.Median, fill_value=3.0),
    })
    # df has 'a' and an unseen 'extra'; 'b' is absent from df
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "extra": pl.Series(["x", "y"], dtype=pl.Utf8),
    })
    with pytest.raises(UnseenColumnError):
        imputer.transform(df)


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


# ---------------------------------------------------------------------------
# _exclusions_applied — initial state
# ---------------------------------------------------------------------------


def test_exclusions_applied_is_false_on_fresh_imputer():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    assert imputer._exclusions_applied is False


# ---------------------------------------------------------------------------
# apply_exclusions — config mutation and flag
# ---------------------------------------------------------------------------


def test_apply_exclusions_adds_dropped_columns_to_config():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=1.0),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert "drop_me" in config.exclude_columns
    assert "keep_me" not in config.exclude_columns


def test_apply_exclusions_sets_exclusions_applied_true():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert imputer._exclusions_applied is True


def test_apply_exclusions_with_no_dropped_columns_is_no_op_on_config():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert config.exclude_columns == []


def test_apply_exclusions_with_no_dropped_columns_still_sets_flag():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert imputer._exclusions_applied is True


def test_apply_exclusions_twice_is_idempotent():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    imputer.apply_exclusions(config)
    assert config.exclude_columns.count("drop_me") == 1


# ---------------------------------------------------------------------------
# apply_exclusions — Soft Exclusion registration for Indicator columns
# ---------------------------------------------------------------------------


def _indicator_record(col: str) -> ColumnImputationRecord:
    return ColumnImputationRecord(
        column=col,
        semantic_type=SemanticType.Boolean,
        strategy=ImputationStrategy.Indicator,
        fill_value=None,
        indicator_added=False,
        signals=[],
    )


def test_apply_exclusions_registers_indicator_column_in_phase_exclusions():
    from dataforge_ml.config import PipelineConfig, PipelinePhase
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
        "a_missing": _indicator_record("a_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert "a_missing" in config.phase_exclusions.get(PipelinePhase.Normalization, [])


def test_apply_exclusions_registers_indicator_for_all_four_soft_phases():
    from dataforge_ml.config import PipelineConfig, PipelinePhase
    imputer = FittedImputer(records={
        "x": _record("x", ImputationStrategy.Constant, fill_value=-1.0),
        "x_missing": _indicator_record("x_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    for phase in [
        PipelinePhase.OutlierDetection,
        PipelinePhase.Normalization,
        PipelinePhase.Encoding,
        PipelinePhase.Scaling,
    ]:
        assert "x_missing" in config.phase_exclusions.get(phase, []), phase


def test_apply_exclusions_registers_multiple_indicator_columns():
    from dataforge_ml.config import PipelineConfig, PipelinePhase
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Constant, fill_value=-1.0),
        "a_missing": _indicator_record("a_missing"),
        "b": _record("b", ImputationStrategy.Constant, fill_value=-1.0),
        "b_missing": _indicator_record("b_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    norm = config.phase_exclusions.get(PipelinePhase.Normalization, [])
    assert "a_missing" in norm
    assert "b_missing" in norm


def test_apply_exclusions_indicator_not_added_to_hard_exclusions():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
        "a_missing": _indicator_record("a_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert "a_missing" not in config.exclude_columns


def test_apply_exclusions_dropped_still_hard_excluded_alongside_indicator():
    from dataforge_ml.config import PipelineConfig, PipelinePhase
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "b": _record("b", ImputationStrategy.Constant, fill_value=-1.0),
        "b_missing": _indicator_record("b_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert "drop_me" in config.exclude_columns
    assert "b_missing" in config.phase_exclusions.get(PipelinePhase.Normalization, [])


def test_apply_exclusions_soft_exclusion_twice_is_idempotent():
    from dataforge_ml.config import PipelineConfig, PipelinePhase
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
        "a_missing": _indicator_record("a_missing"),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    imputer.apply_exclusions(config)
    norm = config.phase_exclusions.get(PipelinePhase.Normalization, [])
    assert norm.count("a_missing") == 1


def test_apply_exclusions_no_indicator_columns_leaves_phase_exclusions_empty():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert config.phase_exclusions == {}


# ---------------------------------------------------------------------------
# apply_exclusions — serialisation round-trip
# ---------------------------------------------------------------------------


def test_exclusions_applied_is_false_after_from_dict_round_trip():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    assert imputer._exclusions_applied is True

    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored._exclusions_applied is False


def test_to_dict_does_not_contain_exclusions_applied_key():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    d = imputer.to_dict()
    assert "_exclusions_applied" not in d
    assert "exclusions_applied" not in d


# ---------------------------------------------------------------------------
# ImputationResult.exclusions_applied — stamped by transform()
# ---------------------------------------------------------------------------


def test_transform_stamps_exclusions_applied_false_when_not_called():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({"a": pl.Series([1.0, None], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.exclusions_applied is False


def test_transform_stamps_exclusions_applied_true_when_called():
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=1.0),
    })
    config = PipelineConfig()
    imputer.apply_exclusions(config)
    df = pl.DataFrame({
        "drop_me": pl.Series([None, 1.0], dtype=pl.Float64),
        "keep_me": pl.Series([1.0, None], dtype=pl.Float64),
    })
    result = imputer.transform(df)
    assert result.exclusions_applied is True


def test_fit_path_apply_exclusions_transform_stamps_field():
    """Demonstrates the fit() → apply_exclusions() → transform() pipeline path."""
    from dataforge_ml.config import PipelineConfig
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "score": _record("score", ImputationStrategy.Mean, fill_value=5.0),
    })
    config = PipelineConfig()
    # caller invokes apply_exclusions on the FittedImputer returned by fit()
    imputer.apply_exclusions(config)
    df = pl.DataFrame({
        "drop_me": pl.Series([None, 1.0], dtype=pl.Float64),
        "score": pl.Series([1.0, None], dtype=pl.Float64),
    })
    result = imputer.transform(df)
    assert result.exclusions_applied is True
    assert "drop_me" in config.exclude_columns


# ---------------------------------------------------------------------------
# DroppedColumnAbsentWarning
# ---------------------------------------------------------------------------


def test_warning_emitted_when_dropped_column_already_absent():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({"keep_me": pl.Series([1.0, None], dtype=pl.Float64)})
    with pytest.warns(DroppedColumnAbsentWarning):
        imputer.transform(df)


def test_warning_names_the_absent_column():
    imputer = FittedImputer(records={
        "target_col": _record("target_col", ImputationStrategy.Dropped),
        "other": _record("other", ImputationStrategy.Mean, fill_value=0.0),
    })
    df = pl.DataFrame({"other": pl.Series([1.0, 2.0], dtype=pl.Float64)})
    with pytest.warns(DroppedColumnAbsentWarning, match="target_col"):
        imputer.transform(df)


def test_transform_completes_despite_warning():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=3.0),
    })
    df = pl.DataFrame({"keep_me": pl.Series([1.0, None], dtype=pl.Float64)})
    with pytest.warns(DroppedColumnAbsentWarning):
        result = imputer.transform(df)
    assert isinstance(result, ImputationResult)
    assert "keep_me" in result.dataframe.columns
    assert result.dataframe["keep_me"].null_count() == 0


def test_no_warning_when_dropped_column_is_present_in_input():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
    })
    df = pl.DataFrame({"drop_me": pl.Series([None, 1.0], dtype=pl.Float64)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedColumnAbsentWarning)
        result = imputer.transform(df)
    assert "drop_me" not in result.dataframe.columns


def test_warning_is_suppressible_via_filter():
    imputer = FittedImputer(records={
        "drop_me": _record("drop_me", ImputationStrategy.Dropped),
        "keep_me": _record("keep_me", ImputationStrategy.Mean, fill_value=1.0),
    })
    df = pl.DataFrame({"keep_me": pl.Series([1.0, None], dtype=pl.Float64)})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DroppedColumnAbsentWarning)
        result = imputer.transform(df)
    assert isinstance(result, ImputationResult)


# ---------------------------------------------------------------------------
# ImputationStrategy.Indicator — enum value and round-trip
# ---------------------------------------------------------------------------


def test_indicator_enum_value_is_indicator_string():
    assert ImputationStrategy.Indicator == "indicator"
    assert str(ImputationStrategy.Indicator) == "indicator"


def test_indicator_round_trips_via_to_dict_from_dict():
    imputer = FittedImputer(records={
        "col_missing": ColumnImputationRecord(
            column="col_missing",
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
            fill_value=None,
            indicator_added=False,
            signals=[],
        ),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["col_missing"].strategy == ImputationStrategy.Indicator


def test_indicator_string_deserialises_to_indicator_enum():
    raw = {
        "records": {
            "x_missing": {
                "column": "x_missing",
                "semantic_type": "boolean",
                "strategy": "indicator",
                "fill_value": None,
                "indicator_added": False,
                "signals": [],
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(raw)
    assert restored.records["x_missing"].strategy is ImputationStrategy.Indicator


# ---------------------------------------------------------------------------
# UnseenColumnError and FittedColumnAbsentError — class identity
# ---------------------------------------------------------------------------


def test_unseen_column_error_is_exception():
    assert issubclass(UnseenColumnError, Exception)


def test_fitted_column_absent_error_is_exception():
    assert issubclass(FittedColumnAbsentError, Exception)


def test_unseen_column_error_is_distinct_from_unfitted_column_error():
    assert UnseenColumnError is not UnfittedColumnError


def test_unseen_column_error_is_distinct_from_fitted_column_absent_error():
    assert UnseenColumnError is not FittedColumnAbsentError


def test_fitted_column_absent_error_is_distinct_from_unfitted_column_error():
    assert FittedColumnAbsentError is not UnfittedColumnError


def test_unseen_column_error_is_instantiable():
    err = UnseenColumnError("unknown: 'foo', 'bar'")
    assert "foo" in str(err)


def test_fitted_column_absent_error_is_instantiable():
    err = FittedColumnAbsentError("absent: 'score'")
    assert "score" in str(err)


def test_new_error_classes_exported_from_package():
    from dataforge_ml.imputation import FittedColumnAbsentError as FCE
    from dataforge_ml.imputation import UnseenColumnError as UCE
    assert UCE is UnseenColumnError
    assert FCE is FittedColumnAbsentError
