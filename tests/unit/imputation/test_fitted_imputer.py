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
# transform() — MNAR data-derived fill + indicator
# ---------------------------------------------------------------------------


def test_mnar_fill_applied():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.MNAR,
                          fill_value=62500.0, indicator_added=True),
    })
    df = pl.DataFrame({"income": pl.Series([50000.0, None, 75000.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["income"].null_count() == 0
    assert result.dataframe["income"][1] == pytest.approx(62500.0)


def test_mnar_indicator_column_appended():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.MNAR,
                          fill_value=62500.0, indicator_added=True),
    })
    df = pl.DataFrame({"income": pl.Series([50000.0, None, 75000.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert "income_missing" in result.dataframe.columns


def test_mnar_indicator_marks_originally_null_rows():
    imputer = FittedImputer(records={
        "income": _record("income", ImputationStrategy.MNAR,
                          fill_value=62500.0, indicator_added=True),
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
        "v": _record("v", ImputationStrategy.MNAR,
                     fill_value=2.0, indicator_added=True),
    })
    df = pl.DataFrame({"v": pl.Series([2.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    # Only the original None should be marked, not the pre-existing 2.0
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


def test_round_trip_preserves_diagnostic_fields():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = ImputationFitDiagnostic(
        r2_train=0.65,
        rmse=1.5,
        mae=1.0,
        converged=True,
        n_iter=6,
        imputed_mean=10.5,
        imputed_std=2.0,
        observed_mean=10.0,
        observed_std=3.0,
        variance_ratio=0.667,
    )
    record = ColumnImputationRecord(
        column="score",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        diagnostic=diag,
    )
    imputer = FittedImputer(records={"score": record})
    restored = FittedImputer.from_dict(imputer.to_dict())

    d = restored.records["score"].diagnostic
    assert d is not None
    assert d.r2_train == pytest.approx(0.65)
    assert d.rmse == pytest.approx(1.5)
    assert d.mae == pytest.approx(1.0)
    assert d.converged is True
    assert d.n_iter == 6
    assert d.variance_ratio == pytest.approx(0.667)
    assert d.imputed_mean == pytest.approx(10.5)
    assert d.imputed_std == pytest.approx(2.0)
    assert d.observed_mean == pytest.approx(10.0)
    assert d.observed_std == pytest.approx(3.0)


def test_round_trip_preserves_none_diagnostic():
    imputer = FittedImputer(records={
        "age": _record("age", ImputationStrategy.Mean, fill_value=30.0),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["age"].diagnostic is None


def test_round_trip_payload_without_diagnostic_key_gives_none():
    payload = {
        "records": {
            "age": {
                "column": "age",
                "semantic_type": "numeric",
                "strategy": "knn",
                "fill_value": None,
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
                # no "diagnostic" key — backward-compat path
            }
        },
        "models": {},
        "model_cols": {},
    }
    fi = FittedImputer.from_dict(payload)
    assert fi.records["age"].diagnostic is None


def test_to_dict_record_includes_diagnostic_key():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    d = imputer.to_dict()
    assert "diagnostic" in d["records"]["a"]


def test_deserialised_imputer_with_diagnostic_produces_identical_output():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = ImputationFitDiagnostic(
        r2_train=0.8, rmse=2.0, mae=1.5, converged=True, n_iter=4,
        imputed_mean=5.0, imputed_std=1.0,
        observed_mean=5.5, observed_std=2.0,
        variance_ratio=0.5,
    )
    record = ColumnImputationRecord(
        column="a",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Mean,
        fill_value=5.0,
        diagnostic=diag,
    )
    imputer = FittedImputer(records={"a": record})
    restored = FittedImputer.from_dict(imputer.to_dict())

    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    r1 = imputer.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_round_trip_preserves_fill_value():
    imputer = FittedImputer(records={
        "x": _record("x", ImputationStrategy.Mean, fill_value=42.5),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["x"].fill_value == pytest.approx(42.5)


def test_round_trip_preserves_indicator_added():
    imputer = FittedImputer(records={
        "mnar_col": _record("mnar_col", ImputationStrategy.MNAR,
                            fill_value=50000.0, indicator_added=True),
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


def test_mnar_round_trip_preserves_strategy_and_fill_value():
    """to_dict/from_dict round-trip for a MNAR column preserves strategy and fill_value."""
    imputer = FittedImputer(records={
        "salary": _record("salary", ImputationStrategy.MNAR,
                          fill_value=62500.0, indicator_added=True),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["salary"].strategy == ImputationStrategy.MNAR
    assert restored.records["salary"].fill_value == pytest.approx(62500.0)
    assert restored.records["salary"].indicator_added is True


# ---------------------------------------------------------------------------
# from_dict() — "constant" strategy deserialisation
# ---------------------------------------------------------------------------


def test_from_dict_constant_strategy_deserialises_as_constant():
    """'constant' strategy string deserialises as ImputationStrategy.Constant."""
    payload = {
        "records": {
            "income": {
                "column": "income",
                "semantic_type": "numeric",
                "strategy": "constant",
                "fill_value": -1.0,
                "indicator_added": True,
                "signals": ["declared MNAR by user configuration"],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.records["income"].strategy == ImputationStrategy.Constant


def test_from_dict_mnar_strategy_string_still_deserialises_correctly():
    """Post-Scope 8 'mnar' strategy string deserialises without error."""
    payload = {
        "records": {
            "salary": {
                "column": "salary",
                "semantic_type": "numeric",
                "strategy": "mnar",
                "fill_value": 55000.0,
                "indicator_added": True,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.records["salary"].strategy == ImputationStrategy.MNAR
    assert restored.records["salary"].fill_value == pytest.approx(55000.0)


def test_from_dict_constant_strategy_preserves_fill_value_and_indicator():
    """fill_value and indicator_added survive 'constant' strategy deserialisation."""
    payload = {
        "records": {
            "age": {
                "column": "age",
                "semantic_type": "numeric",
                "strategy": "constant",
                "fill_value": 38.0,
                "indicator_added": True,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.records["age"].strategy == ImputationStrategy.Constant
    assert restored.records["age"].fill_value == pytest.approx(38.0)
    assert restored.records["age"].indicator_added is True


def test_constant_strategy_round_trips_via_to_dict_from_dict():
    """FittedImputer with Constant-strategy column preserves strategy after to_dict/from_dict."""
    imputer = FittedImputer(records={
        "transaction_count": _record(
            "transaction_count", ImputationStrategy.Constant, fill_value=0.0,
        ),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.records["transaction_count"].strategy == ImputationStrategy.Constant
    assert restored.records["transaction_count"].fill_value == pytest.approx(0.0)


def test_constant_strategy_round_trip_fill_value_applied_correctly():
    """FittedImputer with Constant strategy applies fill_value correctly after round-trip."""
    imputer = FittedImputer(records={
        "transaction_count": _record(
            "transaction_count", ImputationStrategy.Constant, fill_value=0.0,
        ),
    })
    restored = FittedImputer.from_dict(imputer.to_dict())
    df = pl.DataFrame({"transaction_count": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    result = restored.transform(df)
    assert result.dataframe["transaction_count"].null_count() == 0
    assert result.dataframe["transaction_count"][1] == pytest.approx(0.0)


def test_from_dict_migration_does_not_affect_other_strategies():
    """'constant' remapping does not affect other strategy strings."""
    payload = {
        "records": {
            "a": {
                "column": "a", "semantic_type": "numeric",
                "strategy": "mean", "fill_value": 3.0,
                "indicator_added": False, "signals": [], "domain_snap_bounds": None,
            },
            "b": {
                "column": "b", "semantic_type": "numeric",
                "strategy": "median", "fill_value": 2.0,
                "indicator_added": False, "signals": [], "domain_snap_bounds": None,
            },
            "c": {
                "column": "c", "semantic_type": "numeric",
                "strategy": "dropped", "fill_value": None,
                "indicator_added": False, "signals": [], "domain_snap_bounds": None,
            },
            "d": {
                "column": "d", "semantic_type": "numeric",
                "strategy": "passthrough", "fill_value": None,
                "indicator_added": False, "signals": [], "domain_snap_bounds": None,
            },
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.records["a"].strategy == ImputationStrategy.Mean
    assert restored.records["b"].strategy == ImputationStrategy.Median
    assert restored.records["c"].strategy == ImputationStrategy.Dropped
    assert restored.records["d"].strategy == ImputationStrategy.Passthrough


def test_from_dict_legacy_constant_transform_produces_no_nulls():
    """A FittedImputer loaded from a legacy 'constant' payload fills nulls correctly."""
    payload = {
        "records": {
            "income": {
                "column": "income",
                "semantic_type": "numeric",
                "strategy": "constant",
                "fill_value": 50000.0,
                "indicator_added": True,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    df = pl.DataFrame({"income": pl.Series([80000.0, None, 120000.0], dtype=pl.Float64)})
    result = restored.transform(df)
    assert result.dataframe["income"].null_count() == 0
    assert result.dataframe["income"][1] == pytest.approx(50000.0)
    assert "income_missing" in result.dataframe.columns


def test_deserialised_imputer_produces_identical_output():
    """Serialised and deserialised FittedImputer must produce identical transform output."""
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
        "b": _record("b", ImputationStrategy.MNAR, fill_value=20.0, indicator_added=True),
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
    assert config.exclude_columns == ()


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


# ---------------------------------------------------------------------------
# Regression Overhaul Tests (Issue #142)
# ---------------------------------------------------------------------------


def test_regression_new_format_round_trip():
    """Verify that a FittedImputer with a new-format FittedRegression model
    can be serialized, deserialized, and used for transform successfully.
    """
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from dataforge_ml.imputation._numeric_imputer import FittedRegression
    import numpy as np

    # Create dummy data and fit an IterativeImputer
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [np.nan, 8.0]])
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
    imputer.fit(arr)

    # Build FittedRegression
    fitted_reg = FittedRegression(
        model=imputer,
        target_idx=0,
        all_cols=["y", "x"],
    )

    fi = FittedImputer(
        records={
            "y": _record("y", ImputationStrategy.Regression),
            "x": _record("x", ImputationStrategy.Passthrough),
        },
        models={"regression:y": fitted_reg},
        model_cols={"regression:y": ["y", "x"]},
    )

    # Round-trip
    restored = FittedImputer.from_dict(fi.to_dict())

    # Verify model format
    assert isinstance(restored.models["regression:y"], FittedRegression)
    assert restored.model_cols["regression:y"] == ["y", "x"]

    # Transform
    df = pl.DataFrame({
        "y": pl.Series([1.0, None, 5.0], dtype=pl.Float64),
        "x": pl.Series([2.0, 4.0, 6.0], dtype=pl.Float64),
    })
    res = restored.transform(df)
    assert res.dataframe["y"].null_count() == 0
    assert res.dataframe["y"][1] is not None


def test_regression_target_vs_feature_identification():
    """Verify that inference correctly distinguishes between target and feature columns
    using target_idx, even when a feature has a distinctive value pattern.
    """
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from dataforge_ml.imputation._numeric_imputer import FittedRegression
    import numpy as np

    # We want to verify that when target_idx = 1, it fills the column at index 1.
    # Let's say all_cols is ["x", "y"]. The target is "y" (at index 1).
    # Feature "x" has a distinctive pattern (always 999.0).
    arr = np.array([
        [999.0, 1.0],
        [999.0, 2.0],
        [999.0, 3.0],
        [999.0, np.nan],
    ])
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
    imputer.fit(arr)

    fitted_reg = FittedRegression(
        model=imputer,
        target_idx=1,  # target is index 1 ("y")
        all_cols=["x", "y"],
    )

    fi = FittedImputer(
        records={
            "x": _record("x", ImputationStrategy.Passthrough),
            "y": _record("y", ImputationStrategy.Regression),
        },
        models={"regression:y": fitted_reg},
        model_cols={"regression:y": ["x", "y"]},
    )

    df = pl.DataFrame({
        "x": pl.Series([999.0, 999.0, 999.0], dtype=pl.Float64),
        "y": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
    })

    res = fi.transform(df)
    # The imputed value for "y" should be close to 2.0 (based on training)
    # and definitely NOT 999.0 (the feature value).
    imputed_val = res.dataframe["y"][1]
    assert imputed_val is not None
    assert abs(imputed_val - 2.0) < 0.5
    assert imputed_val != 999.0


# ---------------------------------------------------------------------------
# _FittedKNN — transform, scale-sensitivity, round-trip
# ---------------------------------------------------------------------------


def test_fitted_knn_transform_produces_no_nulls():
    import numpy as np
    from sklearn.impute import KNNImputer
    from dataforge_ml.imputation._fitted_imputer import _FittedKNN

    # Training matrix: 4 complete rows, 2 KNN columns.
    train = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [0.2, 0.8],
    ], dtype=np.float64)
    col_means = np.nanmean(train, axis=0)
    col_stds = np.nanstd(train, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    train_scaled = (train - col_means) / col_stds

    knn = KNNImputer(n_neighbors=2)
    knn.fit(train_scaled)

    fitted_knn = _FittedKNN(model=knn, col_means=col_means, col_stds=col_stds)
    imputer = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.KNN),
            "b": _record("b", ImputationStrategy.KNN),
        },
        models={"knn": fitted_knn},
        model_cols={"knn": ["a", "b"]},
    )

    df = pl.DataFrame({
        "a": pl.Series([0.0, None, 0.5, 0.2], dtype=pl.Float64),
        "b": pl.Series([1.0, 0.0, None, 0.8], dtype=pl.Float64),
    })
    result = imputer.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["b"].null_count() == 0


def test_fitted_knn_scale_sensitive_imputed_value_not_dominated_by_large_column():
    """With 1000:1 magnitude ratio, _apply_knn must not let the large-scale
    column dominate neighbour selection: the imputed value must be
    inverse-scaled back to the original small-column range.
    """
    import numpy as np
    from sklearn.impute import KNNImputer
    from dataforge_ml.imputation._fitted_imputer import _FittedKNN

    # `small` in [0, 1]; `large` in [0, 1000]; perfect positive correlation.
    # Query: small=None, large=1000 → nearest scaled neighbour is row3 →
    # imputed small must be ≈ 1.0, not ≈ 1220 (what unscaled-then-no-inverse
    # would produce).
    train = np.array([
        [0.0, 0.0],
        [0.5, 500.0],
        [1.0, 1000.0],
    ], dtype=np.float64)
    col_means = np.nanmean(train, axis=0)
    col_stds = np.nanstd(train, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    train_scaled = (train - col_means) / col_stds

    knn = KNNImputer(n_neighbors=1)
    knn.fit(train_scaled)

    fitted_knn = _FittedKNN(model=knn, col_means=col_means, col_stds=col_stds)
    imputer = FittedImputer(
        records={
            "small": _record("small", ImputationStrategy.KNN),
            "large": _record("large", ImputationStrategy.KNN),
        },
        models={"knn": fitted_knn},
        model_cols={"knn": ["small", "large"]},
    )

    df = pl.DataFrame({
        "small": pl.Series([None], dtype=pl.Float64),
        "large": pl.Series([1000.0], dtype=pl.Float64),
    })
    result = imputer.transform(df)

    imputed_small = result.dataframe["small"][0]
    # Must be in the original column's range [0, 1], not dominated by large.
    assert imputed_small == pytest.approx(1.0, abs=0.01)


def test_fitted_knn_to_dict_from_dict_round_trip():
    import numpy as np
    from sklearn.impute import KNNImputer
    from dataforge_ml.imputation._fitted_imputer import _FittedKNN

    train = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    col_means = np.nanmean(train, axis=0)
    col_stds = np.nanstd(train, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    train_scaled = (train - col_means) / col_stds

    knn = KNNImputer(n_neighbors=1)
    knn.fit(train_scaled)

    original = _FittedKNN(model=knn, col_means=col_means, col_stds=col_stds)
    restored = _FittedKNN.from_dict(original.to_dict())

    np.testing.assert_array_almost_equal(restored.col_means, original.col_means)
    np.testing.assert_array_almost_equal(restored.col_stds, original.col_stds)

    # Restored model must produce identical transform output.
    query = np.array([[None, 1.0]], dtype=np.float64)
    query_scaled = (query - col_means) / col_stds
    out_original = original.model.transform(query_scaled)
    out_restored = restored.model.transform(query_scaled)
    np.testing.assert_array_almost_equal(out_original, out_restored)


# ---------------------------------------------------------------------------
# Issue #162 — MICE-backed FittedImputer with non-default estimator variants
# ---------------------------------------------------------------------------


def test_mice_random_forest_backed_transform_produces_no_nulls():
    """FittedImputer with IterativeImputer(RandomForestRegressor) must produce no nulls."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import IterativeImputer

    train = np.array([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, np.nan], [np.nan, 10.0],
    ])
    mice = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        random_state=0,
    )
    mice.fit(train)

    fi = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.MICE),
            "b": _record("b", ImputationStrategy.MICE),
        },
        models={"mice": mice},
        model_cols={"mice": ["a", "b"]},
    )
    df = pl.DataFrame({
        "a": pl.Series([1.0, None, 5.0, 7.0, None], dtype=pl.Float64),
        "b": pl.Series([2.0, 4.0, None, None, 10.0], dtype=pl.Float64),
    })
    result = fi.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["b"].null_count() == 0


def test_mice_gradient_boosting_backed_transform_produces_no_nulls():
    """FittedImputer with IterativeImputer(GradientBoostingRegressor) must produce no nulls."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import IterativeImputer

    train = np.array([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, np.nan], [np.nan, 10.0],
    ])
    mice = IterativeImputer(
        estimator=GradientBoostingRegressor(n_estimators=10, random_state=0),
        random_state=0,
    )
    mice.fit(train)

    fi = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.MICE),
            "b": _record("b", ImputationStrategy.MICE),
        },
        models={"mice": mice},
        model_cols={"mice": ["a", "b"]},
    )
    df = pl.DataFrame({
        "a": pl.Series([1.0, None, 5.0, 7.0, None], dtype=pl.Float64),
        "b": pl.Series([2.0, 4.0, None, None, 10.0], dtype=pl.Float64),
    })
    result = fi.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["b"].null_count() == 0


def test_mice_random_forest_backed_round_trip():
    """to_dict/from_dict round-trip preserves a RandomForestRegressor-backed IterativeImputer."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import IterativeImputer

    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, np.nan]])
    mice = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=5, random_state=0),
        random_state=0,
    )
    mice.fit(train)

    fi = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.MICE),
            "b": _record("b", ImputationStrategy.MICE),
        },
        models={"mice": mice},
        model_cols={"mice": ["a", "b"]},
    )
    restored = FittedImputer.from_dict(fi.to_dict())

    df = pl.DataFrame({
        "a": pl.Series([1.0, None, 5.0], dtype=pl.Float64),
        "b": pl.Series([2.0, 4.0, None], dtype=pl.Float64),
    })
    r1 = fi.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)
    assert r2.dataframe["a"].null_count() == 0
    assert r2.dataframe["b"].null_count() == 0


def test_mice_gradient_boosting_backed_round_trip():
    """to_dict/from_dict round-trip preserves a GradientBoostingRegressor-backed IterativeImputer."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import IterativeImputer

    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, np.nan]])
    mice = IterativeImputer(
        estimator=GradientBoostingRegressor(n_estimators=5, random_state=0),
        random_state=0,
    )
    mice.fit(train)

    fi = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.MICE),
            "b": _record("b", ImputationStrategy.MICE),
        },
        models={"mice": mice},
        model_cols={"mice": ["a", "b"]},
    )
    restored = FittedImputer.from_dict(fi.to_dict())

    df = pl.DataFrame({
        "a": pl.Series([1.0, None, 5.0], dtype=pl.Float64),
        "b": pl.Series([2.0, 4.0, None], dtype=pl.Float64),
    })
    r1 = fi.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)
    assert r2.dataframe["a"].null_count() == 0
    assert r2.dataframe["b"].null_count() == 0


def test_regression_inference_time_feature_nans():
    """Verify that regression imputation handles feature columns with missing values
    at inference time without errors and without resorting to a feat_means patching loop
    in the apply path.
    """
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from dataforge_ml.imputation._numeric_imputer import FittedRegression
    import numpy as np

    # Train imputer with some missing values in features to support it
    arr = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, np.nan],
        [7.0, 8.0],
    ])
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
    imputer.fit(arr)

    fitted_reg = FittedRegression(
        model=imputer,
        target_idx=0,
        all_cols=["y", "x"],
    )

    fi = FittedImputer(
        records={
            "y": _record("y", ImputationStrategy.Regression),
            "x": _record("x", ImputationStrategy.Regression),
        },
        models={"regression:y": fitted_reg},
        model_cols={"regression:y": ["y", "x"]},
    )

    # During transform, both the target 'y' and the feature 'x' have NaNs
    df = pl.DataFrame({
        "y": pl.Series([1.0, None, 5.0], dtype=pl.Float64),
        "x": pl.Series([2.0, np.nan, 6.0], dtype=pl.Float64),
    })

    # This should run without throwing any exceptions
    res = fi.transform(df)
    assert res.dataframe["y"].null_count() == 0
    assert res.dataframe["y"][1] is not None


# ---------------------------------------------------------------------------
# numeric_sentinels — field, transform wiring, serialisation (issue #174)
# ---------------------------------------------------------------------------


def test_numeric_sentinels_field_defaults_to_empty_dict():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=5.0),
    })
    assert imputer.numeric_sentinels == {}


def test_numeric_sentinels_field_accepts_declared_mapping():
    imputer = FittedImputer(
        records={"age": _record("age", ImputationStrategy.Mean, fill_value=30.0)},
        numeric_sentinels={"age": [-999.0]},
    )
    assert imputer.numeric_sentinels == {"age": [-999.0]}


def test_transform_normalises_int64_sentinel_before_fill():
    imputer = FittedImputer(
        records={"age": _record("age", ImputationStrategy.Mean, fill_value=30.0)},
        numeric_sentinels={"age": [-999.0]},
    )
    df = pl.DataFrame({"age": pl.Series([-999, 25, None], dtype=pl.Int64)})
    result = imputer.transform(df)
    assert result.dataframe["age"].null_count() == 0
    assert result.dataframe["age"][0] == 30
    assert result.dataframe["age"][1] == 25


def test_transform_normalises_float64_sentinel_before_fill():
    imputer = FittedImputer(
        records={"score": _record("score", ImputationStrategy.Median, fill_value=50.0)},
        numeric_sentinels={"score": [-999.0]},
    )
    df = pl.DataFrame({"score": pl.Series([-999.0, 80.0, None], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["score"].null_count() == 0
    assert result.dataframe["score"][0] == pytest.approx(50.0)
    assert result.dataframe["score"][1] == pytest.approx(80.0)


def test_transform_multiple_sentinels_all_normalised():
    imputer = FittedImputer(
        records={"x": _record("x", ImputationStrategy.Mean, fill_value=0.0)},
        numeric_sentinels={"x": [-999.0, 9999.0]},
    )
    df = pl.DataFrame({"x": pl.Series([-999, 9999, 42], dtype=pl.Int32)})
    result = imputer.transform(df)
    assert result.dataframe["x"][0] == 0
    assert result.dataframe["x"][1] == 0
    assert result.dataframe["x"][2] == 42


def test_transform_sentinel_only_affects_declared_column():
    imputer = FittedImputer(
        records={
            "a": _record("a", ImputationStrategy.Mean, fill_value=0.0),
            "b": _record("b", ImputationStrategy.Mean, fill_value=0.0),
        },
        numeric_sentinels={"a": [-999.0]},
    )
    df = pl.DataFrame({
        "a": pl.Series([-999, 1], dtype=pl.Int64),
        "b": pl.Series([-999, 1], dtype=pl.Int64),
    })
    result = imputer.transform(df)
    assert result.dataframe["a"][0] == 0    # sentinel normalised then filled
    assert result.dataframe["b"][0] == -999  # not declared — unchanged


def test_transform_empty_sentinels_behaviour_unchanged():
    imputer = FittedImputer(
        records={"f": _record("f", ImputationStrategy.Mean, fill_value=9.0)},
        numeric_sentinels={},
    )
    df = pl.DataFrame({"f": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    result = imputer.transform(df)
    assert result.dataframe["f"].null_count() == 0
    assert result.dataframe["f"][1] == pytest.approx(9.0)


def test_to_dict_includes_numeric_sentinels_key():
    imputer = FittedImputer(
        records={"age": _record("age", ImputationStrategy.Mean, fill_value=30.0)},
        numeric_sentinels={"age": [-999.0]},
    )
    d = imputer.to_dict()
    assert "numeric_sentinels" in d
    assert d["numeric_sentinels"] == {"age": [-999.0]}


def test_to_dict_numeric_sentinels_empty_when_not_set():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    d = imputer.to_dict()
    assert "numeric_sentinels" in d
    assert d["numeric_sentinels"] == {}


def test_from_dict_restores_numeric_sentinels():
    imputer = FittedImputer(
        records={"age": _record("age", ImputationStrategy.Mean, fill_value=30.0)},
        numeric_sentinels={"age": [-999.0, 9999.0]},
    )
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.numeric_sentinels == {"age": [-999.0, 9999.0]}


def test_from_dict_without_numeric_sentinels_key_defaults_to_empty_dict():
    payload = {
        "records": {
            "age": {
                "column": "age",
                "semantic_type": "numeric",
                "strategy": "mean",
                "fill_value": 30.0,
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.numeric_sentinels == {}


def test_round_trip_with_sentinels_produces_identical_transform_output():
    imputer = FittedImputer(
        records={"age": _record("age", ImputationStrategy.Mean, fill_value=30.0)},
        numeric_sentinels={"age": [-999.0]},
    )
    restored = FittedImputer.from_dict(imputer.to_dict())

    df = pl.DataFrame({"age": pl.Series([-999, 25, None], dtype=pl.Int64)})
    r1 = imputer.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_old_format_deserialised_imputer_transforms_identically_to_empty_sentinels():
    payload_old = {
        "records": {
            "score": {
                "column": "score",
                "semantic_type": "numeric",
                "strategy": "mean",
                "fill_value": 5.0,
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored_old = FittedImputer.from_dict(payload_old)

    empty_sentinels = FittedImputer(
        records={"score": _record("score", ImputationStrategy.Mean, fill_value=5.0)},
        numeric_sentinels={},
    )

    df = pl.DataFrame({"score": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    r_old = restored_old.transform(df)
    r_empty = empty_sentinels.transform(df)
    assert r_old.dataframe.equals(r_empty.dataframe)


# ---------------------------------------------------------------------------
# string_sentinels — field, transform wiring, serialisation (issue #182)
# ---------------------------------------------------------------------------


def test_string_sentinels_field_defaults_to_empty_dict():
    imputer = FittedImputer(records={
        "cat": ColumnImputationRecord(
            column="cat", semantic_type=SemanticType.Categorical,
            strategy=ImputationStrategy.Constant, fill_value="unknown",
            indicator_added=False, signals=[],
        ),
    })
    assert imputer.string_sentinels == {}


def test_string_sentinels_field_accepts_declared_mapping():
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["N/A", "missing"]},
    )
    assert imputer.string_sentinels == {"status": ["N/A", "missing"]}


def test_transform_normalises_declared_string_sentinel_before_fill():
    """Declared string sentinels are converted to null and then filled by the record strategy."""
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["N/A", "missing"]},
    )
    df = pl.DataFrame({"status": pl.Series(["active", "N/A", "missing"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["status"].null_count() == 0
    assert result.dataframe["status"][0] == "active"
    assert result.dataframe["status"][1] == "unknown"
    assert result.dataframe["status"][2] == "unknown"


def test_transform_declared_sentinels_suppress_hardcoded_defaults():
    """When a column has a string_sentinels declaration, hardcoded defaults like 'NA' are NOT treated as null."""
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["MISSING"]},
    )
    # "NA" is a hardcoded default; with replace semantics it must NOT be null-normalised.
    df = pl.DataFrame({"status": pl.Series(["active", "NA", "MISSING"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["status"][0] == "active"
    assert result.dataframe["status"][1] == "NA", '"NA" must pass through unchanged when replaced by declared sentinels'
    assert result.dataframe["status"][2] == "unknown"


def test_transform_declared_sentinels_matched_case_insensitively():
    """Declared string sentinels match data case-insensitively."""
    imputer = FittedImputer(
        records={
            "cat": ColumnImputationRecord(
                column="cat", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="filled",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"cat": ["MISSING"]},
    )
    df = pl.DataFrame({"cat": pl.Series(["x", "missing", "MISSING", "Missing"], dtype=pl.String)})
    result = imputer.transform(df)
    assert result.dataframe["cat"][0] == "x"
    assert result.dataframe["cat"][1] == "filled"
    assert result.dataframe["cat"][2] == "filled"
    assert result.dataframe["cat"][3] == "filled"


def test_transform_empty_string_sentinel_behaviour_unchanged():
    """When string_sentinels is empty, hardcoded default behaviour is unchanged."""
    imputer = FittedImputer(
        records={
            "cat": ColumnImputationRecord(
                column="cat", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={},
    )
    df = pl.DataFrame({"cat": pl.Series(["NA", "hello", "NULL"], dtype=pl.String)})
    result = imputer.transform(df)
    # Hardcoded defaults still apply: "NA" and "NULL" become null → filled
    assert result.dataframe["cat"].null_count() == 0
    assert result.dataframe["cat"][0] == "unknown"
    assert result.dataframe["cat"][1] == "hello"
    assert result.dataframe["cat"][2] == "unknown"


def test_transform_string_sentinel_only_affects_declared_column():
    """string_sentinels declarations for one column do not affect sibling columns."""
    imputer = FittedImputer(
        records={
            "a": ColumnImputationRecord(
                column="a", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="filled",
                indicator_added=False, signals=[],
            ),
            "b": ColumnImputationRecord(
                column="b", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="filled",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"a": ["CUSTOM"]},
    )
    # Column "a" has replace semantics: only "CUSTOM" is null, not "NA"
    # Column "b" has no declaration: "NA" still uses hardcoded defaults
    df = pl.DataFrame({
        "a": pl.Series(["CUSTOM", "NA", "ok"], dtype=pl.String),
        "b": pl.Series(["CUSTOM", "NA", "ok"], dtype=pl.String),
    })
    result = imputer.transform(df)
    assert result.dataframe["a"][0] == "filled"   # CUSTOM matched (declared)
    assert result.dataframe["a"][1] == "NA"        # NA not matched (replaced)
    assert result.dataframe["b"][0] == "CUSTOM"    # CUSTOM not a hardcoded default
    assert result.dataframe["b"][1] == "filled"    # NA matched (hardcoded default)


def test_to_dict_includes_string_sentinels_key():
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["N/A", "missing"]},
    )
    d = imputer.to_dict()
    assert "string_sentinels" in d
    assert d["string_sentinels"] == {"status": ["N/A", "missing"]}


def test_to_dict_string_sentinels_empty_when_not_set():
    imputer = FittedImputer(records={
        "a": _record("a", ImputationStrategy.Mean, fill_value=1.0),
    })
    d = imputer.to_dict()
    assert "string_sentinels" in d
    assert d["string_sentinels"] == {}


def test_from_dict_restores_string_sentinels():
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["N/A", "missing"]},
    )
    restored = FittedImputer.from_dict(imputer.to_dict())
    assert restored.string_sentinels == {"status": ["N/A", "missing"]}


def test_from_dict_without_string_sentinels_key_defaults_to_empty_dict():
    """Old-format payload without 'string_sentinels' key deserialises to empty dict."""
    payload = {
        "records": {
            "status": {
                "column": "status",
                "semantic_type": "categorical",
                "strategy": "constant",
                "fill_value": "unknown",
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored = FittedImputer.from_dict(payload)
    assert restored.string_sentinels == {}


def test_round_trip_with_string_sentinels_produces_identical_transform_output():
    """Serialized and deserialized FittedImputer with string_sentinels produces identical output."""
    imputer = FittedImputer(
        records={
            "status": ColumnImputationRecord(
                column="status", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.Constant, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={"status": ["N/A", "missing"]},
    )
    restored = FittedImputer.from_dict(imputer.to_dict())

    df = pl.DataFrame({"status": pl.Series(["active", "N/A", "missing"], dtype=pl.String)})
    r1 = imputer.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_old_format_deserialised_imputer_transforms_identically_to_empty_string_sentinels():
    """An imputer loaded from an old-format dict behaves identically to one with string_sentinels={}."""
    payload_old = {
        "records": {
            "cat": {
                "column": "cat",
                "semantic_type": "categorical",
                "strategy": "constant",
                "fill_value": "unknown",
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    restored_old = FittedImputer.from_dict(payload_old)

    empty_str_sentinels = FittedImputer(
        records={
            "cat": ColumnImputationRecord(
                column="cat", semantic_type=SemanticType.Categorical,
                strategy=ImputationStrategy.MNAR, fill_value="unknown",
                indicator_added=False, signals=[],
            ),
        },
        string_sentinels={},
    )

    df = pl.DataFrame({"cat": pl.Series(["NA", "hello", "world"], dtype=pl.String)})
    r_old = restored_old.transform(df)
    r_empty = empty_str_sentinels.transform(df)
    assert r_old.dataframe.equals(r_empty.dataframe)

