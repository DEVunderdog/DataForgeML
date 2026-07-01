"""
Unit tests for ImputationOrchestrator.
"""

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, SemanticType
from dataforge_ml.imputation._config import ImputationStrategy, NumericImputationConfig
from dataforge_ml.imputation._fitted_imputer import FittedImputer
from dataforge_ml.imputation.orchestrator import ImputationOrchestrator
from dataforge_ml.profiling._config import (
    ColumnProfile,
    NumericKind,
    StructuralProfileResult,
)
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import NumericStats, SkewSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(cols: dict[str, ColumnProfile]) -> StructuralProfileResult:
    result = StructuralProfileResult()
    result.columns.update(cols)
    return result


def _numeric_cp_with_nulls(col: str, null_count: int = 5, total: int = 100,
                            severity: MissingSeverity = MissingSeverity.Minor,
                            skew: SkewSeverity = SkewSeverity.Normal) -> ColumnProfile:
    return ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column=col, total_rows=total,
            effective_null_count=null_count,
            effective_null_ratio=null_count / total,
            severity=severity,
        ),
        stats=NumericStats(skewness_severity=skew),
    )


def _clean_numeric_cp(col: str) -> ColumnProfile:
    return ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column=col, total_rows=100,
            effective_null_count=0,
        ),
        stats=NumericStats(),
    )


# ---------------------------------------------------------------------------
# fit() — basic contract
# ---------------------------------------------------------------------------


def test_fit_returns_fitted_imputer():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    result = ImputationOrchestrator().fit(df, profile)
    assert isinstance(result, FittedImputer)


def test_fit_does_not_mutate_orchestrator():
    """Calling fit() twice on the same orchestrator should produce independent FittedImputators."""
    orch = ImputationOrchestrator()
    df1 = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    df2 = pl.DataFrame({"a": pl.Series([10.0, None, 30.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    fi1 = orch.fit(df1, profile)
    fi2 = orch.fit(df2, profile)

    # Fill values should differ (computed from different train data)
    assert fi1.records["a"].fill_value != fi2.records["a"].fill_value


def test_fit_records_all_numeric_columns():
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "b": pl.Series([2.0, 3.0], dtype=pl.Float64),
    })
    profile = _make_profile({
        "a": _numeric_cp_with_nulls("a"),
        "b": _clean_numeric_cp("b"),
    })
    fi = ImputationOrchestrator().fit(df, profile)
    assert "a" in fi.records
    assert "b" in fi.records


# ---------------------------------------------------------------------------
# fit() — Text and Identifier columns skipped
# ---------------------------------------------------------------------------


def test_text_columns_in_records_with_passthrough():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "txt": pl.Series(["hello", "world"], dtype=pl.Utf8),
    })
    num_cp = _numeric_cp_with_nulls("num")
    txt_cp = ColumnProfile(name="txt", semantic_type=SemanticType.Text)
    profile = _make_profile({"num": num_cp, "txt": txt_cp})

    fi = ImputationOrchestrator().fit(df, profile)
    assert "txt" in fi.records
    assert fi.records["txt"].strategy == ImputationStrategy.Passthrough
    assert fi.records["txt"].semantic_type == SemanticType.Text


def test_identifier_columns_in_records_with_passthrough():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "id_col": pl.Series(["A001", "A002"], dtype=pl.Utf8),
    })
    num_cp = _numeric_cp_with_nulls("num")
    id_cp = ColumnProfile(name="id_col", semantic_type=SemanticType.Identifier)
    profile = _make_profile({"num": num_cp, "id_col": id_cp})

    fi = ImputationOrchestrator().fit(df, profile)
    assert "id_col" in fi.records
    assert fi.records["id_col"].strategy == ImputationStrategy.Passthrough
    assert fi.records["id_col"].semantic_type == SemanticType.Identifier


# ---------------------------------------------------------------------------
# fit_transform() convenience
# ---------------------------------------------------------------------------


def test_fit_transform_returns_imputation_result_with_no_nulls():
    from dataforge_ml.imputation._config import ImputationResult

    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, None], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    _fitted, result = ImputationOrchestrator().fit_transform(df, profile)
    assert isinstance(result, ImputationResult)
    assert result.dataframe["a"].null_count() == 0


def test_fit_transform_is_equivalent_to_fit_then_transform():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    orch = ImputationOrchestrator()
    _fi, r1 = orch.fit_transform(df, profile)
    r2 = orch.fit(df, profile).transform(df)
    assert r1.dataframe.equals(r2.dataframe)


# ---------------------------------------------------------------------------
# fit_transform() — tuple return (ADR-0021)
# ---------------------------------------------------------------------------


def test_fit_transform_return_value_is_tuple_of_length_two():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    result = ImputationOrchestrator().fit_transform(df, profile)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_fit_transform_first_element_is_fitted_imputer():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    fitted, _result = ImputationOrchestrator().fit_transform(df, profile)
    assert isinstance(fitted, FittedImputer)


def test_fit_transform_second_element_is_imputation_result():
    from dataforge_ml.imputation._config import ImputationResult

    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    _fitted, result = ImputationOrchestrator().fit_transform(df, profile)
    assert isinstance(result, ImputationResult)


def test_fit_transform_fitted_imputer_consistent_with_standalone_fit():
    """FittedImputer from fit_transform must produce the same fill values as standalone fit."""
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, None, 5.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    orch = ImputationOrchestrator()
    fi_from_fit_transform, _ = orch.fit_transform(df, profile)
    fi_from_fit = orch.fit(df, profile)

    assert fi_from_fit_transform.records["a"].fill_value == fi_from_fit.records["a"].fill_value
    assert fi_from_fit_transform.records["a"].strategy == fi_from_fit.records["a"].strategy


# ---------------------------------------------------------------------------
# fit() — schema manifest: Passthrough records for all unhandled columns
# ---------------------------------------------------------------------------


def _categorical_cp(col: str) -> ColumnProfile:
    return ColumnProfile(name=col, semantic_type=SemanticType.Categorical)


def test_text_column_passthrough_record_fill_value_is_none():
    df = pl.DataFrame({"txt": pl.Series(["a", "b"], dtype=pl.Utf8)})
    profile = _make_profile({"txt": ColumnProfile(name="txt", semantic_type=SemanticType.Text)})
    fi = ImputationOrchestrator().fit(df, profile)
    assert fi.records["txt"].fill_value is None


def test_text_column_passthrough_record_indicator_added_is_false():
    df = pl.DataFrame({"txt": pl.Series(["a", "b"], dtype=pl.Utf8)})
    profile = _make_profile({"txt": ColumnProfile(name="txt", semantic_type=SemanticType.Text)})
    fi = ImputationOrchestrator().fit(df, profile)
    assert fi.records["txt"].indicator_added is False


def test_identifier_column_passthrough_record_semantic_type():
    df = pl.DataFrame({"id": pl.Series(["X1", "X2"], dtype=pl.Utf8)})
    profile = _make_profile({"id": ColumnProfile(name="id", semantic_type=SemanticType.Identifier)})
    fi = ImputationOrchestrator().fit(df, profile)
    assert fi.records["id"].semantic_type == SemanticType.Identifier


def test_categorical_column_with_no_fill_strategy_gets_passthrough():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "cat": pl.Series(["a", "b"], dtype=pl.Utf8),
    })
    profile = _make_profile({
        "num": _numeric_cp_with_nulls("num"),
        "cat": _categorical_cp("cat"),
    })
    fi = ImputationOrchestrator().fit(df, profile)
    assert "cat" in fi.records
    assert fi.records["cat"].strategy == ImputationStrategy.Passthrough
    assert fi.records["cat"].semantic_type == SemanticType.Categorical


def test_numeric_column_already_in_records_is_not_overwritten_by_passthrough():
    """Sub-processor result must win over the Passthrough pass for numeric columns."""
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    fi = ImputationOrchestrator().fit(df, profile)
    assert fi.records["a"].strategy != ImputationStrategy.Passthrough


def test_all_train_df_profiled_columns_in_records():
    """Every column in train_df that has a profile entry must appear in records."""
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "txt": pl.Series(["x", "y"], dtype=pl.Utf8),
        "id": pl.Series(["A", "B"], dtype=pl.Utf8),
    })
    profile = _make_profile({
        "num": _numeric_cp_with_nulls("num"),
        "txt": ColumnProfile(name="txt", semantic_type=SemanticType.Text),
        "id": ColumnProfile(name="id", semantic_type=SemanticType.Identifier),
    })
    fi = ImputationOrchestrator().fit(df, profile)
    assert "num" in fi.records
    assert "txt" in fi.records
    assert "id" in fi.records


# ---------------------------------------------------------------------------
# fit() — schema manifest: Indicator records pre-registered at fit time
# ---------------------------------------------------------------------------


def _numeric_cp_mnar(col: str) -> ColumnProfile:
    return ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=None,
        missingness=ColumnMissingnessProfile(
            column=col, total_rows=100,
            effective_null_count=10,
            effective_null_ratio=0.10,
            severity=MissingSeverity.Minor,
        ),
        stats=NumericStats(),
    )


def test_indicator_record_present_in_records_after_fit_before_transform():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = ImputationOrchestrator(cfg).fit(df, profile)

    assert "income_missing" in fi.records


def test_indicator_record_has_strategy_indicator():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = ImputationOrchestrator(cfg).fit(df, profile)

    assert fi.records["income_missing"].strategy == ImputationStrategy.Indicator


def test_indicator_record_has_boolean_semantic_type():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = ImputationOrchestrator(cfg).fit(df, profile)

    assert fi.records["income_missing"].semantic_type == SemanticType.Boolean


def test_indicator_record_indicator_added_is_false():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = ImputationOrchestrator(cfg).fit(df, profile)

    assert fi.records["income_missing"].indicator_added is False


def test_indicator_record_not_present_when_no_indicator_added_columns():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    fi = ImputationOrchestrator().fit(df, profile)
    indicator_cols = [c for c in fi.records if c.endswith("_missing")]
    assert indicator_cols == []


def test_indicator_record_round_trips_via_to_dict_from_dict():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = ImputationOrchestrator(cfg).fit(df, profile)

    from dataforge_ml.imputation._fitted_imputer import FittedImputer
    restored = FittedImputer.from_dict(fi.to_dict())
    assert "income_missing" in restored.records
    assert restored.records["income_missing"].strategy == ImputationStrategy.Indicator
    assert restored.records["income_missing"].semantic_type == SemanticType.Boolean


# ---------------------------------------------------------------------------
# Issue #175 — numeric_sentinels threading through fit()
# ---------------------------------------------------------------------------


def test_fit_threads_numeric_sentinels_to_fitted_imputer():
    """FittedImputer returned by fit() carries the sentinels declared on the profile."""
    df = pl.DataFrame({
        "age": pl.Series([25, -999, 30, -999, 40], dtype=pl.Int64),
    })
    profile = _make_profile({"age": _numeric_cp_with_nulls("age", null_count=2, total=5)})
    profile.numeric_sentinels = {"age": [-999.0]}

    fi = ImputationOrchestrator().fit(df, profile)

    assert fi.numeric_sentinels == {"age": [-999.0]}


def test_fit_numeric_sentinels_empty_when_profile_has_none():
    """FittedImputer has an empty sentinels dict when the profile declares none."""
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    fi = ImputationOrchestrator().fit(df, profile)

    assert fi.numeric_sentinels == {}


# ---------------------------------------------------------------------------
# fit() — per_column_strategy size-guard validation (Seam 3)
# ---------------------------------------------------------------------------


def test_per_column_strategy_regression_size_guard_raises_below_min_rows():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=3)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        regression_min_rows=10,
        _per_column_strategy={"a": ImputationStrategy.Regression},
    )
    with pytest.raises(ValueError):
        ImputationOrchestrator(cfg).fit(df, profile)


def test_per_column_strategy_regression_size_guard_error_message_content():
    n_rows, threshold = 3, 10
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=n_rows)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        regression_min_rows=threshold,
        _per_column_strategy={"a": ImputationStrategy.Regression},
    )
    with pytest.raises(ValueError) as exc_info:
        ImputationOrchestrator(cfg).fit(df, profile)
    msg = str(exc_info.value)
    assert "'a'" in msg
    assert "Regression" in msg
    assert "regression_min_rows" in msg
    assert f"n_rows={n_rows}" in msg
    assert f"regression_min_rows={threshold}" in msg


def test_per_column_strategy_knn_size_guard_raises_above_max_rows():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, 4.0, 5.0, 6.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=6)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        knn_max_rows=3,
        _per_column_strategy={"a": ImputationStrategy.KNN},
    )
    with pytest.raises(ValueError):
        ImputationOrchestrator(cfg).fit(df, profile)


def test_per_column_strategy_knn_size_guard_error_message_content():
    n_rows, threshold = 6, 3
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, 4.0, 5.0, 6.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=n_rows)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        knn_max_rows=threshold,
        _per_column_strategy={"a": ImputationStrategy.KNN},
    )
    with pytest.raises(ValueError) as exc_info:
        ImputationOrchestrator(cfg).fit(df, profile)
    msg = str(exc_info.value)
    assert "'a'" in msg
    assert "KNN" in msg
    assert "knn_max_rows" in msg
    assert f"n_rows={n_rows}" in msg
    assert f"knn_max_rows={threshold}" in msg


def test_per_column_strategy_regression_no_error_when_size_guard_met():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, 4.0, 5.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=5)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        regression_min_rows=3,
        _per_column_strategy={"a": ImputationStrategy.Regression},
    )
    # n_rows=5 >= regression_min_rows=3 — guard passes, no ValueError
    ImputationOrchestrator(cfg).fit(df, profile)


def test_per_column_strategy_knn_no_error_when_size_guard_met():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=1, total=3)})
    cfg = PipelineConfig()
    cfg.imputation.numeric = NumericImputationConfig(
        knn_max_rows=10,
        _per_column_strategy={"a": ImputationStrategy.KNN},
    )
    # n_rows=3 <= knn_max_rows=10 — guard passes, no ValueError
    ImputationOrchestrator(cfg).fit(df, profile)


# ---------------------------------------------------------------------------
# fit() — validate() called before processing
# ---------------------------------------------------------------------------


def test_imputation_orchestrator_raises_on_invalid_config_before_processing():
    """
    Ensure the orchestrator calls config.imputation.validate() before processing.
    We construct a conflicting config via legitimate setters and verify it raises.
    """
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    
    cfg = PipelineConfig()
    # Add MNAR column first
    cfg.imputation.add_mnar_column("a")
    # Set per_column_strategy using legitimate setter, creating conflict
    cfg.imputation.numeric.set_per_column_strategy("a", ImputationStrategy.Median)
    
    orch = ImputationOrchestrator(cfg)
    with pytest.raises(ValueError, match="mutually exclusive: 'a'"):
        orch.fit(df, profile)
