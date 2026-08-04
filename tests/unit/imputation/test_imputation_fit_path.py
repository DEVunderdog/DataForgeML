"""
Unit tests for the layered imputation fit path (decide → execute → build).

These cover the whole-frame contracts the aggregate owns — the full-schema
manifest, Passthrough and Indicator projection, sentinel threading, and the
decide-time size guards.  They previously drove the fused
``ImputationOrchestrator.fit()``; that seam is gone (#368) and the assertions
now drive ``fit_imputer``, the layered drive.

The ``fit_transform`` cases that lived here are gone with it: the fused
convenience it tested (ADR-0021's tuple return) no longer exists, and
``build().transform(df)`` is the explicit two-step that replaced it.
"""

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, SemanticType
from dataforge_ml.imputation._config import ImputationStrategy, NumericImputationConfig
from dataforge_ml.imputation._fitted_imputer import FittedImputer
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
from tests.conftest import fit_imputer

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
    result = fit_imputer(df, profile)
    assert isinstance(result, FittedImputer)


def test_fit_records_all_numeric_columns():
    df = pl.DataFrame({
        "a": pl.Series([1.0, None], dtype=pl.Float64),
        "b": pl.Series([2.0, 3.0], dtype=pl.Float64),
    })
    profile = _make_profile({
        "a": _numeric_cp_with_nulls("a"),
        "b": _clean_numeric_cp("b"),
    })
    fi = fit_imputer(df, profile)
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

    fi = fit_imputer(df, profile)
    assert "txt" in fi.records
    assert fi.records["txt"].decision.strategy == ImputationStrategy.Passthrough
    assert fi.records["txt"].decision.semantic_type == SemanticType.Text


def test_identifier_columns_in_records_with_passthrough():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "id_col": pl.Series(["A001", "A002"], dtype=pl.Utf8),
    })
    num_cp = _numeric_cp_with_nulls("num")
    id_cp = ColumnProfile(name="id_col", semantic_type=SemanticType.Identifier)
    profile = _make_profile({"num": num_cp, "id_col": id_cp})

    fi = fit_imputer(df, profile)
    assert "id_col" in fi.records
    assert fi.records["id_col"].decision.strategy == ImputationStrategy.Passthrough
    assert fi.records["id_col"].decision.semantic_type == SemanticType.Identifier


# ---------------------------------------------------------------------------
# fit_transform() convenience
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fit() — schema manifest: Passthrough records for all unhandled columns
# ---------------------------------------------------------------------------


def _categorical_cp(col: str) -> ColumnProfile:
    return ColumnProfile(name=col, semantic_type=SemanticType.Categorical)


def test_text_column_passthrough_record_fill_value_is_none():
    df = pl.DataFrame({"txt": pl.Series(["a", "b"], dtype=pl.Utf8)})
    profile = _make_profile({"txt": ColumnProfile(name="txt", semantic_type=SemanticType.Text)})
    fi = fit_imputer(df, profile)
    assert fi.records["txt"].fill_value is None


def test_text_column_passthrough_record_indicator_added_is_false():
    df = pl.DataFrame({"txt": pl.Series(["a", "b"], dtype=pl.Utf8)})
    profile = _make_profile({"txt": ColumnProfile(name="txt", semantic_type=SemanticType.Text)})
    fi = fit_imputer(df, profile)
    assert fi.records["txt"].indicator_added is False


def test_identifier_column_passthrough_record_semantic_type():
    df = pl.DataFrame({"id": pl.Series(["X1", "X2"], dtype=pl.Utf8)})
    profile = _make_profile({"id": ColumnProfile(name="id", semantic_type=SemanticType.Identifier)})
    fi = fit_imputer(df, profile)
    assert fi.records["id"].decision.semantic_type == SemanticType.Identifier


def test_categorical_column_with_no_fill_strategy_gets_passthrough():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "cat": pl.Series(["a", "b"], dtype=pl.Utf8),
    })
    profile = _make_profile({
        "num": _numeric_cp_with_nulls("num"),
        "cat": _categorical_cp("cat"),
    })
    fi = fit_imputer(df, profile)
    assert "cat" in fi.records
    assert fi.records["cat"].decision.strategy == ImputationStrategy.Passthrough
    assert fi.records["cat"].decision.semantic_type == SemanticType.Categorical


def test_numeric_column_already_in_records_is_not_overwritten_by_passthrough():
    """Sub-processor result must win over the Passthrough pass for numeric columns."""
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    fi = fit_imputer(df, profile)
    assert fi.records["a"].decision.strategy != ImputationStrategy.Passthrough


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
    fi = fit_imputer(df, profile)
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
    fi = fit_imputer(df, profile, cfg)

    assert "income_missing" in fi.records


def test_indicator_record_has_strategy_indicator():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = fit_imputer(df, profile, cfg)

    assert fi.records["income_missing"].decision.strategy == ImputationStrategy.Indicator


def test_indicator_record_has_boolean_semantic_type():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = fit_imputer(df, profile, cfg)

    assert fi.records["income_missing"].decision.semantic_type == SemanticType.Boolean


def test_indicator_record_indicator_added_is_false():
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = fit_imputer(df, profile, cfg)

    assert fi.records["income_missing"].indicator_added is False


def test_indicator_record_not_present_when_no_indicator_added_columns():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})
    fi = fit_imputer(df, profile)
    indicator_cols = [c for c in fi.records if c.endswith("_missing")]
    assert indicator_cols == []


def test_indicator_record_round_trips_via_to_dict_from_dict(round_trip):
    from dataforge_ml.config import PipelineConfig

    df = pl.DataFrame({"income": pl.Series([1.0, None, 3.0] * 4, dtype=pl.Float64)})
    profile = _make_profile({"income": _numeric_cp_mnar("income")})

    cfg = PipelineConfig()
    cfg.imputation.add_mnar_column("income")
    fi = fit_imputer(df, profile, cfg)

    from dataforge_ml.imputation._fitted_imputer import FittedImputer
    restored = round_trip(fi)
    assert "income_missing" in restored.records
    assert restored.records["income_missing"].decision.strategy == ImputationStrategy.Indicator
    assert restored.records["income_missing"].decision.semantic_type == SemanticType.Boolean


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

    fi = fit_imputer(df, profile)

    assert fi.numeric_sentinels == {"age": [-999.0]}


def test_fit_numeric_sentinels_empty_when_profile_has_none():
    """FittedImputer has an empty sentinels dict when the profile declares none."""
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    fi = fit_imputer(df, profile)

    assert fi.numeric_sentinels == {}


# ---------------------------------------------------------------------------
# fit() — per_column_strategy size-guard validation (Seam 3)
# ---------------------------------------------------------------------------


# The ``per_column_strategy`` model-based size guards
# (``_validate_model_based_size_guards``) were validated inside the fused
# ``ImputationOrchestrator.fit()`` and were tested here.  They are gone with that
# seam (#368), and the layered path does not reproduce them:
#
# - forced **MICE** below ``mice_min_rows`` (the floor formerly named
#   ``regression_min_rows``, before the ADR-0079 collapse folded Regression into
#   MICE) is still caught, but as an execute-time ``UnitNotTrainableError`` rather
#   than a decide-time ``ValueError`` naming the dial to change (ADR-0029/0066);
# - forced **KNN** above ``knn_max_rows`` / ``knn_max_features`` is no longer caught
#   at all — it plans and trains silently.
#
# ``decide()`` deliberately does not carry these guards: its own unit tests pin
# that a forced strategy the shape cannot support is still *planned*, and that
# execution is what refuses it.  Re-homing the KNN guard is therefore a design
# decision (which layer owns a "your config contradicts your data" refusal), left
# to a follow-up rather than settled inside a removal ticket.


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
    
    # ``decide()`` validates the config before it routes anything, so a
    # contradictory config is rejected at plan time rather than at fit time.
    with pytest.raises(ValueError, match="mutually exclusive: 'a'"):
        fit_imputer(df, profile, cfg)
