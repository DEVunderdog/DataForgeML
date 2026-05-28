"""
Unit tests for ImputationOrchestrator and SplitImbalanceWarning.
"""

import warnings

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, SemanticType
from dataforge_ml.imputation._fitted_imputer import FittedImputer
from dataforge_ml.imputation.orchestrator import (
    ImputationOrchestrator,
    SplitImbalanceWarning,
)
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


def test_text_columns_absent_from_records():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "txt": pl.Series(["hello", "world"], dtype=pl.Utf8),
    })
    num_cp = _numeric_cp_with_nulls("num")
    txt_cp = ColumnProfile(name="txt", semantic_type=SemanticType.Text)
    profile = _make_profile({"num": num_cp, "txt": txt_cp})

    fi = ImputationOrchestrator().fit(df, profile)
    assert "txt" not in fi.records


def test_identifier_columns_absent_from_records():
    df = pl.DataFrame({
        "num": pl.Series([1.0, None], dtype=pl.Float64),
        "id_col": pl.Series(["A001", "A002"], dtype=pl.Utf8),
    })
    num_cp = _numeric_cp_with_nulls("num")
    id_cp = ColumnProfile(name="id_col", semantic_type=SemanticType.Identifier)
    profile = _make_profile({"num": num_cp, "id_col": id_cp})

    fi = ImputationOrchestrator().fit(df, profile)
    assert "id_col" not in fi.records


# ---------------------------------------------------------------------------
# SplitImbalanceWarning
# ---------------------------------------------------------------------------


def test_split_imbalance_warning_emitted_when_train_has_no_nulls():
    """Profile reports missingness, train_df has none → warning."""
    # Profile says 'a' has 5% missing, but we give a clean train_df
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=5, total=100)})
    clean_train = pl.DataFrame({"a": pl.Series([float(i) for i in range(10)], dtype=pl.Float64)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ImputationOrchestrator().fit(clean_train, profile)

    split_warnings = [w for w in caught if issubclass(w.category, SplitImbalanceWarning)]
    assert len(split_warnings) == 1
    assert "a" in str(split_warnings[0].message)


def test_split_imbalance_warning_names_all_imbalanced_columns_in_single_warning():
    """All imbalanced columns must appear together in a single SplitImbalanceWarning."""
    profile = _make_profile({
        "a": _numeric_cp_with_nulls("a", null_count=5, total=100),
        "b": _numeric_cp_with_nulls("b", null_count=3, total=100),
    })
    clean_train = pl.DataFrame({
        "a": pl.Series([float(i) for i in range(10)], dtype=pl.Float64),
        "b": pl.Series([float(i) for i in range(10)], dtype=pl.Float64),
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ImputationOrchestrator().fit(clean_train, profile)

    split_warnings = [w for w in caught if issubclass(w.category, SplitImbalanceWarning)]
    assert len(split_warnings) == 1
    msg = str(split_warnings[0].message)
    assert "a" in msg
    assert "b" in msg


def test_no_split_imbalance_warning_when_train_has_nulls():
    """If train_df has nulls matching profile, no warning should fire."""
    profile = _make_profile({"a": _numeric_cp_with_nulls("a", null_count=5, total=100)})
    train = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ImputationOrchestrator().fit(train, profile)

    split_warnings = [w for w in caught if issubclass(w.category, SplitImbalanceWarning)]
    assert len(split_warnings) == 0


def test_no_split_imbalance_warning_when_profile_reports_no_missingness():
    profile = _make_profile({"a": _clean_numeric_cp("a")})
    train = pl.DataFrame({"a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ImputationOrchestrator().fit(train, profile)

    split_warnings = [w for w in caught if issubclass(w.category, SplitImbalanceWarning)]
    assert len(split_warnings) == 0


# ---------------------------------------------------------------------------
# fit_transform() convenience
# ---------------------------------------------------------------------------


def test_fit_transform_returns_imputation_result_with_no_nulls():
    from dataforge_ml.imputation._config import ImputationResult

    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0, None], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    result = ImputationOrchestrator().fit_transform(df, profile)
    assert isinstance(result, ImputationResult)
    assert result.dataframe["a"].null_count() == 0


def test_fit_transform_is_equivalent_to_fit_then_transform():
    df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = _make_profile({"a": _numeric_cp_with_nulls("a")})

    orch = ImputationOrchestrator()
    r1 = orch.fit_transform(df, profile)
    r2 = orch.fit(df, profile).transform(df)
    assert r1.dataframe.equals(r2.dataframe)
