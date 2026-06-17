"""
Unit tests for #79: NumericImputer model-based strategies and FittedImputer model serialisation.

Tests cover:
- Strategy routing for KNN / Regression / MICE (via NumericImputer.fit)
- Model fitting and null-filling correctness
- FittedImputer.to_dict() / from_dict() round-trips with sklearn models
"""

import numpy as np
import polars as pl
import pytest

from dataforge_ml.config import SemanticType
from dataforge_ml.imputation._config import ImputationStrategy, NumericImputationConfig
from dataforge_ml.imputation._fitted_imputer import FittedImputer
from dataforge_ml.imputation._numeric_imputer import NumericImputer
from dataforge_ml.profiling._config import (
    ColumnProfile,
    NumericKind,
    StructuralProfileResult,
)
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import NumericStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(*items: tuple[str, ColumnProfile]) -> StructuralProfileResult:
    result = StructuralProfileResult()
    for col, cp in items:
        result.columns[col] = cp
    return result


def _numeric_cp(
    col: str,
    *,
    null_count: int = 10,
    total_rows: int = 100,
    severity: MissingSeverity = MissingSeverity.High,
    flags: list[MissingnessFlag] | None = None,
    correlated_with: list[str] | None = None,
    numeric_kind: NumericKind = NumericKind.Continuous,
) -> ColumnProfile:
    missingness = ColumnMissingnessProfile(
        column=col,
        total_rows=total_rows,
        effective_null_count=null_count,
        effective_null_ratio=null_count / total_rows,
        severity=severity,
        flags=flags or [],
        correlated_with=correlated_with or [],
    )
    return ColumnProfile(
        name=col,
        semantic_type=SemanticType.Numeric,
        numeric_kind=numeric_kind,
        missingness=missingness,
        stats=NumericStats(),
    )


def _fit(
    df: pl.DataFrame,
    columns: list[str],
    profile: StructuralProfileResult,
    config: NumericImputationConfig | None = None,
    mnar: set[str] | None = None,
):
    return NumericImputer().fit(
        train_df=df,
        columns=columns,
        profile=profile,
        config=config or NumericImputationConfig(),
        mnar_columns=mnar or set(),
    )


def _large_df(n: int = 1000, n_cols: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"c{i}": rng.normal(0, 1, n).tolist() for i in range(n_cols)}
    # Introduce ~10% nulls in first column
    values = data["c0"]
    for idx in range(0, n, 10):
        values[idx] = None
    data["c0"] = values
    return pl.DataFrame({k: pl.Series(v, dtype=pl.Float64) for k, v in data.items()})


# ---------------------------------------------------------------------------
# Strategy routing — MAR path
# ---------------------------------------------------------------------------


def test_mar_severe_gets_mice():
    rng = np.random.default_rng(1)
    n = 200
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 5):
        vals[i] = None  # ~20% missing → Severe
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=40, total_rows=n,
                        severity=MissingSeverity.Severe,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    cp_b = _numeric_cp("b", null_count=0, total_rows=n,
                        severity=MissingSeverity.Minor, flags=[])
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.MICE
    assert "mice" in bundle.models
    assert "a" in bundle.model_cols["mice"]


def test_mar_high_with_corrs_and_large_dataset_gets_regression():
    rng = np.random.default_rng(2)
    n = 2000
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 10):
        vals[i] = None
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=200, total_rows=n,
                        severity=MissingSeverity.High,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    cp_b = _numeric_cp("b", null_count=0, total_rows=n, flags=[])
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    # regression_min_rows=500; n=2000 → Regression guard passes
    config = NumericImputationConfig(
        knn_max_rows=500,  # knn guard fails (n > 500)
        regression_min_rows=500,
    )
    bundle = _fit(df, ["a", "b"], profile, config=config)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.Regression
    assert "regression:a" in bundle.models


def test_mar_high_with_corrs_and_small_dataset_gets_knn():
    rng = np.random.default_rng(3)
    n = 100
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 10):
        vals[i] = None
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=10, total_rows=n,
                        severity=MissingSeverity.High,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    cp_b = _numeric_cp("b", null_count=0, total_rows=n, flags=[])
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    # regression_min_rows=500 > 100 rows → Regression guard fails
    # knn_max_rows=50_000 >= 100 → KNN guard passes
    config = NumericImputationConfig(regression_min_rows=500)
    bundle = _fit(df, ["a", "b"], profile, config=config)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.KNN
    assert "knn" in bundle.models
    assert "a" in bundle.model_cols["knn"]


def test_mar_high_falls_back_to_median_when_all_guards_fail():
    rng = np.random.default_rng(4)
    n = 100
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 10):
        vals[i] = None
    df = pl.DataFrame({"a": pl.Series(vals, dtype=pl.Float64)})
    cp_a = _numeric_cp("a", null_count=10, total_rows=n,
                        severity=MissingSeverity.High,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    profile = _make_profile(("a", cp_a))
    config = NumericImputationConfig(
        knn_max_rows=10,          # fail
        knn_max_features=0,       # fail
        regression_min_rows=10_000,  # fail
    )
    bundle = _fit(df, ["a"], profile, config=config)
    rec_a = bundle.records[0]
    assert rec_a.strategy == ImputationStrategy.Median
    assert rec_a.fill_value is not None


# ---------------------------------------------------------------------------
# Strategy routing — multi-MAR
# ---------------------------------------------------------------------------


def test_multi_mar_all_assigned_mice():
    rng = np.random.default_rng(5)
    n = 300
    df = pl.DataFrame({
        "a": pl.Series([None if i % 10 == 0 else rng.normal() for i in range(n)], dtype=pl.Float64),
        "b": pl.Series([None if i % 10 == 1 else rng.normal() for i in range(n)], dtype=pl.Float64),
        "c": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=30, total_rows=n, severity=MissingSeverity.Moderate,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    cp_b = _numeric_cp("b", null_count=30, total_rows=n, severity=MissingSeverity.Moderate,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["a"])
    cp_c = _numeric_cp("c", null_count=0, total_rows=n)
    cp_c.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b), ("c", cp_c))
    bundle = _fit(df, ["a", "b", "c"], profile)
    rec_a = next(r for r in bundle.records if r.column == "a")
    rec_b = next(r for r in bundle.records if r.column == "b")
    assert rec_a.strategy == ImputationStrategy.MICE
    assert rec_b.strategy == ImputationStrategy.MICE
    assert "mice" in bundle.models
    assert "a" in bundle.model_cols["mice"]
    assert "b" in bundle.model_cols["mice"]


def test_multi_mar_signal_mentions_multi_mar():
    rng = np.random.default_rng(6)
    n = 200
    df = pl.DataFrame({
        "a": pl.Series([None if i % 10 == 0 else rng.normal() for i in range(n)], dtype=pl.Float64),
        "b": pl.Series([None if i % 10 == 1 else rng.normal() for i in range(n)], dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=20, total_rows=n, severity=MissingSeverity.Moderate,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["b"])
    cp_b = _numeric_cp("b", null_count=20, total_rows=n, severity=MissingSeverity.Moderate,
                        flags=[MissingnessFlag.MARSuspect], correlated_with=["a"])
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert any("multi-MAR" in s or "multi_mar" in s.lower() or "≥2" in s for s in rec_a.signals)


# ---------------------------------------------------------------------------
# Strategy routing — MCAR path
# ---------------------------------------------------------------------------


def test_mcar_high_gets_knn():
    rng = np.random.default_rng(7)
    n = 200
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 7):
        vals[i] = None
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=28, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.KNN
    assert "knn" in bundle.models


def test_mcar_high_gets_regression_when_knn_guard_fails():
    rng = np.random.default_rng(8)
    n = 1000
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 7):
        vals[i] = None
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=140, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    config = NumericImputationConfig(
        knn_max_rows=100,          # knn guard: fail (1000 > 100)
        regression_min_rows=500,   # regression guard: pass (1000 >= 500)
    )
    bundle = _fit(df, ["a", "b"], profile, config=config)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.Regression
    assert "regression:a" in bundle.models


def test_mcar_severe_gets_mice():
    rng = np.random.default_rng(9)
    n = 200
    vals = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 4):
        vals[i] = None  # 25% missing → Severe
    df = pl.DataFrame({
        "a": pl.Series(vals, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=50, total_rows=n, severity=MissingSeverity.Severe)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.MICE
    assert "mice" in bundle.models


# ---------------------------------------------------------------------------
# Signals record size-guard fallbacks
# ---------------------------------------------------------------------------


def test_signals_record_size_guard_on_knn_to_median_fallback():
    rng = np.random.default_rng(10)
    n = 100
    df = pl.DataFrame({"a": pl.Series(
        [None if i % 7 == 0 else float(i) for i in range(n)], dtype=pl.Float64
    )})
    cp_a = _numeric_cp("a", null_count=14, total_rows=n, severity=MissingSeverity.High)
    profile = _make_profile(("a", cp_a))
    config = NumericImputationConfig(
        knn_max_rows=10, knn_max_features=0, regression_min_rows=100_000,
    )
    bundle = _fit(df, ["a"], profile, config=config)
    rec = bundle.records[0]
    assert rec.strategy == ImputationStrategy.Median
    signal_text = " ".join(rec.signals)
    assert "guard" in signal_text.lower() or "failed" in signal_text.lower()


def test_signals_record_regression_fallback_to_median_when_no_features():
    # Regression guard passes but no feature columns exist → fallback_to_median signal
    rng = np.random.default_rng(11)
    n = 1000
    vals = [None if i % 7 == 0 else float(rng.normal()) for i in range(n)]
    df = pl.DataFrame({"a": pl.Series(vals, dtype=pl.Float64)})
    cp_a = _numeric_cp("a", null_count=140, total_rows=n, severity=MissingSeverity.High)
    profile = _make_profile(("a", cp_a))
    config = NumericImputationConfig(
        knn_max_rows=100,          # knn guard: fail
        regression_min_rows=500,   # regression guard: pass → Regression selected
    )
    bundle = _fit(df, ["a"], profile, config=config)
    rec = bundle.records[0]
    # Regression was selected by routing but fit failed (no feature cols) → Median
    assert rec.strategy == ImputationStrategy.Median
    assert any("fallback" in s.lower() or "no feature" in s.lower() for s in rec.signals)


# ---------------------------------------------------------------------------
# Model transform correctness
# ---------------------------------------------------------------------------


def test_knn_transform_fills_nulls():
    rng = np.random.default_rng(12)
    n = 300
    vals_a = rng.normal(0, 1, n).tolist()
    for i in range(0, n, 10):
        vals_a[i] = None
    df = pl.DataFrame({
        "a": pl.Series(vals_a, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=30, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    result = fi.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["a"].is_nan().sum() == 0


def test_mice_transform_fills_nulls():
    rng = np.random.default_rng(13)
    n = 300
    vals_a = [None if i % 5 == 0 else float(rng.normal()) for i in range(n)]
    vals_b = [None if i % 7 == 0 else float(rng.normal()) for i in range(n)]
    df = pl.DataFrame({
        "a": pl.Series(vals_a, dtype=pl.Float64),
        "b": pl.Series(vals_b, dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=60, total_rows=n, severity=MissingSeverity.Severe)
    cp_b = _numeric_cp("b", null_count=42, total_rows=n, severity=MissingSeverity.Severe)
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    assert any(r.strategy == ImputationStrategy.MICE for r in bundle.records)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    result = fi.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["b"].null_count() == 0


def test_regression_transform_fills_nulls():
    rng = np.random.default_rng(14)
    n = 1000
    b_vals = rng.normal(0, 1, n)
    a_vals = (2.0 * b_vals + rng.normal(0, 0.1, n)).tolist()
    for i in range(0, n, 10):
        a_vals[i] = None
    df = pl.DataFrame({
        "a": pl.Series(a_vals, dtype=pl.Float64),
        "b": pl.Series(b_vals.tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=100, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    config = NumericImputationConfig(
        knn_max_rows=100,         # knn guard: fail
        regression_min_rows=500,  # regression guard: pass
    )
    bundle = _fit(df, ["a", "b"], profile, config=config)
    rec_a = next(r for r in bundle.records if r.column == "a")
    assert rec_a.strategy == ImputationStrategy.Regression
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    result = fi.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["a"].is_nan().sum() == 0


def test_model_transform_applies_train_time_parameters():
    """Transform on test data must use train-time model, not refit."""
    rng = np.random.default_rng(15)
    n = 500
    b_train = rng.normal(0, 1, n)
    a_train = (3.0 * b_train + rng.normal(0, 0.1, n)).tolist()
    for i in range(0, n, 10):
        a_train[i] = None
    train_df = pl.DataFrame({
        "a": pl.Series(a_train, dtype=pl.Float64),
        "b": pl.Series(b_train.tolist(), dtype=pl.Float64),
    })
    # Entirely different test data
    b_test = rng.normal(10, 1, 100)
    a_test = [None] * 100
    test_df = pl.DataFrame({
        "a": pl.Series(a_test, dtype=pl.Float64),
        "b": pl.Series(b_test.tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=50, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=200)
    bundle = _fit(train_df, ["a", "b"], profile, config=config)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    # Call transform twice — fill values should not change
    r1 = fi.transform(test_df)
    r2 = fi.transform(test_df)
    assert r1.dataframe.equals(r2.dataframe)


# ---------------------------------------------------------------------------
# FittedImputer model serialisation
# ---------------------------------------------------------------------------


def _make_fitted_imputer_with_knn() -> tuple[FittedImputer, pl.DataFrame]:
    rng = np.random.default_rng(20)
    n = 300
    vals_a = [None if i % 10 == 0 else float(rng.normal()) for i in range(n)]
    df = pl.DataFrame({
        "a": pl.Series(vals_a, dtype=pl.Float64),
        "b": pl.Series(rng.normal(0, 1, n).tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=30, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    return fi, df


def _make_fitted_imputer_with_regression() -> tuple[FittedImputer, pl.DataFrame]:
    rng = np.random.default_rng(21)
    n = 1000
    b = rng.normal(0, 1, n)
    a = (2.0 * b + rng.normal(0, 0.1, n)).tolist()
    for i in range(0, n, 10):
        a[i] = None
    df = pl.DataFrame({
        "a": pl.Series(a, dtype=pl.Float64),
        "b": pl.Series(b.tolist(), dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=100, total_rows=n, severity=MissingSeverity.High)
    cp_b = _numeric_cp("b", null_count=0, total_rows=n)
    cp_b.missingness = None
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = _fit(df, ["a", "b"], profile, config=config)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    return fi, df


def _make_fitted_imputer_with_mice() -> tuple[FittedImputer, pl.DataFrame]:
    rng = np.random.default_rng(22)
    n = 300
    vals_a = [None if i % 4 == 0 else float(rng.normal()) for i in range(n)]
    vals_b = [None if i % 5 == 0 else float(rng.normal()) for i in range(n)]
    df = pl.DataFrame({
        "a": pl.Series(vals_a, dtype=pl.Float64),
        "b": pl.Series(vals_b, dtype=pl.Float64),
    })
    cp_a = _numeric_cp("a", null_count=75, total_rows=n, severity=MissingSeverity.Severe)
    cp_b = _numeric_cp("b", null_count=60, total_rows=n, severity=MissingSeverity.Severe)
    profile = _make_profile(("a", cp_a), ("b", cp_b))
    bundle = _fit(df, ["a", "b"], profile)
    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    return fi, df


def test_knn_serialisation_round_trip_identical_output():
    fi, df = _make_fitted_imputer_with_knn()
    restored = FittedImputer.from_dict(fi.to_dict())
    r1 = fi.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_regression_serialisation_round_trip_identical_output():
    fi, df = _make_fitted_imputer_with_regression()
    restored = FittedImputer.from_dict(fi.to_dict())
    r1 = fi.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_mice_serialisation_round_trip_identical_output():
    fi, df = _make_fitted_imputer_with_mice()
    restored = FittedImputer.from_dict(fi.to_dict())
    r1 = fi.transform(df)
    r2 = restored.transform(df)
    assert r1.dataframe.equals(r2.dataframe)


def test_to_dict_stores_models_as_base64_strings():
    fi, _ = _make_fitted_imputer_with_knn()
    d = fi.to_dict()
    assert "models" in d
    assert "knn" in d["models"]
    b64str = d["models"]["knn"]
    assert isinstance(b64str, str)
    # Valid base64 — should decode without error
    import base64
    decoded = base64.b64decode(b64str)
    assert len(decoded) > 0


def test_to_dict_stores_model_cols():
    fi, _ = _make_fitted_imputer_with_knn()
    d = fi.to_dict()
    assert "model_cols" in d
    assert "knn" in d["model_cols"]
    assert isinstance(d["model_cols"]["knn"], list)


def test_from_dict_restores_live_sklearn_objects():
    fi, df = _make_fitted_imputer_with_knn()
    d = fi.to_dict()
    restored = FittedImputer.from_dict(d)
    assert "knn" in restored.models
    # Should be a callable sklearn object
    assert hasattr(restored.models["knn"], "transform")


def test_deserialized_imputer_fills_no_nulls():
    fi, df = _make_fitted_imputer_with_knn()
    restored = FittedImputer.from_dict(fi.to_dict())
    result = restored.transform(df)
    assert result.dataframe["a"].null_count() == 0


def test_regression_model_stored_as_fitted_regression():
    """Regression entry is now a FittedRegression with an IterativeImputer."""
    fi, _ = _make_fitted_imputer_with_regression()
    from dataforge_ml.imputation._numeric_imputer import FittedRegression
    from sklearn.impute import IterativeImputer
    assert "regression:a" in fi.models
    fitted_reg = fi.models["regression:a"]
    assert isinstance(fitted_reg, FittedRegression)
    assert isinstance(fitted_reg.model, IterativeImputer)
    assert fitted_reg.target_idx == 0
    assert fitted_reg.all_cols[0] == "a"
    assert "b" in fitted_reg.all_cols


def test_regression_model_cols_stores_full_all_cols():
    """model_cols['regression:col'] stores [col] + feat_cols, not just feat_cols."""
    fi, _ = _make_fitted_imputer_with_regression()
    all_cols = fi.model_cols["regression:a"]
    assert all_cols[0] == "a"
    assert "b" in all_cols


def test_regression_new_format_round_trip_fills_nulls():
    """Serialising and deserialising a new-format regression entry produces no nulls."""
    fi, df = _make_fitted_imputer_with_regression()
    restored = FittedImputer.from_dict(fi.to_dict())
    result = restored.transform(df)
    assert result.dataframe["a"].null_count() == 0
    assert result.dataframe["a"].is_nan().sum() == 0


def test_regression_legacy_migration_in_from_dict():
    """from_dict() migrates a legacy (BayesianRidge, feat_means) entry to FittedRegression."""
    import base64
    import io
    import joblib
    import numpy as np
    from sklearn.linear_model import BayesianRidge

    from dataforge_ml.config import SemanticType
    from dataforge_ml.imputation._config import (
        ColumnImputationRecord,
        ImputationStrategy,
    )
    from dataforge_ml.imputation._numeric_imputer import FittedRegression

    # Build a minimal legacy dict by hand: models["regression:a"] = (reg, feat_means)
    rng = np.random.default_rng(99)
    X = rng.standard_normal((50, 1))
    y = 2.0 * X[:, 0] + rng.standard_normal(50) * 0.1
    reg = BayesianRidge()
    reg.fit(X, y)
    feat_means = np.array([X.mean()])

    buf = io.BytesIO()
    joblib.dump((reg, feat_means), buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    legacy_dict = {
        "records": {
            "a": {
                "column": "a", "semantic_type": "numeric",
                "strategy": "regression", "fill_value": None,
                "indicator_added": False, "signals": [],
            },
            "b": {
                "column": "b", "semantic_type": "numeric",
                "strategy": "passthrough", "fill_value": None,
                "indicator_added": False, "signals": [],
            },
        },
        "models": {"regression:a": b64},
        # Legacy: model_cols stores only feat_cols, not [col] + feat_cols
        "model_cols": {"regression:a": ["b"]},
    }

    fi = FittedImputer.from_dict(legacy_dict)

    # After migration, models["regression:a"] should be a FittedRegression
    assert isinstance(fi.models["regression:a"], FittedRegression)
    # model_cols should now be ["a", "b"] (target prepended)
    assert fi.model_cols["regression:a"][0] == "a"
    assert "b" in fi.model_cols["regression:a"]
    # The wrapped legacy model carries the (reg, feat_means) tuple
    assert isinstance(fi.models["regression:a"].model, tuple)
