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
    from dataforge_ml.imputation._fitted_imputer import _FittedKNN
    fi, df = _make_fitted_imputer_with_knn()
    d = fi.to_dict()
    restored = FittedImputer.from_dict(d)
    assert "knn" in restored.models
    # models["knn"] is now a _FittedKNN storing model, col_means, col_stds
    assert isinstance(restored.models["knn"], _FittedKNN)
    assert hasattr(restored.models["knn"].model, "transform")


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


# ---------------------------------------------------------------------------
# Domain-snap at transform time for BoundedDiscrete columns
# ---------------------------------------------------------------------------


def test_domain_snap_applied_after_knn_for_bounded_discrete():
    """KNN predictions for a BoundedDiscrete column are clipped and rounded to [min, max]."""
    rng = np.random.default_rng(42)
    n = 200
    # Rating scale 1–5
    raw = rng.integers(1, 6, size=n).tolist()
    null_mask = [i % 10 == 0 for i in range(n)]
    col_vals = [None if null_mask[i] else float(raw[i]) for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()

    df = pl.DataFrame({
        "rating": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })

    profile = _make_profile(
        ("rating", ColumnProfile(
            name="rating",
            semantic_type=SemanticType.Numeric,
            numeric_kind=NumericKind.BoundedDiscrete,
            missingness=ColumnMissingnessProfile(
                column="rating",
                total_rows=n,
                effective_null_count=n // 10,
                effective_null_ratio=0.1,
                severity=MissingSeverity.High,
                flags=[],
                correlated_with=[],
            ),
            stats=NumericStats(min=1.0, max=5.0),
        )),
        ("feat", _numeric_cp("feat", null_count=0, severity=MissingSeverity.High,
                              numeric_kind=NumericKind.Continuous)),
    )

    bundle = _fit(df, ["rating", "feat"], profile)
    rec = next(r for r in bundle.records if r.column == "rating")
    assert rec.strategy == ImputationStrategy.KNN
    assert rec.domain_snap_bounds == (1.0, 5.0)

    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    result = fi.transform(df)
    rating_col = result.dataframe["rating"]

    assert rating_col.null_count() == 0
    vals = rating_col.drop_nulls().to_list()
    assert all(1.0 <= v <= 5.0 for v in vals), f"Values out of [1,5]: {vals}"
    assert all(v == round(v) for v in vals), f"Non-integer values after snap: {vals}"


def test_domain_snap_applied_after_regression_for_bounded_discrete():
    """Regression predictions for a BoundedDiscrete column are clipped and rounded to [min, max]."""
    rng = np.random.default_rng(7)
    n = 600
    raw = rng.integers(1, 6, size=n).tolist()
    null_mask = [i % 10 == 0 for i in range(n)]
    col_vals = [None if null_mask[i] else float(raw[i]) for i in range(n)]
    feat_vals = rng.standard_normal(n).tolist()

    df = pl.DataFrame({
        "rating": pl.Series(col_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })

    profile = _make_profile(
        ("rating", ColumnProfile(
            name="rating",
            semantic_type=SemanticType.Numeric,
            numeric_kind=NumericKind.BoundedDiscrete,
            missingness=ColumnMissingnessProfile(
                column="rating",
                total_rows=n,
                effective_null_count=n // 10,
                effective_null_ratio=0.1,
                severity=MissingSeverity.High,
                flags=[],
                correlated_with=[],
            ),
            stats=NumericStats(min=1.0, max=5.0),
        )),
        ("feat", _numeric_cp("feat", null_count=0, severity=MissingSeverity.High,
                              numeric_kind=NumericKind.Continuous)),
    )

    config = NumericImputationConfig(knn_max_rows=100, regression_min_rows=500)
    bundle = _fit(df, ["rating", "feat"], profile, config=config)
    rec = next(r for r in bundle.records if r.column == "rating")
    assert rec.strategy == ImputationStrategy.Regression
    assert rec.domain_snap_bounds == (1.0, 5.0)

    fi = FittedImputer(
        records={r.column: r for r in bundle.records},
        models=bundle.models,
        model_cols=bundle.model_cols,
    )
    result = fi.transform(df)
    rating_col = result.dataframe["rating"]

    assert rating_col.null_count() == 0
    vals = rating_col.drop_nulls().to_list()
    assert all(1.0 <= v <= 5.0 for v in vals), f"Values out of [1,5]: {vals}"
    assert all(v == round(v) for v in vals), f"Non-integer values after snap: {vals}"


# ---------------------------------------------------------------------------
# Issue #152 — Adaptive KNN hyperparameter selection
# ---------------------------------------------------------------------------


def _parse_knn_n_neighbors(signals: list[str]) -> int:
    """Extract the reported n_neighbors value from a knn_params signal entry."""
    for s in signals:
        if s.startswith("knn_params:"):
            # Format: "knn_params: n_neighbors=K, weights=W | ..."
            part = s.split("n_neighbors=")[1]
            return int(part.split(",")[0].strip())
    raise AssertionError(f"No knn_params signal found in: {signals}")


def _has_weights(signals: list[str], expected_weights: str) -> bool:
    """Return True if any knn_params signal contains weights=<expected_weights>."""
    for s in signals:
        if s.startswith("knn_params:") and f"weights={expected_weights}" in s:
            return True
    return False


def _make_knn_df_and_profile(n: int, n_cols: int, miss_frac: float, rng: np.random.Generator):
    """Build a DataFrame and profile with KNN strategy forced for all columns.

    All columns are MCAR High so they route to KNN (given small n and few features).
    The first `int(n * n_cols * miss_frac)` cells are set to None.
    """
    data = {}
    for i in range(n_cols):
        vals = rng.normal(0, 1, n).tolist()
        # Introduce nulls at the requested fraction
        n_nulls = int(n * miss_frac)
        for j in range(n_nulls):
            vals[j] = None
        data[f"c{i}"] = vals
    df = pl.DataFrame({k: pl.Series(v, dtype=pl.Float64) for k, v in data.items()})

    items = []
    for col in data:
        null_count = int(n * miss_frac)
        cp = _numeric_cp(col, null_count=null_count, total_rows=n, severity=MissingSeverity.High)
        if null_count == 0:
            cp.missingness = None
        items.append((col, cp))
    profile = _make_profile(*items)
    return df, profile, list(data.keys())


def test_knn_high_dim_produces_larger_n_neighbors_than_low_dim():
    """High-dimensional KNN column set → larger n_neighbors than low-dimensional, all else equal.

    Directional test only: we do not assert the exact k value, just the ordering.
    """
    rng = np.random.default_rng(200)
    miss = 0.05  # same missingness for both cases

    # Low-dim: 2 KNN features
    df_low, prof_low, cols_low = _make_knn_df_and_profile(n=500, n_cols=2, miss_frac=miss, rng=rng)
    bundle_low = _fit(df_low, cols_low, prof_low)
    rec_low = next(r for r in bundle_low.records if r.strategy == ImputationStrategy.KNN)
    k_low = _parse_knn_n_neighbors(rec_low.signals)

    # High-dim: 25 KNN features (sqrt(25)=5 > sqrt(2)≈1)
    rng2 = np.random.default_rng(201)
    df_high, prof_high, cols_high = _make_knn_df_and_profile(n=500, n_cols=25, miss_frac=miss, rng=rng2)
    bundle_high = _fit(df_high, cols_high, prof_high)
    rec_high = next(r for r in bundle_high.records if r.strategy == ImputationStrategy.KNN)
    k_high = _parse_knn_n_neighbors(rec_high.signals)

    assert k_high >= k_low, (
        f"Expected high-dim k ({k_high}) >= low-dim k ({k_low})"
    )


def test_knn_high_miss_frac_produces_larger_n_neighbors_than_low_miss_frac():
    """High feature-matrix missingness → larger n_neighbors than low missingness."""
    rng = np.random.default_rng(202)

    # Low missingness: 5%
    df_low, prof_low, cols_low = _make_knn_df_and_profile(n=500, n_cols=4, miss_frac=0.05, rng=rng)
    bundle_low = _fit(df_low, cols_low, prof_low)
    rec_low = next(r for r in bundle_low.records if r.strategy == ImputationStrategy.KNN)
    k_low = _parse_knn_n_neighbors(rec_low.signals)

    # High missingness: 40%
    rng2 = np.random.default_rng(203)
    df_high, prof_high, cols_high = _make_knn_df_and_profile(n=500, n_cols=4, miss_frac=0.40, rng=rng2)
    bundle_high = _fit(df_high, cols_high, prof_high)
    rec_high = next(r for r in bundle_high.records if r.strategy == ImputationStrategy.KNN)
    k_high = _parse_knn_n_neighbors(rec_high.signals)

    assert k_high >= k_low, (
        f"Expected high miss_frac k ({k_high}) >= low miss_frac k ({k_low})"
    )


def test_knn_low_miss_frac_few_features_gives_distance_weights():
    """Low miss_frac + few features → weights=distance in knn_params signal."""
    rng = np.random.default_rng(204)
    # miss_frac < 0.15 and n_features <= 30 → reliability_high=True → distance
    df, profile, cols = _make_knn_df_and_profile(n=500, n_cols=3, miss_frac=0.05, rng=rng)
    config = NumericImputationConfig(
        knn_distance_weight_max_null_ratio=0.15,
        knn_distance_weight_max_features=30,
    )
    bundle = _fit(df, cols, profile, config=config)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        assert _has_weights(rec.signals, "distance"), (
            f"Expected weights=distance in signals; got: {rec.signals}"
        )


def test_knn_high_miss_frac_gives_uniform_weights():
    """High miss_frac (above threshold) → weights=uniform in knn_params signal."""
    rng = np.random.default_rng(205)
    # miss_frac >= 0.15 → reliability_high=False → uniform
    df, profile, cols = _make_knn_df_and_profile(n=500, n_cols=3, miss_frac=0.40, rng=rng)
    config = NumericImputationConfig(
        knn_distance_weight_max_null_ratio=0.15,
        knn_distance_weight_max_features=30,
    )
    bundle = _fit(df, cols, profile, config=config)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        assert _has_weights(rec.signals, "uniform"), (
            f"Expected weights=uniform in signals; got: {rec.signals}"
        )


def test_knn_many_features_gives_uniform_weights():
    """Many features (above knn_distance_weight_max_features) → weights=uniform even with low miss_frac."""
    rng = np.random.default_rng(206)
    # n_features > knn_distance_weight_max_features=5 → reliability_high=False → uniform
    df, profile, cols = _make_knn_df_and_profile(n=500, n_cols=10, miss_frac=0.05, rng=rng)
    config = NumericImputationConfig(
        knn_distance_weight_max_null_ratio=0.15,
        knn_distance_weight_max_features=5,  # threshold below 10 features
    )
    bundle = _fit(df, cols, profile, config=config)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        assert _has_weights(rec.signals, "uniform"), (
            f"Expected weights=uniform in signals; got: {rec.signals}"
        )


def test_knn_signals_contain_knn_scaling_entry():
    """Every KNN-routed column must have a knn_scaling signal entry."""
    rng = np.random.default_rng(207)
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=3, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        assert any("knn_scaling: applied" in s for s in rec.signals), (
            f"Expected knn_scaling signal; got: {rec.signals}"
        )


def test_knn_model_stored_as_fitted_knn_instance():
    """models['knn'] must be a _FittedKNN with model, col_means, col_stds."""
    from dataforge_ml.imputation._fitted_imputer import _FittedKNN
    rng = np.random.default_rng(208)
    df, profile, cols = _make_knn_df_and_profile(n=300, n_cols=2, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    assert "knn" in bundle.models
    fitted = bundle.models["knn"]
    assert isinstance(fitted, _FittedKNN)
    assert hasattr(fitted.model, "transform")
    assert fitted.col_means.shape == (len(cols),)
    assert fitted.col_stds.shape == (len(cols),)


# ---------------------------------------------------------------------------
# Issue #154 — KNN audit log signal format
# ---------------------------------------------------------------------------


def _find_signal(signals: list[str], prefix: str) -> str | None:
    """Return the first signal string that starts with prefix, or None."""
    for s in signals:
        if s.startswith(prefix):
            return s
    return None


def test_knn_both_signals_present_on_every_column_multi_col():
    """Both knn_params and knn_scaling signals must be present on every KNN column.

    Uses a 4-column KNN fit to confirm signals are appended per-column, not
    globally (i.e., all four records must carry their own signal entries).
    """
    rng = np.random.default_rng(210)
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=4, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert len(knn_recs) >= 2, f"Expected ≥2 KNN columns; got {len(knn_recs)}"
    for rec in knn_recs:
        assert _find_signal(rec.signals, "knn_params:") is not None, (
            f"Column '{rec.column}' missing knn_params signal; got: {rec.signals}"
        )
        assert _find_signal(rec.signals, "knn_scaling:") is not None, (
            f"Column '{rec.column}' missing knn_scaling signal; got: {rec.signals}"
        )


def test_knn_params_signal_contains_required_keys():
    """knn_params signal must contain n_neighbors, weights, n_features, miss_frac, complete_frac."""
    rng = np.random.default_rng(211)
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=3, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        sig = _find_signal(rec.signals, "knn_params:")
        assert sig is not None, f"No knn_params signal on '{rec.column}'"
        for key in ("n_neighbors=", "weights=", "n_features=", "miss_frac=", "complete_frac="):
            assert key in sig, (
                f"Key '{key}' missing from knn_params signal on '{rec.column}': {sig}"
            )


def test_knn_params_signal_weights_value_is_valid():
    """weights value in knn_params signal must be either 'distance' or 'uniform'."""
    rng = np.random.default_rng(212)
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=3, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        sig = _find_signal(rec.signals, "knn_params:")
        assert sig is not None
        assert "weights=distance" in sig or "weights=uniform" in sig, (
            f"weights value is neither 'distance' nor 'uniform' in: {sig}"
        )


def test_knn_params_signal_n_neighbors_is_positive_integer():
    """n_neighbors in knn_params signal must parse as a positive integer."""
    rng = np.random.default_rng(213)
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=3, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    for rec in knn_recs:
        k = _parse_knn_n_neighbors(rec.signals)
        assert k >= 1, f"n_neighbors must be ≥ 1; got {k} for '{rec.column}'"


def test_knn_scaling_signal_exact_format():
    """knn_scaling signal must match the exact format defined in Issue #90."""
    import re
    rng = np.random.default_rng(214)
    n_cols = 3
    df, profile, cols = _make_knn_df_and_profile(n=400, n_cols=n_cols, miss_frac=0.10, rng=rng)
    bundle = _fit(df, cols, profile)
    knn_recs = [r for r in bundle.records if r.strategy == ImputationStrategy.KNN]
    assert knn_recs, "Expected at least one KNN column"
    pattern = re.compile(
        r"^knn_scaling: applied StandardScaler \(nanmean/nanstd\) across \d+ feature columns$"
    )
    for rec in knn_recs:
        sig = _find_signal(rec.signals, "knn_scaling:")
        assert sig is not None, f"No knn_scaling signal on '{rec.column}'"
        assert pattern.match(sig), (
            f"knn_scaling signal format mismatch on '{rec.column}': {sig!r}"
        )
