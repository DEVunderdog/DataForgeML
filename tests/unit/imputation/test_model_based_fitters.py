"""
Tests for the joint MICE and KNN fitters — the escalation point's execution
half (#528): KNN widening (ADR-0093), the Estimator Ladder (ADR-0094), and
domain-snapping a BoundedDiscrete column escalated to a model.

Includes synthetic regression guards and a real-frame regression test
reusing ``ames`` (asserting widened KNN beats block-only, the forest beats
BayesianRidge at small sizes, and the booster loses to the forest, per
ADR-0093, ADR-0097, #542, #543).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.impute import KNNImputer

from dataforge_ml import PipelineConfig, StructuralProfiler, derive_units, fit_unit
from dataforge_ml.imputation import ImputationStrategy, resolve_recipe, route
from dataforge_ml.imputation._config import ModelChoice
from dataforge_ml.utils._null_normalization import _resolve_effective_nulls


def _drive(df, cfg):
    """Route, resolve a recipe, and train every unit; return the fitted units."""
    profile = StructuralProfiler(config=cfg).profile(df)
    routing = route(profile, cfg)
    recipe = resolve_recipe(routing, profile, cfg)
    units = derive_units(routing)
    train = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    fitted = {
        unit.unit_id: fit_unit(recipe, unit, train, random_seed=cfg.random_seed).fitted
        for unit in units
    }
    return profile, routing, units, train, fitted


def _config(**per_column_strategy):
    cfg = PipelineConfig()
    cfg.random_seed = 7
    for col, strategy in per_column_strategy.items():
        cfg.imputation.numeric.set_per_column_strategy(col, strategy)
    return cfg


# ---------------------------------------------------------------------------
# KNN widening (ADR-0093): a one-column block no longer degenerates to the
# training mean when a correlated predictor is available.
# ---------------------------------------------------------------------------


def _correlated_frame(n=400, seed=0, missing_frac=0.3):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 10, n)
    t = p * 2.0 + rng.normal(0, 0.5, n)  # tight linear relationship
    t_observed = t.copy()
    missing_idx = rng.choice(n, int(n * missing_frac), replace=False)
    t_masked = t_observed.copy()
    t_masked[missing_idx] = np.nan
    df = pl.DataFrame({"p": p, "t": t_masked})
    return df, t_observed, missing_idx


def test_one_column_knn_block_does_not_collapse_to_the_training_mean():
    """The proven defect (ADR-0093): a block-only KNNImputer(n_neighbors=1)
    on a one-column block with a correlated predictor collapses toward the
    training mean. The library's widened fit must not.
    """
    df, t_true, missing_idx = _correlated_frame()
    cfg = _config(t=ImputationStrategy.KNN)
    _, routing, units, train, fitted = _drive(df, cfg)

    knn_unit = fitted["knn"]
    assert knn_unit.all_cols == ["t", "p"]  # widened past the block's own column
    assert knn_unit.columns == ["t"]

    out = knn_unit.transform(train)
    imputed = out["t"].to_numpy()[missing_idx]
    true_vals = t_true[missing_idx]

    # The block-only defect: KNNImputer trained on ONLY the target column.
    block_only_arr = train["t"].to_numpy().reshape(-1, 1)
    block_only_model = KNNImputer(n_neighbors=5)
    block_only_imputed = block_only_model.fit_transform(block_only_arr)[missing_idx, 0]

    widened_rmse = float(np.sqrt(np.mean((imputed - true_vals) ** 2)))
    block_only_rmse = float(np.sqrt(np.mean((block_only_imputed - true_vals) ** 2)))

    assert widened_rmse < block_only_rmse * 0.5, (
        f"widened RMSE {widened_rmse:.3f} should beat the block-only defect's "
        f"{block_only_rmse:.3f} by a wide margin"
    )
    # The block-only fill collapses toward a near-constant value; the widened
    # fit must vary meaningfully with the correlated predictor.
    assert np.std(imputed) > np.std(block_only_imputed) * 2


def test_knn_widened_beats_block_only_on_rmse_with_noise_columns():
    """Widening still wins with unrelated noise columns in the active set
    (ADR-0093's measured #542 direction, reproduced on synthetic data)."""
    rng = np.random.default_rng(11)
    n = 500
    p = rng.normal(0, 5, n)
    t = p * 1.5 + rng.normal(0, 0.3, n)
    noise_cols = {f"noise_{i}": rng.normal(0, 1, n) for i in range(5)}
    t_true = t.copy()
    missing_idx = rng.choice(n, int(n * 0.25), replace=False)
    t_masked = t.copy()
    t_masked[missing_idx] = np.nan

    data = {"p": p, "t": t_masked, **noise_cols}
    df = pl.DataFrame(data)
    cfg = _config(t=ImputationStrategy.KNN)
    _, _, _, train, fitted = _drive(df, cfg)

    knn_unit = fitted["knn"]
    out = knn_unit.transform(train)
    imputed = out["t"].to_numpy()[missing_idx]
    true_vals = t_true[missing_idx]
    widened_rmse = float(np.sqrt(np.mean((imputed - true_vals) ** 2)))

    block_only_arr = train["t"].to_numpy().reshape(-1, 1)
    block_only_imputed = KNNImputer(n_neighbors=5).fit_transform(block_only_arr)[
        missing_idx, 0
    ]
    block_only_rmse = float(np.sqrt(np.mean((block_only_imputed - true_vals) ** 2)))

    assert widened_rmse < block_only_rmse


def test_knn_block_writes_back_only_its_own_columns():
    """A widened predictor read for distance is never overwritten."""
    df, _, _ = _correlated_frame()
    cfg = _config(t=ImputationStrategy.KNN)
    _, _, _, train, fitted = _drive(df, cfg)
    knn_unit = fitted["knn"]
    out = knn_unit.transform(train)
    # p had no missing values; it must come back bit-for-bit.
    assert out["p"].to_list() == train["p"].to_list()


# ---------------------------------------------------------------------------
# MICE end-to-end: converges, widened, domain-snaps a BoundedDiscrete column
# ---------------------------------------------------------------------------


def test_mice_end_to_end_fits_and_converges():
    rng = np.random.default_rng(5)
    n = 300
    a = rng.normal(0, 1, n)
    b = a * 1.8 + rng.normal(0, 0.2, n)
    c = a * -1.2 + rng.normal(0, 0.2, n)
    b_masked = b.copy()
    missing_idx = rng.choice(n, int(n * 0.2), replace=False)
    b_masked[missing_idx] = np.nan
    c_masked = c.copy()
    missing_idx2 = rng.choice(n, int(n * 0.2), replace=False)
    c_masked[missing_idx2] = np.nan

    df = pl.DataFrame({"a": a, "b": b_masked, "c": c_masked})
    cfg = _config(b=ImputationStrategy.MICE, c=ImputationStrategy.MICE)
    _, routing, units, train, fitted = _drive(df, cfg)

    assert routing.mice_model_choice is not None
    mice_unit = fitted["mice"]
    assert set(mice_unit.columns) == {"b", "c"}
    assert "a" in mice_unit.all_cols

    out = mice_unit.transform(train)
    assert out["b"].null_count() == 0
    assert out["c"].null_count() == 0


def test_mice_domain_snaps_a_bounded_discrete_column():
    """A BoundedDiscrete column escalated to MICE snaps into its observed domain."""
    rng = np.random.default_rng(9)
    n = 300
    a = rng.normal(0, 1, n)
    # Integer-valued, bounded domain [1, 5] -> BoundedDiscrete
    rating = np.clip(np.round(a * 1.5 + 3), 1, 5)
    missing_idx = rng.choice(n, int(n * 0.2), replace=False)
    rating_masked = [
        None if i in set(missing_idx.tolist()) else float(v)
        for i, v in enumerate(rating)
    ]
    b = a * -1.0 + rng.normal(0, 0.2, n)

    df = pl.DataFrame(
        {
            "a": a,
            "rating": pl.Series(rating_masked, dtype=pl.Float64),
            "b": b,
        }
    )
    cfg = _config(rating=ImputationStrategy.MICE)
    _, routing, units, train, fitted = _drive(df, cfg)

    mice_unit = fitted["mice"]
    out = mice_unit.transform(train)
    filled = out["rating"].to_numpy()[missing_idx]
    assert np.all(filled >= 1.0) and np.all(filled <= 5.0)
    assert np.all(filled == np.round(filled))  # snapped to whole numbers


# ---------------------------------------------------------------------------
# Estimator Ladder (ADR-0094): forest beats ridge small, booster loses to
# forest (synthetic reproduction of the ADR-0097 measurement's direction).
# ---------------------------------------------------------------------------


def _nonlinear_frame(n, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(-3, 3, n)
    target = np.sin(a) * np.cos(b) * 5.0 + rng.normal(0, 0.1, n)
    target_masked = target.copy()
    missing_idx = rng.choice(n, int(n * 0.25), replace=False)
    target_masked[missing_idx] = np.nan
    df = pl.DataFrame({"a": a, "b": b, "target": target_masked})
    return df, target, missing_idx


def _mice_rmse_with_estimator(df, target_true, missing_idx, model_choice):
    """Fit MICE forced to ``model_choice`` via with_model_choice and measure RMSE."""
    cfg = _config(target=ImputationStrategy.MICE)
    profile = StructuralProfiler(config=cfg).profile(df)
    routing = route(profile, cfg)
    routing = routing.with_model_choice(model_choice)
    recipe = resolve_recipe(routing, profile, cfg)
    units = derive_units(routing, strategy=ImputationStrategy.MICE)
    train = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    result = fit_unit(recipe, units[0], train, random_seed=cfg.random_seed)
    out = result.fitted.transform(train)
    imputed = out["target"].to_numpy()[missing_idx]
    return float(np.sqrt(np.mean((imputed - target_true[missing_idx]) ** 2)))


def test_forest_beats_bayesian_ridge_at_small_sizes_on_nonlinear_data():
    df, target_true, missing_idx = _nonlinear_frame(n=150)
    rmse_ridge = _mice_rmse_with_estimator(
        df, target_true, missing_idx, ModelChoice.BayesianRidge
    )
    rmse_forest = _mice_rmse_with_estimator(
        df, target_true, missing_idx, ModelChoice.RandomForestRegressor
    )
    assert rmse_forest < rmse_ridge


def test_booster_loses_to_forest_on_small_nonlinear_data():
    df, target_true, missing_idx = _nonlinear_frame(n=150, seed=2)
    rmse_forest = _mice_rmse_with_estimator(
        df, target_true, missing_idx, ModelChoice.RandomForestRegressor
    )
    rmse_booster = _mice_rmse_with_estimator(
        df, target_true, missing_idx, ModelChoice.GradientBoostingRegressor
    )
    assert rmse_forest <= rmse_booster


# ---------------------------------------------------------------------------
# Real-frame regression tests reusing ames (ADR-0093, ADR-0097, #542, #543)
# ---------------------------------------------------------------------------


def test_real_frame_ames_regression():
    """Real-frame regression test reusing ames (ADR-0093, ADR-0097, #542, #543).

    Asserts:
    1. Widened KNN beats block-only fit on masked-cell RMSE.
    2. The forest beats BayesianRidge at small sizes (n=100).
    3. The booster loses to the forest at small sizes.
    """
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        df_raw = bunch.frame
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        df_num = df_raw[num_cols].dropna().iloc[:300].copy()
    except Exception as exc:
        pytest.skip(f"Ames dataset unavailable: {exc}")

    rng = np.random.default_rng(42)
    target = "SalePrice"
    true_vals = df_num[target].to_numpy()
    missing_idx = rng.choice(len(df_num), int(len(df_num) * 0.25), replace=False)

    df_masked = df_num.copy()
    df_masked.loc[df_masked.index[missing_idx], target] = np.nan
    pl_df = pl.from_pandas(df_masked)

    # 1. Widened KNN beats block-only
    cfg = PipelineConfig()
    cfg.random_seed = 42
    cfg.imputation.numeric.set_per_column_strategy(target, ImputationStrategy.KNN)
    profile = StructuralProfiler(config=cfg).profile(pl_df)
    routing = route(profile, cfg)
    recipe = resolve_recipe(routing, profile, cfg)
    unit = derive_units(routing, strategy=ImputationStrategy.KNN)[0]
    train = _resolve_effective_nulls(pl_df)
    res = fit_unit(recipe, unit, train, random_seed=42)
    out_knn = res.fitted.transform(train)
    knn_imputed = out_knn[target].to_numpy()[missing_idx]
    widened_rmse = float(np.sqrt(np.mean((knn_imputed - true_vals[missing_idx]) ** 2)))

    block_only_arr = train[target].to_numpy().reshape(-1, 1)
    block_only_imputed = KNNImputer(n_neighbors=5).fit_transform(block_only_arr)[
        missing_idx, 0
    ]
    block_only_rmse = float(
        np.sqrt(np.mean((block_only_imputed - true_vals[missing_idx]) ** 2))
    )
    assert widened_rmse < block_only_rmse

    # 2 & 3. Small size (n=100): forest beats BayesianRidge, booster loses to forest
    df_small = df_num.iloc[:100].copy()
    true_small = df_small[target].to_numpy()
    miss_small = rng.choice(len(df_small), int(len(df_small) * 0.25), replace=False)
    df_small_masked = df_small.copy()
    df_small_masked.loc[df_small_masked.index[miss_small], target] = np.nan
    pl_small = pl.from_pandas(df_small_masked)

    def fit_mice(choice):
        c = PipelineConfig()
        c.random_seed = 42
        c.imputation.numeric.set_per_column_strategy(target, ImputationStrategy.MICE)
        prof = StructuralProfiler(config=c).profile(pl_small)
        r = route(prof, c).with_model_choice(choice)
        rec = resolve_recipe(r, prof, c)
        u = derive_units(r, strategy=ImputationStrategy.MICE)[0]
        tr = _resolve_effective_nulls(pl_small)
        out = fit_unit(rec, u, tr, random_seed=42).fitted.transform(tr)
        imp = out[target].to_numpy()[miss_small]
        return float(np.sqrt(np.mean((imp - true_small[miss_small]) ** 2)))

    rmse_ridge = fit_mice(ModelChoice.BayesianRidge)
    rmse_forest = fit_mice(ModelChoice.RandomForestRegressor)
    rmse_booster = fit_mice(ModelChoice.GradientBoostingRegressor)

    assert rmse_forest < rmse_ridge
    assert rmse_forest <= rmse_booster
