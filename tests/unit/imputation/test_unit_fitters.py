"""
Tests for #371: the fitter bodies live in unit-shaped functions the surface calls.

The defect these guard against: a model-based unit quietly "succeeding" as a
column of zeros.  So these assert on what the ``fit_unit`` loop actually
*learns* for each unit and what its fitted units then *produce*, not on which
function it called.
"""

import numpy as np
import polars as pl
import pytest

from dataforge_ml import PipelineConfig, StructuralProfiler, fit_unit
from dataforge_ml.imputation import UnitNotTrainableError
from dataforge_ml.imputation._decision_assembler import decide
from dataforge_ml.imputation._fitted_imputer import FittedMICE, _FittedKNN
from dataforge_ml.imputation._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedRegression,
    FittedScalar,
)
from dataforge_ml.utils._null_normalization import _resolve_effective_nulls


def _frame(n=600, seed=0, bimodal=False, grouped=False):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x2 = x1 * 1.4 + rng.normal(0, 2, n)
    x3 = rng.normal(0, 1, n)

    def hole(a, frac):
        a = a.copy()
        a[rng.choice(n, int(n * frac), replace=False)] = np.nan
        return a

    data = {
        "a": hole(x1, 0.15),
        "b": hole(x2, 0.12),
        "c": hole(x3, 0.10),
        "d": x1 * 0.3 + x3,
    }
    if bimodal:
        lo = rng.random(n) < 0.5
        bi = np.where(lo, rng.normal(5, 1, n), rng.normal(40, 1, n))
        data["bi"] = hole(bi, 0.10)
        if grouped:
            data["grp"] = pl.Series(np.where(lo, "lo", "hi"))
    return pl.DataFrame(data)


def _drive(df, cfg):
    """Profile, plan, and train every unit; return the plan, frame, and results.

    ``results`` maps unit id to the fitted unit (unwrapped from the
    ``UnitFitResult`` bundle) so the assertions read the learned state directly.
    """
    profile = StructuralProfiler(config=cfg).profile(df)
    plan = decide(profile, profile.dataset.row_count, cfg)
    train = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    fitted = {
        unit.unit_id: fit_unit(
            plan, unit.unit_id, train, random_seed=cfg.random_seed
        ).fitted
        for unit in plan.units
    }
    return profile, plan, train, fitted


def _config(**per_column_strategy):
    cfg = PipelineConfig()
    cfg.random_seed = 7
    for col, strategy in per_column_strategy.items():
        cfg.imputation.numeric.set_per_column_strategy(col, strategy)
    return cfg


def test_executor_trains_a_real_mice_model_not_a_column_of_zeros():
    cfg = _config(a="mice", b="mice")
    _, _, train, results = _drive(_frame(), cfg)

    fitted = results["mice"]
    assert isinstance(fitted, FittedMICE)

    out = fitted.transform(train)
    filled = out["a"].to_numpy()[train["a"].is_null().to_numpy()]
    assert len(filled) > 0
    assert not np.isnan(filled).any()
    # The regression this guards: every imputed cell was 0.0.
    assert not np.allclose(filled, 0.0)


def test_executor_trains_a_real_knn_model():
    cfg = _config()
    _, plan, train, results = _drive(_frame(), cfg)
    assert any(u.unit_id == "knn" for u in plan.units), "expected KNN routing"

    fitted = results["knn"]
    assert isinstance(fitted, _FittedKNN)

    filled = fitted.transform(train)["a"].to_numpy()[train["a"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_executor_trains_a_real_regression_model():
    cfg = _config(c="regression")
    _, _, train, results = _drive(_frame(), cfg)

    fitted = results["regression:c"]
    assert isinstance(fitted, FittedRegression)

    filled = fitted.transform(train)["c"].to_numpy()[train["c"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_executor_trains_a_real_gmm_sampling_model():
    cfg = _config()
    _, plan, train, results = _drive(_frame(bimodal=True), cfg)
    unit_id = "gmm_sampling:bi"
    assert any(u.unit_id == unit_id for u in plan.units), "expected GMM routing"

    fitted = results[unit_id]
    assert isinstance(fitted, FittedGMMSampling)

    # The two components separate, and every sample lands near one of them.
    assert abs(fitted.center1 - fitted.center2) > 10
    filled = fitted.transform(train)["bi"].to_numpy()[train["bi"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    near = np.minimum(
        np.abs(filled - fitted.center1), np.abs(filled - fitted.center2)
    )
    assert (near < 5).all()


def test_executor_trains_a_real_cluster_conditional_model():
    cfg = _config()
    cfg.imputation.numeric.set_bimodal_grouping_variable("bi", "grp")
    _, plan, train, results = _drive(_frame(bimodal=True, grouped=True), cfg)
    unit_id = "cluster_conditional:bi"
    assert any(u.unit_id == unit_id for u in plan.units), "expected cluster routing"

    fitted = results[unit_id]
    assert isinstance(fitted, FittedClusterConditional)
    assert fitted.grouping_variable == "grp"
    assert set(fitted.group_fills) == {"lo", "hi"}

    filled = fitted.transform(train)["bi"].to_numpy()[train["bi"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_no_model_based_unit_degrades_to_a_scalar_on_a_healthy_frame():
    """A plan that routes model-based work must not silently produce scalars."""
    cfg = _config(a="mice", b="mice", c="regression")
    _, plan, _, results = _drive(_frame(bimodal=True), cfg)

    model_based = {
        u.unit_id
        for u in plan.units
        if u.unit_id in ("mice", "knn") or ":" in u.unit_id and not u.unit_id.startswith(
            ("mean:", "median:", "mode:", "constant:", "mnar:")
        )
    }
    assert model_based, "expected the plan to route some model-based work"
    for unit_id in model_based:
        assert not isinstance(results[unit_id], FittedScalar), (
            f"unit '{unit_id}' degraded to a scalar on a frame it should train on"
        )


def test_forced_strategy_that_cannot_train_raises_rather_than_degrading():
    """A user's deliberate choice is never silently swapped out (ADR-0029)."""
    cfg = _config(a="regression")
    df = pl.DataFrame({"a": [1.0] + [None] * 5})

    profile = StructuralProfiler(config=cfg).profile(df)
    plan = decide(profile, profile.dataset.row_count, cfg)

    unit_id = "regression:a"
    if any(u.unit_id == unit_id for u in plan.units):
        with pytest.raises(UnitNotTrainableError):
            fit_unit(plan, unit_id, df)


def test_hyperparameter_override_reaches_the_fitter():
    """``with_hyperparameters`` is exactly the plan that fits (ADR-0073)."""
    cfg = _config(a="mice", b="mice")
    df = _frame()
    profile = StructuralProfiler(config=cfg).profile(df)
    plan = decide(profile, profile.dataset.row_count, cfg)

    edited = plan.with_hyperparameters("mice", {"max_iter": 3})
    fitted = fit_unit(plan, "mice", df).fitted
    fitted_edited = fit_unit(edited, "mice", df).fitted

    # The decided base is untouched; only the edited plan carries the override.
    assert fitted_edited.model.max_iter == 3
    assert fitted.model.max_iter != 3
