"""The stateless user-orchestrated execution surface (#385 / ADR-0071).

Covers ``fit_unit`` and ``FittedImputer.compose`` as the drive the user actually
holds: plan with ``decide``, train each unit through the user's own loop
(batch scheduling is user-owned, ADR-0075), and compose the result. The
single-track failure contract, the forced-oversize allowance, and exact
coverage are all asserted here.

``core_budget`` (#432 / ADR-0081) joins the same seam: it is the arithmetic the
user calls once before that loop, so it belongs beside the loop it prices.
"""

from __future__ import annotations

import inspect
import warnings

import joblib
import numpy as np
import polars as pl
import pytest

from dataforge_ml import (
    FittedImputer,
    ModelChoice,
    PipelineConfig,
    StructuralProfiler,
    core_budget,
    decide,
    fit_unit,
)
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import (
    ColumnImputationDecision,
    FitSignals,
    ImputationDecision,
    ImputationFitWarning,
    ImputationStrategy,
    UnitFitResult,
    UnitNotTrainableError,
)
from dataforge_ml.imputation._fitted_units import FittedScalar
from dataforge_ml.profiling._config import ProfileConfig


def _holey_frame(n=250, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    data = {}
    for name in ("a", "b", "c"):
        vals = (base + rng.normal(0.0, 0.5, n)).tolist()
        for i in range(n):
            if rng.random() < 0.12:
                vals[i] = None
        data[name] = pl.Series(vals, dtype=pl.Float64)
    return pl.DataFrame(data)


def _plan(df, config=None):
    config = config or PipelineConfig()
    profile = StructuralProfiler().profile(df)
    return decide(profile, len(df), config)


def _fit_all(plan, df):
    """The user-owned drive: one ``fit_unit`` call per planned unit."""
    return {u.unit_id: fit_unit(plan, u.unit_id, df) for u in plan.units}


def test_fit_many_is_absent_from_both_public_namespaces():
    """Batch scheduling is user-owned (ADR-0075): fit_many no longer exists."""
    import dataforge_ml
    import dataforge_ml.imputation

    for namespace in (dataforge_ml, dataforge_ml.imputation):
        assert "fit_many" not in namespace.__all__
        assert not hasattr(namespace, "fit_many")


def test_full_flow_fit_unit_loop_compose_transform_imputes_the_frame():
    df = _holey_frame()
    plan = _plan(df)
    results = _fit_all(plan, df)
    assert set(results) == {u.unit_id for u in plan.units}
    imputer = FittedImputer.compose(plan, results)
    out = imputer.transform(df).dataframe
    for col in ("a", "b", "c"):
        assert out[col].null_count() == 0


def test_fit_unit_returns_a_bundle_for_one_planned_unit():
    df = _holey_frame()
    plan = _plan(df)
    unit_id = plan.units[0].unit_id
    result = fit_unit(plan, unit_id, df)
    assert isinstance(result, UnitFitResult)
    assert result.unit_id == unit_id
    assert result.strategy == plan.units[0].strategy
    assert result.fitted is not None


def test_compose_accepts_raw_fitted_units_and_unit_fit_results():
    df = _holey_frame()
    plan = _plan(df)
    results = _fit_all(plan, df)

    from_bundles = FittedImputer.compose(plan, results)
    from_raw = FittedImputer.compose(plan, [r.fitted for r in results.values()])

    a = from_bundles.transform(df).dataframe
    b = from_raw.transform(df).dataframe
    assert a.equals(b)


def test_compose_rejects_a_set_that_does_not_cover_the_plan():
    df = _holey_frame()
    plan = _plan(df)
    results = _fit_all(plan, df)

    dropped = dict(results)
    dropped.pop(next(iter(dropped)))
    with pytest.raises(ValueError, match="cover the plan"):
        FittedImputer.compose(plan, dropped)


def test_application_order_is_plan_order():
    # Force columns onto distinct model-based blocks so the composed imputer
    # carries more than one unit, then assert the ordered ``units`` list tracks
    # the plan's model-unit order rather than any training/completion order.
    df = _holey_frame(seed=3)
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(["a", "b"], "mice")
    config.imputation.numeric.set_per_column_strategy("c", "knn")
    plan = _plan(df, config)
    results = _fit_all(plan, df)
    imputer = FittedImputer.compose(plan, results)

    plan_model_unit_ids = [
        u.unit_id
        for u in plan.units
        if u.strategy.value not in ("mean", "median", "mode", "constant", "mnar")
    ]
    assert len(plan_model_unit_ids) >= 2
    expected = [results[unit_id].fitted for unit_id in plan_model_unit_ids]
    assert len(imputer.units) == len(expected)
    assert all(a is b for a, b in zip(imputer.units, expected))


def _unpredictable_mice_plan():
    # A MICE column whose values are white noise, uncorrelated with every
    # other numeric column, profiles Unpredictable (near-zero R²_RF) — the
    # block resolves no model choice and its fitter reports it cannot train.
    # Nonlinearity profiling must be explicitly enabled (off by default) and
    # driven straight into StructuralProfiler, bypassing ``_plan``'s
    # default-config profile.
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = np.random.default_rng(99).normal(0.0, 1.0, n)
    holes = rng.choice(n, int(n * 0.12), replace=False)
    y[holes] = np.nan
    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    config = PipelineConfig(profiling=ProfileConfig(compute_nonlinearity=True))
    config.imputation.numeric.set_per_column_strategy("y", "mice")
    profile = StructuralProfiler(config=config).profile(df)
    plan = decide(profile, len(df), config)
    return df, plan


def test_untrainable_unit_raises_with_structured_payload():
    df, plan = _unpredictable_mice_plan()
    unit_id = "mice"
    assert any(u.unit_id == unit_id for u in plan.units)
    assert plan.column_decisions["y"].model_choice is None

    with pytest.raises(UnitNotTrainableError) as exc:
        fit_unit(plan, unit_id, df)
    err = exc.value
    assert err.unit_id == unit_id
    assert err.columns == ("y",)
    assert err.reason


def test_forcing_a_strategy_past_its_routing_threshold_succeeds():
    df = _holey_frame(n=200)
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", "knn")
    config.imputation.numeric.knn_max_rows = 10  # 200 rows is far past the guard
    plan = _plan(df, config)
    # No block: the forced KNN trains and composes cleanly.
    imputer = FittedImputer.compose(plan, _fit_all(plan, df))
    assert imputer.transform(df).dataframe["a"].null_count() == 0


def test_forced_oversize_warns_structurally_and_records_it():
    """A forced-oversize fit warns on the standard channel AND records it (ADR-0074)."""
    df = _holey_frame(n=200)
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", "knn")
    config.imputation.numeric.knn_max_rows = 10  # 200 rows is far past the cap
    plan = _plan(df, config)
    unit_id = next(u.unit_id for u in plan.units if u.strategy.value == "knn")

    with pytest.warns(ImputationFitWarning, match="forced past its routing threshold"):
        result = fit_unit(plan, unit_id, df)

    # Dual-channelled: the same warning is also recorded on the structured record.
    assert result.signals.warnings
    assert any("forced past" in w for w in result.signals.warnings)


def test_genuine_failure_still_raises_not_warns():
    """A forced but untrainable unit raises; it does not degrade to a warning."""
    df, plan = _unpredictable_mice_plan()
    with pytest.raises(UnitNotTrainableError):
        fit_unit(plan, "mice", df)


def test_fit_result_carries_a_populated_fit_signals_record():
    df = _holey_frame()
    plan = _plan(df)
    results = _fit_all(plan, df)

    for result in results.values():
        signals = result.signals
        assert isinstance(signals, FitSignals)
        assert signals.unit_id == result.unit_id
        assert signals.strategy == result.strategy
        # Duration is measured for every fit and is non-negative.
        assert signals.duration_s >= 0.0
        assert isinstance(signals.warnings, tuple)
        assert isinstance(signals.notes, tuple)


def test_fit_signals_reports_estimator_and_convergence_for_a_model_unit():
    df = _holey_frame(n=600)
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", "mice")
    plan = _plan(df, config)
    unit_id = "mice"
    assert any(u.unit_id == unit_id for u in plan.units)

    signals = fit_unit(plan, unit_id, df).signals
    assert signals.estimator is not None
    assert signals.converged is not None
    assert signals.n_iter is not None


def test_scalar_units_land_on_records_and_model_units_on_the_list():
    df = _holey_frame()
    plan = _plan(df)
    results = _fit_all(plan, df)
    imputer = FittedImputer.compose(plan, results)
    # No FittedScalar leaks into the ordered unit list; scalar fills are on
    # records.
    assert not any(isinstance(u, FittedScalar) for u in imputer.units)


# ---------------------------------------------------------------------------
# core_budget — the whole-plan inner-parallelism arithmetic (#432 / ADR-0081)
#
# Plans are built straight from ``ColumnImputationDecision`` entries so the
# arithmetic is under test rather than the router: the budget reads only
# ``units`` and ``column_decisions``, and an explicit ``total_cores`` keeps every
# expected value an exact integer with no monkeypatching of core detection.
# ---------------------------------------------------------------------------


def _synthetic_plan(*decisions: ColumnImputationDecision) -> ImputationDecision:
    return ImputationDecision(
        column_decisions={d.column: d for d in decisions},
        config_snapshot={},
    )


def _numeric(column, strategy, **kw) -> ColumnImputationDecision:
    return ColumnImputationDecision(
        column=column, semantic_type=SemanticType.Numeric, strategy=strategy, **kw
    )


def _mice_col(column, choice=ModelChoice.RandomForestRegressor):
    return _numeric(column, ImputationStrategy.MICE, model_choice=choice)


def _budget_plan(choice=ModelChoice.RandomForestRegressor, knn=True, scalar=True):
    """A plan shaped like a real one: a MICE block, optionally KNN, optionally dust."""
    decisions = [_mice_col("m1", choice), _mice_col("m2", choice)]
    if knn:
        decisions.append(_numeric("k1", ImputationStrategy.KNN))
    if scalar:
        decisions.append(_numeric("s1", ImputationStrategy.Median))
    return _synthetic_plan(*decisions)


def test_core_budget_is_exported_from_both_public_namespaces():
    import dataforge_ml
    import dataforge_ml.imputation

    for namespace in (dataforge_ml, dataforge_ml.imputation):
        assert "core_budget" in namespace.__all__
        assert namespace.core_budget is core_budget


def test_core_budget_keys_are_exactly_the_plans_unit_ids():
    df = _holey_frame()
    plan = _plan(df)
    budget = core_budget(plan, max_workers=4, total_cores=8)
    assert set(budget) == {u.unit_id for u in plan.units}


def test_parallel_drive_reserves_one_core_for_the_knn_block():
    plan = _budget_plan()
    assert core_budget(plan, max_workers=8, total_cores=12) == {
        "mice": 11,
        "knn": 1,
        "median:s1": 1,
    }


def test_parallel_drive_without_a_knn_block_gives_mice_every_core():
    plan = _budget_plan(knn=False)
    assert core_budget(plan, max_workers=8, total_cores=12) == {
        "mice": 12,
        "median:s1": 1,
    }


def test_max_workers_one_is_a_sequential_drive_and_gives_mice_minus_one():
    """Outer degree one: the inner layer takes the whole machine (ADR-0069)."""
    plan = _budget_plan()
    assert core_budget(plan, max_workers=1, total_cores=12) == {
        "mice": -1,
        "knn": 1,
        "median:s1": 1,
    }


def test_max_workers_none_is_a_pool_of_unknown_degree_not_a_sequential_drive():
    """``ThreadPoolExecutor(max_workers=None)`` is legal, common, and not sequential."""
    plan = _budget_plan()
    parallel = core_budget(plan, max_workers=8, total_cores=12)
    assert core_budget(plan, max_workers=None, total_cores=12) == parallel


def test_one_core_box_with_a_knn_block_still_floors_mice_at_one():
    plan = _budget_plan()
    assert core_budget(plan, max_workers=8, total_cores=1)["mice"] == 1


@pytest.mark.parametrize(
    "choice",
    [ModelChoice.GradientBoostingRegressor, ModelChoice.BayesianRidge],
)
def test_a_mice_block_with_no_inner_parallelism_to_spend_is_priced_at_one(choice):
    """Neither estimator has an ``n_jobs``, so ``1`` is the truthful answer."""
    plan = _budget_plan(choice=choice)
    assert core_budget(plan, max_workers=8, total_cores=12) == {
        "mice": 1,
        "knn": 1,
        "median:s1": 1,
    }


def test_no_warning_is_emitted_for_a_block_that_cannot_spend_a_budget():
    """A warning here would fire on a correct plan; the returned ``1`` is the answer."""
    plan = _budget_plan(choice=ModelChoice.GradientBoostingRegressor)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert core_budget(plan, max_workers=8, total_cores=12)["mice"] == 1


@pytest.mark.parametrize("max_workers", [1, 4, 8, None])
def test_a_custom_estimator_is_priced_at_one_in_every_branch(max_workers):
    """The library spends no cores on an estimator it did not build (ADR-0083).

    Including the sequential ``max_workers=1`` branch, which hands a
    library-built MICE block ``-1``: there is nothing here to open up, since the
    library never sets a foreign estimator's parameters.
    """
    plan = _budget_plan(choice=ModelChoice.Custom)
    assert core_budget(plan, max_workers=max_workers, total_cores=12) == {
        "mice": 1,
        "knn": 1,
        "median:s1": 1,
    }


def test_a_plan_with_no_mice_unit_is_all_ones():
    plan = _synthetic_plan(
        _numeric("k1", ImputationStrategy.KNN),
        _numeric("s1", ImputationStrategy.Median),
    )
    budget = core_budget(plan, max_workers=8, total_cores=12)
    assert budget == {"knn": 1, "median:s1": 1}


def test_total_cores_overrides_detection():
    plan = _budget_plan(knn=False)
    assert core_budget(plan, max_workers=8, total_cores=4)["mice"] == 4
    assert core_budget(plan, max_workers=8, total_cores=64)["mice"] == 64


def test_the_default_path_detects_cores_with_joblib_cpu_count():
    """Detection is ``joblib.cpu_count()`` — cgroup-aware, and what sklearn uses."""
    plan = _budget_plan(knn=False)
    detected = core_budget(plan, max_workers=8)
    assert detected == core_budget(plan, max_workers=8, total_cores=joblib.cpu_count())


def test_fit_unit_signature_is_unchanged_and_n_jobs_inner_still_defaults_to_minus_one():
    """``core_budget`` is purely additive: the sequential loop reads the same -1."""
    params = inspect.signature(fit_unit).parameters
    assert list(params) == [
        "decision",
        "unit_id",
        "df",
        "random_seed",
        "n_jobs_inner",
    ]
    assert params["n_jobs_inner"].default == -1


def test_a_reloaded_plan_fits_identically_with_no_config_in_hand():
    """Every fact the fitters need rides on the plan itself (#466 / ADR-0083).

    The saved plan is stripped of its config snapshot before reloading, so the
    only thing left to fit from is ``column_decisions`` — the whole point of
    lowering the grouping variable and the constant fill onto the column.
    """
    rng = np.random.default_rng(3)
    n = 300
    lo = rng.random(n) < 0.5
    bi = np.where(lo, rng.normal(5.0, 1.0, n), rng.normal(40.0, 1.0, n)).tolist()
    for i in range(n):
        if rng.random() < 0.1:
            bi[i] = None
    const = (rng.normal(0.0, 1.0, n)).tolist()
    for i in range(n):
        if rng.random() < 0.1:
            const[i] = None
    df = pl.DataFrame(
        {
            "bi": pl.Series(bi, dtype=pl.Float64),
            "grp": pl.Series(np.where(lo, "lo", "hi")),
            "k": pl.Series(const, dtype=pl.Float64),
        }
    )

    config = PipelineConfig()
    config.random_seed = 11
    config.imputation.numeric.set_bimodal_grouping_variable("bi", "grp")
    config.imputation.numeric.set_per_column_constant_fill("k", -9.0)
    plan = _plan(df, config)
    assert any(u.unit_id == "cluster_conditional:bi" for u in plan.units)
    assert any(u.unit_id == "constant:k" for u in plan.units)

    reloaded = ImputationDecision.from_dict({**plan.to_dict(), "config_snapshot": {}})
    assert reloaded.config_snapshot == {}

    in_process = _fit_all(plan, df)
    from_disk = _fit_all(reloaded, df)
    assert set(in_process) == set(from_disk)

    cluster = from_disk["cluster_conditional:bi"].fitted
    assert cluster.grouping_variable == "grp"
    assert (
        cluster.group_fills
        == in_process["cluster_conditional:bi"].fitted.group_fills
    )
    assert from_disk["constant:k"].fitted.fill_value == -9.0
    for unit_id, result in from_disk.items():
        assert result.fitted.transform(df).equals(
            in_process[unit_id].fitted.transform(df)
        )
