"""The stateless user-orchestrated execution surface (#385 / ADR-0071, ADR-0084).

Covers ``fit_unit`` and ``FittedImputer.compose`` as the drive the user
actually holds: route + resolve a recipe, train each unit through the user's
own loop (batch scheduling is user-owned, ADR-0075), and compose the result.
The single-track failure contract and exact coverage are asserted here, for
every non-model-based strategy (KNN/MICE are the escalation-point ticket's
concern).
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from dataforge_ml import (
    FittedImputer,
    PipelineConfig,
    StructuralProfiler,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
)
from dataforge_ml.imputation import (
    FitSignals,
    ImputationFitWarning,  # noqa: F401 — re-exported, importability check
    ImputationStrategy,
    ImputationUnit,
    UnitFitResult,
    UnitNotTrainableError,
)
from dataforge_ml.imputation._fitted_units import FittedScalar


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


def _routing_and_recipe(df, config=None):
    config = config or PipelineConfig()
    profile = StructuralProfiler(config=config).profile(df)
    routing = route(profile, config)
    recipe = resolve_recipe(routing, profile, config)
    return profile, routing, recipe


def _fit_all(recipe, units, df):
    """The user-owned drive: one ``fit_unit`` call per derived unit."""
    return {u.unit_id: fit_unit(recipe, u, df) for u in units}


def _bimodal_frame(n=400, seed=3, grouped=False):
    """One column that routes to ClusterConditional or GMMSampling."""
    rng = np.random.default_rng(seed)
    lo = rng.random(n) < 0.5
    bi = np.where(lo, rng.normal(5.0, 1.0, n), rng.normal(40.0, 1.0, n)).tolist()
    for i in range(n):
        if rng.random() < 0.1:
            bi[i] = None
    data = {"bi": pl.Series(bi, dtype=pl.Float64)}
    if grouped:
        data["grp"] = pl.Series(np.where(lo, "lo", "hi"))
    return pl.DataFrame(data)


def test_fit_many_is_absent_from_both_public_namespaces():
    """Batch scheduling is user-owned (ADR-0075): fit_many no longer exists."""
    import dataforge_ml
    import dataforge_ml.imputation

    for namespace in (dataforge_ml, dataforge_ml.imputation):
        assert "fit_many" not in namespace.__all__
        assert not hasattr(namespace, "fit_many")


def test_full_flow_fit_unit_loop_compose_transform_imputes_the_frame():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)
    assert set(results) == {u.unit_id for u in units}
    imputer = FittedImputer.compose(recipe, results)
    out = imputer.transform(df).dataframe
    for col in ("a", "b", "c"):
        assert out[col].null_count() == 0


def test_fit_unit_returns_a_bundle_for_one_derived_unit():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    unit = units[0]
    result = fit_unit(recipe, unit, df)
    assert isinstance(result, UnitFitResult)
    assert result.unit_id == unit.unit_id
    assert result.strategy == unit.strategy
    assert result.fitted is not None


def test_compose_accepts_raw_fitted_units_and_unit_fit_results():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)

    from_bundles = FittedImputer.compose(recipe, results)
    from_raw = FittedImputer.compose(recipe, [r.fitted for r in results.values()])

    a = from_bundles.transform(df).dataframe
    b = from_raw.transform(df).dataframe
    assert a.equals(b)


def test_compose_rejects_a_set_that_does_not_cover_the_routing():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)

    dropped = dict(results)
    dropped.pop(next(iter(dropped)))
    with pytest.raises(ValueError, match="cover the routing"):
        FittedImputer.compose(recipe, dropped)


def test_application_order_is_routing_order():
    # Two independent bimodal columns — one grouped (ClusterConditional), one
    # not (GMMSampling) — so the composed imputer carries more than one
    # non-scalar unit, then assert the ordered ``units`` list tracks the
    # routing's unit order rather than any training/completion order.
    grouped = _bimodal_frame(seed=3, grouped=True).rename({"bi": "bi_grouped"})
    ungrouped = _bimodal_frame(seed=11, grouped=False).rename({"bi": "bi_free"})
    df = pl.concat([grouped, ungrouped], how="horizontal_extend")

    config = PipelineConfig()
    config.imputation.numeric.set_bimodal_grouping_variable("bi_grouped", "grp")
    profile, routing, recipe = _routing_and_recipe(df, config)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)
    imputer = FittedImputer.compose(recipe, results)

    routing_model_unit_ids = [
        u.unit_id
        for u in units
        if u.strategy
        not in (
            ImputationStrategy.Mean,
            ImputationStrategy.Median,
            ImputationStrategy.Mode,
            ImputationStrategy.Constant,
            ImputationStrategy.MNAR,
        )
    ]
    assert len(routing_model_unit_ids) >= 2
    expected = [results[unit_id].fitted for unit_id in routing_model_unit_ids]
    assert len(imputer.units) == len(expected)
    assert all(a is b for a, b in zip(imputer.units, expected))


def test_untrainable_unit_raises_with_structured_payload():
    # A hand-authored ClusterConditional column with no grouping variable and
    # no profile-measured bimodal centres cannot train: resolve_recipe never
    # raises over the missing estimate, so the failure surfaces at fit_unit.
    from dataforge_ml import author

    df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    profile, _, _ = _routing_and_recipe(df)
    routing = author({"x": ImputationStrategy.ClusterConditional}, profile=profile)
    recipe = resolve_recipe(routing, profile, PipelineConfig())
    (unit,) = derive_units(routing)
    assert recipe.column_estimates["x"].center1 is None

    with pytest.raises(UnitNotTrainableError) as exc:
        fit_unit(recipe, unit, df)
    err = exc.value
    assert err.unit_id == unit.unit_id
    assert err.columns == ("x",)
    assert "with_estimates" in err.reason


def test_fit_result_carries_a_populated_fit_signals_record():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)

    for result in results.values():
        signals = result.signals
        assert isinstance(signals, FitSignals)
        assert signals.unit_id == result.unit_id
        assert signals.strategy == result.strategy
        # Duration is measured for every fit and is non-negative.
        assert signals.duration_s >= 0.0
        assert isinstance(signals.warnings, tuple)
        assert isinstance(signals.notes, tuple)


def test_fit_signals_reports_the_estimator_for_a_gmm_unit():
    df = _bimodal_frame(grouped=False)
    profile, routing, recipe = _routing_and_recipe(df)
    (unit,) = derive_units(routing, strategy=ImputationStrategy.GMMSampling)

    signals = fit_unit(recipe, unit, df, random_seed=3).signals
    assert signals.estimator == "GaussianMixture"


def test_scalar_units_land_on_records_and_bimodal_units_on_the_list():
    df = _bimodal_frame(grouped=False)
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)
    results = _fit_all(recipe, units, df)
    imputer = FittedImputer.compose(recipe, results)
    # No FittedScalar leaks into the ordered unit list; scalar fills are on
    # records.
    assert not any(isinstance(u, FittedScalar) for u in imputer.units)


def test_fit_unit_signature_takes_a_recipe_and_defaults_n_jobs_inner_to_minus_one():
    params = inspect.signature(fit_unit).parameters
    assert list(params) == [
        "recipe",
        "unit",
        "df",
        "random_seed",
        "n_jobs_inner",
    ]
    assert params["n_jobs_inner"].default == -1


def test_derive_units_selects_by_strategy_without_naming_a_unit_id():
    """The selection surface: an enum the checker sees, not a typed literal."""
    df = _bimodal_frame(grouped=True)
    config = PipelineConfig()
    config.imputation.numeric.set_bimodal_grouping_variable("bi", "grp")
    _, routing, _ = _routing_and_recipe(df, config)

    (unit,) = derive_units(routing, strategy=ImputationStrategy.ClusterConditional)
    assert unit.strategy == ImputationStrategy.ClusterConditional
    assert not unit.is_block
    assert set(unit.columns) == {"bi"}


def test_derive_units_returns_empty_when_nothing_routed_to_the_strategy():
    """The empty case is ordinary, so it is a zero-length tuple — never None, never a raise."""
    df = _holey_frame()
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(["a", "b", "c"], "median")
    _, routing, _ = _routing_and_recipe(df, config)

    assert derive_units(routing, strategy=ImputationStrategy.MICE) == ()
    assert derive_units(routing, strategy=ImputationStrategy.Dropped) == ()


def test_fit_unit_always_reads_the_recipes_current_hyperparameters():
    """A unit carries no dials, so it cannot go stale (ADR-0084 amendment).

    Editing the recipe with ``with_hyperparameters`` and re-fitting the exact
    same unit object picks up the edit — there is nothing on the unit for a
    prior edit to leave behind.
    """
    df = pl.DataFrame(
        {"x": [1.0, 2.0, None, None, 3.0, 4.0, 5.0, None, 6.0, 7.0] * 5}
    )
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    _, routing, recipe = _routing_and_recipe(df, config)
    (unit,) = derive_units(routing, strategy=ImputationStrategy.MNAR)

    original_tendency = recipe.hyperparameters(unit.unit_id)["central_tendency"]
    other = "mean" if original_tendency != "mean" else "median"
    edited = recipe.with_hyperparameters(unit.unit_id, {"central_tendency": other})

    original_fill = fit_unit(recipe, unit, df).fitted.fill_value
    edited_fill = fit_unit(edited, unit, df).fitted.fill_value
    assert original_fill != edited_fill


def test_mnar_signals_say_the_fill_is_computed_not_applied():
    """Nulls in the output are otherwise the only clue (ADR-0098)."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, None, None, 3.0, 4.0, 5.0, None, 6.0, 7.0] * 5}
    )
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    _, routing, recipe = _routing_and_recipe(df, config)
    (unit,) = derive_units(routing, strategy=ImputationStrategy.MNAR)

    notes = fit_unit(recipe, unit, df).signals.notes

    assert any("fill computed, not applied" in note for note in notes)


def test_an_mnar_units_own_transform_is_the_opt_in_fill():
    """The composed imputer leaves MNAR nulls; the bare unit applies the fill."""
    df = pl.DataFrame(
        {"x": [1.0, 2.0, None, None, 3.0, 4.0, 5.0, None, 6.0, 7.0] * 5}
    )
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    _, routing, recipe = _routing_and_recipe(df, config)
    (unit,) = derive_units(routing, strategy=ImputationStrategy.MNAR)
    fitted = fit_unit(recipe, unit, df).fitted

    composed = FittedImputer.compose(recipe, {unit.unit_id: fitted}).transform(df)
    assert composed.dataframe["x"].null_count() == df["x"].null_count()

    opted_in = fitted.transform(df)
    assert opted_in["x"].null_count() == 0
    assert opted_in["x"].filter(df["x"].is_null()).unique().to_list() == [
        fitted.fill_value
    ]


def test_passing_a_unit_id_string_raises_a_pointed_type_error():
    """The pre-4.x call shape fails loudly, naming the replacement."""
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    units = derive_units(routing)

    with pytest.raises(TypeError, match="takes an ImputationUnit, not the id"):
        fit_unit(recipe, units[0].unit_id, df)


def test_fit_unit_raises_key_error_for_a_unit_unknown_to_the_recipe():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    foreign = ImputationUnit(
        unit_id="mice",
        strategy=ImputationStrategy.MICE,
        columns=("a",),
        is_block=True,
    )
    assert derive_units(routing, strategy=ImputationStrategy.MICE) == ()

    with pytest.raises(KeyError, match="carries no unit 'mice'"):
        fit_unit(recipe, foreign, df)


def test_fit_unit_raises_value_error_for_a_unit_from_a_different_routing():
    df = _holey_frame()
    _, routing, recipe = _routing_and_recipe(df)
    real_unit = derive_units(routing)[0]
    mismatched = ImputationUnit(
        unit_id=real_unit.unit_id,
        strategy=ImputationStrategy.Mean,
        columns=("does-not-exist",),
    )
    with pytest.raises(ValueError, match="does not match the recipe's routing"):
        fit_unit(recipe, mismatched, df)


def test_a_reloaded_routing_resolves_and_fits_identically():
    """Every fact the fitters need rides on the routing + recipe.

    The routing round-trips through bare-bytes persistence; resolving a fresh
    recipe against the same profile and fitting from that reproduces the
    in-process fit exactly.
    """
    from dataforge_ml import deserialize, serialize

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
    profile, routing, recipe = _routing_and_recipe(df, config)
    units = derive_units(routing)
    assert any(u.unit_id == "cluster_conditional:bi" for u in units)
    assert any(u.unit_id == "constant:k" for u in units)

    reloaded_routing = deserialize(serialize(routing))
    reloaded_recipe = resolve_recipe(reloaded_routing, profile, config)
    reloaded_units = derive_units(reloaded_routing)

    in_process = _fit_all(recipe, units, df)
    from_disk = _fit_all(reloaded_recipe, reloaded_units, df)
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
