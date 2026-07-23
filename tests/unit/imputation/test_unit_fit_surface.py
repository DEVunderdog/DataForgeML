"""The stateless user-orchestrated execution surface (#385 / ADR-0071).

Covers ``fit_unit`` and ``FittedImputer.compose`` as the drive the user actually
holds: plan with ``decide``, train each unit through the user's own loop
(batch scheduling is user-owned, ADR-0075), and compose the result. The
single-track failure contract, the forced-oversize allowance, and exact
coverage are all asserted here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dataforge_ml import (
    FittedImputer,
    PipelineConfig,
    StructuralProfiler,
    decide,
    fit_unit,
)
from dataforge_ml.imputation import (
    FitSignals,
    ImputationFitWarning,
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
    # Force several columns to model-based strategies so the composed imputer
    # carries more than one unit, then assert the ordered ``units`` list tracks
    # the plan's model-unit order rather than any training/completion order.
    df = _holey_frame(seed=3)
    config = PipelineConfig()
    for col in ("a", "b", "c"):
        config.imputation.numeric.set_per_column_strategy(col, "regression")
    plan = _plan(df, config)
    imputer = FittedImputer.compose(plan, _fit_all(plan, df))

    plan_model_units = [
        u.unit_id
        for u in plan.units
        if u.strategy.value not in ("mean", "median", "mode", "constant", "mnar")
    ]
    fitted_targets = [u.target_columns[0] for u in imputer.units]
    fitted_ids = [f"regression:{c}" for c in fitted_targets]
    assert fitted_ids == plan_model_units


def _single_column_forced_regression_plan():
    # A regression forced on the only column has no feature columns to predict
    # from, so its fitter reports it cannot train — with the column well under
    # the drop threshold so the plan really does route it to a regression unit.
    rng = np.random.default_rng(1)
    vals = rng.normal(0.0, 1.0, 250).tolist()
    for i in range(250):
        if rng.random() < 0.12:
            vals[i] = None
    df = pl.DataFrame({"a": pl.Series(vals, dtype=pl.Float64)})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", "regression")
    return df, _plan(df, config)


def test_untrainable_unit_raises_with_structured_payload():
    df, plan = _single_column_forced_regression_plan()
    unit_id = "regression:a"
    assert any(u.unit_id == unit_id for u in plan.units)

    with pytest.raises(UnitNotTrainableError) as exc:
        fit_unit(plan, unit_id, df)
    err = exc.value
    assert err.unit_id == unit_id
    assert err.columns == ("a",)
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
    df, plan = _single_column_forced_regression_plan()
    with pytest.raises(UnitNotTrainableError):
        fit_unit(plan, "regression:a", df)


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
    config.imputation.numeric.set_per_column_strategy("a", "regression")
    plan = _plan(df, config)
    unit_id = "regression:a"
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
