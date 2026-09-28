"""Observed-value preservation across fitted-unit ``transform`` (#400, #401).

A user imputing a column gets every observed measurement back untouched —
bit-for-bit, original dtype — while only the cells that were actually missing
(Effective Null: null, NaN, or infinite for float columns; null otherwise)
receive fill values. The contract is enforced inside each unit's own
``transform``, so it holds on the stateless user-orchestrated door (ADR-0071)
where a user holds a single Fitted Unit with no ``FittedImputer`` involved.

Parametrized over every non-model-based Fitted Unit type (KNN and MICE are the
escalation-point ticket's concern and are not covered here). The scalar
control is forced per column; the bimodal strategies cannot be forced, so
their cases route naturally off bimodal fixtures (grouping variable declared →
Cluster-Conditional, no correlated features → GMM Sampling). ``median`` rides
along as the no-op control: ``FittedScalar`` fills via ``fill_null`` and
already satisfies the invariant, so it must keep passing unchanged. Everything
runs through the public ``route`` → ``resolve_recipe`` → ``derive_units`` →
``fit_unit`` → ``transform`` path.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dataforge_ml import (
    PipelineConfig,
    StructuralProfiler,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
)

STRATEGY_CASES = [
    pytest.param("median", id="scalar-control"),
    pytest.param("cluster_conditional", id="cluster-conditional"),
    pytest.param("gmm_sampling", id="gmm-sampling"),
]

_FORCED_STRATEGIES = ("median",)

_TARGET_COLS = ("rating", "stock", "noisy")

_BIMODAL_COLS = ("bi_float", "bi_int")


def _train_frame(n=240, seed=7):
    """Training data whose columns cover every dtype case of the contract.

    ``rating``: Float64 whole numbers 1..5. ``stock``: Int64. ``noisy``:
    continuous Float64. Each carries genuine nulls.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)

    rating = np.clip(np.round(base * 1.2 + 3.0), 1.0, 5.0)
    rating[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]  # every domain slot present
    count = np.round(base * 10.0 + 50.0).astype(np.int64)
    noisy = base * 2.0 + rng.normal(0.0, 0.3, n)

    data = {
        "rating": pl.Series(rating, dtype=pl.Float64),
        "stock": pl.Series(count, dtype=pl.Int64),
        "noisy": pl.Series(noisy, dtype=pl.Float64),
    }
    df = pl.DataFrame(data)
    holes = {
        col: rng.choice(np.arange(5, n), size=n // 10, replace=False)
        for col in _TARGET_COLS
    }
    return df.with_columns(
        pl.when(pl.arange(0, n).is_in(list(idx)))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col, idx in holes.items()
    )


def _probe_frame(raw_nan=True):
    """Frame handed to ``transform``: the cells preservation is about.

    ``rating`` carries fractional *observed* values; ``stock`` is Int64 with
    observed values and holes; ``noisy`` carries a raw ``NaN`` alongside
    genuine nulls, so a bare ``is_null()`` predicate would restore the ``NaN``
    over the fill.

    The scalar control passes ``raw_nan=False``: ``FittedScalar`` fills genuine
    nulls only (``fill_null``), because on the composed door raw ``NaN`` is
    normalised to null before any fill (``_resolve_effective_nulls``), and on
    the standalone door the unit is deliberately left untouched (#401).
    """
    noisy_nan = float("nan") if raw_nan else None
    return pl.DataFrame(
        {
            "rating": pl.Series(
                [3.6, 1.0, None, 4.4, 2.0, None, 5.0, 1.2], dtype=pl.Float64
            ),
            "stock": pl.Series(
                [37, None, 50, 61, None, 44, 58, 49], dtype=pl.Int64
            ),
            "noisy": pl.Series(
                [-1.25, 0.5, noisy_nan, None, 2.75, noisy_nan, None, 0.0],
                dtype=pl.Float64,
            ),
        }
    )


def _bimodal_train_frame(n=600, seed=3, grouped=False):
    """Training data that routes to the bimodal strategies.

    ``bi_float``: Float64 with well-separated modes. ``bi_int``: Int64,
    likewise bimodal — the dtype the numpy write-back used to widen. With
    ``grouped=True`` both columns split on the same ``grp`` label (branch 1,
    Cluster-Conditional); without it the mode masks are independent, so the
    columns do not correlate and branch 4 (GMM Sampling) fires.
    """
    rng = np.random.default_rng(seed)
    lo_f = rng.random(n) < 0.5
    lo_i = lo_f if grouped else rng.random(n) < 0.5

    bi_float = np.where(lo_f, rng.normal(5.0, 1.0, n), rng.normal(40.0, 1.0, n))
    bi_int = np.round(
        np.where(lo_i, rng.normal(100.0, 2.0, n), rng.normal(160.0, 2.0, n))
    ).astype(np.int64)

    data = {
        "bi_float": pl.Series(bi_float, dtype=pl.Float64),
        "bi_int": pl.Series(bi_int, dtype=pl.Int64),
    }
    if grouped:
        data["grp"] = pl.Series(np.where(lo_f, "lo", "hi"))
    df = pl.DataFrame(data)
    holes = {
        col: rng.choice(np.arange(0, n), size=n // 10, replace=False)
        for col in _BIMODAL_COLS
    }
    return df.with_columns(
        pl.when(pl.arange(0, n).is_in(list(idx)))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col, idx in holes.items()
    )


def _bimodal_probe_frame(grouped=False):
    """Probe for the bimodal strategies: fractional observed floats, an Int64
    column the write-back used to widen, and genuine nulls to fill. No raw
    ``NaN``: the bimodal units fill Polars nulls, and on the composed door raw
    ``NaN`` is normalised to null before they run (``_resolve_effective_nulls``).
    """
    data = {
        "bi_float": pl.Series(
            [5.3, 39.7, None, 41.2, 4.6, None, 6.1, 38.4], dtype=pl.Float64
        ),
        "bi_int": pl.Series(
            [101, 158, None, 162, 97, None, 103, 161], dtype=pl.Int64
        ),
    }
    if grouped:
        data["grp"] = pl.Series(["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"])
    return pl.DataFrame(data)


def _observed_mask(s: pl.Series) -> pl.Series:
    if s.dtype in (pl.Float32, pl.Float64):
        return ~(s.is_null() | s.is_nan() | s.is_infinite())
    return s.is_not_null()


def _select_units(units, strategy_name, target_cols):
    """The units the case is about — block ids are bare, per-column ids
    are ``strategy:column``."""
    selected = [
        u
        for u in units
        if u.unit_id == strategy_name or u.unit_id.startswith(f"{strategy_name}:")
    ]
    assert {c for u in selected for c in u.columns} == set(target_cols), (
        f"expected {strategy_name} to own exactly {target_cols}"
    )
    return selected


def _fit_all(recipe, units, df):
    return [(u, fit_unit(recipe, u, df, random_seed=7).fitted) for u in units]


def _case(strategy_name):
    """route → resolve_recipe → fit_unit for the case's strategy; returns
    (unit, fitted) pairs and the probe frame the assertions run against."""
    if strategy_name in _FORCED_STRATEGIES:
        df = _train_frame()
        config = PipelineConfig()
        config.random_seed = 7
        for col in _TARGET_COLS:
            config.imputation.numeric.set_per_column_strategy(col, strategy_name)
        profile = StructuralProfiler(config=config).profile(df)
        routing = route(profile, config)
        recipe = resolve_recipe(routing, profile, config)
        units = _select_units(derive_units(routing), strategy_name, _TARGET_COLS)
        return _fit_all(recipe, units, df), _probe_frame(raw_nan=False)

    grouped = strategy_name == "cluster_conditional"
    df = _bimodal_train_frame(grouped=grouped)
    config = PipelineConfig()
    config.random_seed = 7
    if grouped:
        for col in _BIMODAL_COLS:
            config.imputation.numeric.set_bimodal_grouping_variable(col, "grp")
    profile = StructuralProfiler(config=config).profile(df)
    routing = route(profile, config)
    recipe = resolve_recipe(routing, profile, config)
    units = _select_units(derive_units(routing), strategy_name, _BIMODAL_COLS)
    return _fit_all(recipe, units, df), _bimodal_probe_frame(grouped=grouped)


def _authored_case():
    """The same contract on a routing no profile produced (#468, ADR-0090).

    The guarantee lives inside each unit's own ``transform``, below the
    routing, so it must hold identically for a hand-authored one.
    """
    from dataforge_ml import AuthoredColumn, ImputationStrategy, author

    df = _train_frame()
    profile = StructuralProfiler(config=PipelineConfig()).profile(df)
    routing = author(
        {
            "rating": ImputationStrategy.Median,
            "stock": ImputationStrategy.Median,
            "noisy": ImputationStrategy.Median,
        },
        profile=profile,
    )
    recipe = resolve_recipe(routing, profile, PipelineConfig())
    units = _select_units(derive_units(routing), "median", _TARGET_COLS)
    return _fit_all(recipe, units, df), _probe_frame(raw_nan=False)


def test_authored_routing_preserves_observed_cells_and_dtypes():
    pairs, probe = _authored_case()
    for unit, fitted in pairs:
        out = fitted.transform(probe)
        assert out.schema == probe.schema, f"{unit.unit_id}: schema changed"
        for col in unit.columns:
            mask = _observed_mask(probe[col])
            assert out[col].filter(mask).equals(probe[col].filter(mask)), (
                f"{unit.unit_id}: observed cells of '{col}' changed"
            )
            filled = out[col].filter(~mask)
            assert len(filled) > 0, f"fixture rot: '{col}' has no missing cells"
            assert _observed_mask(filled).all(), (
                f"{unit.unit_id}: missing cells of '{col}' were not filled"
            )


@pytest.mark.parametrize("strategy_name", STRATEGY_CASES)
def test_observed_cells_come_back_bit_for_bit(strategy_name):
    pairs, probe = _case(strategy_name)
    for unit, fitted in pairs:
        out = fitted.transform(probe)
        for col in unit.columns:
            mask = _observed_mask(probe[col])
            assert out[col].filter(mask).equals(probe[col].filter(mask)), (
                f"{unit.unit_id}: observed cells of '{col}' changed"
            )


@pytest.mark.parametrize("strategy_name", STRATEGY_CASES)
def test_dtypes_are_unchanged(strategy_name):
    pairs, probe = _case(strategy_name)
    for unit, fitted in pairs:
        out = fitted.transform(probe)
        assert out.schema == probe.schema, f"{unit.unit_id}: schema changed"


@pytest.mark.parametrize("strategy_name", STRATEGY_CASES)
def test_columns_outside_the_unit_targets_are_untouched(strategy_name):
    pairs, probe = _case(strategy_name)
    for unit, fitted in pairs:
        out = fitted.transform(probe)
        for col in probe.columns:
            if col in fitted.target_columns:
                continue
            assert out[col].equals(probe[col]), (
                f"{unit.unit_id}: non-target column '{col}' changed"
            )


@pytest.mark.parametrize("strategy_name", STRATEGY_CASES)
def test_missing_cells_are_filled(strategy_name):
    # Preservation must not be satisfiable by a unit that does nothing: every
    # Effective Null in the input carries a real value on the way out.
    pairs, probe = _case(strategy_name)
    for unit, fitted in pairs:
        out = fitted.transform(probe)
        for col in unit.columns:
            filled = out[col].filter(~_observed_mask(probe[col]))
            assert len(filled) > 0, f"fixture rot: '{col}' has no missing cells"
            assert _observed_mask(filled).all(), (
                f"{unit.unit_id}: missing cells of '{col}' were not filled"
            )
