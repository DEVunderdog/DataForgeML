"""
Tests for #371: the fitter bodies live in unit-shaped functions the surface calls.

Covers only the non-model-based fitters — scalar (Constant), GMM Sampling and
Cluster-Conditional. KNN and MICE are covered separately in
``test_model_based_fitters.py``.

The defect these guard against: a bimodal unit quietly "succeeding" as a
column of zeros. So these assert on what the ``fit_unit`` loop actually
*learns* for each unit and what its fitted units then *produce*, not on which
function it called.
"""

import numpy as np
import polars as pl

from dataforge_ml import PipelineConfig, StructuralProfiler, derive_units, fit_unit
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import ImputationStrategy, resolve_recipe, route
from dataforge_ml.imputation._config import ColumnRouting, ImputationUnit
from dataforge_ml.imputation._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)
from dataforge_ml.imputation._fitters import (
    UnitFitContext,
    fit_cluster_unit,
    fit_gmm_unit,
    fit_scalar_unit,
)
from dataforge_ml.imputation._recipe import ColumnEstimates
from dataforge_ml.utils._null_normalization import _resolve_effective_nulls


def _frame(n=600, seed=0, bimodal=False, grouped=False):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x3 = rng.normal(0, 1, n)

    def hole(a, frac):
        a = a.copy()
        a[rng.choice(n, int(n * frac), replace=False)] = np.nan
        return a

    data = {
        "a": hole(x1, 0.15),
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
    """Route, resolve a recipe, and train every unit; return everything as
    learned so the assertions read the learned state directly."""
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


def test_executor_trains_a_real_gmm_sampling_model():
    cfg = _config()
    _, routing, units, train, results = _drive(_frame(bimodal=True), cfg)
    unit_id = "gmm_sampling:bi"
    assert any(u.unit_id == unit_id for u in units), "expected GMM routing"

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
    _, routing, units, train, results = _drive(_frame(bimodal=True, grouped=True), cfg)
    unit_id = "cluster_conditional:bi"
    assert any(u.unit_id == unit_id for u in units), "expected cluster routing"

    fitted = results[unit_id]
    assert isinstance(fitted, FittedClusterConditional)
    assert fitted.grouping_variable == "grp"
    assert set(fitted.group_fills) == {"lo", "hi"}

    filled = fitted.transform(train)["bi"].to_numpy()[train["bi"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


# ---------------------------------------------------------------------------
# The fitters read the facts off the routing / recipe, never config (#466)
# ---------------------------------------------------------------------------


def _bimodal_frame(grouped: bool = False) -> pl.DataFrame:
    """A bimodal frame whose NaN holes are real nulls, as a fitter always sees."""
    df = _frame(bimodal=True, grouped=grouped)
    return df.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dtype in df.schema.items()
            if dtype.is_float()
        ]
    )


def _fact_ctx(
    routing: ColumnRouting,
    estimates: ColumnEstimates | None = None,
    hyperparameters: dict | None = None,
) -> UnitFitContext:
    return UnitFitContext(
        column_routings={routing.column: routing},
        column_estimates=(
            {routing.column: estimates} if estimates is not None else {}
        ),
        unit_hyperparameters=hyperparameters or {},
        random_seed=7,
    )


def test_constant_fill_is_read_off_the_routing():
    routing = ColumnRouting(
        column="a",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Constant,
        constant_fill=3.5,
    )
    unit = ImputationUnit(
        unit_id="constant:a", strategy=ImputationStrategy.Constant, columns=("a",)
    )
    outcome = fit_scalar_unit(
        unit, pl.DataFrame({"a": [1.0, None, 2.0]}), _fact_ctx(routing)
    )
    assert isinstance(outcome.fitted, FittedScalar)
    assert outcome.fitted.fill_value == 3.5


def test_constant_fill_in_config_alone_no_longer_reaches_the_fitter():
    """The behaviour ADR-0088 gives up knowingly: config is read at route-time only."""
    routing = ColumnRouting(
        column="a",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Constant,
    )
    unit = ImputationUnit(
        unit_id="constant:a", strategy=ImputationStrategy.Constant, columns=("a",)
    )
    outcome = fit_scalar_unit(
        unit, pl.DataFrame({"a": [1.0, None, 2.0]}), _fact_ctx(routing)
    )
    assert outcome.fitted is None
    assert "constant_fill" in outcome.fallback_reason


def test_gmm_centres_are_read_off_the_recipes_column_estimates():
    routing = ColumnRouting(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.GMMSampling,
    )
    estimates = ColumnEstimates(center1=5.0, center2=40.0)
    unit = ImputationUnit(
        unit_id="gmm_sampling:bi",
        strategy=ImputationStrategy.GMMSampling,
        columns=("bi",),
    )
    df = _bimodal_frame()
    outcome = fit_gmm_unit(unit, df, _fact_ctx(routing, estimates))
    assert isinstance(outcome.fitted, FittedGMMSampling)
    assert abs(outcome.fitted.center1 - outcome.fitted.center2) > 10


def test_gmm_without_centres_on_the_recipe_cannot_train():
    routing = ColumnRouting(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.GMMSampling,
    )
    estimates = ColumnEstimates()  # centres missing
    unit = ImputationUnit(
        unit_id="gmm_sampling:bi",
        strategy=ImputationStrategy.GMMSampling,
        columns=("bi",),
    )
    outcome = fit_gmm_unit(unit, _bimodal_frame(), _fact_ctx(routing, estimates))
    assert outcome.fitted is None
    assert "bimodal centres" in outcome.fallback_reason
    assert "with_estimates" in outcome.fallback_reason


def test_cluster_grouping_variable_is_read_off_the_routing():
    routing = ColumnRouting(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
        grouping_variable="grp",
    )
    estimates = ColumnEstimates(center1=5.0, center2=40.0)
    unit = ImputationUnit(
        unit_id="cluster_conditional:bi",
        strategy=ImputationStrategy.ClusterConditional,
        columns=("bi",),
    )
    outcome = fit_cluster_unit(
        unit,
        _bimodal_frame(grouped=True),
        _fact_ctx(
            routing,
            estimates,
            hyperparameters={"cluster_conditional:bi": {"central_tendency": "median"}},
        ),
    )
    assert isinstance(outcome.fitted, FittedClusterConditional)
    assert outcome.fitted.grouping_variable == "grp"
    assert set(outcome.fitted.group_fills) == {"lo", "hi"}


def test_cluster_feature_cols_are_read_off_the_recipes_column_estimates():
    routing = ColumnRouting(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
    )
    estimates = ColumnEstimates(center1=5.0, center2=40.0, feature_cols=("a", "d"))
    unit = ImputationUnit(
        unit_id="cluster_conditional:bi",
        strategy=ImputationStrategy.ClusterConditional,
        columns=("bi",),
    )
    outcome = fit_cluster_unit(
        unit,
        _bimodal_frame(),
        _fact_ctx(
            routing,
            estimates,
            hyperparameters={"cluster_conditional:bi": {"central_tendency": "median"}},
        ),
    )
    assert isinstance(outcome.fitted, FittedClusterConditional)
    assert outcome.fitted.grouping_variable is None
    assert outcome.fitted.feature_cols == ["a", "d"]
    assert outcome.fitted.feature_centroid_1 is not None
