"""The manual authoring door — ``author()`` builds a real plan (#468/#469, ADR-0083).

Every assertion drives the public API only: ``author`` in, an
:class:`ImputationDecision` out, and where a plan has to *work* it is worked
through ``fit_unit`` → ``compose`` → ``transform``. Nothing here reaches for
``_derive_units``, a fitter internal, or the dial table's identity — the door's
promise is that a hand-authored plan is indistinguishable downstream, and that
is only testable from downstream.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.tree import DecisionTreeRegressor

import dataforge_ml
from dataforge_ml import (
    AuthoredColumn,
    FittedImputer,
    ImputationDecision,
    ImputationStrategy,
    ModelChoice,
    SemanticType,
    author,
    core_budget,
    fit_unit,
)

COLUMNS = ["score", "revenue", "rating", "label"]


def _frame(n=240, seed=11):
    """Two correlated numeric columns, a third numeric one, and a string column."""
    rng = np.random.default_rng(seed)
    score = rng.normal(50.0, 10.0, n)
    revenue = score * 4.0 + rng.normal(0.0, 5.0, n)
    rating = np.clip(np.round(rng.normal(3.0, 1.0, n)), 1.0, 5.0)

    df = pl.DataFrame(
        {
            "score": pl.Series(score, dtype=pl.Float64),
            "revenue": pl.Series(revenue, dtype=pl.Float64),
            "rating": pl.Series(rating, dtype=pl.Float64),
            "label": pl.Series(["A" if i % 2 else "B" for i in range(n)]),
        }
    )
    idx = pl.arange(0, n, eager=True)
    return df.with_columns(
        pl.when(idx % 9 == 0).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(idx % 7 == 0).then(None).otherwise(pl.col("revenue")).alias("revenue"),
        pl.when(idx % 11 == 0).then(None).otherwise(pl.col("rating")).alias("rating"),
    )


def _drive(plan, df):
    """The user-orchestrated loop: train every planned unit, then compose."""
    results = [fit_unit(plan, unit, df, random_seed=7) for unit in plan.units]
    return FittedImputer.compose(plan, results), results


# ---------------------------------------------------------------------------
# The door is part of the Public API (ADR-0050)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["author", "AuthoredColumn"])
def test_exported_from_package_root_and_subpackage(name):
    from dataforge_ml import imputation

    assert name in dataforge_ml.__all__
    assert name in imputation.__all__
    assert getattr(dataforge_ml, name) is getattr(imputation, name)


# ---------------------------------------------------------------------------
# What the door builds
# ---------------------------------------------------------------------------


def test_names_alone_are_enough_no_data_anywhere_in_the_call():
    plan = author(
        {"score": ImputationStrategy.Median},
        columns=["score", "revenue"],
    )
    assert isinstance(plan, ImputationDecision)
    assert set(plan.column_decisions) == {"score", "revenue"}


def test_unnamed_columns_take_the_default():
    columns = [f"c{i}" for i in range(40)]
    named = {c: ImputationStrategy.Median for c in columns[:3]}
    plan = author(named, columns=columns, default=ImputationStrategy.Mean)

    assert [c for c in plan.column_decisions] == columns
    for col in columns[:3]:
        assert plan.column_decisions[col].strategy == ImputationStrategy.Median
    others = [plan.column_decisions[c].strategy for c in columns[3:]]
    assert others == [ImputationStrategy.Mean] * 37


def test_default_is_passthrough():
    plan = author({}, columns=["a", "b"])
    assert all(
        d.strategy == ImputationStrategy.Passthrough
        for d in plan.column_decisions.values()
    )


def test_semantic_type_is_stamped_numeric():
    plan = author({"score": ImputationStrategy.Mean}, columns=COLUMNS)
    assert plan.column_decisions["score"].semantic_type == SemanticType.Numeric


def test_signals_are_empty_and_config_snapshot_may_be_empty():
    plan = author({"score": ImputationStrategy.Mean}, columns=COLUMNS)
    assert all(d.signals == () for d in plan.column_decisions.values())
    assert plan.config_snapshot == {}


def test_authored_column_facts_land_on_the_decision():
    plan = author(
        {
            "rating": AuthoredColumn(
                strategy=ImputationStrategy.ClusterConditional,
                center1=1.0,
                center2=5.0,
                feature_cols=("score", "revenue"),
                domain_snap_bounds=(1.0, 5.0),
            )
        },
        columns=COLUMNS,
    )
    decision = plan.column_decisions["rating"]
    assert decision.center1 == 1.0
    assert decision.center2 == 5.0
    assert decision.feature_cols == ("score", "revenue")
    assert decision.domain_snap_bounds == (1.0, 5.0)


def test_constant_fill_lands_and_is_applied():
    plan = author(
        {"rating": AuthoredColumn(ImputationStrategy.Constant, constant_fill=3.0)},
        columns=COLUMNS,
    )
    assert plan.column_decisions["rating"].constant_fill == 3.0
    imputer, _ = _drive(plan, _frame())
    out = imputer.transform(_frame())
    assert out.dataframe["rating"].null_count() == 0


# ---------------------------------------------------------------------------
# Derived, never authored
# ---------------------------------------------------------------------------


def test_mnar_derives_its_flags_and_registers_an_indicator_column():
    plan = author({"score": ImputationStrategy.MNAR}, columns=COLUMNS)

    decision = plan.column_decisions["score"]
    assert decision.mnar is True
    assert decision.indicator_flag is True
    assert decision.drop is False

    indicator = plan.column_decisions["score_missing"]
    assert indicator.strategy == ImputationStrategy.Indicator
    assert indicator.semantic_type == SemanticType.Boolean


def test_dropped_derives_the_drop_flag():
    plan = author({"rating": ImputationStrategy.Dropped}, columns=COLUMNS)
    assert plan.column_decisions["rating"].drop is True
    assert plan.dropped_columns == ("rating",)


def test_non_mnar_columns_get_no_indicator_entry():
    plan = author({"score": ImputationStrategy.Median}, columns=COLUMNS)
    assert "score_missing" not in plan.column_decisions


# ---------------------------------------------------------------------------
# What the door refuses, at authoring time, from the names alone
# ---------------------------------------------------------------------------


def test_map_key_absent_from_columns_raises():
    with pytest.raises(ValueError, match="typo"):
        author({"typo": ImputationStrategy.Mean}, columns=COLUMNS)


def test_mice_for_a_single_column_raises():
    with pytest.raises(ValueError, match="score"):
        author({"score": ImputationStrategy.MICE}, columns=COLUMNS)


def test_constant_without_a_fill_raises():
    with pytest.raises(ValueError, match="rating"):
        author({"rating": ImputationStrategy.Constant}, columns=COLUMNS)


@pytest.mark.parametrize(
    "strategy",
    [ImputationStrategy.GMMSampling, ImputationStrategy.ClusterConditional],
)
@pytest.mark.parametrize("centres", [{}, {"center1": 1.0}, {"center2": 5.0}])
def test_bimodal_without_both_centres_raises(strategy, centres):
    with pytest.raises(ValueError, match="rating"):
        author(
            {"rating": AuthoredColumn(strategy, grouping_variable="label", **centres)},
            columns=COLUMNS,
        )


@pytest.mark.parametrize("feature_cols", [None, ()])
def test_cluster_conditional_without_a_partition_raises(feature_cols):
    with pytest.raises(ValueError, match="rating"):
        author(
            {
                "rating": AuthoredColumn(
                    ImputationStrategy.ClusterConditional,
                    center1=1.0,
                    center2=5.0,
                    feature_cols=feature_cols,
                )
            },
            columns=COLUMNS,
        )


def test_indicator_anywhere_in_the_map_raises():
    with pytest.raises(ValueError, match="rating"):
        author({"rating": ImputationStrategy.Indicator}, columns=COLUMNS)


def test_indicator_as_the_default_raises():
    with pytest.raises(ValueError, match="Indicator"):
        author({}, columns=COLUMNS, default=ImputationStrategy.Indicator)


def test_a_non_numeric_semantic_type_raises():
    # A column decision is AuthoredColumn-shaped, so it is the one thing a user
    # can hand the door that carries a semantic type of its own.
    foreign = dataforge_ml.ColumnImputationDecision(
        column="label",
        semantic_type=SemanticType.Categorical,
        strategy=ImputationStrategy.Mode,
    )
    with pytest.raises(ValueError, match="label"):
        author({"label": foreign}, columns=COLUMNS)


def test_an_unknown_strategy_name_raises():
    with pytest.raises(ValueError, match="score"):
        author({"score": "interpolate"}, columns=COLUMNS)


# ---------------------------------------------------------------------------
# The decided base, and the dial grammar on top of it
# ---------------------------------------------------------------------------


def test_with_hyperparameters_works_on_a_hand_authored_plan():
    plan = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        columns=COLUMNS,
    )
    edited = plan.with_hyperparameters("mice", {"max_iter": 3})
    hyp = dict(next(u for u in edited.units if u.unit_id == "mice").hyperparameters)

    assert hyp["max_iter"] == 3
    # The base is complete: the keys the author did not name survive the edit.
    assert set(hyp) == {"max_iter", "tol", "initial_strategy", "n_nearest_features"}


def test_an_undialable_key_raises_like_it_does_on_a_decided_plan():
    plan = author(
        {"score": ImputationStrategy.KNN, "revenue": ImputationStrategy.KNN},
        columns=COLUMNS,
    )
    with pytest.raises(ValueError, match="max_iter"):
        plan.with_hyperparameters("knn", {"max_iter": 3})


def test_the_decided_base_survives_a_round_trip():
    plan = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        columns=COLUMNS,
    )
    assert ImputationDecision.from_dict(plan.to_dict()) == plan


# ---------------------------------------------------------------------------
# The plan trains, transforms, and prices
# ---------------------------------------------------------------------------


def test_a_bare_mice_plan_resolves_bayesian_ridge_and_trains():
    plan = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        columns=COLUMNS,
    )
    assert plan.column_decisions["score"].model_choice == ModelChoice.BayesianRidge

    df = _frame()
    imputer, results = _drive(plan, df)
    mice = next(r for r in results if r.unit_id == "mice")
    assert mice.signals.estimator == "Pipeline(StandardScaler+BayesianRidge)"

    out = imputer.transform(df)
    assert out.dataframe["score"].null_count() == 0
    assert out.dataframe["revenue"].null_count() == 0


def test_a_hand_authored_cluster_conditional_fills_its_nulls():
    # The silent no-op the door's authoring-time check closes: a unit with no
    # partition fits cleanly and fills zero cells. This one has centres and
    # features, so it must actually fill.
    plan = author(
        {
            "rating": AuthoredColumn(
                ImputationStrategy.ClusterConditional,
                center1=2.0,
                center2=4.0,
                feature_cols=("score", "revenue"),
            ),
            "score": ImputationStrategy.Median,
            "revenue": ImputationStrategy.Median,
        },
        columns=COLUMNS,
    )
    df = _frame()
    assert df["rating"].null_count() > 0

    imputer, _ = _drive(plan, df)
    out = imputer.transform(df)
    assert out.dataframe["rating"].null_count() == 0


def test_passthrough_leaves_a_column_and_its_nulls_alone():
    plan = author({}, columns=COLUMNS)
    df = _frame()
    imputer, _ = _drive(plan, df)
    out = imputer.transform(df)
    assert out.dataframe.equals(df)


def test_sentinels_normalise_including_on_a_passthrough_column():
    df = _frame().with_columns(
        pl.col("rating").fill_null(-999.0),
        pl.col("score").fill_null(-999.0),
    )
    plan = author(
        {"score": ImputationStrategy.Median},
        columns=COLUMNS,
        numeric_sentinels={"score": [-999.0], "rating": [-999.0]},
    )
    assert plan.numeric_sentinels == {"score": [-999.0], "rating": [-999.0]}

    imputer, _ = _drive(plan, df)
    out = imputer.transform(df)

    # The imputed column's sentinels became fills, not observations.
    assert -999.0 not in out.dataframe["score"].to_list()
    assert out.dataframe["score"].null_count() == 0
    # The Passthrough column's sentinels became nulls and were then left alone.
    assert -999.0 not in out.dataframe["rating"].to_list()
    assert out.dataframe["rating"].null_count() > 0


def test_a_scalar_fill_is_learned_from_sentinel_normalised_data():
    df = _frame().with_columns(pl.col("score").fill_null(-999.0))
    plan = author(
        {"score": ImputationStrategy.Mean},
        columns=COLUMNS,
        numeric_sentinels={"score": [-999.0]},
    )
    imputer, _ = _drive(plan, df)
    assert 20.0 < imputer.records["score"].fill_value < 80.0


def test_core_budget_prices_a_hand_authored_plan():
    plan = author(
        {
            "score": ImputationStrategy.MICE,
            "revenue": ImputationStrategy.MICE,
            "rating": ImputationStrategy.Median,
        },
        columns=COLUMNS,
    )
    budget = core_budget(plan, max_workers=1, total_cores=8)
    assert set(budget) == {u.unit_id for u in plan.units}
    assert budget["mice"] == -1
    assert budget["median:rating"] == 1


# ---------------------------------------------------------------------------
# The estimators channel — a foreign estimator on the plan (#469, ADR-0083)
# ---------------------------------------------------------------------------


def _mice_plan(**kwargs):
    return author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        columns=COLUMNS,
        **kwargs,
    )


def test_a_supplied_estimator_stamps_custom_on_every_column_of_the_unit():
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    plan = _mice_plan(estimators={"mice": model})

    assert plan.column_decisions["score"].model_choice == ModelChoice.Custom
    assert plan.column_decisions["revenue"].model_choice == ModelChoice.Custom
    # Columns outside the unit are untouched.
    assert plan.column_decisions["rating"].model_choice is None


def test_the_plan_holds_the_callers_object_itself_never_a_clone():
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    plan = _mice_plan(estimators={"mice": model})

    assert plan.custom_estimators["mice"] is model


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p.with_hyperparameters("mice", {"max_iter": 4}),
        lambda p: p.with_model_choice("rating", ModelChoice.BayesianRidge),
    ],
    ids=["with_hyperparameters", "with_model_choice"],
)
def test_identity_survives_every_derived_copy(edit):
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    plan = _mice_plan(estimators={"mice": model})

    assert edit(plan).custom_estimators["mice"] is model


def test_the_unit_itself_carries_no_estimator():
    """``ImputationUnit`` stays estimator-free, so a two-estimator block is
    structurally impossible rather than checked (ADR-0083)."""
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    plan = _mice_plan(estimators={"mice": model})
    unit = next(u for u in plan.units if u.unit_id == "mice")

    assert not any(
        getattr(unit, name) is model for name in unit.__dataclass_fields__
    )


def test_the_supplied_estimator_is_what_actually_trains():
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    plan = _mice_plan(estimators={"mice": model})

    df = _frame()
    imputer, results = _drive(plan, df)
    mice = next(r for r in results if r.unit_id == "mice")

    assert mice.signals.estimator == "DecisionTreeRegressor"
    out = imputer.transform(df)
    assert out.dataframe["score"].null_count() == 0
    assert out.dataframe["revenue"].null_count() == 0


def test_an_already_fitted_estimator_is_accepted_without_raise_or_warning():
    fitted = DecisionTreeRegressor(max_depth=3, random_state=0)
    fitted.fit(np.arange(20.0).reshape(-1, 1), np.arange(20.0))
    assert hasattr(fitted, "tree_")  # it really is fitted

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = _mice_plan(estimators={"mice": fitted})

    # And it trains: sklearn refits from scratch, so the prior state is inert.
    df = _frame()
    imputer, _ = _drive(plan, df)
    assert imputer.transform(df).dataframe["score"].null_count() == 0


def test_an_unknown_unit_id_raises():
    with pytest.raises(ValueError, match="no such unit"):
        _mice_plan(estimators={"knn": DecisionTreeRegressor()})


def test_a_unit_with_no_estimator_slot_raises():
    with pytest.raises(ValueError, match="trains no estimator"):
        author(
            {"score": ImputationStrategy.Median},
            columns=COLUMNS,
            estimators={"median:score": DecisionTreeRegressor()},
        )


def test_a_none_estimator_raises_rather_than_stamping_an_empty_slot():
    with pytest.raises(ValueError, match="no estimator"):
        _mice_plan(estimators={"mice": None})


def test_core_budget_prices_a_custom_unit_at_one_in_every_branch():
    plan = _mice_plan(estimators={"mice": DecisionTreeRegressor()})

    for max_workers in (1, 4, None):
        budget = core_budget(plan, max_workers=max_workers, total_cores=12)
        assert budget["mice"] == 1, max_workers


def test_mnar_end_to_end_appends_the_indicator_column():
    plan = author(
        {"score": ImputationStrategy.MNAR},
        columns=COLUMNS,
    )
    df = _frame()
    imputer, _ = _drive(plan, df)
    out = imputer.transform(df)

    assert "score_missing" in out.dataframe.columns
    assert out.dataframe["score"].null_count() == 0
    assert out.dataframe["score_missing"].sum() == df["score"].null_count()
