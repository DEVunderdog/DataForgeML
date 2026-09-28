"""The manual authoring door — ``author()`` builds a real routing (ADR-0090).

Every assertion drives the public API only: ``author`` in, an
``ImputationRouting`` out, and where a routing has to *work* it is worked
through ``resolve_recipe`` → ``derive_units`` → ``fit_unit`` → ``compose`` →
``transform``. The door's promise is that a hand-authored routing is
indistinguishable downstream from a routed one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import dataforge_ml
from dataforge_ml import (
    AuthoredColumn,
    FittedImputer,
    ImputationRouting,
    ImputationStrategy,
    PipelineConfig,
    SemanticType,
    StructuralProfiler,
    UnitNotTrainableError,
    author,
    derive_units,
    fit_unit,
    resolve_recipe,
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


def _profile(df=None, config=None):
    df = df if df is not None else _frame()
    config = config or PipelineConfig()
    return StructuralProfiler(config=config).profile(df)


def _drive(routing, df, config=None):
    """The user-orchestrated loop: route → recipe → train every unit → compose."""
    config = config or PipelineConfig()
    profile = _profile(df, config)
    recipe = resolve_recipe(routing, profile, config)
    units = derive_units(routing)
    results = [fit_unit(recipe, unit, df, random_seed=7) for unit in units]
    return FittedImputer.compose(recipe, results), results


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


def test_names_and_types_alone_are_enough_no_statistics_touched():
    profile = _profile()
    routing = author({"score": ImputationStrategy.Median}, profile=profile)
    assert isinstance(routing, ImputationRouting)
    assert set(routing.column_routings) == set(COLUMNS)


def test_author_takes_no_data_dependent_arguments():
    import inspect

    sig = inspect.signature(author)
    param_names = list(sig.parameters.keys())
    assert param_names == ["columns_map", "profile", "base", "default"]
    assert sig.parameters["profile"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["base"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["default"].kind == inspect.Parameter.KEYWORD_ONLY


def test_unnamed_columns_take_the_default():
    profile = _profile()
    named = {"score": ImputationStrategy.Median}
    routing = author(named, profile=profile, default=ImputationStrategy.Passthrough)

    assert routing.column_routings["score"].strategy == ImputationStrategy.Median
    for col in ("revenue", "rating", "label"):
        assert routing.column_routings[col].strategy == ImputationStrategy.Passthrough


def test_default_is_passthrough():
    routing = author({}, profile=_profile())
    assert all(
        r.strategy == ImputationStrategy.Passthrough
        for r in routing.column_routings.values()
    )


def test_semantic_type_is_read_off_the_profile():
    profile = _profile()
    routing = author({"score": ImputationStrategy.Mean}, profile=profile)
    assert routing.column_routings["score"].semantic_type == SemanticType.Numeric
    assert routing.column_routings["label"].semantic_type != SemanticType.Numeric


def test_signals_are_empty():
    routing = author({"score": ImputationStrategy.Mean}, profile=_profile())
    assert all(r.signals == () for r in routing.column_routings.values())


def test_constant_fill_lands_and_is_applied():
    df = _frame()
    routing = author(
        {"rating": AuthoredColumn(ImputationStrategy.Constant, constant_fill=3.0)},
        profile=_profile(df),
    )
    assert routing.column_routings["rating"].constant_fill == 3.0
    imputer, _ = _drive(routing, df)
    out = imputer.transform(df)
    assert out.dataframe["rating"].null_count() == 0


def test_grouping_variable_lands_on_the_routing():
    routing = author(
        {
            "rating": AuthoredColumn(
                ImputationStrategy.ClusterConditional, grouping_variable="label"
            )
        },
        profile=_profile(),
    )
    assert routing.column_routings["rating"].grouping_variable == "label"


# ---------------------------------------------------------------------------
# Derived, never authored
# ---------------------------------------------------------------------------


def test_mnar_derives_its_flags_and_registers_an_indicator_column():
    routing = author({"score": ImputationStrategy.MNAR}, profile=_profile())

    entry = routing.column_routings["score"]
    assert entry.mnar is True
    assert entry.indicator_flag is True
    assert entry.drop is False

    indicator = routing.column_routings["score_missing"]
    assert indicator.strategy == ImputationStrategy.Indicator
    assert indicator.semantic_type == SemanticType.Boolean


def test_dropped_derives_the_drop_flag():
    routing = author({"rating": ImputationStrategy.Dropped}, profile=_profile())
    assert routing.column_routings["rating"].drop is True


def test_excluded_is_carried_from_the_declaration():
    routing = author(
        {
            "rating": AuthoredColumn(
                strategy=ImputationStrategy.Passthrough, excluded=True
            )
        },
        profile=_profile(),
    )
    assert routing.column_routings["rating"].excluded is True
    assert routing.column_routings["score"].excluded is False


def test_an_excluded_column_is_never_a_knn_predictor():
    df = _frame()
    routing = author(
        {
            "score": ImputationStrategy.KNN,
            "revenue": ImputationStrategy.KNN,
            "rating": AuthoredColumn(
                strategy=ImputationStrategy.Passthrough, excluded=True
            ),
        },
        profile=_profile(df),
    )
    _, results = _drive(routing, df)
    (knn,) = [r.fitted for r in results if r.unit_id == "knn"]
    assert "rating" not in knn.all_cols


def test_non_mnar_columns_get_no_indicator_entry():
    routing = author({"score": ImputationStrategy.Median}, profile=_profile())
    assert "score_missing" not in routing.column_routings


# ---------------------------------------------------------------------------
# What the door refuses, at authoring time, from the names alone
# ---------------------------------------------------------------------------


def test_neither_profile_nor_base_raises():
    with pytest.raises(ValueError, match="exactly one"):
        author({"score": ImputationStrategy.Mean})


def test_both_profile_and_base_raises():
    profile = _profile()
    base = author({}, profile=profile)
    with pytest.raises(ValueError, match="exactly one"):
        author({}, profile=profile, base=base)


def test_map_key_absent_from_the_universe_raises():
    with pytest.raises(ValueError, match="typo"):
        author({"typo": ImputationStrategy.Mean}, profile=_profile())


def test_mice_for_a_single_column_raises():
    with pytest.raises(ValueError, match="score"):
        author({"score": ImputationStrategy.MICE}, profile=_profile())


def test_constant_without_a_fill_raises():
    with pytest.raises(ValueError, match="rating"):
        author({"rating": ImputationStrategy.Constant}, profile=_profile())


def test_excluded_with_a_strategy_other_than_passthrough_raises():
    with pytest.raises(ValueError, match="rating"):
        author(
            {"rating": AuthoredColumn(strategy=ImputationStrategy.Median, excluded=True)},
            profile=_profile(),
        )


def test_indicator_anywhere_in_the_map_raises():
    with pytest.raises(ValueError, match="rating"):
        author({"rating": ImputationStrategy.Indicator}, profile=_profile())


def test_indicator_as_the_default_raises():
    with pytest.raises(ValueError, match="Indicator"):
        author({}, profile=_profile(), default=ImputationStrategy.Indicator)


def test_a_non_numeric_column_declared_anything_but_passthrough_or_dropped_raises():
    with pytest.raises(ValueError, match="label"):
        author({"label": ImputationStrategy.Mode}, profile=_profile())


def test_a_non_numeric_column_may_be_declared_passthrough_or_dropped():
    routing = author(
        {"label": ImputationStrategy.Dropped}, profile=_profile()
    )
    assert routing.column_routings["label"].strategy == ImputationStrategy.Dropped


def test_an_unknown_strategy_name_raises():
    with pytest.raises(ValueError, match="score"):
        author({"score": "interpolate"}, profile=_profile())


def test_an_unrecognised_value_type_raises_type_error():
    with pytest.raises(TypeError, match="score"):
        author({"score": 42}, profile=_profile())


# ---------------------------------------------------------------------------
# Re-authoring: base=
# ---------------------------------------------------------------------------


def test_reauthoring_carries_unnamed_columns_verbatim():
    profile = _profile()
    base = author({"score": ImputationStrategy.Median}, profile=profile)
    edited = author({"revenue": ImputationStrategy.Mean}, base=base)

    assert edited.column_routings["score"] is base.column_routings["score"]
    assert edited.column_routings["revenue"].strategy == ImputationStrategy.Mean


def test_reauthoring_an_unknown_column_raises():
    base = author({}, profile=_profile())
    with pytest.raises(ValueError, match="typo"):
        author({"typo": ImputationStrategy.Mean}, base=base)


def test_reauthoring_replaces_indicator_entries():
    profile = _profile()
    base = author({"score": ImputationStrategy.MNAR}, profile=profile)
    assert "score_missing" in base.column_routings

    edited = author({"score": ImputationStrategy.Median}, base=base)
    assert "score_missing" not in edited.column_routings


def test_reauthoring_non_numeric_column_with_invalid_strategy_raises():
    profile = _profile()
    base = author({"score": ImputationStrategy.Median}, profile=profile)
    with pytest.raises(ValueError, match="label"):
        author({"label": ImputationStrategy.Mean}, base=base)


def test_reauthoring_carries_mice_estimator_when_mice_columns_remain():
    from sklearn.linear_model import Ridge

    from dataforge_ml.imputation._config import ModelChoice

    profile = _profile()
    base = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=profile,
    ).with_model_choice(ModelChoice.RandomForestRegressor)

    # Edit an unrelated column: MICE columns remain
    edited = author({"rating": ImputationStrategy.Mean}, base=base)
    assert edited.mice_model_choice == ModelChoice.RandomForestRegressor
    assert edited.mice_estimator is None

    # Custom estimator instance carries by identity
    estimator = Ridge()
    base_custom = base.with_model_choice(estimator)
    edited_custom = author({"rating": ImputationStrategy.Mean}, base=base_custom)
    assert edited_custom.mice_model_choice == ModelChoice.Custom
    assert edited_custom.mice_estimator is estimator


def test_reauthoring_creates_mice_block_stamped_bayesian_ridge():
    from dataforge_ml.imputation._config import ModelChoice

    profile = _profile()
    base = author({"score": ImputationStrategy.Median}, profile=profile)
    assert base.mice_model_choice is None

    # Edit creates the MICE block
    edited = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        base=base,
    )
    assert edited.mice_model_choice == ModelChoice.BayesianRidge
    assert edited.mice_estimator is None


def test_reauthoring_dissolves_mice_block_clears_estimator():
    from dataforge_ml.imputation._config import ModelChoice

    profile = _profile()
    base = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=profile,
    ).with_model_choice(ModelChoice.RandomForestRegressor)

    # Edit dissolves the MICE block
    edited = author(
        {"score": ImputationStrategy.Median, "revenue": ImputationStrategy.Mean},
        base=base,
    )
    assert edited.mice_model_choice is None
    assert edited.mice_estimator is None


def test_base_edited_routing_round_trips_through_resolve_recipe():
    from dataforge_ml import route

    df = _frame()
    profile = _profile(df)
    base = route(profile)

    # Re-author one column
    edited = author({"revenue": ImputationStrategy.Median}, base=base)
    recipe = resolve_recipe(edited, profile, PipelineConfig())

    assert recipe.routing is edited
    assert (
        recipe.routing.column_routings["revenue"].strategy
        == ImputationStrategy.Median
    )
    # Unnamed columns keep base ColumnRouting verbatim, signals included
    for col, r in base.column_routings.items():
        if col not in ("revenue", "revenue_missing"):
            assert recipe.routing.column_routings[col] is r


# ---------------------------------------------------------------------------
# The routing trains, transforms end to end
# ---------------------------------------------------------------------------


def test_a_hand_authored_cluster_conditional_fills_its_nulls():
    df = _frame()
    profile = _profile(df)
    routing = author(
        {
            "rating": ImputationStrategy.ClusterConditional,
            "score": ImputationStrategy.Median,
            "revenue": ImputationStrategy.Median,
        },
        profile=profile,
    )
    assert df["rating"].null_count() > 0

    imputer, _ = _drive(routing, df)
    out = imputer.transform(df)
    assert out.dataframe["rating"].null_count() == 0


def test_passthrough_leaves_a_column_and_its_nulls_alone():
    df = _frame()
    routing = author({}, profile=_profile(df))
    imputer, _ = _drive(routing, df)
    out = imputer.transform(df)
    assert out.dataframe.equals(df)


def test_mnar_end_to_end_appends_the_indicator_column():
    df = _frame()
    routing = author({"score": ImputationStrategy.MNAR}, profile=_profile(df))
    imputer, _ = _drive(routing, df)
    out = imputer.transform(df)

    assert "score_missing" in out.dataframe.columns
    assert out.dataframe["score"].null_count() == df["score"].null_count()
    assert out.dataframe["score_missing"].sum() == df["score"].null_count()


def test_a_bare_mice_declaration_structurally_survives_authoring_but_cannot_fit():
    """MICE may be declared at the door (>=2 columns) with no model choice —
    author() resolves no Estimator Ladder pick (ADR-0090), so the block
    cannot train until with_model_choice sets one explicitly."""
    routing = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=_profile(),
    )
    assert routing.mice_model_choice is None
    df = _frame()
    recipe = resolve_recipe(routing, _profile(df), PipelineConfig())
    (unit,) = derive_units(routing, strategy=ImputationStrategy.MICE)
    with pytest.raises(UnitNotTrainableError):
        fit_unit(recipe, unit, df)


def test_with_model_choice_sets_a_library_estimator_family():
    from dataforge_ml.imputation._config import ModelChoice

    routing = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=_profile(),
    )
    updated = routing.with_model_choice(ModelChoice.GradientBoostingRegressor)
    assert updated.mice_model_choice == ModelChoice.GradientBoostingRegressor
    assert updated.mice_estimator is None
    # The original is untouched — with_model_choice returns a new routing.
    assert routing.mice_model_choice is None


def test_with_model_choice_sets_a_custom_estimator_instance():
    from sklearn.linear_model import Ridge

    from dataforge_ml.imputation._config import ModelChoice

    routing = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=_profile(),
    )
    estimator = Ridge()
    updated = routing.with_model_choice(estimator)
    assert updated.mice_model_choice == ModelChoice.Custom
    assert updated.mice_estimator is estimator


def test_with_model_choice_raises_on_bare_custom_label():
    from dataforge_ml.imputation._config import ModelChoice

    routing = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=_profile(),
    )
    with pytest.raises(ValueError, match="Custom"):
        routing.with_model_choice(ModelChoice.Custom)


def test_with_model_choice_raises_when_routing_has_no_mice_block():
    from dataforge_ml.imputation._config import ModelChoice

    routing = author({"score": ImputationStrategy.Median}, profile=_profile())
    with pytest.raises(ValueError, match="MICE"):
        routing.with_model_choice(ModelChoice.RandomForestRegressor)


def test_with_model_choice_works_identically_on_routed_and_authored_routings():
    from sklearn.linear_model import Ridge

    from dataforge_ml import route
    from dataforge_ml.imputation._config import ModelChoice

    df = _frame()
    profile = _profile(df)

    # 1. Routings with MICE block: authored vs routed
    authored_mice = author(
        {"score": ImputationStrategy.MICE, "revenue": ImputationStrategy.MICE},
        profile=profile,
    )
    cfg_mice = PipelineConfig()
    cfg_mice.imputation.numeric.set_per_column_strategy(
        ["score", "revenue"], ImputationStrategy.MICE
    )
    routed_mice = route(profile, cfg_mice)

    for r in (authored_mice, routed_mice):
        # Setting a library family produces a new routing with choice set
        updated_lib = r.with_model_choice(ModelChoice.GradientBoostingRegressor)
        assert updated_lib.mice_model_choice == ModelChoice.GradientBoostingRegressor
        assert updated_lib.mice_estimator is None
        assert r.mice_model_choice != ModelChoice.GradientBoostingRegressor

        # Setting a custom estimator instance sets ModelChoice.Custom and holds instance by identity
        est = Ridge()
        updated_custom = r.with_model_choice(est)
        assert updated_custom.mice_model_choice == ModelChoice.Custom
        assert updated_custom.mice_estimator is est
        assert r.mice_estimator is None

        # Bare ModelChoice.Custom raises ValueError
        with pytest.raises(ValueError, match="Custom"):
            r.with_model_choice(ModelChoice.Custom)

    # 2. Routings with NO MICE block: authored vs routed
    authored_no_mice = author({"score": ImputationStrategy.Median}, profile=profile)
    routed_no_mice = route(profile)

    for r in (authored_no_mice, routed_no_mice):
        with pytest.raises(ValueError, match="MICE"):
            r.with_model_choice(ModelChoice.RandomForestRegressor)
        with pytest.raises(ValueError, match="MICE"):
            r.with_model_choice(Ridge())


def test_hand_authored_routing_with_forced_mice_and_bimodal_estimates_end_to_end():
    from sklearn.linear_model import Ridge

    from dataforge_ml.imputation._config import ModelChoice

    df = _frame()
    profile = _profile(df)

    # score is normally distributed in _frame(), so center1/center2 will be None
    assert profile.columns["score"].stats.bimodal_stats is None

    # Hand-author routing with MICE and GMMSampling
    routing = author(
        {
            "rating": ImputationStrategy.MICE,
            "revenue": ImputationStrategy.MICE,
            "score": ImputationStrategy.GMMSampling,
        },
        profile=profile,
    )
    # Set forced MICE estimator via with_model_choice
    custom_estimator = Ridge()
    routing = routing.with_model_choice(custom_estimator)
    assert routing.mice_model_choice == ModelChoice.Custom
    assert routing.mice_estimator is custom_estimator

    # resolve_recipe leaves bimodal centres as None
    recipe = resolve_recipe(routing, profile, PipelineConfig())
    assert recipe.column_estimates["score"].center1 is None
    assert recipe.column_estimates["score"].center2 is None

    units = derive_units(recipe.routing)
    gmm_unit = next(u for u in units if u.strategy == ImputationStrategy.GMMSampling)

    # Attempting to fit before supplying estimates raises UnitNotTrainableError
    with pytest.raises(UnitNotTrainableError) as exc_info:
        fit_unit(recipe, gmm_unit, df)
    assert "with_estimates" in exc_info.value.reason

    # Supply the missing bimodal centres via with_estimates
    recipe = recipe.with_estimates("score", center1=45.0, center2=55.0)
    assert recipe.column_estimates["score"].center1 == 45.0
    assert recipe.column_estimates["score"].center2 == 55.0

    # Now fits successfully end to end
    results = [fit_unit(recipe, unit, df, random_seed=42) for unit in units]
    imputer = FittedImputer.compose(recipe, results)
    out = imputer.transform(df)

    assert out.dataframe["score"].null_count() == 0
    assert out.dataframe["revenue"].null_count() == 0
    assert out.dataframe["rating"].null_count() == 0
    assert out.dataframe.shape == df.shape

