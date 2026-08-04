"""
Unit tests for the ImputationDecision plan-edit API (ADR-0060, ADR-0082).

Covers ``with_model_choice`` / ``with_hyperparameters`` — the *dial* edits that
survive ADR-0082: immutable edits returning a new valid plan, and unit
re-derivation on the returned plan. Strategy is no longer editable on a built
plan; declaring one is ``per_column_strategy``'s job and is covered by the
config and assembler suites. Plans are built directly from
``ColumnImputationDecision`` entries so the edit methods (not the assembler) are
under test.
"""

import pytest
from sklearn.tree import DecisionTreeRegressor

from dataforge_ml import ModelChoice
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import (
    ColumnImputationDecision,
    ImputationDecision,
    ImputationStrategy,
    author,
)

S = ImputationStrategy


def _plan(*decisions: ColumnImputationDecision) -> ImputationDecision:
    return ImputationDecision(
        column_decisions={d.column: d for d in decisions},
        config_snapshot={},
    )


def _numeric(column: str, strategy: ImputationStrategy, **kw) -> ColumnImputationDecision:
    return ColumnImputationDecision(
        column=column, semantic_type=SemanticType.Numeric, strategy=strategy, **kw
    )


# ---------------------------------------------------------------------------
# Immutability — a new plan is returned; the original is unchanged
# ---------------------------------------------------------------------------


def test_with_model_choice_returns_new_plan_original_unchanged() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.BayesianRidge))
    edited = plan.with_model_choice("a", ModelChoice.RandomForestRegressor)
    assert edited is not plan
    assert plan.column_decisions["a"].model_choice == ModelChoice.BayesianRidge
    assert edited.column_decisions["a"].model_choice == ModelChoice.RandomForestRegressor


def test_strategy_is_not_editable_on_a_built_plan() -> None:
    """Structural edits belong to ``decide`` alone (ADR-0082).

    Guards the collapse: a strategy edit re-derives which units exist, so it can
    strand a unit with no decided hyperparameter base. The only declaration
    surface is ``per_column_strategy``, consumed before routing.
    """
    plan = _plan(_numeric("a", S.Median))
    assert not hasattr(plan, "with_strategy")


def test_user_declared_strategy_leaves_only_a_signal() -> None:
    """Forcing a strategy is prose on the plan, never a machine predicate.

    ``forced`` is deleted (ADR-0083), so a declared strategy is distinguishable
    from an auto-routed one only by the router's signal, which nothing parses.
    """
    plan = _plan(_numeric("a", S.MICE, signals=("per_column_strategy_override: x",)))
    restored = ColumnImputationDecision.from_dict(plan.column_decisions["a"].to_dict())
    assert restored.signals == ("per_column_strategy_override: x",)
    assert not hasattr(restored, "forced")


def test_with_model_choice_accepts_none_and_string() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.BayesianRidge))
    assert plan.with_model_choice("a", None).column_decisions["a"].model_choice is None
    assert (
        plan.with_model_choice("a", "random_forest_regressor").column_decisions["a"].model_choice
        == ModelChoice.RandomForestRegressor
    )


# ---------------------------------------------------------------------------
# units re-derive on the returned plan
# ---------------------------------------------------------------------------


def test_units_are_derived_from_the_column_map() -> None:
    """``units`` is always a projection of ``column_decisions`` (ADR-0060)."""
    plan = _plan(_numeric("a", S.MICE), _numeric("b", S.MICE), _numeric("c", S.Median))
    assert {u.unit_id for u in plan.units} == {"mice", "median:c"}
    mice_unit = next(u for u in plan.units if u.unit_id == "mice")
    assert mice_unit.columns == ("a", "b")


def test_with_model_choice_edit_visible_on_unit_owner() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.BayesianRidge))
    edited = plan.with_model_choice("a", ModelChoice.RandomForestRegressor)
    # the estimator family is reachable through the plan the unit belongs to
    assert edited.units[0].unit_id == "mice"
    assert edited.column_decisions["a"].model_choice == ModelChoice.RandomForestRegressor


# ---------------------------------------------------------------------------
# Unknown column
# ---------------------------------------------------------------------------


def test_with_model_choice_unknown_column_raises_keyerror() -> None:
    plan = _plan(_numeric("a", S.MICE))
    with pytest.raises(KeyError):
        plan.with_model_choice("nope", ModelChoice.BayesianRidge)


# ---------------------------------------------------------------------------
# with_hyperparameters — per-key merge onto the decided base (ADR-0073)
# ---------------------------------------------------------------------------


def _mice_plan() -> ImputationDecision:
    """A two-column MICE plan whose ``"mice"`` unit carries a complete base."""
    return ImputationDecision(
        column_decisions={
            d.column: d for d in (_numeric("a", S.MICE), _numeric("b", S.MICE))
        },
        config_snapshot={},
        decided_hyperparameters={
            "mice": (
                ("max_iter", 10),
                ("tol", 1e-3),
                ("initial_strategy", "mean"),
            )
        },
    )


def _unit_hyp(plan: ImputationDecision, unit_id: str) -> dict:
    unit = next(u for u in plan.units if u.unit_id == unit_id)
    return dict(unit.hyperparameters or ())


def test_with_hyperparameters_overrides_only_named_keys() -> None:
    plan = _mice_plan()
    edited = plan.with_hyperparameters("mice", {"max_iter": 50})

    merged = _unit_hyp(edited, "mice")
    # named key overridden
    assert merged["max_iter"] == 50
    # every other decided dial kept
    assert merged["tol"] == 1e-3
    assert merged["initial_strategy"] == "mean"


def test_with_hyperparameters_returns_new_plan_original_unchanged() -> None:
    plan = _mice_plan()
    edited = plan.with_hyperparameters("mice", {"max_iter": 50})
    assert edited is not plan
    assert _unit_hyp(plan, "mice")["max_iter"] == 10
    assert _unit_hyp(edited, "mice")["max_iter"] == 50


def test_with_hyperparameters_none_resets_to_decided_base() -> None:
    plan = _mice_plan()
    edited = plan.with_hyperparameters("mice", {"max_iter": 50})
    reset = edited.with_hyperparameters("mice", None)

    assert _unit_hyp(reset, "mice") == {
        "max_iter": 10,
        "tol": 1e-3,
        "initial_strategy": "mean",
    }
    assert reset.override_hyperparameters == {}


def test_with_hyperparameters_successive_edits_accumulate_per_key() -> None:
    plan = _mice_plan()
    edited = plan.with_hyperparameters("mice", {"max_iter": 50}).with_hyperparameters(
        "mice", {"tol": 1e-5}
    )

    merged = _unit_hyp(edited, "mice")
    assert merged["max_iter"] == 50
    assert merged["tol"] == 1e-5
    assert merged["initial_strategy"] == "mean"


def test_with_hyperparameters_unknown_key_raises_naming_unit_and_key() -> None:
    plan = _mice_plan()
    with pytest.raises(ValueError) as err:
        plan.with_hyperparameters("mice", {"nope": 1})
    assert "mice" in str(err.value)
    assert "nope" in str(err.value)


@pytest.mark.parametrize(
    ("strategy", "key"),
    [
        (S.GMMSampling, "center1"),
        (S.GMMSampling, "central_tendency"),
        (S.Constant, "fill_value"),
    ],
)
def test_with_hyperparameters_raises_on_every_key_for_a_dialless_unit(
    strategy: ImputationStrategy, key: str
) -> None:
    """A strategy with no dial-table row has nothing to dial (#464).

    Both bimodal centres and a declared constant are facts, not dials — the
    decided base may carry them for the fitter to read, but the override schema
    is the dial table, so every key is rejected for these units.
    """
    plan = ImputationDecision(
        column_decisions={"a": _numeric("a", strategy)},
        config_snapshot={},
        decided_hyperparameters={
            f"{strategy}:a": (("center1", 1.0), ("central_tendency", "median"))
        },
    )
    with pytest.raises(ValueError, match=key):
        plan.with_hyperparameters(f"{strategy}:a", {key: 2.0})


def test_with_hyperparameters_rejects_a_base_key_that_is_not_a_dial() -> None:
    """The dial table is the override schema, not the decided base (#464).

    A Cluster-Conditional unit carries its centres alongside its one dial;
    ``central_tendency`` is dialable and ``center1`` is not.
    """
    plan = ImputationDecision(
        column_decisions={"a": _numeric("a", S.ClusterConditional)},
        config_snapshot={},
        decided_hyperparameters={
            "cluster_conditional:a": (
                ("central_tendency", "median"),
                ("center1", 1.0),
            )
        },
    )
    assert (
        _unit_hyp(
            plan.with_hyperparameters("cluster_conditional:a", {"central_tendency": "mean"}),
            "cluster_conditional:a",
        )["central_tendency"]
        == "mean"
    )
    with pytest.raises(ValueError, match="center1"):
        plan.with_hyperparameters("cluster_conditional:a", {"center1": 2.0})


def test_with_hyperparameters_unknown_unit_raises_keyerror() -> None:
    plan = _mice_plan()
    with pytest.raises(KeyError):
        plan.with_hyperparameters("knn", {"n_neighbors": 3})


def test_with_hyperparameters_override_survives_save_load_round_trip() -> None:
    plan = _mice_plan().with_hyperparameters("mice", {"max_iter": 50})
    restored = ImputationDecision.from_dict(plan.to_dict())
    assert restored.override_hyperparameters == plan.override_hyperparameters
    assert _unit_hyp(restored, "mice")["max_iter"] == 50


# ---------------------------------------------------------------------------
# Re-authoring — author(map, base=plan) (#470)
#
# The structural edit ADR-0082 removed, returned as a door that re-derives what
# ``with_strategy`` left stale. Carry-over is ruled field by field, so it is
# tested field by field.
# ---------------------------------------------------------------------------


def _reauthoring_base() -> ImputationDecision:
    """A KNN block, a MICE block and a categorical column, with dials on both.

    The KNN dials stand in for router-*computed* values: ``n_neighbors`` is 17,
    which no dial-table default would ever produce.
    """
    return ImputationDecision(
        column_decisions={
            "a": _numeric("a", S.KNN),
            "b": _numeric("b", S.KNN),
            "c": _numeric("c", S.MICE),
            "d": _numeric("d", S.MICE),
            "e": ColumnImputationDecision(
                column="e", semantic_type=SemanticType.Categorical, strategy=S.Mode
            ),
        },
        config_snapshot={"numeric": {"knn_k": 17}},
        decided_hyperparameters={
            "knn": (("n_neighbors", 17), ("weights", "distance")),
            "mice": (("max_iter", 25), ("tol", 1e-3)),
        },
        override_hyperparameters={"knn": (("n_neighbors", 9),), "mice": (("max_iter", 50),)},
        numeric_sentinels={"c": [-999.0]},
        string_sentinels={"e": ["NA"]},
    )


def test_base_keeps_unnamed_columns_and_takes_the_map_for_named_ones() -> None:
    base = _reauthoring_base()
    edited = author({"c": S.Median, "d": S.Median}, base=base)

    assert edited.column_decisions["c"].strategy == S.Median
    assert edited.column_decisions["d"].strategy == S.Median
    for col in ("a", "b", "e"):
        assert edited.column_decisions[col] == base.column_decisions[col]
    assert list(edited.column_decisions) == list(base.column_decisions)


def test_base_needs_no_column_universe_and_refuses_one() -> None:
    base = _reauthoring_base()
    with pytest.raises(ValueError, match="'columns'"):
        author({"c": S.Median}, base=base, columns=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match="'default'"):
        author({"c": S.Median}, base=base, default=S.Median)


def test_base_carries_config_snapshot_and_sentinels_verbatim() -> None:
    base = _reauthoring_base()
    edited = author({"c": S.Median, "d": S.Median}, base=base)

    assert edited.config_snapshot == base.config_snapshot
    assert edited.numeric_sentinels == base.numeric_sentinels
    assert edited.string_sentinels == base.string_sentinels


def test_base_sentinel_argument_overwrites_only_the_column_it_names() -> None:
    base = _reauthoring_base()
    edited = author(
        {"c": S.Median, "d": S.Median},
        base=base,
        numeric_sentinels={"c": [-1.0]},
        string_sentinels={"a": ["?"]},
    )
    assert edited.numeric_sentinels == {"c": [-1.0]}
    assert edited.string_sentinels == {"e": ["NA"], "a": ["?"]}


def test_base_carries_custom_estimators_and_the_argument_overwrites_them() -> None:
    first, second = DecisionTreeRegressor(), DecisionTreeRegressor()
    base = author(
        {"c": S.MICE, "d": S.MICE}, columns=["c", "d"], estimators={"mice": first}
    )
    assert base.custom_estimators["mice"] is first

    carried = author({}, base=base)
    assert carried.custom_estimators["mice"] is first
    assert carried.column_decisions["c"].model_choice == ModelChoice.Custom

    replaced = author({}, base=base, estimators={"mice": second})
    assert replaced.custom_estimators["mice"] is second


def test_base_drops_the_estimator_of_a_unit_the_edit_dissolved() -> None:
    base = author(
        {"c": S.MICE, "d": S.MICE},
        columns=["c", "d"],
        estimators={"mice": DecisionTreeRegressor()},
    )
    edited = author({"c": S.Median, "d": S.Median}, base=base)
    assert edited.custom_estimators == {}


def test_base_carries_a_surviving_delta_and_removes_an_orphaned_one() -> None:
    base = _reauthoring_base()
    edited = author({"c": S.Median, "d": S.Median}, base=base)

    assert edited.override_hyperparameters["knn"] == (("n_neighbors", 9),)
    assert _unit_hyp(edited, "knn")["n_neighbors"] == 9
    assert "mice" not in edited.override_hyperparameters
    assert "mice" not in edited.to_dict()["override_hyperparameters"]


def test_base_does_not_resurrect_an_orphaned_delta_when_a_column_returns() -> None:
    base = _reauthoring_base()
    dissolved = author({"c": S.Median, "d": S.Median}, base=base)
    restored = author({"c": S.MICE, "d": S.MICE}, base=dissolved)

    assert "mice" not in restored.override_hyperparameters
    assert _unit_hyp(restored, "mice")["max_iter"] == 10


def test_base_gap_fills_the_decided_base_and_never_overwrites_it() -> None:
    base = _reauthoring_base()
    edited = author({"c": S.Median, "d": S.Median}, base=base)

    # Carried: the router's own 17 survives a structural edit elsewhere.
    assert dict(edited.decided_hyperparameters["knn"]) == {
        "n_neighbors": 17,
        "weights": "distance",
    }
    # Gap-filled: a partial row is completed from the dial table without
    # touching what the base already decided.
    widened = author({"a": S.MICE, "b": S.MICE, "c": S.MICE, "d": S.MICE}, base=base)
    assert dict(widened.decided_hyperparameters["mice"]) == {
        "max_iter": 25,
        "tol": 1e-3,
        "initial_strategy": "mean",
        "n_nearest_features": None,
    }
    assert "knn" not in widened.decided_hyperparameters


def test_base_keeps_the_semantic_type_of_a_re_authored_column() -> None:
    base = _reauthoring_base()
    edited = author({"e": S.Mode}, base=base)
    assert edited.column_decisions["e"].semantic_type == SemanticType.Categorical


def test_base_re_derives_indicator_entries_from_what_survives_the_edit() -> None:
    base = author({"a": S.MNAR, "b": S.Median}, columns=["a", "b"])
    assert "a_missing" in base.column_decisions

    edited = author({"a": S.Median, "b": S.MNAR}, base=base)
    assert "a_missing" not in edited.column_decisions
    assert edited.column_decisions["b_missing"].strategy == S.Indicator


def test_base_accepts_a_deserialized_plan() -> None:
    base = ImputationDecision.from_dict(_reauthoring_base().to_dict())
    edited = author({"c": S.Median, "d": S.Median}, base=base)
    assert edited.column_decisions["c"].strategy == S.Median
    assert dict(edited.decided_hyperparameters["knn"])["n_neighbors"] == 17
