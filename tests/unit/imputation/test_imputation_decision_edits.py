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

from dataforge_ml import ModelChoice
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import (
    ColumnImputationDecision,
    ImputationDecision,
    ImputationStrategy,
)

S = ImputationStrategy


def _plan(*decisions: ColumnImputationDecision) -> ImputationDecision:
    return ImputationDecision(
        column_decisions={d.column: d for d in decisions},
        decided_for_shape=(100, len(decisions), tuple(d.column for d in decisions)),
        config_snapshot={},
        profile_provenance={},
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


def test_forced_survives_round_trip_without_config() -> None:
    """A store-loaded force is still forced, config or no config (ADR-0066)."""
    forced = _numeric("a", S.MICE, forced=True)
    restored = ColumnImputationDecision.from_dict(forced.to_dict())
    assert restored.forced is True


def test_forced_is_not_read_from_the_signal_string() -> None:
    """The bool is the machine predicate; the signal is prose (ADR-0066)."""
    decision = _numeric("a", S.MICE, forced=True, signals=())
    assert decision.forced is True
    assert not any("per_column_strategy_override" in s for s in decision.signals)


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
        decided_for_shape=(100, 2, ("a", "b")),
        config_snapshot={},
        profile_provenance={},
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


def test_with_hyperparameters_unknown_unit_raises_keyerror() -> None:
    plan = _mice_plan()
    with pytest.raises(KeyError):
        plan.with_hyperparameters("knn", {"n_neighbors": 3})


def test_with_hyperparameters_override_survives_save_load_round_trip() -> None:
    plan = _mice_plan().with_hyperparameters("mice", {"max_iter": 50})
    restored = ImputationDecision.from_dict(plan.to_dict())
    assert restored.override_hyperparameters == plan.override_hyperparameters
    assert _unit_hyp(restored, "mice")["max_iter"] == 50
