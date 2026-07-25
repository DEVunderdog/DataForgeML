"""
Unit tests for the ImputationDecision plan-edit API (ADR-0060, issue #346).

Covers ``with_strategy`` / ``with_model_choice``: immutable edits returning a new
valid plan, unit re-derivation on the returned plan, and edit-time legality
validation — output-only labels and non-declarable-for-semantic-type strategies
rejected with the same redirect messages the config surfaces. Plans are built
directly from ``ColumnImputationDecision`` entries so the edit methods (not the
assembler) are under test.
"""

import pytest

from dataforge_ml import ModelChoice, PipelineConfig
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


def test_with_strategy_returns_new_plan_original_unchanged() -> None:
    plan = _plan(_numeric("a", S.Median))
    edited = plan.with_strategy("a", S.Mean)
    assert edited is not plan
    assert plan.column_decisions["a"].strategy == S.Median
    assert edited.column_decisions["a"].strategy == S.Mean


def test_with_model_choice_returns_new_plan_original_unchanged() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.BayesianRidge))
    edited = plan.with_model_choice("a", ModelChoice.RandomForestRegressor)
    assert edited is not plan
    assert plan.column_decisions["a"].model_choice == ModelChoice.BayesianRidge
    assert edited.column_decisions["a"].model_choice == ModelChoice.RandomForestRegressor


def test_with_strategy_marks_the_column_forced() -> None:
    """Editing a strategy is forcing it, and the plan says so (ADR-0066).

    Pins the latent bug: this force never touches ``per_column_strategy``, so a
    config-derived check could not see it and the column degraded silently
    instead of raising.
    """
    plan = _plan(_numeric("a", S.Median))
    assert plan.column_decisions["a"].forced is False

    edited = plan.with_strategy("a", S.MICE)
    assert edited.column_decisions["a"].forced is True
    assert plan.column_decisions["a"].forced is False


def test_with_strategy_forced_survives_round_trip_without_config() -> None:
    """A store-loaded plan-edited force is still forced, config or no config."""
    edited = _plan(_numeric("a", S.Median)).with_strategy("a", S.MICE)
    restored = ColumnImputationDecision.from_dict(edited.column_decisions["a"].to_dict())
    assert restored.forced is True


def test_with_strategy_forced_is_not_read_from_the_signal_string() -> None:
    """The bool is the machine predicate; the signal is prose (ADR-0066)."""
    decision = _numeric("a", S.MICE, forced=True, signals=())
    assert decision.forced is True
    assert not any("per_column_strategy_override" in s for s in decision.signals)


def test_with_strategy_resets_model_choice() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.GradientBoostingRegressor))
    edited = plan.with_strategy("a", S.Median)
    assert edited.column_decisions["a"].model_choice is None


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


def test_units_reflect_the_edit() -> None:
    plan = _plan(_numeric("a", S.MICE), _numeric("b", S.MICE))
    assert {u.unit_id for u in plan.units} == {"mice"}

    edited = plan.with_strategy("a", S.Median)
    assert {u.unit_id for u in edited.units} == {"mice", "median:a"}
    # b remains the (now singleton) MICE block
    mice_unit = next(u for u in edited.units if u.unit_id == "mice")
    assert mice_unit.columns == ("b",)
    # original untouched
    assert {u.unit_id for u in plan.units} == {"mice"}


def test_with_model_choice_edit_visible_on_unit_owner() -> None:
    plan = _plan(_numeric("a", S.MICE, model_choice=ModelChoice.BayesianRidge))
    edited = plan.with_model_choice("a", ModelChoice.RandomForestRegressor)
    # the estimator family is reachable through the plan the unit belongs to
    assert edited.units[0].unit_id == "mice"
    assert edited.column_decisions["a"].model_choice == ModelChoice.RandomForestRegressor


# ---------------------------------------------------------------------------
# Legality — output-only labels rejected, message identical to the config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [S.Dropped, S.MNAR, S.Passthrough, S.Indicator, S.ClusterConditional, S.GMMSampling],
)
def test_output_only_labels_rejected_with_config_message(strategy) -> None:
    plan = _plan(_numeric("a", S.Median))

    with pytest.raises(ValueError) as edit_err:
        plan.with_strategy("a", strategy)

    config = PipelineConfig()
    with pytest.raises(ValueError) as config_err:
        config.imputation.numeric.set_per_column_strategy("a", strategy)

    assert str(edit_err.value) == str(config_err.value)


def test_constant_rejected_with_config_redirect() -> None:
    plan = _plan(_numeric("a", S.Median))

    with pytest.raises(ValueError) as edit_err:
        plan.with_strategy("a", S.Constant)

    # The config redirects a fill-less Constant to per_column_constant_fill.
    config = PipelineConfig()
    with pytest.raises(ValueError) as config_err:
        config.imputation.numeric.set_per_column_strategy("a", S.Constant)

    assert str(edit_err.value) == str(config_err.value)


# ---------------------------------------------------------------------------
# Legality — strategy must be declarable for the column's semantic type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy", [S.MICE, S.KNN, S.Mean, S.Median, S.Mode]
)
def test_numeric_input_strategies_are_declarable(strategy) -> None:
    plan = _plan(_numeric("a", S.Median))
    edited = plan.with_strategy("a", strategy)
    assert edited.column_decisions["a"].strategy == strategy


@pytest.mark.parametrize(
    "semantic_type",
    [SemanticType.Categorical, SemanticType.Boolean, SemanticType.Datetime, SemanticType.Text],
)
def test_non_numeric_column_cannot_declare_imputation_strategy(semantic_type) -> None:
    plan = _plan(
        ColumnImputationDecision(
            column="c", semantic_type=semantic_type, strategy=S.Passthrough
        )
    )
    with pytest.raises(ValueError, match="not a declarable strategy"):
        plan.with_strategy("c", S.Median)


# ---------------------------------------------------------------------------
# Unknown column
# ---------------------------------------------------------------------------


def test_with_strategy_unknown_column_raises_keyerror() -> None:
    plan = _plan(_numeric("a", S.Median))
    with pytest.raises(KeyError):
        plan.with_strategy("nope", S.Mean)


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
