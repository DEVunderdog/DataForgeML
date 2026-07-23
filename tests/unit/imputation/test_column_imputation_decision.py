"""
Unit tests for ColumnImputationDecision — the pure, value-free per-column
imputation plan entry (ADR-0060, issue #343).

All tests are pure — no DataFrames, no StructuralProfiler. They assert the
structural guarantees a user relies on: the exact decided field set, value-free
construction (no fill value / fitted-model home), and immutability.
"""

import dataclasses
import json

import pytest

from dataforge_ml import ColumnImputationDecision as RootColumnImputationDecision
from dataforge_ml import ModelChoice as RootModelChoice
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import (
    ColumnImputationDecision,
    ImputationStrategy,
    ModelChoice,
)

# ---------------------------------------------------------------------------
# Importability / public surface
# ---------------------------------------------------------------------------


def test_exported_from_package_root() -> None:
    assert RootColumnImputationDecision is ColumnImputationDecision
    assert RootModelChoice is ModelChoice


# ---------------------------------------------------------------------------
# Exact decided field set
# ---------------------------------------------------------------------------


def test_has_exactly_the_decided_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(ColumnImputationDecision)}
    assert field_names == {
        "column",
        "semantic_type",
        "strategy",
        "signals",
        "model_choice",
        "domain_snap_bounds",
        "indicator_flag",
        "mnar",
        "drop",
        "forced",
    }


@pytest.mark.parametrize(
    "learned_field",
    ["fill_value", "fitted_model", "model", "coefficients", "n_iter", "convergence"],
)
def test_no_learned_value_field(learned_field: str) -> None:
    """The decision structurally cannot carry anything learned from data."""
    field_names = {f.name for f in dataclasses.fields(ColumnImputationDecision)}
    assert learned_field not in field_names


def test_model_choice_is_a_value_free_label() -> None:
    """model_choice holds an enum name, never a live/fitted estimator object."""
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        model_choice=ModelChoice.GradientBoostingRegressor,
    )
    assert isinstance(decision.model_choice, ModelChoice)
    assert str(decision.model_choice) == "gradient_boosting_regressor"


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_is_frozen() -> None:
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Median,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.strategy = ImputationStrategy.Mean  # type: ignore[misc]


def test_is_hashable() -> None:
    """Frozen + immutable fields make the decision hashable and set-safe."""
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        signals=("nonlinear", "high-corr"),
        model_choice=ModelChoice.RandomForestRegressor,
        domain_snap_bounds=(0.0, 120.0),
    )
    assert decision in {decision}


def test_signals_default_is_immutable_tuple() -> None:
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Median,
    )
    assert decision.signals == ()
    assert isinstance(decision.signals, tuple)


# ---------------------------------------------------------------------------
# to_dict serialisation shape
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_shape() -> None:
    decision = ColumnImputationDecision(
        column="income",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        signals=("linear",),
        model_choice=ModelChoice.BayesianRidge,
        domain_snap_bounds=(0.0, 1_000.0),
        indicator_flag=True,
        mnar=False,
        drop=False,
        forced=True,
    )
    assert decision.to_dict() == {
        "column": "income",
        "semantic_type": SemanticType.Numeric.name,
        "strategy": "Regression",
        "signals": ["linear"],
        "model_choice": "BayesianRidge",
        "domain_snap_bounds": [0.0, 1_000.0],
        "indicator_flag": True,
        "mnar": False,
        "drop": False,
        "forced": True,
    }


def test_to_dict_none_optionals() -> None:
    decision = ColumnImputationDecision(
        column="city",
        semantic_type=SemanticType.Categorical,
        strategy=ImputationStrategy.Mode,
    )
    result = decision.to_dict()
    assert result["model_choice"] is None
    assert result["domain_snap_bounds"] is None
    assert result["signals"] == []


# ---------------------------------------------------------------------------
# forced — a decide-time fact carried on the plan (ADR-0066)
# ---------------------------------------------------------------------------


def test_forced_defaults_to_false() -> None:
    """An auto-routed column is not forced."""
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Median,
    )
    assert decision.forced is False


@pytest.mark.parametrize("forced", [True, False])
def test_forced_round_trips_without_a_config(forced: bool) -> None:
    """A plan reloaded from a store reports ``forced`` with no config present.

    Serialisation is the deciding argument for the field's existence: forced-ness
    must survive into a process that never sees the originating config.
    """
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.MICE,
        forced=forced,
    )
    payload = json.loads(json.dumps(decision.to_dict()))
    restored = ColumnImputationDecision.from_dict(payload)
    assert restored.forced is forced
    assert restored == decision


def test_forced_is_immutable() -> None:
    """``forced`` is decided, so it cannot be flipped after the fact."""
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.MICE,
        forced=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.forced = False  # type: ignore[misc]
