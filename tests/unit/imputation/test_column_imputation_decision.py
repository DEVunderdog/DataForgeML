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
        "center1",
        "center2",
        "feature_cols",
        "grouping_variable",
        "constant_fill",
        "indicator_flag",
        "mnar",
        "drop",
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
        strategy=ImputationStrategy.MICE,
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
        strategy=ImputationStrategy.MICE,
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
        strategy=ImputationStrategy.MICE,
        signals=("linear",),
        model_choice=ModelChoice.BayesianRidge,
        domain_snap_bounds=(0.0, 1_000.0),
        indicator_flag=True,
        mnar=False,
        drop=False,
    )
    assert decision.to_dict() == {
        "column": "income",
        "semantic_type": SemanticType.Numeric.name,
        "strategy": "MICE",
        "signals": ["linear"],
        "model_choice": "BayesianRidge",
        "domain_snap_bounds": [0.0, 1_000.0],
        "center1": None,
        "center2": None,
        "feature_cols": None,
        "grouping_variable": None,
        "constant_fill": None,
        "indicator_flag": True,
        "mnar": False,
        "drop": False,
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
    assert result["center1"] is None
    assert result["center2"] is None
    assert result["feature_cols"] is None
    assert result["grouping_variable"] is None
    assert result["constant_fill"] is None


# ---------------------------------------------------------------------------
# Facts about the data (ADR-0083)
# ---------------------------------------------------------------------------


def test_facts_about_the_data_round_trip_through_json() -> None:
    """The five lowered facts survive to_dict → JSON → from_dict unchanged."""
    decision = ColumnImputationDecision(
        column="income",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
        center1=10.0,
        center2=90.0,
        feature_cols=("age", "score"),
        grouping_variable="region",
        constant_fill=-1.0,
    )
    payload = json.loads(json.dumps(decision.to_dict()))
    assert payload["center1"] == 10.0
    assert payload["center2"] == 90.0
    assert payload["feature_cols"] == ["age", "score"]
    assert payload["grouping_variable"] == "region"
    assert payload["constant_fill"] == -1.0

    restored = ColumnImputationDecision.from_dict(payload)
    assert restored == decision
    assert isinstance(restored.feature_cols, tuple)


def test_facts_keep_the_decision_hashable() -> None:
    """``feature_cols`` is a tuple, so a decision carrying facts stays set-safe."""
    decision = ColumnImputationDecision(
        column="income",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
        center1=10.0,
        center2=90.0,
        feature_cols=("age",),
    )
    assert decision in {decision}


def test_facts_default_to_none() -> None:
    """A column with no measured facts carries none of them."""
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Median,
    )
    assert decision.center1 is None
    assert decision.center2 is None
    assert decision.feature_cols is None
    assert decision.grouping_variable is None
    assert decision.constant_fill is None


# ---------------------------------------------------------------------------
# forced — deleted outright (ADR-0083)
# ---------------------------------------------------------------------------


def test_forced_is_not_a_field() -> None:
    """``forced`` is gone, not defaulted: it had no reader in ``src/``."""
    field_names = {f.name for f in dataclasses.fields(ColumnImputationDecision)}
    assert "forced" not in field_names
    with pytest.raises(TypeError):
        ColumnImputationDecision(
            column="age",
            semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.MICE,
            forced=True,  # type: ignore[call-arg]
        )


def test_forced_is_absent_from_the_wire_and_ignored_on_the_way_back() -> None:
    """A payload carrying the retired key round-trips without resurrecting it.

    Backward compatibility is a non-concern (ADR-0083); what matters is that a
    stale key cannot smuggle a field back onto the decision.
    """
    decision = ColumnImputationDecision(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.MICE,
    )
    payload = json.loads(json.dumps(decision.to_dict()))
    assert "forced" not in payload

    restored = ColumnImputationDecision.from_dict({**payload, "forced": True})
    assert restored == decision
    assert not hasattr(restored, "forced")
