"""The typed hyperparameter and estimator records, and how they convert (ADR-0101)."""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from dataforge_ml._boundary._schema import strict_subset_problems, to_strict_subset
from dataforge_ml._seed import DEFAULT_SEED
from dataforge_ml.imputation import ImputationStrategy
from dataforge_ml.imputation._config import _STRATEGY_DIALS, ModelChoice, _dial_defaults
from dataforge_ml.imputation._regression_estimator_factory import (
    RegressionEstimatorFactory,
)
from dataforge_ml.operations._hyperparameters import (
    _ESTIMATORS,
    _RECORDS,
    EstimatorName,
    Hyperparameters,
    MiceEstimator,
    MiceHyperparameters,
    estimator_record,
    hyperparameters_record,
)
from dataforge_ml.records import WireRecord


def test_mice_record_converts_to_the_dials_the_recipe_reads() -> None:
    record = MiceHyperparameters(kind="mice", max_iter=25, tol=1e-4)

    assert record.to_hyperparameters() == {"max_iter": 25, "tol": 1e-4}


def test_mice_full_dial_row_round_trips() -> None:
    dials = {"max_iter": 10, "tol": 1e-3, "initial_strategy": "median", "n_nearest_features": None}

    record = hyperparameters_record(ImputationStrategy.MICE, dials)

    assert record.n_nearest_features == "all"
    assert record.to_hyperparameters() == dials


def test_every_strategy_with_dials_has_a_record_with_exactly_its_keys() -> None:
    assert set(_RECORDS) == set(_STRATEGY_DIALS)
    for strategy, record in _RECORDS.items():
        assert set(record.model_fields) - {"kind"} == set(_STRATEGY_DIALS[strategy])
        assert record.model_fields["kind"].annotation == Literal[strategy.value]


@pytest.mark.parametrize(
    ("strategy", "dials"),
    [
        (ImputationStrategy.MICE, {"max_iter": 40, "tol": 0.01, "initial_strategy": "mean", "n_nearest_features": 7}),
        (ImputationStrategy.KNN, {"n_neighbors": 9, "weights": "distance"}),
        (ImputationStrategy.MNAR, {"central_tendency": "mode"}),
        (ImputationStrategy.ClusterConditional, {"central_tendency": "mean"}),
    ],
)
def test_each_strategy_round_trips(strategy: ImputationStrategy, dials: dict[str, Any]) -> None:
    record = hyperparameters_record(strategy, dials)

    assert record.to_hyperparameters() == dials
    assert hyperparameters_record(strategy, record.to_hyperparameters()) == record
    assert TypeAdapter(Hyperparameters).validate_python(record.to_dict()) == record


def test_the_neutral_dial_rows_round_trip() -> None:
    for strategy in _STRATEGY_DIALS:
        dials = _dial_defaults(strategy)
        assert hyperparameters_record(strategy, dials).to_hyperparameters() == dials


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "mice", "max_iters": 5},  # unknown key
        {"kind": "knn", "max_iter": 5},  # another strategy's key
        {"kind": "knn", "weights": "gaussian"},  # unknown value
        {"kind": "mice", "max_iter": "5"},  # a quoted number is never a number
        {"kind": "mice", "max_iter": 0},  # out of range
        {"kind": "mice", "initial_strategy": "constant"},  # the fitter reports only mean or median
        {"kind": "cluster_conditional", "central_tendency": "mode"},
        {"kind": "gmm_sampling"},  # a strategy with no dials has no record
    ],
)
def test_the_record_rejects_what_its_strategy_does_not_have(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Hyperparameters).validate_python(payload)


# --- Estimators -------------------------------------------------------------------


def test_the_estimator_enum_is_every_model_choice() -> None:
    assert set(get_args(EstimatorName)) == {c.value for c in ModelChoice}
    assert set(_ESTIMATORS) == {ModelChoice(name) for name in get_args(EstimatorName)}


@pytest.mark.parametrize(
    ("choice", "params"),
    [
        (ModelChoice.BayesianRidge, {}),
        (ModelChoice.BayesianRidge, {"max_iter": 500, "alpha_1": 1e-5, "lambda_2": 1e-4}),
        (ModelChoice.RandomForestRegressor, {"n_estimators": 50, "max_depth": 8, "max_features": "sqrt"}),
        (ModelChoice.RandomForestRegressor, {"max_features": 0.5, "min_samples_leaf": 3}),
        (ModelChoice.GradientBoostingRegressor, {"learning_rate": 0.05, "subsample": 0.8}),
    ],
)
def test_each_estimator_round_trips(choice: ModelChoice, params: dict[str, Any]) -> None:
    record = estimator_record(choice, params)

    assert record.to_estimator() == (choice, params)
    assert TypeAdapter(MiceEstimator).validate_python(record.to_dict()) == record


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "custom"},
        {"kind": "extra_trees_regressor"},
        {"kind": "bayesian_ridge", "n_estimators": 10},
        {"kind": "random_forest_regressor", "random_state": 3},  # the seed is not a parameter
        {"kind": "random_forest_regressor", "n_jobs": 4},  # n_jobs_inner is not on the wire
        {"kind": "gradient_boosting_regressor", "subsample": 1.5},
        {"kind": "random_forest_regressor", "max_features": "auto"},
    ],
)
def test_the_estimator_record_rejects_what_it_does_not_have(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(MiceEstimator).validate_python(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"kind": "bayesian_ridge", "max_iter": 500, "lambda_1": 1e-4}, {"model__max_iter": 500, "model__lambda_1": 1e-4}),
        ({"kind": "random_forest_regressor", "n_estimators": 7, "max_features": "log2"}, {"n_estimators": 7, "max_features": "log2"}),
        ({"kind": "gradient_boosting_regressor", "learning_rate": 0.3, "max_depth": 5}, {"learning_rate": 0.3, "max_depth": 5}),
    ],
)
def test_the_factory_builds_the_estimator_the_record_names(
    payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    choice, params = TypeAdapter(MiceEstimator).validate_python(payload).to_estimator()

    built = RegressionEstimatorFactory.build_from_choice(choice, params=params).get_params()

    assert {key: built[key] for key in expected} == expected


def test_a_forest_with_parameters_still_fits_and_predicts_pinned() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 3)), rng.normal(size=60)
    forest = RegressionEstimatorFactory.build_from_choice(
        ModelChoice.RandomForestRegressor, n_jobs=2, params={"n_estimators": 5, "max_depth": 2}
    )

    forest.fit(X, y)

    assert len(forest.forest_.estimators_) == 5
    assert forest.forest_.n_jobs == 1
    assert max(tree.get_depth() for tree in forest.forest_.estimators_) <= 2
    assert clone(forest).get_params()["max_depth"] == 2


# --- Strict Subset ----------------------------------------------------------------


class _Carrier(WireRecord):
    """The two unions as a resolve_recipe-style argument would carry them."""

    hyperparameters: list[Hyperparameters] | None = None
    """Per-unit dials."""
    estimator: MiceEstimator | None = None
    """The MICE block's estimator."""


@pytest.mark.parametrize(
    "model",
    [*_RECORDS.values(), *_ESTIMATORS.values(), _Carrier],
    ids=lambda model: model.__name__,
)
def test_every_record_is_in_the_strict_subset(model: type[WireRecord]) -> None:
    schema = to_strict_subset(model.model_json_schema())

    assert strict_subset_problems(schema) == []
    for name, field in schema["properties"].items():
        assert field.get("description"), f"{model.__name__}.{name} has no description"


def test_a_forest_with_no_parameters_keeps_sklearns_defaults() -> None:
    rng = np.random.default_rng(0)
    forest = RegressionEstimatorFactory.build_from_choice(ModelChoice.RandomForestRegressor)

    forest.fit(rng.normal(size=(30, 2)), rng.normal(size=30))

    assert forest.forest_.get_params() == RandomForestRegressor(random_state=DEFAULT_SEED, n_jobs=1).get_params()
