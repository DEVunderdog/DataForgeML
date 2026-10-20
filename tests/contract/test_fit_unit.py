"""``fit_unit`` end to end, in both deliveries: one unit Handle and its Fit Signals."""

from __future__ import annotations

from typing import Any

from dataforge_ml._boundary._workspace import payload_path, read_record
from dataforge_ml._serialization import deserialize

from .conftest import Harness, records
from .test_route import _profile


def _unit_handles(harness: Harness) -> list[str]:
    return [h for h in harness.handles() if h.startswith("unit_")]


def _stored_unit(handle: str) -> Any:
    record = read_record(handle, "unit", param="unit")
    return deserialize(payload_path(handle, record).read_bytes())


def _recipe(harness: Harness, strategies: list[dict[str, Any]], **arguments: Any) -> str:
    routing = harness.code_mode(
        "route", profile=_profile(harness), strategies=strategies
    ).json["routing"]
    return harness.code_mode("resolve_recipe", routing=routing, **arguments).json["recipe"]


def _mice_and_median_recipe(harness: Harness, **arguments: Any) -> str:
    return _recipe(
        harness,
        [{"column": "income", "strategy": "mice"}, {"column": "age", "strategy": "median"}],
        **arguments,
    )


def _gmm_recipe(harness: Harness) -> str:
    # The fixture's ``income`` has one mode, so the forced bimodal column needs centres set.
    return _recipe(
        harness,
        [{"column": "income", "strategy": "gmm_sampling"}],
        estimates=[{"column": "income", "center1": 45000.0, "center2": 62000.0}],
    )


# --- The fit ----------------------------------------------------------------------


def test_it_returns_a_unit_handle_and_typed_fit_signals(harness: Harness) -> None:
    recipe = _mice_and_median_recipe(harness)

    out = harness.call("fit_unit", recipe=recipe, unit_id="mice")

    assert not out.is_error
    assert out.json["unit"] in _unit_handles(harness)
    assert out.json["recipe"] == recipe
    assert out.json["frame"] == read_record(recipe, "recipe", param="recipe")["frame"]
    assert (out.json["unit_id"], out.json["strategy"], out.json["columns"]) == (
        "mice", "mice", ["income"],
    )
    signals = out.json["signals"]
    assert isinstance(signals["estimator"], str)
    assert isinstance(signals["converged"], bool)
    assert isinstance(signals["n_iter"], int)
    assert signals["hyperparameters"]["kind"] == "mice"


def test_the_unit_trains_with_the_recipes_merged_dials(harness: Harness) -> None:
    routing = harness.code_mode(
        "route", profile=_profile(harness), strategies=[{"column": "income", "strategy": "mice"}]
    ).json["routing"]
    resolved = harness.code_mode(
        "resolve_recipe",
        routing=routing,
        hyperparameters=[{"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iter": 7}}],
    ).json
    (unit,) = records(resolved["units"])

    out = harness.call("fit_unit", recipe=resolved["recipe"], unit_id="mice")

    assert out.json["signals"]["hyperparameters"] == unit["hyperparameters"]
    assert unit["hyperparameters"]["max_iter"] == 7


def test_a_scalar_unit_has_no_estimator_iterations_or_dials(harness: Harness) -> None:
    out = harness.call("fit_unit", recipe=_mice_and_median_recipe(harness), unit_id="median:age")

    assert out.json["signals"] == {
        "estimator": None, "converged": None, "n_iter": None, "hyperparameters": None,
    }
    assert records(out.json["warnings"]) == []


def test_the_unit_handle_holds_the_fitted_unit_and_its_lineage(harness: Harness) -> None:
    recipe = _mice_and_median_recipe(harness)

    unit = harness.call("fit_unit", recipe=recipe, unit_id="median:age").json["unit"]

    record = read_record(unit, "unit", param="unit")
    frame = read_record(recipe, "recipe", param="recipe")["frame"]
    assert record["lineage"]["operation"] == "fit_unit"
    assert record["lineage"]["arguments"] == {"recipe": recipe, "unit_id": "median:age"}
    assert record["lineage"]["inputs"] == [recipe, frame]
    assert (record["unit_id"], record["strategy"], record["frame"]) == ("median:age", "median", frame)
    assert _stored_unit(unit).target_columns == ["age"]


# --- Warnings ---------------------------------------------------------------------


def test_a_model_that_hits_its_iteration_cap_comes_back_as_a_coded_warning(
    harness: Harness,
) -> None:
    recipe = _mice_and_median_recipe(
        harness,
        hyperparameters=[{"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iter": 1}}],
    )

    out = harness.call("fit_unit", recipe=recipe, unit_id="mice")

    assert not out.is_error and out.json["unit"] in _unit_handles(harness)
    assert out.json["signals"]["converged"] is False
    (warning,) = records(out.json["warnings"])
    assert warning["code"] == "mice_not_converged"
    assert warning["columns"] == ["income"]
    assert warning["hint"]


# --- The seed ---------------------------------------------------------------------


def test_a_null_seed_gives_the_same_unit_twice(harness: Harness) -> None:
    recipe = _gmm_recipe(harness)

    first, second = (
        harness.call("fit_unit", recipe=recipe, unit_id="gmm_sampling:income").json
        for _ in range(2)
    )

    assert first["seed"] == second["seed"] == 42
    assert first["signals"] == second["signals"]
    assert _stored_unit(first["unit"]) == _stored_unit(second["unit"])


def test_a_seed_passed_is_the_seed_used(harness: Harness) -> None:
    out = harness.call(
        "fit_unit", recipe=_gmm_recipe(harness), unit_id="gmm_sampling:income", seed=7
    )

    assert out.json["seed"] == 7
    assert _stored_unit(out.json["unit"]).random_seed == 7


# --- Errors -----------------------------------------------------------------------


def test_an_unknown_unit_is_refused_with_candidates(harness: Harness) -> None:
    out = harness.call("fit_unit", recipe=_mice_and_median_recipe(harness), unit_id="mcie")

    assert out.code == "unknown_unit"
    assert out.json["param"] == "unit_id"
    assert out.json["candidates"] == ["mice"]
    assert _unit_handles(harness) == []


def test_a_unit_that_cannot_train_is_refused(harness: Harness) -> None:
    # A forced bimodal column whose profile found one mode has no centres.
    recipe = _recipe(harness, [{"column": "income", "strategy": "gmm_sampling"}])

    out = harness.call("fit_unit", recipe=recipe, unit_id="gmm_sampling:income")

    assert out.code == "unit_not_trainable"
    assert out.json["columns"] == ["income"]
    assert _unit_handles(harness) == []


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    out = harness.call("fit_unit", recipe=_profile(harness), unit_id="mice")

    assert out.code == "wrong_handle_kind"
    assert _unit_handles(harness) == []


def test_a_seed_out_of_range_is_refused(harness: Harness) -> None:
    out = harness.call(
        "fit_unit", recipe=_mice_and_median_recipe(harness), unit_id="mice", seed=-1
    )

    assert out.code == "invalid_argument"
    assert out.json["param"] == "seed"
