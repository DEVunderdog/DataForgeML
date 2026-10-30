"""``compose_imputer`` end to end, in both deliveries: an imputer manifest Handle."""

from __future__ import annotations

from typing import Any

from dataforge_ml._boundary._workspace import read_frame, read_imputer, read_record

from .conftest import Harness, records
from .test_route import _profile


def _kind(harness: Harness, kind: str) -> list[str]:
    return [h for h in harness.handles() if h.startswith(f"{kind}_")]


def _recipe(harness: Harness) -> str:
    routing = harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "mice"}, {"column": "age", "strategy": "median"}],
    ).json["routing"]
    return harness.code_mode("resolve_recipe", routing=routing).json["recipe"]


def _fit(harness: Harness, recipe: str, unit_id: str, **arguments: Any) -> str:
    return harness.code_mode("fit_unit", recipe=recipe, unit_id=unit_id, **arguments).json["unit"]


def _fitted(harness: Harness) -> tuple[str, str, str]:
    recipe = _recipe(harness)
    return recipe, _fit(harness, recipe, "mice"), _fit(harness, recipe, "median:age")


# --- The manifest -----------------------------------------------------------------


def test_it_writes_an_imputer_handle_naming_the_recipe_and_its_units(harness: Harness) -> None:
    recipe, mice, median = _fitted(harness)

    out = harness.call("compose_imputer", recipe=recipe, units=[mice, median])

    assert not out.is_error, out.json
    assert out.json["imputer"] in _kind(harness, "imputer")
    assert out.json["recipe"] == recipe
    # Routing order, ``age`` before ``income``, whatever order the units were passed in.
    assert records(out.json["units"]) == [
        {"unit_id": "median:age", "strategy": "median", "unit": median},
        {"unit_id": "mice", "strategy": "mice", "unit": mice},
    ]


def test_the_imputer_handle_is_a_manifest_with_its_lineage(harness: Harness) -> None:
    recipe, mice, median = _fitted(harness)

    imputer = harness.code_mode("compose_imputer", recipe=recipe, units=[mice, median]).json[
        "imputer"
    ]

    record = read_record(imputer, "imputer", param="imputer")
    assert record["payload"] is None
    assert list((harness.workspace / "handles" / imputer).iterdir()) == [
        harness.workspace / "handles" / imputer / "record.json"
    ]
    assert record["recipe"] == recipe
    assert record["units"] == [
        {"unit_id": "median:age", "unit": median},
        {"unit_id": "mice", "unit": mice},
    ]
    assert record["lineage"]["operation"] == "compose_imputer"
    assert record["lineage"]["inputs"] == [recipe, mice, median]


def test_the_manifest_composes_again_into_an_imputer_that_fills_the_frame(
    harness: Harness,
) -> None:
    recipe, mice, median = _fitted(harness)
    imputer = harness.code_mode("compose_imputer", recipe=recipe, units=[mice, median]).json[
        "imputer"
    ]
    data, _ = read_frame(read_record(recipe, "recipe", param="recipe")["frame"], param="frame")

    composed, _ = read_imputer(imputer, param="imputer")

    filled = composed.transform(data).dataframe
    assert filled.height == data.height
    assert filled.select(["age", "income"]).null_count().sum_horizontal().item() == 0


def test_composing_again_after_one_unit_changes_writes_only_a_new_manifest(
    harness: Harness,
) -> None:
    recipe, mice, median = _fitted(harness)
    first = harness.code_mode("compose_imputer", recipe=recipe, units=[mice, median]).json
    refit = _fit(harness, recipe, "mice", seed=7)
    before = set(harness.handles())

    second = harness.code_mode("compose_imputer", recipe=recipe, units=[refit, median]).json

    assert set(harness.handles()) - before == {second["imputer"]}
    assert second["imputer"] != first["imputer"]
    assert read_record(first["imputer"], "imputer", param="imputer")["units"][1]["unit"] == mice
    assert read_record(second["imputer"], "imputer", param="imputer")["units"][1]["unit"] == refit


# --- Errors -----------------------------------------------------------------------


def test_a_unit_from_another_recipe_is_refused(harness: Harness) -> None:
    recipe, mice, _ = _fitted(harness)
    other = _recipe(harness)
    foreign = _fit(harness, other, "median:age")

    out = harness.call("compose_imputer", recipe=recipe, units=[mice, foreign])

    assert out.code == "unit_recipe_mismatch"
    assert out.json["param"] == "units[1]"
    assert _kind(harness, "imputer") == []


def test_a_missing_unit_is_refused_with_its_columns(harness: Harness) -> None:
    recipe, _, median = _fitted(harness)

    out = harness.call("compose_imputer", recipe=recipe, units=[median])

    assert out.code == "unit_coverage_mismatch"
    assert out.json["columns"] == ["income"]
    assert "`mice`" in out.json["message"]
    assert _kind(harness, "imputer") == []


def test_two_units_for_the_same_unit_are_refused(harness: Harness) -> None:
    recipe, mice, median = _fitted(harness)
    again = _fit(harness, recipe, "median:age", seed=3)

    out = harness.call("compose_imputer", recipe=recipe, units=[mice, median, again])

    assert out.code == "duplicate_key"
    assert out.json["param"] == "units[2]"
    assert _kind(harness, "imputer") == []


def test_a_units_entry_of_another_kind_is_refused(harness: Harness) -> None:
    recipe, mice, _ = _fitted(harness)

    out = harness.call("compose_imputer", recipe=recipe, units=[mice, recipe])

    assert out.code == "wrong_handle_kind"
    assert out.json["param"] == "units[1]"
    assert _kind(harness, "imputer") == []


def test_an_unknown_unit_handle_is_refused_with_candidates(harness: Harness) -> None:
    recipe, mice, median = _fitted(harness)

    out = harness.call("compose_imputer", recipe=recipe, units=[mice, "unit_00000000"])

    assert out.code == "unknown_handle"
    assert out.json["param"] == "units[1]"
    assert set(out.json["candidates"]) == {mice, median}
