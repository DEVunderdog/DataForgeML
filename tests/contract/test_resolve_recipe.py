"""``resolve_recipe`` end to end, in both deliveries: the recipe Handle and its units by value."""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

from dataforge_ml._boundary._workspace import payload_path, read_record
from dataforge_ml._serialization import deserialize
from dataforge_ml.imputation._config import _STRUCTURAL_STRATEGIES, ImputationStrategy
from dataforge_ml.operations.resolve_recipe import UnitStrategy

from .conftest import Harness, records, write_flagged_wide_csv
from .test_route import _profile, _stored_routing


def _recipe_handles(harness: Harness) -> list[str]:
    return [h for h in harness.handles() if h.startswith("recipe_")]


# --- The recipe -------------------------------------------------------------------


def test_it_returns_a_recipe_handle_and_its_units_by_value(harness: Harness) -> None:
    profile = _profile(harness)
    routing = harness.code_mode(
        "route",
        profile=profile,
        strategies=[
            {"column": "income", "strategy": "mice"},
            {"column": "age", "strategy": "median"},
        ],
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    assert not out.is_error
    assert out.json["recipe"] in _recipe_handles(harness)
    assert (out.json["routing"], out.json["profile"]) == (routing, profile)
    assert records(out.json["units"]) == [
        {"unit_id": "median:age", "strategy": "median", "columns": ["age"], "hyperparameters": None},
        {"unit_id": "mice", "strategy": "mice", "columns": ["income"],
         "hyperparameters": records(out.json["units"])[1]["hyperparameters"]},
    ]


def _stored_recipe(handle: str) -> Any:
    record = read_record(handle, "recipe", param="recipe")
    return deserialize(payload_path(handle, record).read_bytes())


def test_lineage_records_the_routing_read(harness: Harness) -> None:
    profile = _profile(harness)
    routing = harness.code_mode("route", profile=profile).json["routing"]

    recipe = harness.call("resolve_recipe", routing=routing).json["recipe"]

    lineage = read_record(recipe, "recipe", param="recipe")["lineage"]
    assert lineage["operation"] == "resolve_recipe"
    assert lineage["arguments"] == {"routing": routing}
    assert lineage["inputs"] == [routing, profile]


def test_the_recipe_holds_the_routing_it_was_resolved_from(harness: Harness) -> None:
    routing = harness.code_mode(
        "route", profile=_profile(harness), strategies=[{"column": "income", "strategy": "knn"}]
    ).json["routing"]

    recipe = _stored_recipe(harness.call("resolve_recipe", routing=routing).json["recipe"])

    assert recipe.routing == _stored_routing(routing)
    assert set(recipe.hyperparameters("knn")) == {"n_neighbors", "weights"}


def _wide_numeric_profile(harness: Harness, n_columns: int = 12) -> str:
    # ``n_columns`` correlated numeric columns, the first with gaps: wide
    # enough that MICE counts its nearest features rather than using all.
    csv = harness.inputs / "wide_numeric.csv"
    names = [f"x{i}" for i in range(n_columns)]
    lines = [",".join(names)]
    for r in range(60):
        base = (r % 17) + 0.5
        cells = [f"{base * (i + 1) + (r * (i + 3)) % 7 * 0.01:.3f}" for i in range(n_columns)]
        if r % 6 == 0:
            cells[0] = ""
        lines.append(",".join(cells))
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return _profile(harness, str(csv))


def test_a_dial_set_on_route_holds_in_the_recipe(harness: Harness) -> None:
    profile = _wide_numeric_profile(harness)
    mice = [{"column": "x0", "strategy": "mice"}]
    default = harness.code_mode("route", profile=profile, strategies=mice).json["routing"]
    capped = harness.code_mode(
        "route",
        profile=profile,
        strategies=mice,
        thresholds={"numeric": {"mice_max_nearest_features": 2}},
    ).json["routing"]

    by_default, by_dial = (
        _stored_recipe(harness.call("resolve_recipe", routing=r).json["recipe"])
        for r in (default, capped)
    )

    assert by_default.hyperparameters("mice")["n_nearest_features"] == 11
    assert by_dial.hyperparameters("mice")["n_nearest_features"] == 2


def test_a_routing_that_fills_nothing_has_no_units(harness: Harness) -> None:
    routing = harness.code_mode(
        "route", profile=_profile(harness), default_strategy="passthrough"
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    assert not out.is_error and out.json["recipe"] in _recipe_handles(harness)
    assert records(out.json["units"]) == []


# --- Hyperparameters --------------------------------------------------------------


def _mice_and_median_routing(harness: Harness) -> str:
    return harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[
            {"column": "income", "strategy": "mice"},
            {"column": "age", "strategy": "median"},
        ],
    ).json["routing"]


def test_hyperparameters_land_on_the_named_unit_and_show_in_the_result(
    harness: Harness,
) -> None:
    routing = _mice_and_median_routing(harness)

    out = harness.call(
        "resolve_recipe",
        routing=routing,
        hyperparameters=[
            {"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iter": 7, "n_nearest_features": "all"}},
        ],
    )

    assert not out.is_error
    median, mice = records(out.json["units"])
    assert median["hyperparameters"] is None
    assert mice["hyperparameters"]["kind"] == "mice"
    assert (mice["hyperparameters"]["max_iter"], mice["hyperparameters"]["n_nearest_features"]) == (7, "all")
    recipe = _stored_recipe(out.json["recipe"])
    assert recipe.hyperparameters("mice")["max_iter"] == 7
    assert recipe.hyperparameters("mice")["n_nearest_features"] is None


def test_without_hyperparameters_each_unit_shows_its_resolved_dials(harness: Harness) -> None:
    routing = _mice_and_median_routing(harness)

    out = harness.call("resolve_recipe", routing=routing)

    mice = records(out.json["units"])[1]["hyperparameters"]
    stored = _stored_recipe(out.json["recipe"]).hyperparameters("mice")
    assert mice["max_iter"] == stored["max_iter"] and mice["initial_strategy"] in ("mean", "median")


def _tune(harness: Harness, *entries: dict[str, Any]) -> Any:
    return harness.call(
        "resolve_recipe",
        routing=_mice_and_median_routing(harness),
        hyperparameters=list(entries),
    )


def test_an_unknown_unit_is_refused_with_candidates(harness: Harness) -> None:
    out = _tune(harness, {"unit_id": "mcie", "hyperparameters": {"kind": "mice", "max_iter": 7}})

    assert out.code == "unknown_unit"
    assert out.json["param"] == "hyperparameters[0].unit_id"
    assert out.json["candidates"] == ["mice"]
    assert _recipe_handles(harness) == []


def test_a_unit_named_twice_is_a_duplicate_key(harness: Harness) -> None:
    entry = {"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iter": 7}}

    out = _tune(harness, entry, entry)

    assert out.code == "duplicate_key"
    assert out.json["params"] == ["hyperparameters[1].unit_id"]
    assert _recipe_handles(harness) == []


def test_a_record_of_another_strategy_is_refused(harness: Harness) -> None:
    out = _tune(harness, {"unit_id": "mice", "hyperparameters": {"kind": "knn", "n_neighbors": 3}})

    assert out.code == "invalid_argument"
    assert out.json["param"] == "hyperparameters[0].hyperparameters.kind"
    assert out.json["allowed"] == ["mice"]
    assert _recipe_handles(harness) == []


def test_a_unit_with_no_hyperparameters_is_refused(harness: Harness) -> None:
    out = _tune(harness, {"unit_id": "median:age", "hyperparameters": {"kind": "mnar"}})

    assert out.code == "invalid_argument"
    assert out.json["param"] == "hyperparameters[0].unit_id"
    assert _recipe_handles(harness) == []


def test_an_unknown_key_or_value_is_refused_before_any_work(harness: Harness) -> None:
    out = _tune(
        harness,
        {"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iters": 7}},
        {"unit_id": "knn", "hyperparameters": {"kind": "knn", "weights": "gaussian"}},
    )

    assert out.code == "invalid_argument"
    assert out.json["params"] == [
        "hyperparameters[0].hyperparameters.mice.max_iters",
        "hyperparameters[1].hyperparameters.knn.weights",
    ]
    assert _recipe_handles(harness) == []


def test_every_problem_entry_comes_back_in_one_error(harness: Harness) -> None:
    out = _tune(
        harness,
        {"unit_id": "mcie", "hyperparameters": {"kind": "mice"}},
        {"unit_id": "median:age", "hyperparameters": {"kind": "mnar"}},
    )

    assert out.code == "unknown_unit"
    assert out.json["params"] == ["hyperparameters[0].unit_id", "hyperparameters[1].unit_id"]
    assert len(out.json["issues"]) == 2
    assert _recipe_handles(harness) == []


# --- Caps -------------------------------------------------------------------------


def _flagged_wide_profile(harness: Harness, tmp_path: Path) -> str:
    wide = write_flagged_wide_csv(tmp_path / "flagged.csv", 150, name_length=8)
    harness.declare(wide)
    frame = harness.code_mode("load", path=str(wide)).json["frame"]
    return harness.code_mode("profile", frame=frame).json["profile"]


def test_units_are_capped_at_a_hundred(harness: Harness, tmp_path: Path) -> None:
    routing = harness.code_mode(
        "route", profile=_flagged_wide_profile(harness, tmp_path), default_strategy="median"
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    units = records(out.json["units"])
    assert len(units) == 100
    assert units[0] == {
        "unit_id": "median:c0000_xx", "strategy": "median", "columns": ["c0000_xx"],
        "hyperparameters": None,
    }
    assert records(out.json["omitted"]) == [
        {"field": "units", "shown": 100, "total": 150, "operation": "explain_routing",
         "reason": "cap"},
    ]


def test_a_block_lists_at_most_a_hundred_columns(harness: Harness, tmp_path: Path) -> None:
    routing = harness.code_mode(
        "route", profile=_flagged_wide_profile(harness, tmp_path), default_strategy="mice"
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    (unit,) = records(out.json["units"])
    assert (unit["unit_id"], len(unit["columns"])) == ("mice", 100)
    assert unit["columns"][-1] == "c0099_xx"
    assert records(out.json["omitted"]) == [
        {"field": "units[0].columns", "shown": 100, "total": 150,
         "operation": "explain_routing", "reason": "cap"},
    ]


# --- Errors -----------------------------------------------------------------------


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    out = harness.call("resolve_recipe", routing=_profile(harness))

    assert out.code == "wrong_handle_kind"
    assert _recipe_handles(harness) == []


def test_an_unknown_routing_lists_routing_candidates(harness: Harness) -> None:
    routing = harness.code_mode("route", profile=_profile(harness)).json["routing"]

    out = harness.call("resolve_recipe", routing="routing_00000000")

    assert out.code == "unknown_handle"
    assert out.json["candidates"] == [routing]
    assert _recipe_handles(harness) == []


# --- The wire vocabulary ----------------------------------------------------------


def test_the_wire_unit_strategies_are_every_strategy_that_trains() -> None:
    trains = {s.value for s in ImputationStrategy} - {s.value for s in _STRUCTURAL_STRATEGIES}
    assert set(typing.get_args(UnitStrategy)) == trains
