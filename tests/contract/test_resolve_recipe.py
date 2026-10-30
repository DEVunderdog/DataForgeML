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


# --- Estimates --------------------------------------------------------------------


def _estimates_routing(harness: Harness, **profile_arguments: Any) -> str:
    # ``income`` is forced onto a bimodal strategy, so it carries estimates;
    # ``age`` takes ``median``, which reads none.
    return harness.code_mode(
        "route",
        profile=_profile(harness, **profile_arguments),
        strategies=[
            {"column": "income", "strategy": "cluster_conditional"},
            {"column": "age", "strategy": "median"},
        ],
    ).json["routing"]


def _estimate(harness: Harness, *entries: dict[str, Any]) -> Any:
    return harness.call(
        "resolve_recipe", routing=_estimates_routing(harness), estimates=list(entries)
    )


def test_estimates_land_on_the_recipe_and_show_in_the_result(harness: Harness) -> None:
    out = _estimate(
        harness,
        {"column": "income", "center1": 45000.0, "center2": 60000, "feature_cols": ["age"]},
    )

    assert not out.is_error
    assert records(out.json["estimates"]) == [
        {"column": "income", "center1": 45000.0, "center2": 60000.0, "feature_cols": ["age"],
         "domain_snap_bounds": None},
    ]
    stored = _stored_recipe(out.json["recipe"]).column_estimates["income"]
    assert (stored.center1, stored.center2, stored.feature_cols) == (45000.0, 60000.0, ("age",))


def test_a_field_left_null_keeps_its_resolved_value(harness: Harness) -> None:
    routing = _estimates_routing(harness)
    resolved = records(harness.call("resolve_recipe", routing=routing).json["estimates"])

    out = harness.call(
        "resolve_recipe",
        routing=routing,
        estimates=[{"column": "income", "center1": 1.5, "center2": 2.5}],
    )

    (row,) = records(out.json["estimates"])
    assert row == {**resolved[0], "center1": 1.5, "center2": 2.5}


def test_without_estimates_each_column_shows_its_resolved_ones(harness: Harness) -> None:
    out = harness.call("resolve_recipe", routing=_estimates_routing(harness))

    stored = _stored_recipe(out.json["recipe"]).column_estimates
    rows = records(out.json["estimates"])
    assert [r["column"] for r in rows] == list(stored) == ["income"]
    assert rows[0]["feature_cols"] == list(stored["income"].feature_cols)


def test_domain_snap_bounds_show_and_override_as_low_and_high(harness: Harness) -> None:
    profile = _profile(
        harness, numeric_kinds=[{"column": "income", "numeric_kind": "bounded_discrete"}]
    )
    routing = harness.code_mode(
        "route",
        profile=profile,
        strategies=[
            {"column": "income", "strategy": "mice"},
            {"column": "age", "strategy": "median"},
        ],
    ).json["routing"]
    resolved = records(harness.call("resolve_recipe", routing=routing).json["estimates"])
    low, high = _stored_recipe(
        harness.call("resolve_recipe", routing=routing).json["recipe"]
    ).column_estimates["income"].domain_snap_bounds
    assert resolved == [
        {"column": "income", "center1": None, "center2": None, "feature_cols": None,
         "domain_snap_bounds": {"low": low, "high": high}},
    ]

    out = harness.call(
        "resolve_recipe",
        routing=routing,
        estimates=[{"column": "income", "domain_snap_bounds": {"low": 0, "high": 100000}}],
    )

    assert records(out.json["estimates"])[0]["domain_snap_bounds"] == {"low": 0.0, "high": 100000.0}
    stored = _stored_recipe(out.json["recipe"]).column_estimates["income"]
    assert stored.domain_snap_bounds == (0.0, 100000.0)


def test_an_unknown_column_is_refused_with_candidates(harness: Harness) -> None:
    out = _estimate(harness, {"column": "incme", "feature_cols": ["age"]})

    assert out.code == "unknown_column"
    assert out.json["param"] == "estimates[0].column"
    assert out.json["candidates"] == ["income"]
    assert _recipe_handles(harness) == []


def test_a_column_named_twice_is_a_duplicate_key(harness: Harness) -> None:
    entry = {"column": "income", "center1": 1.0, "center2": 2.0}

    out = _estimate(harness, entry, entry)

    assert out.code == "duplicate_key"
    assert out.json["params"] == ["estimates[1].column"]
    assert out.json["columns"] == ["income"]
    assert _recipe_handles(harness) == []


def test_a_column_that_reads_no_estimates_is_refused(harness: Harness) -> None:
    out = _estimate(harness, {"column": "age", "center1": 1.0})

    assert out.code == "estimates_not_applicable"
    assert out.json["param"] == "estimates[0].column"
    assert out.json["columns"] == ["age"]
    assert _recipe_handles(harness) == []


def test_a_field_the_strategy_does_not_read_is_refused(harness: Harness) -> None:
    routing = harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "gmm_sampling"}],
    ).json["routing"]

    out = harness.call(
        "resolve_recipe",
        routing=routing,
        estimates=[
            {"column": "income", "feature_cols": ["age"],
             "domain_snap_bounds": {"low": 0, "high": 1}},
        ],
    )

    assert out.code == "estimates_not_applicable"
    assert out.json["params"] == [
        "estimates[0].feature_cols",
        "estimates[0].domain_snap_bounds",
    ]
    assert _recipe_handles(harness) == []


def test_an_unknown_feature_column_is_refused(harness: Harness) -> None:
    out = _estimate(harness, {"column": "income", "feature_cols": ["age", "agee"]})

    assert out.code == "unknown_column"
    assert out.json["param"] == "estimates[0].feature_cols[1]"
    assert out.json["candidates"] == ["age"]
    assert _recipe_handles(harness) == []


def test_a_column_cannot_be_its_own_feature_or_a_non_numeric_one(harness: Harness) -> None:
    out = _estimate(harness, {"column": "income", "feature_cols": ["income", "city"]})

    assert out.code == "invalid_argument"
    assert out.json["params"] == [
        "estimates[0].feature_cols[0]",
        "estimates[0].feature_cols[1]",
    ]
    assert _recipe_handles(harness) == []


def test_bounds_with_low_above_high_are_refused(harness: Harness) -> None:
    profile = _profile(
        harness, numeric_kinds=[{"column": "income", "numeric_kind": "bounded_discrete"}]
    )
    routing = harness.code_mode(
        "route", profile=profile, strategies=[{"column": "income", "strategy": "mice"}]
    ).json["routing"]

    out = harness.call(
        "resolve_recipe",
        routing=routing,
        estimates=[{"column": "income", "domain_snap_bounds": {"low": 5, "high": 1}}],
    )

    assert out.code == "invalid_argument"
    assert out.json["param"] == "estimates[0].domain_snap_bounds"
    assert _recipe_handles(harness) == []


def test_one_centre_over_a_missing_pair_is_refused(harness: Harness) -> None:
    # The fixture's ``income`` has one mode, so a forced bimodal column has no centres.
    out = _estimate(harness, {"column": "income", "center1": 1.0})

    assert out.code == "invalid_argument"
    assert out.json["param"] == "estimates[0].center2"
    assert _recipe_handles(harness) == []


def test_hyperparameter_and_estimate_problems_come_back_in_one_error(harness: Harness) -> None:
    out = harness.call(
        "resolve_recipe",
        routing=_estimates_routing(harness),
        hyperparameters=[{"unit_id": "mcie", "hyperparameters": {"kind": "mice"}}],
        estimates=[{"column": "incme", "center1": 1.0}],
    )

    assert out.code == "unknown_unit"
    assert out.json["params"] == ["hyperparameters[0].unit_id", "estimates[0].column"]
    assert len(out.json["issues"]) == 2
    assert _recipe_handles(harness) == []


# --- Caps -------------------------------------------------------------------------


def _flagged_wide_profile(harness: Harness, tmp_path: Path) -> str:
    wide = write_flagged_wide_csv(tmp_path / "flagged.csv", 150, name_length=8)
    harness.declare(wide)
    frame = harness.code_mode("load", path=str(wide)).json["frame"]
    return harness.code_mode("profile", frame=frame).json["profile"]


def test_units_are_capped_at_fifty(harness: Harness, tmp_path: Path) -> None:
    routing = harness.code_mode(
        "route", profile=_flagged_wide_profile(harness, tmp_path), default_strategy="median"
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    units = records(out.json["units"])
    assert len(units) == 50
    assert units[0] == {
        "unit_id": "median:c0000_xx", "strategy": "median", "columns": ["c0000_xx"],
        "hyperparameters": None,
    }
    assert records(out.json["omitted"]) == [
        {"field": "units", "shown": 50, "total": 150, "operation": "explain_routing",
         "reason": "cap"},
    ]


def test_estimates_are_capped_at_fifteen_columns_of_ten_features(
    harness: Harness, tmp_path: Path
) -> None:
    # Every column of the wide table correlates with every other, so each
    # ``cluster_conditional`` column resolves 149 features.
    routing = harness.code_mode(
        "route",
        profile=_flagged_wide_profile(harness, tmp_path),
        default_strategy="cluster_conditional",
    ).json["routing"]

    out = harness.call("resolve_recipe", routing=routing)

    rows = records(out.json["estimates"])
    assert [r["column"] for r in rows] == [f"c{i:04d}_xx" for i in range(15)]
    assert rows[0]["feature_cols"] == [f"c{i:04d}_xx" for i in range(1, 11)]
    omitted = records(out.json["omitted"])
    assert {"field": "estimates", "shown": 15, "total": 150, "operation": None,
            "reason": "cap"} in omitted
    assert {"field": "estimates[0].feature_cols", "shown": 10, "total": 149,
            "operation": None, "reason": "cap"} in omitted


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
