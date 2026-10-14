"""``route`` end to end, in both deliveries: the routing Handle and its Routing Reasons."""

from __future__ import annotations

import re
import typing
from pathlib import Path
from typing import Any

import pytest

from dataforge_ml._boundary._route import ROUTING_DIALS, dial_defaults
from dataforge_ml._boundary._workspace import payload_path, read_record
from dataforge_ml._serialization import deserialize
from dataforge_ml.imputation import ImputationStrategy, RoutingReason
from dataforge_ml.operations.route import (
    DeclarableStrategy,
    ReasonName,
    RouteThresholds,
    RoutingDials,
    StrategyName,
)
from dataforge_ml.records import Conflict, Infeasible, InvalidInput, UnknownName

from .conftest import AMES, Harness, records

#: Rows in the routing fixture: enough that ``age`` reads as numeric, not an identifier.
N_ROWS = 40


def _routing_csv(harness: Harness) -> Path:
    # ``age`` is numeric and complete, ``income`` numeric with gaps, ``city``
    # categorical with gaps.
    csv = harness.inputs / "routing.csv"
    cities = ("north", "south", "east")
    lines = ["age,city,income"] + [
        f"{20.5 + i % 15},{'' if i % 7 == 3 else cities[i % 3]},"
        f"{'' if i % 5 == 0 else 40000 + 750.25 * i}"
        for i in range(N_ROWS)
    ]
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return csv


def _profile(harness: Harness, path: str | None = None, **arguments: Any) -> str:
    path = path or str(_routing_csv(harness))
    frame = harness.code_mode("load", path=path).json["frame"]
    return harness.code_mode("profile", frame=frame, **arguments).json["profile"]


def _routing_handles(harness: Harness) -> list[str]:
    return [h for h in harness.handles() if h.startswith("routing_")]


def _stored_routing(handle: str) -> Any:
    return deserialize(payload_path(handle, read_record(handle, "routing", param="routing")).read_bytes())


def _by_column(out: Any) -> dict[str, dict[str, Any]]:
    return {row["column"]: row for row in records(out.json["columns"])}


# --- The routing ------------------------------------------------------------------


def test_route_returns_a_routing_handle_and_a_reason_for_every_column(harness: Harness) -> None:
    profile = _profile(harness)

    out = harness.call("route", profile=profile)

    assert not out.is_error
    assert out.json["routing"] in _routing_handles(harness)
    assert out.json["profile"] == profile
    rows = _by_column(out)
    assert set(rows) == {"age", "city", "income"}
    assert all(row["reason"] in RoutingReason.__members__ for row in rows.values())
    assert rows["age"] == {
        "column": "age", "semantic_type": "numeric", "strategy": "passthrough",
        "reason": "no_missing_values",
    }
    assert (rows["city"]["strategy"], rows["city"]["reason"]) == ("passthrough", "not_imputed_type")
    assert rows["income"]["strategy"] != "passthrough"
    assert next(r["column"] for r in records(out.json["columns"])) == "income", "fills first"
    assert sum(c["count"] for c in records(out.json["census"])) == 3


def test_non_numeric_missing_values_earn_one_warning(harness: Harness) -> None:
    out = harness.call("route", profile=_profile(harness))

    warnings = records(out.json["warnings"])
    assert [w["code"] for w in warnings] == ["non_numeric_missing_kept"]
    assert warnings[0]["columns"] == ["city"]


def test_ames_routing_fits_the_result_budget_whole(harness: Harness) -> None:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES), targets=["SalePrice"])

    out = harness.call("route", profile=profile)  # budget checked inside

    rows = records(out.json["columns"])
    assert len(rows) == 82 and records(out.json["omitted"]) == []
    assert all(r["reason"] for r in rows)
    passthrough = [r["strategy"] == "passthrough" for r in rows]
    assert passthrough == sorted(passthrough), "columns that fill come first"
    assert [w["code"] for w in records(out.json["warnings"])] == ["non_numeric_missing_kept"]


# --- Authoring --------------------------------------------------------------------


def test_authored_strategies_take_the_full_vocabulary(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[
            {"column": "age", "strategy": "mnar"},
            {"column": "income", "strategy": "constant", "constant_fill": 0},
            {"column": "city", "strategy": "dropped"},
        ],
    )

    rows = _by_column(out)
    assert {c: (r["strategy"], r["reason"]) for c, r in rows.items()} == {
        "age": ("mnar", "authored"),
        "income": ("constant", "authored"),
        "city": ("dropped", "authored"),
        "age_missing": ("indicator", "authored"),
    }
    assert rows["age_missing"]["semantic_type"] == "boolean"
    assert out.json["warnings"]["rows"] == [], "every column's nulls are addressed"


def test_indicator_is_refused_with_a_coded_error(harness: Harness) -> None:
    out = harness.call(
        "route", profile=_profile(harness), strategies=[{"column": "age", "strategy": "indicator"}]
    )

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["strategies[0].strategy"]
    assert "indicator" not in out.json["allowed"] and "passthrough" in out.json["allowed"]
    assert _routing_handles(harness) == []


def test_null_default_strategy_auto_routes_unlisted_columns(harness: Harness) -> None:
    profile = _profile(harness)
    auto = _by_column(harness.code_mode("route", profile=profile))

    out = harness.call(
        "route", profile=profile, strategies=[{"column": "age", "strategy": "median"}]
    )

    rows = _by_column(out)
    assert (rows["age"]["strategy"], rows["age"]["reason"]) == ("median", "authored")
    assert rows["income"] == auto["income"] and rows["city"] == auto["city"]


def test_a_default_strategy_stamps_every_unlisted_numeric_column(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "median"}],
        default_strategy="mean",
    )

    rows = _by_column(out)
    assert (rows["age"]["strategy"], rows["age"]["reason"]) == ("mean", "authored")
    assert rows["income"]["strategy"] == "median"
    assert (rows["city"]["strategy"], rows["city"]["reason"]) == ("passthrough", "not_imputed_type")


# --- Feasibility ------------------------------------------------------------------


def test_forcing_past_the_feasibility_floor_warns_once(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[
            {"column": "age", "strategy": "gmm_sampling"},
            {"column": "income", "strategy": "knn"},
        ],
        thresholds={"numeric": {"knn_max_rows": 1}},
    )

    warnings = records(out.json["warnings"])
    forced = [w for w in warnings if w["code"] == "forced_past_feasibility"]
    assert len(forced) == 1 and forced[0]["columns"] == ["age", "income"]
    assert forced[0]["param"] == "strategies"
    assert next(w["code"] for w in warnings) == "forced_past_feasibility"


def test_a_forced_column_stores_its_failed_terms_with_dial_pointers(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "knn"}],
        thresholds={"numeric": {"knn_max_rows": 1}},
    )

    # explain_routing (#598) will read this; until then, read the Handle itself.
    routing = _stored_routing(out.json["routing"])
    (fact,) = routing.column_routings["income"].evidence
    assert (fact.kind, fact.strategy, fact.passed) == ("feasibility", "knn", False)
    (term,) = fact.failed_terms
    assert (term.term, term.value, term.limit) == ("row_count", N_ROWS, 1)
    assert term.dial.to_dict() == {
        "operation": "route", "dial": "thresholds.numeric.knn_max_rows",
        "value": 1, "source": "argument",
    }


def test_a_column_forced_within_the_floor_does_not_warn(harness: Harness) -> None:
    out = harness.call(
        "route", profile=_profile(harness), strategies=[{"column": "income", "strategy": "knn"}]
    )

    assert "forced_past_feasibility" not in [w["code"] for w in records(out.json["warnings"])]


# --- The MICE estimator -----------------------------------------------------------


def test_the_mice_estimator_and_its_reason_are_inline(harness: Harness) -> None:
    profile = _profile(harness)
    mice = [{"column": c, "strategy": "mice"} for c in ("age", "income")]

    laddered = harness.call("route", profile=profile, strategies=mice)
    chosen = harness.call(
        "route", profile=profile, strategies=mice, model_choice="gradient_boosting_regressor"
    )

    reason = laddered.json["estimator_reason"]
    assert laddered.json["mice_estimator"] in ("bayesian_ridge", "random_forest_regressor")
    assert reason["source"] == "ladder" and reason["column"] in ("age", "income")
    assert chosen.json["mice_estimator"] == "gradient_boosting_regressor"
    assert chosen.json["estimator_reason"] == {**reason, "source": "argument"}


def test_no_mice_block_means_no_estimator(harness: Harness) -> None:
    out = harness.call(
        "route", profile=_profile(harness), strategies=[{"column": "income", "strategy": "mean"}]
    )

    assert out.json["mice_estimator"] is None and out.json["estimator_reason"] is None


def test_a_model_choice_without_a_mice_block_is_infeasible(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "mean"}],
        model_choice="bayesian_ridge",
    )

    assert out.code == "no_mice_block" and isinstance(out.error, Infeasible)
    assert out.json["param"] == "model_choice"
    assert _routing_handles(harness) == []


# --- Exclusions -------------------------------------------------------------------


def test_excluded_columns_leave_and_skipped_columns_pass_through(harness: Harness) -> None:
    out = harness.call(
        "route", profile=_profile(harness), exclude_columns=["income"], skip_columns=["age"]
    )

    rows = _by_column(out)
    assert "income" not in rows
    assert (rows["age"]["strategy"], rows["age"]["reason"]) == ("passthrough", "soft_excluded")


def test_a_column_profiling_skipped_passes_through_untyped(harness: Harness) -> None:
    out = harness.call("route", profile=_profile(harness, skip_columns=["city"]))

    city = _by_column(out)["city"]
    assert city == {
        "column": "city", "semantic_type": None, "strategy": "passthrough",
        "reason": "soft_excluded",
    }
    refused = harness.call(
        "route",
        profile=_profile(harness, skip_columns=["city"]),
        strategies=[{"column": "city", "strategy": "mode"}],
    )
    assert refused.code == "strategy_not_declarable"


def test_the_profiles_exclusions_carry_into_the_routing(harness: Harness) -> None:
    profile = _profile(harness, exclude_columns=["income"])

    out = harness.call("route", profile=profile, exclude_columns=["income"])

    assert "income" not in _by_column(out)
    record = read_record(out.json["routing"], "routing", param="routing")
    assert record["declarations"]["exclusions"] == ["income"]


# --- Errors -----------------------------------------------------------------------


def test_every_bad_argument_comes_back_in_one_error(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[
            {"column": "incom", "strategy": "mean"},
            {"column": "city", "strategy": "mean"},
            {"column": "income", "strategy": "constant"},
        ],
        exclude_columns=["agee"],
        thresholds={"numeric": {"signal_score_middle_tier_min": 0.9}},
    )

    assert out.code == "unknown_column" and isinstance(out.error, UnknownName)
    assert out.json["params"] == [
        "strategies[0].column",
        "strategies[1].strategy",
        "strategies[2].constant_fill",
        "exclude_columns[0]",
        "thresholds.numeric.signal_score_middle_tier_min",
    ]
    assert out.json["candidates"] == ["income", "age"]
    assert _routing_handles(harness) == []


def test_a_column_listed_twice_is_a_duplicate_key(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        strategies=[
            {"column": "age", "strategy": "mean"},
            {"column": "age", "strategy": "median"},
        ],
    )

    assert out.code == "duplicate_key" and isinstance(out.error, InvalidInput)
    assert out.json["param"] == "strategies[1].column"


@pytest.mark.parametrize(
    ("arguments", "param"),
    [
        (
            {"strategies": [{"column": "age", "strategy": "mean"}], "exclude_columns": ["age"]},
            "strategies[0].column",
        ),
        (
            {"strategies": [{"column": "age", "strategy": "mean"}], "skip_columns": ["age"]},
            "strategies[0].column",
        ),
        ({"exclude_columns": ["income"], "skip_columns": ["income"]}, "skip_columns[0]"),
    ],
)
def test_contradicting_declarations_are_a_conflict(
    harness: Harness, arguments: dict[str, Any], param: str
) -> None:
    out = harness.call("route", profile=_profile(harness), **arguments)

    assert out.code == "declaration_conflict" and isinstance(out.error, Conflict)
    assert out.json["param"] == param


def test_excluding_a_target_is_a_conflict(harness: Harness) -> None:
    out = harness.call(
        "route", profile=_profile(harness, targets=["income"]), exclude_columns=["income"]
    )

    assert out.code == "declaration_conflict"


@pytest.mark.parametrize(
    ("entry", "code", "param"),
    [
        ({"column": "city", "strategy": "mode"}, "strategy_not_declarable", "strategies[0].strategy"),
        ({"column": "age", "strategy": "constant"}, "constant_fill_missing", "strategies[0].constant_fill"),
        (
            {"column": "age", "strategy": "mean", "constant_fill": 1.5},
            "invalid_argument",
            "strategies[0].constant_fill",
        ),
        (
            {"column": "age", "strategy": "mean", "grouping_variable": "city"},
            "invalid_argument",
            "strategies[0].grouping_variable",
        ),
        (
            {"column": "age", "strategy": "cluster_conditional", "grouping_variable": "cty"},
            "unknown_column",
            "strategies[0].grouping_variable",
        ),
    ],
)
def test_a_malformed_strategy_entry_names_its_field(
    harness: Harness, entry: dict[str, Any], code: str, param: str
) -> None:
    out = harness.call("route", profile=_profile(harness), strategies=[entry])

    assert out.code == code
    assert out.json["param"] == param


def test_dials_out_of_order_are_an_invalid_dial_value(harness: Harness) -> None:
    out = harness.call(
        "route",
        profile=_profile(harness),
        thresholds={"numeric": {"signal_score_top_tier_min": 0.1}},
    )

    assert out.code == "invalid_dial_value"
    assert out.json["param"] == "thresholds.numeric.signal_score_top_tier_min"


def test_a_profile_argument_of_another_kind_is_refused(harness: Harness) -> None:
    frame = harness.code_mode("load", path=str(_routing_csv(harness))).json["frame"]

    out = harness.call("route", profile=frame)

    assert out.code == "wrong_handle_kind" and out.json["allowed"] == ["profile"]


# --- Literal containment ----------------------------------------------------------


def test_no_cell_value_reaches_any_route_output(harness: Harness, tmp_path: Path) -> None:
    # route has no Literal Field, so a canary cell may appear nowhere: not in
    # a result, a warning or an error, in either delivery.
    csv = tmp_path / "canary.csv"
    lines = ["label,amount,score"] + [
        f"{'CANARY_LABEL_Q7' if i % 2 else ''},{31337.271828 if i % 3 else ''},{i * 1.5}"
        for i in range(30)
    ]
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    profile = _profile(harness, str(csv))

    outputs = [
        harness.call("route", profile=profile),
        harness.call(
            "route",
            profile=profile,
            strategies=[{"column": "amount", "strategy": "gmm_sampling"}],
            thresholds={"numeric": {"knn_max_rows": 1}},
        ),
        harness.call("route", profile=profile, strategies=[{"column": "label", "strategy": "mean"}]),
    ]

    text = " ".join(str(o.json) for o in outputs)
    assert "CANARY_LABEL_Q7" not in text and "31337" not in text
    assert outputs[2].is_error


# --- The wire vocabulary stays in step with the library ---------------------------


def test_every_dial_on_the_wire_states_the_library_default() -> None:
    assert list(RoutingDials.model_fields) == list(ROUTING_DIALS)
    assert list(RouteThresholds.model_fields) == ["numeric"]
    defaults = dial_defaults()
    for name, dial in RoutingDials.model_fields.items():
        stated = re.search(r"Null means (-?[0-9.]+)\.$", dial.description or "")
        assert stated, f"{name}: description must end with its default"
        assert float(stated.group(1)) == pytest.approx(defaults[f"numeric.{name}"]), name


def test_the_wire_enums_match_the_library_enums() -> None:
    assert set(typing.get_args(ReasonName)) == {r.value for r in RoutingReason}
    assert set(typing.get_args(StrategyName)) == {s.value for s in ImputationStrategy}
    assert set(typing.get_args(DeclarableStrategy)) == {
        s.value for s in ImputationStrategy
    } - {"indicator"}
