"""``explain_routing`` end to end, in both deliveries: stored Routing Evidence, paged."""

from __future__ import annotations

import inspect
import re
import typing

from dataforge_ml.imputation import _escalation
from dataforge_ml.imputation._evidence import EntryCause
from dataforge_ml.operations.explain_routing import EntryCauseName, TermName
from dataforge_ml.records import InvalidInput, UnknownName

from .conftest import AMES, Harness, records
from .test_route import N_ROWS, _by_column, _profile


def _ames_routing(harness: Harness) -> str:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES), targets=["SalePrice"])
    return harness.code_mode("route", profile=profile).json["routing"]


# --- The evidence -----------------------------------------------------------------


def test_a_routed_column_carries_its_whole_walk(harness: Harness) -> None:
    routing = harness.code_mode("route", profile=_profile(harness)).json["routing"]

    out = harness.call("explain_routing", routing=routing, columns=["income"])

    assert not out.is_error
    (row,) = records(out.json["columns"])
    assert (row["column"], row["strategy"], row["reason"]) == (
        "income", "mean", "bottom_tier_fill",
    )
    assert row["entry"] == ["mcar_high_severity"]
    score = row["signal_score"]
    assert 0 <= score["score"] <= 1 and score["predictors"] == 1
    assert row["tier"] == "bottom"
    assert [(c["strategy"], c["passed"]) for c in row["feasibility"]] == [
        ("mice", True), ("knn", True),
    ]


def test_a_failed_term_names_its_dial(harness: Harness) -> None:
    routing = harness.code_mode(
        "route", profile=_profile(harness), thresholds={"numeric": {"knn_max_rows": 1}}
    ).json["routing"]

    out = harness.call("explain_routing", routing=routing, columns=["income"])

    knn = next(c for c in _by_column(out)["income"]["feasibility"] if c["strategy"] == "knn")
    assert knn["passed"] is False
    (term,) = knn["failed_terms"]
    assert (term["term"], term["value"], term["limit"]) == ("row_count", N_ROWS, 1)
    assert term["dial"] == {
        "operation": "route", "dial": "thresholds.numeric.knn_max_rows",
        "value": 1, "source": "argument",
    }


def test_an_authored_column_has_empty_evidence(harness: Harness) -> None:
    routing = harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "median"}],
    ).json["routing"]

    out = harness.call("explain_routing", routing=routing)

    row = _by_column(out)["income"]
    assert row["reason"] == "authored"
    assert (row["entry"], row["signal_score"], row["tier"]) == ([], None, None)
    assert row["feasibility"] == []


def test_a_forced_column_keeps_its_one_feasibility_check(harness: Harness) -> None:
    routing = harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "knn"}],
        thresholds={"numeric": {"knn_max_rows": 1}},
    ).json["routing"]

    out = harness.call("explain_routing", routing=routing, columns=["income"])

    row = _by_column(out)["income"]
    assert (row["entry"], row["signal_score"], row["tier"]) == ([], None, None)
    (check,) = row["feasibility"]
    assert (check["strategy"], check["passed"]) == ("knn", False)
    assert [t["term"] for t in check["failed_terms"]] == ["row_count"]


def test_indicator_and_passthrough_rows_are_explained_too(harness: Harness) -> None:
    routing = harness.code_mode(
        "route",
        profile=_profile(harness),
        strategies=[{"column": "income", "strategy": "mnar"}],
    ).json["routing"]

    rows = _by_column(harness.call("explain_routing", routing=routing))

    assert rows["income_missing"]["strategy"] == "indicator"
    assert rows["city"]["reason"] == "not_imputed_type"
    assert all(r["feasibility"] == [] for r in (rows["income_missing"], rows["city"]))


# --- Another process --------------------------------------------------------------


def test_it_explains_a_routing_another_process_made(harness: Harness) -> None:
    profile = _profile(harness)
    made_by_cli = harness.cli("route", profile=profile)
    assert made_by_cli.exit_code == 0, made_by_cli.stdout
    routing = made_by_cli.json["routing"]

    out = harness.call("explain_routing", routing=routing)  # Code Mode and a fresh CLI

    assert out.json["routing"] == routing and out.json["profile"] == profile
    assert set(_by_column(out)) == {"age", "city", "income"}


# --- Order and paging -------------------------------------------------------------


def test_columns_follow_routes_order(harness: Harness) -> None:
    profile = _profile(harness)
    routed = harness.code_mode(
        "route", profile=profile, strategies=[{"column": "age", "strategy": "mnar"}]
    ).json

    out = harness.call("explain_routing", routing=routed["routing"])

    assert [r["column"] for r in records(out.json["columns"])] == [
        r["column"] for r in records(routed["columns"])
    ]


def test_pages_are_capped_at_fifteen_and_name_the_rest(harness: Harness) -> None:
    routing = _ames_routing(harness)

    first = harness.call("explain_routing", routing=routing)

    assert first.json["total"] == 82 and first.json["offset"] == 0
    assert len(records(first.json["columns"])) == 15
    assert records(first.json["omitted"]) == [
        {"field": "columns", "shown": 15, "total": 82, "operation": "explain_routing",
         "reason": "cap"},
    ]
    seen: list[str] = []
    for offset in range(0, 82, 15):
        page = harness.code_mode("explain_routing", routing=routing, offset=offset).json
        seen += [r["column"] for r in records(page["columns"])]
    last = harness.code_mode("explain_routing", routing=routing, offset=75).json
    assert records(last["omitted"]) == []
    assert len(seen) == len(set(seen)) == 82


def test_a_limit_past_the_cap_is_invalid(harness: Harness) -> None:
    routing = harness.code_mode("route", profile=_profile(harness)).json["routing"]

    out = harness.call("explain_routing", routing=routing, limit=16)

    assert out.code == "invalid_argument" and isinstance(out.error, InvalidInput)


# --- Errors -----------------------------------------------------------------------


def test_unknown_columns_are_named_with_candidates(harness: Harness) -> None:
    routing = harness.code_mode("route", profile=_profile(harness)).json["routing"]

    out = harness.call("explain_routing", routing=routing, columns=["income", "incme", "cty"])

    assert out.code == "unknown_column" and isinstance(out.error, UnknownName)
    assert out.json["params"] == ["columns[1]", "columns[2]"]
    assert out.json["candidates"][0] == "income"


def test_an_excluded_column_is_unknown_and_says_why(harness: Harness) -> None:
    routing = harness.code_mode(
        "route", profile=_profile(harness), exclude_columns=["age"]
    ).json["routing"]

    out = harness.call("explain_routing", routing=routing, columns=["age"])

    assert out.code == "unknown_column"
    assert "excluded" in out.json["message"]


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    out = harness.call("explain_routing", routing=_profile(harness))

    assert out.code == "wrong_handle_kind"


def test_an_unknown_routing_lists_routing_candidates(harness: Harness) -> None:
    routing = harness.code_mode("route", profile=_profile(harness)).json["routing"]

    out = harness.call("explain_routing", routing="routing_00000000")

    assert out.code == "unknown_handle"
    assert out.json["candidates"] == [routing]


# --- The wire vocabulary ----------------------------------------------------------


def test_the_wire_entry_causes_match_the_library_enum() -> None:
    assert set(typing.get_args(EntryCauseName)) == {c.value for c in EntryCause}


def test_every_failed_term_the_router_writes_is_on_the_wire() -> None:
    written = set(re.findall(r'term="(\w+)"', inspect.getsource(_escalation)))
    assert written == set(typing.get_args(TermName))
