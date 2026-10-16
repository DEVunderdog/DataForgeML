"""The five profile detail Operations end to end, in both deliveries.

``describe_columns``, ``top_associations``, ``missingness``, ``describe_target``
and ``value_summary`` read a stored profile Handle. Each test checks only
what crosses the boundary: the result JSON and the Library Error record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataforge_ml._boundary._operation import canonical_json, operations
from dataforge_ml.records import UnknownName, literal_field_paths

from .conftest import AMES, Harness, records, write_flagged_wide_csv


def _profile(harness: Harness, path: str = "train.csv", **arguments: Any) -> str:
    frame = harness.code_mode("load", path=path).json["frame"]
    out = harness.code_mode("profile", frame=frame, **arguments)
    assert not out.is_error, out.json
    return out.json["profile"]


def _csv(harness: Harness, path: Path, text: str) -> str:
    path.write_text(text)
    harness.declare(path)
    return str(path)


def _wide(harness: Harness, tmp_path: Path, n_columns: int = 30) -> str:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", n_columns)
    harness.declare(wide)
    return _profile(harness, str(wide))


def _mixed_csv(harness: Harness, tmp_path: Path) -> str:
    # 60 rows: two numeric columns that go missing together, an independent
    # one, a categorical with a rare label, and a -999 sentinel nobody declared.
    lines = ["a,b,c,grade,reading"]
    for i in range(60):
        gap = i % 6 == 0
        a = "" if gap else str(i * 1.5)
        b = "" if gap else str(100 - i * 0.7 + (i % 3))
        c = "" if i % 7 == 3 else str((i * 37) % 11)
        grade = "rare" if i == 0 else ("high" if i % 2 else "low")
        reading = "-999" if i % 10 == 0 else str(20 + (i % 9))
        lines.append(f"{a},{b},{c},{grade},{reading}")
    return _csv(harness, tmp_path / "mixed.csv", "\n".join(lines) + "\n")


# --- Every Operation, on Ames, within budget ------------------------------------------


def test_all_five_work_on_ames_within_the_result_budget(harness: Harness) -> None:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES), targets=["SalePrice"])

    # Each call checks the budget and that both deliveries agree.
    columns = harness.call("describe_columns", profile=profile)
    pairs = harness.call("top_associations", profile=profile)
    missing = harness.call("missingness", profile=profile)
    target = harness.call("describe_target", profile=profile, name="SalePrice")
    values = harness.call("value_summary", profile=profile, column="Neighborhood")

    assert columns.json["total"] == 82 and len(records(columns.json["columns"])) == 15
    assert len(records(pairs.json["pairs"])) == 100 and pairs.json["total"] > 100
    assert missing.json["columns_with_missing"] > 0
    assert target.json["problem_type"] == "regression"
    assert records(values.json["top_values"])[0]["value"] == "NAmes"


# --- describe_columns ---------------------------------------------------------------------


def test_describe_columns_gives_full_profiles_without_cell_values(
    harness: Harness,
) -> None:
    profile = _profile(harness, targets=["income"])

    out = harness.call("describe_columns", profile=profile)

    assert (out.json["total"], out.json["offset"]) == (3, 0)
    income = out.record.get("income")
    assert (income.role, income.semantic_type, income.cardinality) == (
        "target",
        "numeric",
        5,
    )
    assert income.missingness.null_count == 1 and income.missingness.null_ratio == 1 / 6
    assert income.stats.kind == "numeric" and income.stats.mean == pytest.approx(
        60000.3
    )
    city = out.record.get("city")
    assert city.stats.kind == "categorical" and city.stats.mode_share == pytest.approx(
        2 / 6
    )
    assert literal_field_paths(operations()["describe_columns"].result_model) == []


def test_describe_columns_pages_flagged_first_and_statelessly(harness: Harness) -> None:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES))

    whole: list[dict[str, Any]] = []
    for offset in range(0, 82, 15):
        page = harness.code_mode("describe_columns", profile=profile, offset=offset)
        whole += records(page.json["columns"])
    first = harness.call("describe_columns", profile=profile, offset=15, limit=15)
    again = harness.call("describe_columns", profile=profile, offset=15, limit=15)

    assert len(whole) == 82 and len({c["column"] for c in whole}) == 82
    flagged = [bool(c["flags"]) for c in whole]
    assert flagged == sorted(flagged, reverse=True), "flagged columns come first"
    assert canonical_json(first.json) == canonical_json(again.json)
    assert records(first.json["omitted"]) == [
        {"field": "columns", "shown": 15, "total": 82, "operation": "describe_columns", "reason": "cap"}
    ]
    assert records(first.json["columns"]) == whole[15:30]


def test_describe_columns_filters_by_names_and_flags(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    named = harness.call("describe_columns", profile=profile, names=["grade", "a"])
    flagged = harness.call("describe_columns", profile=profile, flagged_only=True)

    assert [c["column"] for c in records(named.json["columns"])] == ["a", "grade"]
    shown = records(flagged.json["columns"])
    assert shown and all(c["flags"] for c in shown)
    raised = {f for c in shown for f in c["flags"]}
    assert {e["flag"] for e in records(flagged.json["flag_dials"])} == raised


def test_describe_columns_points_each_flag_at_its_dial(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", 3)
    harness.declare(wide)
    profile = _profile(
        harness, str(wide), thresholds={"missingness": {"col_drop_threshold": 0.4}}
    )

    out = harness.call("describe_columns", profile=profile)

    pointers = {e["flag"]: e["dials"] for e in records(out.json["flag_dials"])}
    assert pointers["missingness.drop_candidate"] == [
        {
            "operation": "profile",
            "dial": "thresholds.missingness.col_drop_threshold",
            "value": 0.4,
            "source": "argument",
        }
    ]


def test_long_co_missing_lists_are_capped_naming_missingness(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _wide(harness, tmp_path, 30)

    out = harness.call("describe_columns", profile=profile, limit=2)

    first = records(out.json["columns"])[0]
    assert len(first["missingness"]["co_missing_with"]) == 10
    assert {
        "field": "columns[0].missingness.co_missing_with",
        "shown": 10,
        "total": 29,
        "operation": "missingness",
        "reason": "cap",
    } in records(out.json["omitted"])


def test_skipped_columns_are_described_without_statistics(harness: Harness) -> None:
    profile = _profile(harness, skip_columns=["city"])

    out = harness.call("describe_columns", profile=profile, names=["city"])

    city = out.record.get("city")
    assert (city.role, city.semantic_type, city.missingness, city.stats) == (
        "skipped",
        None,
        None,
        None,
    )


def test_unknown_names_come_back_in_one_error_with_candidates(harness: Harness) -> None:
    profile = _profile(harness, exclude_columns=["city"])

    out = harness.call(
        "describe_columns", profile=profile, names=["agee", "age", "city"]
    )

    assert out.code == "unknown_column" and isinstance(out.error, UnknownName)
    assert out.json["params"] == ["names[0]", "names[2]"]
    assert out.json["columns"] == ["agee", "city"]
    assert out.json["candidates"][0] == "age"
    assert "excluded" in out.json["issues"][1]["problem"]


# --- top_associations ---------------------------------------------------------------------


def test_associations_are_one_list_strongest_first(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("top_associations", profile=profile)

    pairs = records(out.json["pairs"])
    strengths = [abs(p["value"]) for p in pairs]
    assert strengths == sorted(strengths, reverse=True)
    assert {"pearson", "eta_squared"} <= {p["metric"] for p in pairs}
    assert [d["metric"] for d in records(out.json["near_redundant_dials"])] == [
        "pearson",
        "cramers_v",
        "eta_squared",
    ]


def test_associations_against_a_target_put_it_first(harness: Harness) -> None:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES), targets=["SalePrice"])

    out = harness.call("top_associations", profile=profile, column="SalePrice", limit=5)

    pairs = records(out.json["pairs"])
    assert len(pairs) == 5 and all(p["column_a"] == "SalePrice" for p in pairs)
    assert pairs[0] == {
        "column_a": "SalePrice",
        "column_b": "Gr Liv Area",
        "metric": "pearson",
        "value": pytest.approx(0.7068, abs=1e-4),
        "near_redundant": False,
    }
    assert records(out.json["omitted"]) == [
        {
            "field": "pairs",
            "shown": 5,
            "total": out.json["total"],
            "operation": "top_associations",
            "reason": "cap",
        }
    ]


def test_associations_filter_by_metric_and_strength(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    pearson = harness.call("top_associations", profile=profile, metric="pearson")
    strong = harness.call("top_associations", profile=profile, min_strength=0.9)

    assert {p["metric"] for p in records(pearson.json["pairs"])} == {"pearson"}
    assert all(abs(p["value"]) >= 0.9 for p in records(strong.json["pairs"]))
    assert strong.json["total"] < pearson.json["total"] + 10


def test_associations_cap_at_100(harness: Harness, tmp_path: Path) -> None:
    profile = _wide(harness, tmp_path, 20)  # 190 Pearson pairs

    out = harness.call("top_associations", profile=profile)

    assert len(records(out.json["pairs"])) == 100
    assert records(out.json["omitted"]) == [
        {"field": "pairs", "shown": 100, "total": 190, "operation": "top_associations", "reason": "cap"}
    ]


def test_an_unknown_association_column_has_candidates(harness: Harness) -> None:
    out = harness.call("top_associations", profile=_profile(harness), column="incme")

    assert out.code == "unknown_column"
    assert (out.json["param"], out.json["candidates"][0]) == ("column", "income")


# --- missingness ----------------------------------------------------------------------------


def test_missingness_gives_rows_and_co_missing_pairs(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("missingness", profile=profile)

    assert out.json["columns_with_missing"] == 3
    rows = out.json["rows"]
    assert (
        rows["drop_candidate_dial"]["dial"]
        == "thresholds.missingness.row_drop_threshold"
    )
    pairs = records(out.json["pairs"])
    assert pairs[0] == {
        "column_a": "a",
        "column_b": "b",
        "correlation": 1.0,
        "mar_suspect": True,
    }
    assert [p["correlation"] for p in pairs] == sorted(
        (p["correlation"] for p in pairs), reverse=True
    )
    assert all(not p["mar_suspect"] for p in pairs[1:])
    assert out.json["mar_suspect_dial"]["value"] == 0.6


def test_missingness_filters_by_column(harness: Harness, tmp_path: Path) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("missingness", profile=profile, column="b")

    pairs = records(out.json["pairs"])
    assert [(p["column_a"], p["column_b"]) for p in pairs] == [("b", "a"), ("b", "c")]


def test_missingness_pairs_cap_at_100(harness: Harness, tmp_path: Path) -> None:
    profile = _wide(harness, tmp_path, 20)  # 190 co-missing pairs

    first = harness.call("missingness", profile=profile)
    last = harness.call("missingness", profile=profile, offset=100)

    assert records(first.json["omitted"]) == [
        {"field": "pairs", "shown": 100, "total": 190, "operation": "missingness", "reason": "cap"}
    ]
    assert (
        len(records(last.json["pairs"])) == 90 and records(last.json["omitted"]) == []
    )


# --- describe_target ------------------------------------------------------------------------


def test_describe_target_gives_a_regression_targets_distribution(
    harness: Harness,
) -> None:
    out = harness.call(
        "describe_target", profile=_profile(harness, targets=["income"]), name="income"
    )

    assert (out.json["problem_type"], out.json["missing_count"]) == ("regression", 1)
    assert out.json["flags"] == ["contains_missing"]
    assert out.json["distribution"]["kind"] == "numeric" and out.json["classes"] is None


def test_describe_target_lists_a_classification_targets_classes(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path), targets=["grade"])

    out = harness.call("describe_target", profile=profile, name="grade")

    assert out.json["problem_type"] == "multiclass_classification"
    classes = out.json["classes"]
    assert classes["class_count"] == 3
    assert [c["label"] for c in records(classes["counts"])] == ["high", "low", "rare"]
    assert literal_field_paths(operations()["describe_target"].result_model) == [
        "classes.counts[].label"
    ]


def test_describe_target_caps_classes_naming_value_summary(harness: Harness) -> None:
    harness.declare(AMES)
    profile = _profile(harness, str(AMES), targets=["Neighborhood"])

    out = harness.call("describe_target", profile=profile, name="Neighborhood")

    assert len(records(out.json["classes"]["counts"])) == 10
    assert records(out.json["omitted"]) == [
        {
            "field": "classes.counts",
            "shown": 10,
            "total": 28,
            "operation": "value_summary",
            "reason": "cap",
        }
    ]


def test_a_column_that_is_not_a_target_is_an_unknown_column(harness: Harness) -> None:
    with_target = _profile(harness, targets=["income"])
    without = _profile(harness)

    wrong = harness.call("describe_target", profile=with_target, name="age")
    none = harness.call("describe_target", profile=without, name="income")

    assert wrong.code == "unknown_column" and wrong.json["candidates"] == ["income"]
    assert none.code == "unknown_column" and "no targets" in none.json["message"]


# --- value_summary --------------------------------------------------------------------------


def test_value_summary_shows_a_sentinel_nobody_declared(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("value_summary", profile=profile, column="reading")

    assert out.json["bounds"] == {"min": -999.0, "max": 28.0}
    assert records(out.json["percentiles"])[0] == {"percentile": 1, "value": -999.0}
    assert -999.0 in [v["value"] for v in records(out.json["top_values"])]


def test_value_summary_gives_labels_and_rare_labels(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("value_summary", profile=profile, column="grade")

    assert [v["value"] for v in records(out.json["top_values"])] == [
        "high",
        "low",
        "rare",
    ]
    assert out.json["rare_labels"] == ["rare"]
    assert (out.json["mode"], out.json["bounds"], out.json["median"]) == (
        "high",
        None,
        None,
    )


def test_value_summary_k_cuts_each_list_naming_value_summary(
    harness: Harness, tmp_path: Path
) -> None:
    profile = _profile(harness, _mixed_csv(harness, tmp_path))

    out = harness.call("value_summary", profile=profile, column="grade", k=1)

    assert [v["value"] for v in records(out.json["top_values"])] == ["high"]
    assert records(out.json["omitted"]) == [
        {"field": "top_values", "shown": 1, "total": 3, "operation": "value_summary", "reason": "cap"}
    ]


def test_value_summary_fields_holding_cell_values_are_literal_fields() -> None:
    assert literal_field_paths(operations()["value_summary"].result_model) == [
        "top_values[].value",
        "rare_labels",
        "bounds.min",
        "bounds.max",
        "median",
        "mode",
        "percentiles[].value",
    ]


@pytest.mark.parametrize(
    "name", ["describe_columns", "top_associations", "missingness"]
)
def test_the_other_detail_results_hold_no_literal_fields(name: str) -> None:
    assert literal_field_paths(operations()[name].result_model) == []


def test_value_summary_k_is_capped_at_20(harness: Harness) -> None:
    out = harness.call("value_summary", profile=_profile(harness), column="city", k=21)

    assert (out.code, out.json["param"]) == ("invalid_argument", "k")


# --- Handles ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("describe_columns", {}),
        ("top_associations", {}),
        ("missingness", {}),
        ("describe_target", {"name": "income"}),
        ("value_summary", {"column": "age"}),
    ],
)
def test_each_takes_only_a_profile_handle(
    harness: Harness, name: str, arguments: dict[str, Any]
) -> None:
    frame = harness.code_mode("load", path="train.csv").json["frame"]
    _profile(harness)

    wrong = harness.call(name, profile=frame, **arguments)
    unknown = harness.call(name, profile="profile_00000000", **arguments)

    assert wrong.code == "wrong_handle_kind"
    assert unknown.code == "unknown_handle" and unknown.json["candidates"]


def test_detail_operations_write_no_handles(harness: Harness) -> None:
    profile = _profile(harness, targets=["income"])
    before = harness.handles()

    harness.code_mode("describe_columns", profile=profile)
    harness.code_mode("top_associations", profile=profile)
    harness.code_mode("missingness", profile=profile)
    harness.code_mode("describe_target", profile=profile, name="income")
    harness.code_mode("value_summary", profile=profile, column="city")

    assert harness.handles() == before
