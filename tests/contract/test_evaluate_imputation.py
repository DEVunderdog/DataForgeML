"""``evaluate_imputation`` end to end, in both deliveries: a ``c2st_`` Handle and a bounded summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataforge_ml._boundary._settings import CORE_BUDGET_ENV
from dataforge_ml._boundary._workspace import payload_path, read_record
from dataforge_ml._seed import DEFAULT_SEED

from .conftest import Harness, records, write_wide_csv
from .test_c2st import _spy_threads

#: Rows in the evaluation fixture.
N_ROWS = 150
#: ``b`` is missing on every third row.
B_MISSING = N_ROWS // 3


def _csv(harness: Harness) -> Path:
    # ``a`` is numeric and complete, ``b`` numeric, tied to ``a`` and missing
    # on every third row, ``city`` a string column with gaps.
    csv = harness.inputs / "evaluate.csv"
    lines = ["a,b,city"]
    for i in range(N_ROWS):
        a = (i * 13) % 47 + 0.5
        b = "" if i % 3 == 1 else f"{2 * a + (i % 5)}"
        city = "" if i % 10 == 7 else "nse"[i % 3]
        lines.append(f"{a},{b},{city}")
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return csv


def _profile(harness: Harness, csv: Path, **arguments: Any) -> str:
    frame = harness.code_mode("load", path=str(csv)).json["frame"]
    return harness.code_mode("profile", frame=frame, **arguments).json["profile"]


def _source(profile: str) -> str:
    return read_record(profile, "profile", param="profile")["frame"]


def _imputer(harness: Harness, profile: str, **route: Any) -> str:
    routing = harness.code_mode("route", profile=profile, **route).json["routing"]
    resolved = harness.code_mode("resolve_recipe", routing=routing).json
    recipe = resolved["recipe"]
    units = [
        harness.code_mode("fit_unit", recipe=recipe, unit_id=unit["unit_id"]).json["unit"]
        for unit in records(resolved["units"])
    ]
    return harness.code_mode("compose_imputer", recipe=recipe, units=units).json["imputer"]


def _imputed(harness: Harness, profile: str, frame: str | None = None, **route: Any) -> str:
    imputer = _imputer(harness, profile, **route)
    frame = frame or _source(profile)
    return harness.code_mode("impute", imputer=imputer, frame=frame).json["frame"]


def _median(harness: Harness, **profile: Any) -> tuple[str, str]:
    profile_handle = _profile(harness, _csv(harness), **profile)
    imputed = _imputed(
        harness, profile_handle, strategies=[{"column": "b", "strategy": "median"}]
    )
    return profile_handle, imputed


def _by_column(out: Any) -> dict[str, dict[str, Any]]:
    return {row["column"]: row for row in records(out.json["columns"])}


def _dials(out: Any) -> dict[str, list[dict[str, Any]]]:
    return {row["outcome"]: row["dials"] for row in records(out.json["outcome_dials"])}


# --- The result ---------------------------------------------------------------------


def test_it_writes_a_c2st_handle_and_an_outcome_per_column(harness: Harness) -> None:
    profile, imputed = _median(harness)

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    assert not out.is_error, out.json
    result = out.json
    assert result["c2st"].startswith("c2st_") and result["c2st"] in harness.handles()
    assert (result["profile"], result["imputed"]) == (profile, imputed)
    assert result["source_frame"] == _source(profile)
    assert (result["rows"], result["seed"]) == (N_ROWS, DEFAULT_SEED)
    rows = _by_column(out)
    assert set(rows) == {"a", "b", "city"}
    assert (rows["a"]["outcome"], rows["a"]["verdict"]) == ("no_filled_cells", "no_verdict")
    assert rows["a"]["mean_frame_z"] is None and rows["a"]["annotations"] == []
    assert rows["city"]["outcome"] == "unfilled"  # passthrough keeps its gaps
    b = rows["b"]
    assert (b["outcome"], b["n_filled"], b["n_observed"]) == (
        "tested", B_MISSING, N_ROWS - B_MISSING,
    )
    assert b["mean_frame_z"] > 3, "a median fill is caught"
    assert b["verdict"] == "flagged"
    assert "low_power" in b["annotations"]
    assert result["warnings"]["rows"] == [], "outcomes are never warnings"


def test_the_census_comes_first_and_flagged_columns_lead(harness: Harness) -> None:
    profile, imputed = _median(harness)

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    fields = [k for k in out.json if k not in ("warnings", "omitted")]
    assert fields.index("census") < fields.index("columns")
    census = records(out.json["census"])
    assert sum(row["count"] for row in census) == 3
    assert {"outcome": "tested", "verdict": "flagged", "count": 1} in census
    assert records(out.json["columns"])[0]["column"] == "b"


def test_threshold_decided_outcomes_carry_dial_pointers(harness: Harness) -> None:
    profile, imputed = _median(harness)

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    dials = _dials(out)
    assert dials["flagged"] == [
        {
            "operation": "evaluate_imputation",
            "dial": "thresholds.fdr_alpha",
            "value": 0.05,
            "source": "default",
        }
    ]
    assert dials["low_power"][0]["dial"] == "thresholds.low_power_below"
    assert "no_filled_cells" not in dials and "unfilled" not in dials, "decided by definition"


def test_a_threshold_moves_the_outcome_and_its_pointer_says_so(harness: Harness) -> None:
    profile, imputed = _median(harness)

    out = harness.call(
        "evaluate_imputation",
        profile=profile,
        imputed=imputed,
        thresholds={"min_filled_cells": B_MISSING + 1},
    )

    assert not out.is_error, out.json
    assert _by_column(out)["b"]["outcome"] == "below_sample_floor"
    assert _by_column(out)["b"]["mean_frame_z"] is None
    assert _dials(out)["below_sample_floor"] == [
        {
            "operation": "evaluate_imputation",
            "dial": "thresholds.min_filled_cells",
            "value": B_MISSING + 1,
            "source": "argument",
        }
    ]


def test_the_seed_is_an_argument(harness: Harness) -> None:
    profile, imputed = _median(harness)

    first = harness.call("evaluate_imputation", profile=profile, imputed=imputed, seed=7)
    second = harness.code_mode("evaluate_imputation", profile=profile, imputed=imputed, seed=7)

    assert first.json["seed"] == 7
    assert _by_column(second) == _by_column(first)


def test_excluded_and_dropped_columns_are_not_evaluated(harness: Harness) -> None:
    profile = _profile(harness, _csv(harness), exclude_columns=["city"])
    imputed = _imputed(harness, profile, strategies=[{"column": "b", "strategy": "dropped"}])

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    assert not out.is_error, out.json
    assert set(_by_column(out)) == {"a"}


def test_a_filled_fold_is_scored_against_the_fold_it_was_filled_from(
    harness: Harness,
) -> None:
    profile = _profile(harness, _csv(harness), exclude_columns=["city"])
    split = harness.code_mode(
        "split", profile=profile, scheme={"kind": "random", "test_size": 0.3}
    ).json
    imputed = _imputed(
        harness, profile, split["test"], strategies=[{"column": "b", "strategy": "median"}]
    )

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    assert not out.is_error, out.json
    assert out.json["source_frame"] == split["test"]
    assert out.json["rows"] == read_record(split["test"], "frame", param="f")["size"]["rows"]
    b = _by_column(out)["b"]
    assert b["n_observed"] + b["n_filled"] == out.json["rows"]


def test_the_handle_holds_the_per_frame_results_and_the_provenance(
    harness: Harness,
) -> None:
    profile, imputed = _median(harness)

    handle = harness.code_mode(
        "evaluate_imputation", profile=profile, imputed=imputed
    ).json["c2st"]

    record = read_record(handle, "c2st", param="report")
    assert record["lineage"]["operation"] == "evaluate_imputation"
    assert record["lineage"]["inputs"][:2] == [profile, imputed]
    assert (record["profile"], record["imputed"]) == (profile, imputed)
    assert record["size"] == {"columns": 3}
    stored = json.loads(payload_path(handle, record).read_text("utf-8"))
    by_column = {c["column"]: c for c in stored["columns"]}
    (frame,) = by_column["b"]["score"]["frames"]
    assert len(frame["repeat_z"]) == 5
    assert by_column["a"]["score"] is None
    provenance = stored["provenance"]
    assert provenance["classifier_params"]["random_state"] == str(DEFAULT_SEED)
    assert provenance["config"]["fdr_alpha"] == 0.05


# --- More than 100 columns -----------------------------------------------------------


def test_past_100_columns_omitted_names_describe_c2st_columns(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_wide_csv(tmp_path / "wide.csv", 101)
    harness.declare(wide)
    profile = _profile(harness, wide)
    imputed = _imputed(harness, profile)

    out = harness.call("evaluate_imputation", profile=profile, imputed=imputed)

    assert not out.is_error, out.json
    assert len(out.json["columns"]["rows"]) == 100
    assert records(out.json["omitted"]) == [
        {
            "field": "columns",
            "shown": 100,
            "total": 101,
            "operation": "describe_c2st_columns",
            "reason": "cap",
        }
    ]
    assert sum(row["count"] for row in records(out.json["census"])) == 101


# --- Errors -------------------------------------------------------------------------


def test_a_frame_impute_did_not_make_is_refused(harness: Harness) -> None:
    profile = _profile(harness, _csv(harness))

    out = harness.call("evaluate_imputation", profile=profile, imputed=_source(profile))

    assert out.code == "invalid_argument"
    assert out.json["param"] == "imputed"


def test_a_split_frame_is_refused_though_it_has_a_source(harness: Harness) -> None:
    profile = _profile(harness, _csv(harness), exclude_columns=["city"])
    test = harness.code_mode(
        "split", profile=profile, scheme={"kind": "random", "test_size": 0.3}
    ).json["test"]

    out = harness.call("evaluate_imputation", profile=profile, imputed=test)

    assert out.code == "invalid_argument"
    assert out.json["param"] == "imputed"


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    profile, imputed = _median(harness)

    out = harness.call("evaluate_imputation", profile=imputed, imputed=profile)

    assert out.code == "wrong_handle_kind"


# --- The core budget ----------------------------------------------------------------


def test_the_classifier_spends_the_core_budget_in_openmp_threads(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, imputed = _median(harness)
    monkeypatch.setenv(CORE_BUDGET_ENV, "1")
    seen = _spy_threads(monkeypatch)

    out = harness.code_mode("evaluate_imputation", profile=profile, imputed=imputed)

    assert not out.is_error, out.json
    assert seen and set(seen) == {1}
