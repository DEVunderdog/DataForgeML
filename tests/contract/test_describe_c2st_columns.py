"""``describe_c2st_columns`` end to end, in both deliveries: pages of a stored C2ST report."""

from __future__ import annotations

from typing import Any

from dataforge_ml._seed import DEFAULT_SEED

from .conftest import Harness, records
from .test_evaluate_imputation import _csv, _imputed, _median, _profile


def _report(harness: Harness, **profile: Any) -> str:
    profile_handle, imputed = _median(harness, **profile)
    return harness.code_mode(
        "evaluate_imputation", profile=profile_handle, imputed=imputed
    ).json["c2st"]


def _by_column(out: Any) -> dict[str, dict[str, Any]]:
    return {row["column"]: row for row in records(out.json["columns"])}


# --- The result ---------------------------------------------------------------------


def test_it_pages_per_frame_results_and_provenance(harness: Harness) -> None:
    report = _report(harness)

    out = harness.call("describe_c2st_columns", report=report)

    assert not out.is_error, out.json
    result = out.json
    assert (result["report"], result["total"], result["offset"]) == (report, 3, 0)
    assert records(result["omitted"]) == []
    assert records(result["columns"])[0]["column"] == "b", "flagged first"
    columns = _by_column(out)
    assert columns["a"]["score"] is None and columns["a"]["outcome"] == "no_filled_cells"
    b = columns["b"]
    assert (b["outcome"], b["verdict"]) == ("tested", "flagged")
    assert "low_power" in b["annotations"]
    (frame,) = records(b["score"]["frames"])
    assert frame["frame"] == 0
    assert len(frame["repeat_z"]) == 5
    assert frame["mean_repeat_z"] == b["score"]["mean_frame_z"], "one frame"
    provenance = result["provenance"]
    params = {p["name"]: p["value"] for p in records(provenance["classifier_params"])}
    assert params["random_state"] == str(DEFAULT_SEED)
    assert provenance["config"]["fdr_alpha"] == 0.05
    assert provenance["config"]["random_state"] == DEFAULT_SEED
    assert provenance["sklearn_version"]


def test_it_agrees_with_evaluate_imputation(harness: Harness) -> None:
    profile, imputed = _median(harness)
    evaluated = harness.code_mode("evaluate_imputation", profile=profile, imputed=imputed)

    out = harness.call("describe_c2st_columns", report=evaluated.json["c2st"])

    summary = {row["column"]: row for row in records(evaluated.json["columns"])}
    for name, row in _by_column(out).items():
        assert row["outcome"] == summary[name]["outcome"]
        z = None if row["score"] is None else row["score"]["mean_frame_z"]
        assert z == summary[name]["mean_frame_z"]


def test_names_filter_the_columns(harness: Harness) -> None:
    report = _report(harness)

    out = harness.call("describe_c2st_columns", report=report, names=["a", "city"])

    assert out.json["total"] == 2
    assert [row["column"] for row in records(out.json["columns"])] == ["a", "city"]


def test_a_cut_page_counts_the_rest_in_omitted(harness: Harness) -> None:
    report = _report(harness)

    first = harness.call("describe_c2st_columns", report=report, limit=2)
    rest = harness.call("describe_c2st_columns", report=report, offset=2, limit=2)

    assert [row["column"] for row in records(first.json["columns"])] == ["b", "a"]
    assert records(first.json["omitted"]) == [
        {
            "field": "columns",
            "shown": 2,
            "total": 3,
            "operation": "describe_c2st_columns",
            "reason": "cap",
        }
    ]
    assert [row["column"] for row in records(rest.json["columns"])] == ["city"]
    assert (rest.json["offset"], records(rest.json["omitted"])) == (2, [])


# --- Errors -------------------------------------------------------------------------


def test_unknown_names_raise_unknown_column_with_candidates(harness: Harness) -> None:
    report = _report(harness)

    out = harness.call("describe_c2st_columns", report=report, names=["a", "bb", "citty"])

    assert out.code == "unknown_column"
    assert out.json["params"] == ["names[1]", "names[2]"]
    assert set(out.json["candidates"]) >= {"b", "city"}


def test_an_excluded_column_is_unknown(harness: Harness) -> None:
    profile = _profile(harness, _csv(harness), exclude_columns=["city"])
    imputed = _imputed(harness, profile, strategies=[{"column": "b", "strategy": "median"}])
    report = harness.code_mode(
        "evaluate_imputation", profile=profile, imputed=imputed
    ).json["c2st"]

    out = harness.call("describe_c2st_columns", report=report, names=["city"])

    assert out.code == "unknown_column"


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    profile, _ = _median(harness)

    out = harness.call("describe_c2st_columns", report=profile)

    assert out.code == "wrong_handle_kind"
