"""``list_handles``, ``export_frame`` and the Workspace guarantees of ADR-0103.

Handles survive a process restart, two processes share one Workspace without
locks, and no Operation ever deletes anything from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import typing
from pathlib import Path

import polars as pl
import pytest
import tiktoken

from dataforge_ml._boundary import _list_handles, _workspace
from dataforge_ml._boundary._list_handles import INPUTS_CAP, PAGE_CAP
from dataforge_ml._boundary._operation import canonical_json, operations
from dataforge_ml.operations.list_handles import HandleKind
from dataforge_ml.records import InvalidInput, UnknownName

from .conftest import Harness


def _load(harness: Harness, path: str = "train.csv") -> str:
    return harness.code_mode("load", path=path).json["frame"]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )


# --- list_handles ----------------------------------------------------------


def test_list_handles_is_newest_first_with_lineage_and_size(harness: Harness) -> None:
    older = _load(harness)
    newer = _load(harness, str(harness.inputs / "train.csv"))

    out = harness.call("list_handles", kind="frame")

    assert out.json["total"] == 2 and out.json["offset"] == 0
    assert [h["handle"] for h in out.json["handles"]] == [newer, older]
    assert out.json["handles"][1] == {
        "handle": older,
        "kind": "frame",
        "operation": "load",
        "arguments": [{"name": "path", "value": '"train.csv"', "value_cut": False}],
        "inputs": [],
        "size": [{"measure": "rows", "value": 6}, {"measure": "columns", "value": 3}],
    }
    assert out.json["omitted"] == [] and out.json["warnings"] == []


def test_list_handles_pages_with_offset_and_names_itself_in_omitted(
    harness: Harness,
) -> None:
    made = [_load(harness) for _ in range(5)][::-1]

    first = harness.call("list_handles", kind="frame", limit=2)
    second = harness.call("list_handles", kind="frame", offset=2, limit=2)
    last = harness.call("list_handles", kind="frame", offset=4, limit=2)

    assert [h["handle"] for h in first.json["handles"]] == made[:2]
    assert [h["handle"] for h in second.json["handles"]] == made[2:4]
    assert [h["handle"] for h in last.json["handles"]] == made[4:]
    assert first.json["omitted"] == [
        {"field": "handles", "shown": 2, "total": 5, "operation": "list_handles"}
    ]
    assert last.json["omitted"] == []


def test_list_handles_without_kind_lists_every_kind_and_filters_by_kind(
    harness: Harness,
) -> None:
    frame = _load(harness)

    assert [h["handle"] for h in harness.call("list_handles").json["handles"]] == [
        frame
    ]
    assert harness.call("list_handles", kind="profile").json == {
        "warnings": [],
        "omitted": [],
        "total": 0,
        "offset": 0,
        "handles": [],
    }


@pytest.mark.parametrize(
    ("arguments", "param"),
    [
        ({"kind": "frames"}, "kind"),
        ({"offset": -1}, "offset"),
        ({"limit": 21}, "limit"),
    ],
)
def test_list_handles_rejects_bad_paging_and_kinds(
    harness: Harness, arguments: dict[str, object], param: str
) -> None:
    out = harness.call("list_handles", **arguments)
    assert out.code == "invalid_argument"
    assert out.json["param"] == param


def test_list_handles_cuts_long_argument_values_and_long_input_lists(
    harness: Harness,
) -> None:
    frame = _load(harness)
    record_path = harness.workspace / "handles" / frame / "record.json"
    record = json.loads(record_path.read_text())
    record["lineage"]["arguments"] = {"path": "x" * 500}
    record["lineage"]["inputs"] = [f"frame_{i:08x}" for i in range(25)]
    record_path.write_text(json.dumps(record))

    out = harness.call("list_handles")

    [argument] = out.json["handles"][0]["arguments"]
    assert argument["value_cut"] is True
    assert len(argument["value"]) == 80 and argument["value"].endswith("…")
    assert len(out.json["handles"][0]["inputs"]) == 10
    assert out.json["omitted"] == [
        {"field": "handles[0].inputs", "shown": 10, "total": 25, "operation": None}
    ]


def test_a_full_page_at_every_cap_fits_the_result_budget(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixtures cannot reach this shape yet: each Handle made by the
    # registered Operation with the most arguments, every value cut, every
    # inputs list full. A new Operation with a longer signature re-checks it.
    widest = max(len(spec.args_model.model_fields) for spec in operations().values())
    value = json.dumps([{"column": f"col_{i}", "strategy": "MICE"} for i in range(9)])
    record = {
        "handle": "frame_7f3a9c2e",
        "kind": "frame",
        "lineage": {
            "operation": "export_frame",
            "arguments": {f"argument_{i:02d}": value for i in range(widest)},
            "inputs": [f"frame_{i:08x}" for i in range(INPUTS_CAP + 1)],
        },
        "size": {"rows": 1_000_000, "columns": 1_000},
    }
    monkeypatch.setattr(
        _list_handles,
        "list_records",
        lambda kind: _workspace.HandleListing(
            records=[record] * (PAGE_CAP + 1), unreadable=1
        ),
    )

    out = harness.call("list_handles")

    tokens = len(tiktoken.get_encoding("o200k_base").encode(canonical_json(out.json)))
    assert len(out.json["handles"]) == PAGE_CAP
    assert tokens <= operations()["list_handles"].result_budget_tokens, tokens


def test_unreadable_handles_are_skipped_with_a_warning(harness: Harness) -> None:
    kept = _load(harness)
    damaged = _load(harness)
    (harness.workspace / "handles" / damaged / "record.json").write_text("{not json")
    (harness.workspace / "handles" / "frame_0000beef").mkdir()  # still being written

    out = harness.call("list_handles")

    assert [h["handle"] for h in out.json["handles"]] == [kept]
    assert [w["code"] for w in out.json["warnings"]] == ["handles_unreadable"]


def test_handle_kind_literal_matches_the_workspace_kinds() -> None:
    assert typing.get_args(HandleKind) == _workspace.HANDLE_KINDS


# --- export_frame ----------------------------------------------------------


def test_export_frame_writes_csv_into_exports_and_returns_its_path(
    harness: Harness,
) -> None:
    frame = _load(harness)

    out = harness.call("export_frame", frame=frame)

    path = Path(out.json["path"])
    assert path == harness.workspace.resolve() / "exports" / f"{frame}.csv"
    assert pl.read_csv(path).equals(pl.read_csv(harness.inputs / "train.csv"))
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_export_frame_is_declared_as_an_export(harness: Harness) -> None:
    spec = operations()["export_frame"]
    assert (spec.effect, spec.phase, spec.scales_with_data) == (
        "export",
        "workspace",
        True,
    )
    spec = operations()["list_handles"]
    assert (spec.effect, spec.phase, spec.scales_with_data) == (
        "read",
        "workspace",
        False,
    )


def test_exporting_twice_returns_the_same_file_and_writes_no_handle(
    harness: Harness,
) -> None:
    frame = _load(harness)
    first = harness.code_mode("export_frame", frame=frame).json["path"]
    stamp = os.stat(first).st_mtime_ns

    second = harness.code_mode("export_frame", frame=frame).json["path"]

    assert first == second and os.stat(second).st_mtime_ns == stamp
    assert harness.handles() == [frame]


# --- Handle errors ---------------------------------------------------------


def test_unknown_handle_offers_at_most_five_same_kind_candidates_newest_first(
    harness: Harness,
) -> None:
    made = [_load(harness) for _ in range(7)][::-1]

    out = harness.call("export_frame", frame="frame_00000000")

    assert out.code == "unknown_handle"
    assert isinstance(out.error, UnknownName)
    assert out.json["param"] == "frame"
    assert out.json["candidates"] == made[:5]


def test_wrong_handle_kind_names_expected_given_and_argument(harness: Harness) -> None:
    out = harness.call("export_frame", frame="profile_0000abcd")

    assert out.code == "wrong_handle_kind"
    assert isinstance(out.error, InvalidInput)
    assert out.json["param"] == "frame"
    assert out.json["allowed"] == ["frame"]
    assert "`frame`" in out.json["message"] and "`profile`" in out.json["message"]


# --- ADR-0103 guarantees -----------------------------------------------------


def test_a_new_process_lists_and_exports_handles_made_by_an_earlier_one(
    harness: Harness,
) -> None:
    frame = _load(harness)
    script = (
        "from dataforge_ml import export_frame, list_handles\n"
        "page = list_handles(kind='frame')\n"
        "print(page.handles[0].handle)\n"
        f"print(export_frame(frame={frame!r}).path)\n"
    )

    listed, exported = _run(script).stdout.split()

    assert listed == frame
    assert pl.read_csv(exported).equals(pl.read_csv(harness.inputs / "train.csv"))


def test_two_processes_share_one_workspace_without_collisions(harness: Harness) -> None:
    shared = _load(harness)
    script = (
        "from dataforge_ml import export_frame, load\n"
        "for _ in range(15):\n"
        "    print(load(path='train.csv').frame)\n"
        f"    export_frame(frame={shared!r})\n"
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    made = []
    for worker in workers:
        out, _ = worker.communicate(timeout=300)
        assert worker.returncode == 0
        made += out.split()

    assert len(made) == len(set(made)) == 30
    assert sorted(harness.handles()) == sorted([shared, *made])
    expected = pl.read_csv(harness.inputs / "train.csv")
    for handle in made:
        frame, _ = _workspace.read_frame(handle, param="frame")
        assert frame.equals(expected)
    page = harness.code_mode("list_handles", kind="frame").json
    assert page["total"] == 31 and page["warnings"] == []
    exports = list((harness.workspace / "exports").iterdir())
    assert [p.name for p in exports] == [f"{shared}.csv"]
    assert pl.read_csv(exports[0]).equals(expected)


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_no_operation_deletes_or_changes_anything_in_the_workspace(
    harness: Harness, tmp_path: Path
) -> None:
    from .test_operation_contract import WORST_CASE

    _load(harness)
    frame = _load(harness)
    harness.code_mode("export_frame", frame=frame)

    for name in sorted(operations()):
        assert name in WORST_CASE, f"add a worst-case call for {name} to WORST_CASE"
        before = _snapshot(harness.workspace)
        (tmp_path / name).mkdir()
        WORST_CASE[name](harness, tmp_path / name)
        after = _snapshot(harness.workspace)
        missing = sorted(set(before) - set(after))
        changed = sorted(k for k in before if k in after and before[k] != after[k])
        assert missing == [] and changed == [], (
            f"{name} deleted {missing} changed {changed}"
        )
