"""``load`` end to end, in both deliveries, plus the boundary rules it stands on."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from dataforge_ml._boundary import _load, _settings, _workspace
from dataforge_ml._boundary._operation import _build_spec, invoke_json, invoke_python
from dataforge_ml._cli import run as run_cli
from dataforge_ml.records import (
    DataForgeError,
    IncompatibleArtifact,
    InvalidInput,
    OutputRecord,
    Unexpected,
    UnknownName,
)

from .conftest import Harness, run_dataforge, write_wide_csv


def test_load_returns_frame_handle_counts_and_dtypes(harness: Harness) -> None:
    out = harness.call("load", path="train.csv")

    assert not out.is_error
    assert out.json["frame"].startswith("frame_")
    assert (out.json["n_rows"], out.json["n_columns"]) == (6, 3)
    assert out.json["columns"] == [
        {"column": "age", "dtype": "Int64"},
        {"column": "city", "dtype": "String"},
        {"column": "income", "dtype": "Float64"},
    ]
    assert out.json["warnings"] == [] and out.json["omitted"] == []
    assert out.record.get("income").dtype == "Float64"
    assert out.record.get("missing") is None


def test_load_accepts_an_absolute_declared_path(harness: Harness) -> None:
    out = harness.call("load", path=str(harness.inputs / "train.csv"))
    assert out.json["n_rows"] == 6


def test_load_caps_columns_at_100_and_names_the_rest_in_omitted(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_wide_csv(tmp_path / "wide.csv", 130)
    harness.declare(wide)

    out = harness.call("load", path=str(wide))

    assert out.json["n_columns"] == 130
    assert len(out.json["columns"]) == 100
    assert out.json["columns"][0]["column"].startswith("c0000_")
    assert out.json["omitted"] == [
        {"field": "columns", "shown": 100, "total": 130, "operation": "describe_columns"}
    ]


def test_undeclared_path_is_refused_and_writes_no_handle(
    harness: Harness, tmp_path: Path
) -> None:
    edited = tmp_path / "edited.csv"
    edited.write_text("a\n1\n")

    out = harness.call("load", path=str(edited))

    assert out.code == "input_not_declared"
    assert out.json["param"] == "path"
    assert isinstance(out.error, InvalidInput)
    assert harness.handles() == []


def test_schema_invalid_arguments_come_back_in_one_error(harness: Harness) -> None:
    out = harness.call("load", sheet="x")

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["path", "sheet"]
    assert {i["param"] for i in out.json["issues"]} == {"path", "sheet"}
    assert harness.handles() == []


def test_a_value_of_the_wrong_type_is_named_in_one_error(harness: Harness) -> None:
    # A shell flag is always text, so only Code Mode can pass a number here.
    out = harness.code_mode("load", path=5, sheet="x")

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["path", "sheet"]
    assert harness.handles() == []


def test_missing_argument_is_invalid_argument(harness: Harness) -> None:
    out = harness.call("load")
    assert out.code == "invalid_argument"
    assert out.json["param"] == "path"


class _ProbeResult(OutputRecord):
    total: float
    """Sum of the inputs."""


def _probe(count: int, scale: float | None = None, names: list[str] | None = None) -> _ProbeResult:
    """Probe Operation for strictness, never registered.

    Parameters
    ----------
    count : int
        An integer.
    scale : float or None
        A float.
    names : list of str or None
        Strings.
    """
    return _ProbeResult(total=count * (scale or 1.0) + len(names or []))


_PROBE = _build_spec(_probe, effect="read", phase="workspace", scales_with_data=False, result_budget_tokens=10_000)


def _both(arguments: dict) -> tuple[dict, dict]:
    outs = []
    for run in (
        lambda: invoke_python(_PROBE, arguments),
        lambda: invoke_json(_PROBE, json.dumps(arguments)),
    ):
        try:
            outs.append(run().to_dict())
        except DataForgeError as err:
            outs.append(err.to_dict())
    return outs[0], outs[1]


def test_values_are_strict_quoted_number_is_never_an_int(harness: Harness) -> None:
    code, tool = _both({"count": "5", "scale": "2.0"})
    assert code == tool
    assert code["code"] == "invalid_argument"
    assert code["params"] == ["count", "scale"]


def test_shapes_are_loose_int_for_float_and_tuple_for_array(harness: Harness) -> None:
    code, tool = _both({"count": 2, "scale": 3, "names": ["a"]})
    assert code == tool == {"warnings": [], "omitted": [], "total": 7.0}
    assert invoke_python(_PROBE, {"count": 2, "names": ("a", "b")}).total == 4.0


def test_foreign_exception_becomes_unexpected_with_a_cut_message(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(raw: bytes) -> None:
        raise ValueError("bad cell " * 200)

    monkeypatch.setattr(_load, "_load_csv", boom)
    out = harness.call_in_process("load", path="train.csv")

    assert out.code == "unexpected"
    assert isinstance(out.error, Unexpected)
    assert out.json["message"].startswith("ValueError: bad cell")
    assert len(out.json["message"]) <= 320
    assert harness.handles() == []


@pytest.mark.parametrize("exc", [MemoryError, KeyboardInterrupt])
def test_memory_error_and_interrupt_pass_through_untouched(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]
) -> None:
    def boom(raw: bytes) -> None:
        raise exc()

    monkeypatch.setattr(_load, "_load_csv", boom)
    with pytest.raises(exc):
        harness.code_mode("load", path="train.csv")
    with pytest.raises(exc):
        run_cli(["load", "--path", "train.csv"])


def test_the_workspace_defaults_to_dot_dataforge_created_by_the_first_write(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_settings.WORKSPACE_ENV)
    default = harness.inputs / ".dataforge"

    assert harness.call("list_handles").json["total"] == 0
    assert not default.exists(), "a read must not create the Workspace"

    out = harness.call("load", path="train.csv")

    assert (default / "handles" / out.json["frame"] / "record.json").exists()
    assert harness.call("list_handles").json["total"] == 2


def test_with_no_declared_inputs_any_path_loads(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(_settings.DECLARED_INPUTS_ENV)
    anywhere = tmp_path / "anywhere.csv"
    anywhere.write_text("a,b\n1,2\n")

    out = harness.call("load", path=str(anywhere))

    assert (out.json["n_rows"], out.json["n_columns"]) == (1, 2)


def test_once_any_input_is_declared_every_other_path_is_refused(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allowed = tmp_path / "allowed.csv"
    other = tmp_path / "other.csv"
    for path in (allowed, other):
        path.write_text("a\n1\n")
    monkeypatch.setenv(_settings.DECLARED_INPUTS_ENV, str(allowed))

    assert not harness.call("load", path=str(allowed)).is_error
    refused = harness.call("load", path=str(other))
    assert refused.code == "input_not_declared"
    assert run_dataforge(["load", "--path", str(other)]).exit_code == 2


def test_a_missing_file_is_input_not_found(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_settings.DECLARED_INPUTS_ENV)

    out = harness.call("load", path="no_such.csv")

    assert out.code == "input_not_found"
    assert out.json["param"] == "path"
    assert harness.handles() == []


def test_frame_handle_is_written_once_with_lineage(harness: Harness) -> None:
    first = harness.code_mode("load", path="train.csv").json["frame"]
    second = harness.code_mode("load", path="train.csv").json["frame"]

    assert first != second
    directory = harness.workspace / "handles" / first
    assert sorted(p.name for p in directory.iterdir()) == ["frame.parquet", "record.json"]
    record = json.loads((directory / "record.json").read_text())
    assert record["kind"] == "frame"
    assert record["lineage"] == {
        "operation": "load",
        "arguments": {"path": "train.csv"},
        "inputs": [],
    }
    assert record["size"] == {"rows": 6, "columns": 3}
    assert record["declarations"] == {
        "targets": [],
        "type_overrides": [],
        "sentinels": [],
        "exclusions": [],
    }


def test_frame_handle_survives_being_read_back_from_disk(harness: Harness) -> None:
    handle = harness.code_mode("load", path="train.csv").json["frame"]
    original = pl.read_csv("train.csv")
    _workspace._CACHE.clear()

    frame, declarations = _workspace.read_frame(handle, param="frame")

    assert frame.equals(original)
    assert declarations == _workspace.InheritedDeclarations()


def test_frame_handle_survives_a_process_restart(harness: Harness) -> None:
    handle = harness.code_mode("load", path="train.csv").json["frame"]
    script = (
        "from dataforge_ml._boundary._workspace import read_frame\n"
        f"frame, _ = read_frame({handle!r}, param='frame')\n"
        "print(frame.shape)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "(6, 3)"


def test_bad_handles_raise_typed_errors(harness: Harness) -> None:
    handle = harness.code_mode("load", path="train.csv").json["frame"]

    with pytest.raises(UnknownName) as unknown:
        _workspace.read_record("frame_00000000", "frame", param="frame")
    assert unknown.value.code == "unknown_handle"
    assert unknown.value.candidates == [handle]

    with pytest.raises(InvalidInput) as wrong:
        _workspace.read_record(handle, "profile", param="profile")
    assert wrong.value.code == "wrong_handle_kind"
    assert wrong.value.allowed == ["profile"]

    with pytest.raises(UnknownName):
        _workspace.read_record("../../etc", "frame", param="frame")

    record = harness.workspace / "handles" / handle / "record.json"
    stored = json.loads(record.read_text())
    record.write_text(json.dumps({**stored, "format_schema_version": 999}))
    with pytest.raises(IncompatibleArtifact):
        _workspace.read_record(handle, "frame", param="frame")


def test_error_list_fields_are_capped_with_omitted() -> None:
    err = DataForgeError.from_code(
        "unknown_handle", "No such Handle.", candidates=[f"frame_{i:08x}" for i in range(9)]
    )
    assert len(err.candidates) == 5
    assert err.to_dict()["omitted"] == [
        {"field": "candidates", "shown": 5, "total": 9, "operation": None}
    ]
