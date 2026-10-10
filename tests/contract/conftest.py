"""The shared Operation test driver: one call, both deliveries, one JSON.

A test calls an Operation the way an Agent would and checks only what crosses
the boundary. ``harness.call(name, **arguments)`` runs the Operation twice:
once as a Code Mode Python call, once through the Tool Calls delivery. It
asserts both give the same JSON (Handle ids aside, since each run writes its
own Handles) and returns the Code Mode outcome.

The Harness settings are set the way a Harness sets them, through the
environment: a temporary Workspace, and every CSV under the fixture directory
(copied to a temporary input directory, which is also the working directory)
as a Declared Input.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import dataforge_ml
from dataforge_ml._boundary._operation import canonical_json
from dataforge_ml._boundary._workspace import (
    DECLARED_INPUTS_ENV,
    HANDLE_KINDS,
    WORKSPACE_ENV,
)
from dataforge_ml._harness.tool_calls import call_tool
from dataforge_ml.records import DataForgeError

FIXTURES = Path(__file__).parent / "fixtures"
_HANDLE = re.compile(rf"\b({'|'.join(HANDLE_KINDS)})_[0-9a-f]{{8}}\b")


def normalise_handles(text: str) -> str:
    """Replace each distinct Handle id with ``<kind>_#<n>`` in order of appearance."""
    seen: dict[str, str] = {}

    def sub(match: re.Match[str]) -> str:
        return seen.setdefault(match.group(0), f"{match.group(1)}_#{len(seen) + 1}")

    return _HANDLE.sub(sub, text)


@dataclass(frozen=True)
class Outcome:
    """What one Operation call returned, as the Agent sees it."""

    json: dict[str, Any]
    is_error: bool
    record: Any = None
    error: DataForgeError | None = None

    @property
    def code(self) -> str | None:
        return self.json["code"] if self.is_error else None


class Harness:
    """A Harness stand-in: Workspace, Declared Inputs, and the two deliveries."""

    def __init__(self, workspace: Path, inputs: Path) -> None:
        self.workspace = workspace
        self.inputs = inputs

    def declare(self, *paths: Path) -> None:
        """Declare extra input paths, alongside the fixture CSVs."""
        current = os.environ.get(DECLARED_INPUTS_ENV, "")
        os.environ[DECLARED_INPUTS_ENV] = os.pathsep.join(
            [current, *map(str, paths)] if current else map(str, paths)
        )

    def handles(self) -> list[str]:
        """Every Handle directory currently in the Workspace."""
        root = self.workspace / "handles"
        return sorted(p.name for p in root.iterdir()) if root.exists() else []

    def code_mode(self, name: str, **arguments: Any) -> Outcome:
        """Call the Operation as the Agent's code would: imported from ``dataforge_ml``."""
        try:
            record = getattr(dataforge_ml, name)(**arguments)
        except DataForgeError as err:
            return Outcome(json=err.to_dict(), is_error=True, error=err)
        return Outcome(json=record.to_dict(), is_error=False, record=record)

    def tool_call(self, name: str, **arguments: Any) -> Outcome:
        """Call the Operation through the Tool Calls delivery."""
        result = call_tool(name, arguments)
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        return Outcome(json=result["structuredContent"], is_error=result["isError"])

    def call(self, name: str, **arguments: Any) -> Outcome:
        """Call through both deliveries, assert identical JSON, return Code Mode's."""
        code = self.code_mode(name, **arguments)
        tool = self.tool_call(name, **arguments)
        assert code.is_error == tool.is_error
        assert normalise_handles(canonical_json(code.json)) == normalise_handles(
            canonical_json(tool.json)
        )
        return code


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    shutil.copytree(FIXTURES, inputs)
    monkeypatch.setenv(WORKSPACE_ENV, str(workspace))
    monkeypatch.setenv(
        DECLARED_INPUTS_ENV,
        os.pathsep.join(str(p) for p in sorted(inputs.glob("*.csv"))),
    )
    monkeypatch.chdir(inputs)
    return Harness(workspace, inputs)


def write_wide_csv(path: Path, n_columns: int, name_length: int = 64) -> Path:
    """Write a 3-row CSV whose header has ``n_columns`` long column names."""
    names = [f"c{i:04d}_".ljust(name_length, "x") for i in range(n_columns)]
    rows = [",".join(names)] + [",".join(str(r * i) for i in range(n_columns)) for r in range(3)]
    path.write_text("\n".join(rows) + "\n")
    return path
