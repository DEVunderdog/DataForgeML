"""The shared Operation test driver: one call, both deliveries, one JSON.

A test calls an Operation the way an Agent would and checks only what crosses
the boundary. ``harness.call(name, **arguments)`` runs the Operation twice:
once as a Code Mode Python call, once as a ``dataforge`` subprocess. It
asserts both give the same JSON (Handle ids aside, since each run writes its
own Handles), that the exit code names the error's category, that stderr
is empty, and that the JSON is within the Result Budget. It returns the Code
Mode outcome.

A test that patches library internals cannot reach a subprocess, so
``harness.call_in_process`` runs the CLI's ``run`` in this process instead.

The Workspace Settings are set the way an Operator sets them, through env
vars: a temporary Workspace, and every CSV under the fixture directory
(copied to a temporary input directory, which is also the working directory)
as a Declared Input.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tiktoken

import dataforge_ml
from dataforge_ml._boundary._error_codes import REGISTRY
from dataforge_ml._boundary._operation import (
    ERROR_BUDGET_CHARACTERS,
    RESULT_BUDGET_CHARACTERS,
    canonical_json,
    operations,
)
from dataforge_ml._boundary._settings import DECLARED_INPUTS_ENV, WORKSPACE_ENV
from dataforge_ml._boundary._workspace import HANDLE_KINDS
from dataforge_ml._cli import EXIT_CODES, kebab, run
from dataforge_ml.records import DataForgeError

FIXTURES = Path(__file__).parent / "fixtures"
#: Ames (House Prices), the real-data fixture for Result Budgets. Declare it to load it.
AMES = Path(__file__).parent / "data" / "ames.csv"
#: The installed console script, next to the interpreter running the tests.
DATAFORGE = Path(sys.executable).parent / "dataforge"
_HANDLE = re.compile(rf"\b({'|'.join(HANDLE_KINDS)})_[0-9a-f]{{8}}\b")


def normalise_handles(text: str) -> str:
    """Replace each distinct Handle id with ``<kind>_#<n>`` in order of appearance."""
    seen: dict[str, str] = {}

    def sub(match: re.Match[str]) -> str:
        return seen.setdefault(match.group(0), f"{match.group(1)}_#{len(seen) + 1}")

    return _HANDLE.sub(sub, text)


def assert_within_budget(name: str, json_value: dict[str, Any], *, is_error: bool) -> None:
    """Fail when a result or Library Error is over its Result Budget (ADR-0112).

    A result may have at most its Operation's token budget in ``o200k_base``
    tokens and :data:`RESULT_BUDGET_CHARACTERS` characters; a Library Error at
    most :data:`ERROR_BUDGET_CHARACTERS` characters.
    """
    text = canonical_json(json_value)
    if is_error:
        assert len(text) <= ERROR_BUDGET_CHARACTERS, f"{name}: error of {len(text)} characters"
        return
    tokens = len(tiktoken.get_encoding("o200k_base").encode(text))
    assert tokens <= operations()[name].result_budget_tokens, f"{name}: {tokens} tokens"
    assert len(text) <= RESULT_BUDGET_CHARACTERS, f"{name}: {len(text)} characters"


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


@dataclass(frozen=True)
class CliOutcome:
    """What one ``dataforge`` process printed and returned."""

    json: dict[str, Any]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def is_error(self) -> bool:
        return self.exit_code != 0


def expected_exit_code(error_code: str | None) -> int:
    """The exit code a CLI result must have: 0, or its Error Code's category's code."""
    return 0 if error_code is None else EXIT_CODES[REGISTRY[error_code].category]


def to_argv(name: str, arguments: dict[str, Any]) -> list[str]:
    """Spell a Code Mode call as ``dataforge`` arguments.

    A scalar becomes its text, a list of scalars a comma list, and anything
    nested inline JSON. A null argument is left out, which is its default.
    """
    argv = [kebab(name)]
    for key, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            text = str(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value
        ):
            text = ",".join(map(str, value))
        else:
            text = json.dumps(value)
        argv += [f"--{kebab(key)}", text]
    return argv


def run_dataforge(
    argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> CliOutcome:
    """Run the ``dataforge`` console script as its own process."""
    assert DATAFORGE.exists(), f"{DATAFORGE} is missing; run `uv sync` to install it"
    done = subprocess.run(
        [str(DATAFORGE), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=300,
        check=False,
    )
    assert done.stdout.endswith("\n") and done.stdout.count("\n") == 1, done.stdout
    return CliOutcome(
        json=json.loads(done.stdout),
        exit_code=done.returncode,
        stdout=done.stdout,
        stderr=done.stderr,
    )


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

    def cli(self, name: str, **arguments: Any) -> CliOutcome:
        """Call the Operation as a ``dataforge`` subprocess, as an Agent's shell would."""
        return run_dataforge(to_argv(name, arguments))

    def cli_in_process(self, name: str, **arguments: Any) -> CliOutcome:
        """Run the CLI's ``run`` in this process, so patched internals apply."""
        stdout = io.StringIO()
        exit_code = run(to_argv(name, arguments), stdout=stdout)
        return CliOutcome(
            json=json.loads(stdout.getvalue()),
            exit_code=exit_code,
            stdout=stdout.getvalue(),
            stderr="",
        )

    def _compare(self, name: str, code: Outcome, cli: CliOutcome) -> Outcome:
        assert_within_budget(name, code.json, is_error=code.is_error)
        assert code.is_error == cli.is_error
        assert cli.exit_code == expected_exit_code(code.code)
        assert cli.stderr == ""
        assert normalise_handles(canonical_json(code.json)) == normalise_handles(
            canonical_json(cli.json)
        )
        assert cli.stdout == canonical_json(cli.json) + "\n"
        return code

    def call(self, name: str, **arguments: Any) -> Outcome:
        """Call through both deliveries, assert identical JSON, return Code Mode's."""
        return self._compare(
            name, self.code_mode(name, **arguments), self.cli(name, **arguments)
        )

    def call_in_process(self, name: str, **arguments: Any) -> Outcome:
        """As :meth:`call`, with the CLI run in this process."""
        return self._compare(
            name, self.code_mode(name, **arguments), self.cli_in_process(name, **arguments)
        )


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


def records(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a list of records back out of its Table Layout JSON, nested tables too."""
    assert list(table) == ["fields", "rows"], table

    def cell(value: Any) -> Any:
        is_table = isinstance(value, dict) and list(value) == ["fields", "rows"]
        return records(value) if is_table else value

    return [
        {name: cell(value) for name, value in zip(table["fields"], row, strict=True)}
        for row in table["rows"]
    ]


def write_wide_csv(path: Path, n_columns: int, name_length: int = 64) -> Path:
    """Write a 3-row CSV whose header has ``n_columns`` long column names."""
    names = [f"c{i:04d}_".ljust(name_length, "x") for i in range(n_columns)]
    rows = [",".join(names)] + [",".join(str(r * i) for i in range(n_columns)) for r in range(3)]
    path.write_text("\n".join(rows) + "\n")
    return path


def write_flagged_wide_csv(path: Path, n_columns: int, name_length: int = 64) -> Path:
    """Write a 40-row CSV of ``n_columns`` long-named numeric columns that raise flags.

    Every column misses the same 60% of rows, so each is flagged
    ``missingness.drop_candidate`` and ``missingness.mar_suspect``, and too few
    complete rows remain to probe nonlinearity, which falls back with a
    warning naming every column.
    """
    names = [f"c{i:04d}_".ljust(name_length, "x") for i in range(n_columns)]
    lines = [",".join(names)]
    for r in range(40):
        missing = r % 5 < 3
        lines.append(",".join("" if missing else str(r + 0.5 + i / 1000) for i in range(n_columns)))
    path.write_text("\n".join(lines) + "\n")
    return path
