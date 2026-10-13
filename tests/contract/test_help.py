"""The layered help: ``dataforge --help``, ``<command> --help`` and ``dataforge schema``.

Every test is driven by the Operation registry, so an Operation added later
gets its help checked with no change here. The help is generated into the
committed ``_help.json``; refresh it after an intended change with
``uv run python -m dataforge_ml._boundary._help_build`` (or
``DATAFORGE_ML_UPDATE_SNAPSHOTS=1 uv run pytest tests/contract``).
"""

from __future__ import annotations

import doctest
import io
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Annotated, Any

import pytest
import tiktoken
from pydantic import Field

from dataforge_ml import _help
from dataforge_ml._boundary._help_build import (
    LAYER_BUDGET_TOKENS,
    PHASE_ORDER,
    build_help,
    example_steps,
    help_json,
)
from dataforge_ml._boundary._operation import PHASES, _build_spec, operations
from dataforge_ml._cli import kebab, run
from dataforge_ml.records import OutputRecord, WireRecord

from .conftest import Harness, run_dataforge

OPERATIONS = operations()
NAMES = sorted(OPERATIONS)
COMMANDS = sorted(kebab(name) for name in NAMES)
_HEAVY = ("pandas", "polars", "sklearn", "scipy", "numpy")
_HANDLE = re.compile(r"\b[a-z0-9]+_[0-9a-f]{8}\b")


def _layers(data: dict[str, Any]) -> dict[str, str]:
    """Every help layer's text, keyed by the command line that prints it."""
    layers = {"dataforge --help": data["overview"]}
    for command, text in data["commands"].items():
        layers[f"dataforge {command} --help"] = text
    for command, pages in data["schemas"].items():
        for path, text in pages.items():
            layers[_help.schema_pointer(command, path)] = text
    return layers


def _cli(*argv: str) -> tuple[int, str]:
    stdout = io.StringIO()
    return run(list(argv), stdout=stdout), stdout.getvalue()


# --- Generated and fresh ---------------------------------------------------------


def test_committed_help_is_fresh() -> None:
    current = help_json()
    if os.environ.get("DATAFORGE_ML_UPDATE_SNAPSHOTS"):
        _help.HELP_FILE.write_text(current, "utf-8")
    assert _help.HELP_FILE.read_text("utf-8") == current, (
        "_help.json is stale; run `uv run python -m dataforge_ml._boundary._help_build`"
    )


def test_every_operation_has_all_three_layers() -> None:
    data = json.loads(_help.HELP_FILE.read_text("utf-8"))
    assert sorted(data["commands"]) == COMMANDS
    for name in NAMES:
        command = kebab(name)
        for summary_line in (OPERATIONS[name].docstring.summary,):
            assert summary_line in data["overview"]
        arguments = list(OPERATIONS[name].args_model.model_fields)
        assert [p for p in data["schemas"][command] if "." not in p] == arguments


def test_every_phase_is_listed_in_the_overview_order() -> None:
    assert sorted(PHASE_ORDER) == sorted(PHASES)


def test_no_operation_is_named_like_the_schema_command() -> None:
    assert _help.SCHEMA_COMMAND not in COMMANDS


# --- Cheap to read -----------------------------------------------------------------


def test_help_and_schema_import_no_heavy_dependency() -> None:
    argvs = [["--help"], ["schema", "--help"]]
    for command in COMMANDS:
        argvs.append([command, "--help"])
        argvs += [["schema", command, *p.split(".", 1)] for p in _help.schema_paths(command)]
    script = (
        "import io, json, sys\n"
        "from dataforge_ml._cli import run\n"
        f"for argv in json.loads({json.dumps(json.dumps(argvs))}):\n"
        "    assert run(argv, stdout=io.StringIO()) == 0, argv\n"
        f"print(json.dumps(sorted(m for m in {_HEAVY!r} if m in sys.modules)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


def test_every_layer_is_under_its_budget() -> None:
    encoding = tiktoken.get_encoding("o200k_base")
    layers = _layers(json.loads(_help.HELP_FILE.read_text("utf-8")))
    layers["dataforge schema --help"] = _help.SCHEMA_USAGE
    over = {
        line: tokens
        for line, text in layers.items()
        if (tokens := len(encoding.encode(text))) > LAYER_BUDGET_TOKENS
    }
    # Over budget: split the layer into a narrower one. Never cut its text.
    assert over == {}


# --- Every name the help gives is real ---------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_see_also_name_is_a_real_command(name: str) -> None:
    see_also = OPERATIONS[name].docstring.see_also()
    assert see_also, "an Operation lists at least one See Also"
    assert [n for n in see_also if n not in OPERATIONS] == []
    assert all(see_also.values()), "every See Also entry says why"


def _pointer_problems(data: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for line, text in _layers(data).items():
        for match in re.finditer(r"dataforge schema ([a-z][a-z-]*) ([a-z_][\w.]*)(?: ([\w.]+))?", text):
            command, argument, rest = match.groups()
            path = f"{argument}.{rest}" if rest else argument
            if path not in data["schemas"].get(command, {}):
                problems.append(f"{line}: `{match.group(0)}` opens nothing")
        for match in re.finditer(r"dataforge ([a-z][a-z-]*) --help", text):
            if match.group(1) not in data["commands"]:
                problems.append(f"{line}: `{match.group(0)}` is no command")
    return problems


def test_every_pointer_in_the_help_opens_a_real_layer() -> None:
    assert _pointer_problems(json.loads(_help.HELP_FILE.read_text("utf-8"))) == []


def test_the_pointer_check_catches_what_it_should() -> None:
    data = {
        "overview": "Run `dataforge nope --help`.",
        "commands": {"demo": "Fields: dataforge schema demo thresholds missing"},
        "schemas": {"demo": {"thresholds": ""}},
    }
    assert len(_pointer_problems(data)) == 2


# --- The Example is a shell session ------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_example_runs_as_shell_commands(name: str, harness: Harness) -> None:
    steps = example_steps(OPERATIONS[name])
    assert steps, "the Example has `$ dataforge ...` lines"
    assert any(shlex.split(c)[1] == kebab(name) for c, _ in steps), "it runs its own command"
    bound: dict[str, str] = {}  # Handle in the docstring -> Handle this run made
    checker = doctest.OutputChecker()
    for command, expected in steps:
        for shown, real in bound.items():
            command = command.replace(shown, real)
        argv = shlex.split(command)
        assert argv[0] == "dataforge", command
        out = run_dataforge(argv[1:])
        assert out.exit_code == 0, out.stdout
        shown_handles = list(dict.fromkeys(_HANDLE.findall(expected)))
        real_handles = list(dict.fromkeys(_HANDLE.findall(out.stdout)))
        for shown, real in zip(shown_handles, real_handles):
            bound.setdefault(shown, real)
        for shown, real in bound.items():
            expected = expected.replace(shown, real)
        assert checker.check_output(expected + "\n", out.stdout, doctest.ELLIPSIS), (
            f"$ {command}\nexpected: {expected}\ngot:      {out.stdout}"
        )


# --- A nested argument opens one level at a time ------------------------------------


class _NumericDials(WireRecord):
    """Dials for numeric columns."""

    high_skew: Annotated[float, Field(gt=0)] | None = None
    """Skew above which a column counts as skewed. Null means 1.0."""


class _Thresholds(WireRecord):
    """Every dial group. Null keeps every default."""

    numeric: _NumericDials | None = None
    """Numeric dials. Each one turns a threshold."""
    seed: int | None = None
    """Random seed. Null means 0."""


class _Strategy(WireRecord):
    """One column's declared strategy."""

    column: str
    """Column name."""
    dials: _NumericDials | None = None
    """Dials for this column only."""


class _DemoResult(OutputRecord):
    """A demo result."""

    echo: str
    """The arguments, as JSON."""


def _demo(
    thresholds: _Thresholds | None = None,
    strategies: list[_Strategy] | None = None,
    limit: int | None = None,
) -> _DemoResult:
    """Demo Operation with nested arguments, never registered.

    Use when
    --------
    Never.

    Parameters
    ----------
    thresholds : _Thresholds or None
        The dial groups. Each group opens on its own.
    strategies : list of _Strategy or None
        Declared strategies, one per column.
    limit : int or None
        A flat flag.

    Returns
    -------
    _DemoResult
        An echo.

    Errors
    ------
    invalid_argument
        A bad argument.

    See Also
    --------
    load : Make a frame Handle.

    Example
    -------
    $ dataforge demo
    {...}
    """
    return _DemoResult(echo="")


_DEMO = _build_spec(
    _demo, effect="read", phase="profile", scales_with_data=True, result_budget_tokens=10_000
)


@pytest.fixture
def demo_help(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data = build_help({"demo": _DEMO})
    monkeypatch.setattr(_help, "_data", lambda: data)
    return data


def test_a_nested_flag_gets_one_line_and_the_command_that_opens_it(
    demo_help: dict[str, Any],
) -> None:
    page = demo_help["commands"]["demo"]
    assert "  --thresholds <JSON, inline or @file.json>, optional\n" in page
    assert "      The dial groups.\n      Fields: dataforge schema demo thresholds\n" in page
    assert "      Fields: dataforge schema demo strategies\n" in page
    assert "  --limit <integer>, optional\n      A flat flag.\n" in page
    assert "Cost: grows with the data." in page
    assert "Next" not in page


def test_schema_opens_one_level_at_a_time(demo_help: dict[str, Any]) -> None:
    pages = demo_help["schemas"]["demo"]
    assert sorted(pages) == [
        "limit",
        "strategies",
        "strategies.dials",
        "thresholds",
        "thresholds.numeric",
    ]
    top = pages["thresholds"]
    assert "  numeric: object, optional\n" in top
    assert "Fields: dataforge schema demo thresholds numeric" in top
    assert "high_skew" not in top  # one level only
    assert "  seed: integer, optional\n      Random seed. Null means 0.\n" in top
    numeric = " ".join(pages["thresholds.numeric"].split())
    assert "high_skew: number, optional" in numeric and "Constraints: greater than 0." in numeric
    assert "list of object" in pages["strategies"] and "  column: string" in pages["strategies"]
    assert _pointer_problems(demo_help) == []


def test_the_schema_command_takes_the_path_dotted_or_spaced(demo_help: dict[str, Any]) -> None:
    spaced = _cli("schema", "demo", "thresholds", "numeric")
    dotted = _cli("schema", "demo", "thresholds.numeric")
    assert spaced == dotted == (0, demo_help["schemas"]["demo"]["thresholds.numeric"])


@pytest.mark.parametrize(
    ("argv", "code", "allowed"),
    [
        (["schema"], "invalid_argument", ["demo"]),
        (["schema", "dmo", "limit"], "unknown_operation", None),
        (["schema", "demo"], "invalid_argument", ["thresholds", "strategies", "limit"]),
        (["schema", "demo", "nope"], "invalid_argument", ["thresholds", "strategies", "limit"]),
        (["schema", "demo", "thresholds", "nope"], "invalid_argument", ["thresholds.numeric"]),
        (["schema", "demo", "thresholds", "numeric", "x"], "invalid_argument", None),
        (["nope", "--help"], "unknown_operation", None),
    ],
)
def test_a_wrong_schema_path_names_what_opens(
    demo_help: dict[str, Any], argv: list[str], code: str, allowed: list[str] | None
) -> None:
    exit_code, stdout = _cli(*argv)
    error = json.loads(stdout)
    assert error["code"] == code and exit_code in (2, 3)
    if allowed is not None:
        assert error["allowed"] == allowed
    if code == "unknown_operation":
        assert error["candidates"]


def test_help_layers_print_text_and_exit_zero(harness: Harness) -> None:
    for argv in (["--help"], ["schema", "--help"], ["load", "--help"], ["schema", "load", "path"]):
        done = subprocess.run(
            [str(sys.executable), "-m", "dataforge_ml", *argv],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0 and done.stderr == "", argv
        assert not done.stdout.startswith("{"), argv
    assert harness.handles() == []
