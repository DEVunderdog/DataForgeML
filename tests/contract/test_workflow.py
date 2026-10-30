"""The whole workflow end to end, in both deliveries, each on a fresh Workspace.

load → profile → split → route → resolve_recipe → fit_unit → compose_imputer →
impute → evaluate_imputation, run once as Code Mode calls and once as
``dataforge`` subprocesses. Each delivery chains its own Handles in its own
temporary Workspace, so the CLI run can never read a Handle Code Mode wrote.
Every step's two results must then pass the shared driver's comparison.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dataforge_ml._boundary._settings import WORKSPACE_ENV

from .conftest import Harness, records, run_dataforge, to_argv

#: Rows in the workflow fixture.
N_ROWS = 150


def _csv(harness: Harness) -> Path:
    # ``a`` is numeric and complete, ``b`` numeric, tied to ``a`` and missing
    # on every third row, ``c`` numeric with sparser gaps, ``city`` a string
    # column with gaps.
    csv = harness.inputs / "workflow.csv"
    lines = ["a,b,c,city"]
    for i in range(N_ROWS):
        a = (i * 13) % 47 + 0.5
        b = "" if i % 3 == 1 else f"{2 * a + (i % 5)}"
        c = "" if i % 7 == 3 else f"{a / 2 + (i % 11)}"
        city = "" if i % 10 == 7 else "nse"[i % 3]
        lines.append(f"{a},{b},{c},{city}")
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return csv


def _workflow(call: Callable[..., Any], csv: Path) -> dict[str, Any]:
    """Run every step through ``call``, chaining the Handles each step returns.

    ``call(name, **arguments)`` returns an outcome with a ``json`` attribute;
    the outcomes come back keyed by step, a ``fit_unit`` step by its unit too.
    """
    steps: dict[str, Any] = {}

    def step(name: str, key: str | None = None, **arguments: Any) -> dict[str, Any]:
        out = steps[key or name] = call(name, **arguments)
        assert not out.is_error, (name, out.json)
        return out.json

    frame = step("load", path=str(csv))["frame"]
    profile = step("profile", frame=frame)["profile"]
    split = step("split", profile=profile, scheme={"kind": "random", "test_size": 0.3})
    train = records(split["folds"])[0]["train"]
    routing = step(
        "route",
        profile=profile,
        strategies=[
            {"column": "b", "strategy": "mice"},
            {"column": "c", "strategy": "median"},
        ],
    )["routing"]
    resolved = step("resolve_recipe", routing=routing)
    recipe = resolved["recipe"]
    units = [
        step(
            "fit_unit", f"fit_unit {unit['unit_id']}",
            recipe=recipe, unit_id=unit["unit_id"], frame=train,
        )["unit"]
        for unit in records(resolved["units"])
    ]
    imputer = step("compose_imputer", recipe=recipe, units=units)["imputer"]
    imputed = step("impute", imputer=imputer, frame=split["test"])["frame"]
    step("evaluate_imputation", profile=profile, imputed=imputed)
    return steps


def test_the_workflow_gives_the_same_json_in_both_deliveries(
    harness: Harness, tmp_path: Path
) -> None:
    cli_workspace = tmp_path / "cli_workspace"
    assert not harness.workspace.exists() and not cli_workspace.exists()
    csv = _csv(harness)

    def cli(name: str, **arguments: Any) -> Any:
        env = {**os.environ, WORKSPACE_ENV: str(cli_workspace)}
        return run_dataforge(to_argv(name, arguments), env=env)

    in_code = _workflow(harness.code_mode, csv)
    in_cli = _workflow(cli, csv)

    assert list(in_code) == list(in_cli)
    assert len([k for k in in_code if k.startswith("fit_unit ")]) == 2, "one mice, one median"
    for key, code in in_code.items():
        harness.compare(key.split()[0], code, in_cli[key])
    assert (cli_workspace / "handles" / in_cli["evaluate_imputation"].json["c2st"]).is_dir()
    assert in_code["evaluate_imputation"].json["c2st"] in harness.handles()
