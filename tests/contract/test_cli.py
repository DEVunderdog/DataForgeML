"""The ``dataforge`` command itself: flags, exit codes, streams and Workspace Settings.

The shared driver already runs every Operation test through the CLI and checks
its JSON, exit code and empty stderr. These tests cover what is the CLI's own:
how flags become arguments, the exit code of each category, ``--progress``,
and the order in which each Workspace Setting is resolved (ADR-0112).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dataforge_ml._boundary import _settings
from dataforge_ml._boundary._error_codes import CATEGORIES
from dataforge_ml._boundary._operation import _build_spec, invoke_json, operations
from dataforge_ml._boundary._settings import (
    ENV_VARS,
    SETTING_NAMES,
    flag_overrides,
    resolve_settings,
)
from dataforge_ml._cli import (
    _HELP_FLAGS,
    _PROGRESS_FLAG,
    _SETTING_FLAGS,
    EXIT_CODES,
    build_arguments,
    kebab,
    takes_json,
)
from dataforge_ml.records import DataForgeError, OutputRecord, WireRecord

from .conftest import Harness, run_dataforge


def _load(harness: Harness) -> str:
    return harness.code_mode("load", path="train.csv").json["frame"]


# --- Exit codes and streams --------------------------------------------------


def test_every_error_category_has_its_exit_code() -> None:
    assert EXIT_CODES == {
        "InvalidInput": 2,
        "UnknownName": 3,
        "Conflict": 4,
        "Infeasible": 5,
        "IncompatibleArtifact": 6,
        "Unexpected": 1,
    }
    assert set(EXIT_CODES) == set(CATEGORIES) - {"warning"}


def test_each_reachable_category_exits_with_its_code(harness: Harness) -> None:
    frame = _load(harness)
    assert run_dataforge(["list-handles"]).exit_code == 0
    assert run_dataforge(["list-handles", "--limit", "0"]).exit_code == 2
    assert run_dataforge(["export-frame", "--frame", "frame_00000000"]).exit_code == 3

    payload = harness.workspace / "handles" / frame / "frame.parquet"
    payload.write_bytes(b"not parquet")
    out = run_dataforge(["export-frame", "--frame", frame])
    assert (out.exit_code, out.json["code"]) == (1, "unexpected")

    record = harness.workspace / "handles" / frame / "record.json"
    record.write_text(json.dumps({**json.loads(record.read_text()), "format_schema_version": 999}))
    out = run_dataforge(["export-frame", "--frame", frame])
    assert (out.exit_code, out.json["code"]) == (6, "incompatible_artifact")


def test_success_with_warnings_still_exits_zero(harness: Harness) -> None:
    damaged = _load(harness)
    (harness.workspace / "handles" / damaged / "record.json").write_text("{not json")

    out = run_dataforge(["list-handles"])

    assert out.exit_code == 0
    assert [w["code"] for w in out.json["warnings"]] == ["handles_unreadable"]


def test_unknown_command_is_unknown_operation_with_candidates(harness: Harness) -> None:
    out = run_dataforge(["lod", "--path", "train.csv"])

    assert (out.exit_code, out.json["code"]) == (3, "unknown_operation")
    assert out.json["candidates"] == ["load"]
    assert out.stderr == ""


def test_snake_case_command_is_not_the_cli_spelling(harness: Harness) -> None:
    out = run_dataforge(["list_handles"])
    assert out.json["code"] == "unknown_operation"
    assert out.json["candidates"][0] == "list-handles"


def test_no_command_is_invalid_argument_listing_the_commands(harness: Harness) -> None:
    out = run_dataforge([])
    assert (out.exit_code, out.json["code"]) == (2, "invalid_argument")
    assert out.json["allowed"] == sorted(kebab(n) for n in operations())


def test_stderr_is_empty_without_progress_and_one_line_per_event_with_it(
    harness: Harness,
) -> None:
    quiet = run_dataforge(["load", "--path", "train.csv"])
    loud = run_dataforge(["load", "--path", "train.csv", "--progress"])
    before = run_dataforge(["--progress", "list-handles"])

    assert quiet.stderr == ""
    assert loud.stderr.splitlines() == ["load: started", "load: finished"]
    assert before.stderr.splitlines() == ["list_handles: started", "list_handles: finished"]
    assert loud.json["n_rows"] == quiet.json["n_rows"] == 6
    failed = run_dataforge(["export-frame", "--frame", "frame_00000000"])
    assert failed.stderr == ""


def test_help_prints_text_and_exits_zero(harness: Harness) -> None:
    import subprocess

    from .conftest import DATAFORGE

    for argv in (["--help"], ["load", "--help"]):
        done = subprocess.run(
            [str(DATAFORGE), *argv], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0 and done.stderr == ""
        assert done.stdout.startswith("usage: dataforge")
    assert harness.handles() == []


def test_no_operation_parameter_is_shadowed_by_a_cli_flag() -> None:
    own = {*_SETTING_FLAGS, _PROGRESS_FLAG, *_HELP_FLAGS}
    for name, spec in operations().items():
        for param in spec.args_model.model_fields:
            assert f"--{kebab(param)}" not in own, (name, param)


# --- Flags become arguments ----------------------------------------------------


class _Scheme(WireRecord):
    """A nested argument."""

    kind: str
    """Scheme kind."""
    folds: int
    """Fold count."""


class _ProbeResult(OutputRecord):
    echo: str
    """The arguments, as JSON."""


def _probe(
    scheme: _Scheme,
    columns: list[str] | None = None,
    seeds: list[int] | None = None,
    limit: int | None = None,
    scale: float | None = None,
) -> _ProbeResult:
    """Probe Operation for flag parsing, never registered.

    Parameters
    ----------
    scheme : _Scheme
        Nested.
    columns : list of str or None
        A simple list.
    seeds : list of int or None
        A simple list of integers.
    limit : int or None
        An integer.
    scale : float or None
        A float.
    """
    return _ProbeResult(echo=json.dumps([scheme.folds, columns, seeds, limit, scale]))


_PROBE = _build_spec(
    _probe, effect="read", phase="workspace", scales_with_data=False, result_budget_tokens=10_000
)


def _cli_call(*tokens: str) -> dict:
    try:
        return invoke_json(_PROBE, json.dumps(build_arguments(_PROBE, tokens))).to_dict()
    except DataForgeError as err:
        return err.to_dict()


def test_only_nested_arguments_take_json() -> None:
    assert [takes_json(_PROBE, n) for n in _PROBE.args_model.model_fields] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_flat_flags_are_typed_by_their_parameter() -> None:
    out = _cli_call(
        "--scheme", '{"kind": "kfold", "folds": 5}',
        "--columns", "a,b",
        "--seeds=1,2",
        "--limit", "-3",
        "--scale", "2",
    )
    assert json.loads(out["echo"]) == [5, ["a", "b"], [1, 2], -3, 2.0]


def test_a_nested_flag_reads_json_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "scheme.json"
    path.write_text('{"kind": "kfold", "folds": 3}')
    assert json.loads(_cli_call("--scheme", f"@{path}")["echo"])[0] == 3


def test_malformed_json_is_invalid_argument_naming_the_flag(tmp_path: Path) -> None:
    out = _cli_call("--scheme", "{kind: kfold")
    assert out["code"] == "invalid_argument"
    assert out["param"] == "scheme"

    out = _cli_call("--scheme", f"@{tmp_path / 'absent.json'}")
    assert (out["code"], out["param"]) == ("invalid_argument", "scheme")


def test_nested_json_is_checked_by_the_args_model_with_its_param_path() -> None:
    out = _cli_call("--scheme", '{"kind": "kfold", "folds": "5"}', "--limit", "x")
    assert out["code"] == "invalid_argument"
    assert out["params"] == ["scheme.folds", "limit"]


@pytest.mark.parametrize(
    ("tokens", "param"),
    [
        (("stray",), None),
        (("--limit",), "limit"),
        (("--limit", "1", "--limit", "2"), "limit"),
        (("--no-such", "1", "--scheme", '{"kind": "a", "folds": 1}'), "no_such"),
    ],
)
def test_bad_flag_shapes_are_invalid_argument(tokens: tuple[str, ...], param: str | None) -> None:
    out = _cli_call(*tokens)
    assert out["code"] == "invalid_argument"
    assert out.get("param") == param


# --- Workspace Settings --------------------------------------------------------

#: Per setting: a value for each source as text (flag, env), as TOML, and the
#: resolved value it must produce.
_LAYERS = {
    "workspace": {
        "flag": ("ws_flag", "ws_flag"),
        "env": ("ws_env", "ws_env"),
        "config": ('"ws_config"', "ws_config"),
    },
    "literal_policy": {
        "flag": ("deny", "deny"),
        "env": ("allow", "allow"),
        "config": ('"deny"', "deny"),
    },
    "declared_inputs": {
        "flag": ("flag.csv", ("flag.csv",)),
        "env": (f"env.csv{os.pathsep}env2.csv", ("env.csv", "env2.csv")),
        "config": ('["config.csv"]', ("config.csv",)),
    },
    "core_budget": {
        "flag": ("3", 3),
        "env": ("5", 5),
        "config": ("7", 7),
    },
}


def _resolved(settings: _settings.WorkspaceSettings, name: str, cwd: Path):
    value = getattr(settings, name)
    if name == "workspace":
        return value.relative_to(cwd).as_posix()
    if name == "declared_inputs":
        return tuple(p.relative_to(cwd).as_posix() for p in value)
    return value


def _defaults(cwd: Path) -> dict:
    return {
        "workspace": ".dataforge",
        "literal_policy": "allow",
        "declared_inputs": (),
        "core_budget": _settings._default_core_budget(),
    }


@pytest.fixture
def clean_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for env in ENV_VARS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("name", SETTING_NAMES)
@pytest.mark.parametrize("top", ["flag", "env", "config", "default"])
def test_each_setting_resolves_flag_then_env_then_config_then_default(
    name: str, top: str, clean_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every source below ``top`` is set too, so ``top`` must win over each of them.
    order = ["flag", "env", "config", "default"]
    layers = order[order.index(top):]
    values = _LAYERS[name]
    if "config" in layers:
        config = clean_settings / ".dataforge" / "config.toml"
        config.parent.mkdir()
        config.write_text(f"{name} = {values['config'][0]}\n")
    if "env" in layers:
        monkeypatch.setenv(ENV_VARS[name], values["env"][0])
    flags = {name: values["flag"][0]} if "flag" in layers else {}

    with flag_overrides(flags):
        settings = resolve_settings()

    expected = _defaults(clean_settings)[name] if top == "default" else values[top][1]
    assert _resolved(settings, name, clean_settings) == expected


def test_the_settings_file_lives_in_the_workspace_the_flag_or_env_names(
    clean_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = clean_settings / "elsewhere"
    moved.mkdir()
    (moved / "config.toml").write_text('literal_policy = "deny"\n')
    monkeypatch.setenv(_settings.WORKSPACE_ENV, str(moved))

    assert resolve_settings().literal_policy == "deny"


def test_a_workspace_set_in_the_default_config_moves_the_workspace_end_to_end(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_settings.WORKSPACE_ENV)
    (harness.inputs / ".dataforge").mkdir()
    (harness.inputs / ".dataforge" / "config.toml").write_text('workspace = "from_config"\n')

    out = run_dataforge(["load", "--path", "train.csv"])
    assert (harness.inputs / "from_config" / "handles" / out.json["frame"]).is_dir()

    env = {**os.environ, _settings.WORKSPACE_ENV: str(harness.inputs / "from_env")}
    out = run_dataforge(["load", "--path", "train.csv"], env=env)
    assert (harness.inputs / "from_env" / "handles" / out.json["frame"]).is_dir()

    flag = str(harness.inputs / "from_flag")
    out = run_dataforge(["load", "--path", "train.csv", "--workspace", flag], env=env)
    assert (harness.inputs / "from_flag" / "handles" / out.json["frame"]).is_dir()


def test_declared_inputs_flag_beats_the_env_var_end_to_end(
    harness: Harness, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_text("a\n1\n")

    refused = run_dataforge(["load", "--path", str(elsewhere)])
    allowed = run_dataforge(["--declared-inputs", str(elsewhere), "load", "--path", str(elsewhere)])
    opened = run_dataforge(["load", "--path", str(elsewhere), "--declared-inputs="])

    assert refused.json["code"] == "input_not_declared"
    assert allowed.exit_code == 0 and opened.exit_code == 0


@pytest.mark.parametrize(
    ("env", "config", "param"),
    [
        ({_settings.LITERAL_POLICY_ENV: "maybe"}, None, "literal_policy"),
        ({_settings.CORE_BUDGET_ENV: "0"}, None, "core_budget"),
        ({_settings.CORE_BUDGET_ENV: "two"}, None, "core_budget"),
        ({_settings.WORKSPACE_ENV: " "}, None, "workspace"),
        ({}, "core_budget = 2.5\n", "core_budget"),
        ({}, 'declared_inputs = "a.csv"\n', "declared_inputs"),
        ({}, 'literal = "deny"\n', "literal"),
        ({}, "not toml [\n", None),
    ],
)
def test_a_bad_setting_in_any_source_is_invalid_setting(
    clean_settings: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    config: str | None,
    param: str | None,
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if config is not None:
        (clean_settings / ".dataforge").mkdir()
        (clean_settings / ".dataforge" / "config.toml").write_text(config)

    out = run_dataforge(["list-handles"], cwd=clean_settings)

    assert (out.exit_code, out.json["code"]) == (2, "invalid_setting")
    assert out.json.get("param") == param
    assert out.stderr == ""


def test_a_bad_setting_flag_names_the_flag(harness: Harness) -> None:
    out = run_dataforge(["list-handles", "--literal-policy", "maybe"])
    assert out.json["code"] == "invalid_setting"
    assert "`--literal-policy`" in out.json["message"]
    assert out.json["allowed"] == ["allow", "deny"]
