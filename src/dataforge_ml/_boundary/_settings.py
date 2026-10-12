"""Workspace Settings: where they come from, and in what order (ADR-0112).

The Operator sets four Workspace Settings: the Workspace, the Literal Policy,
the Declared Inputs and the core budget. Each resolves in one order: a CLI
flag, then an env var, then ``config.toml``, then the default. Code Mode has
no flags, so it starts at the env var.

The settings file is ``config.toml`` inside the Workspace that the flag, the
env var or the default names, so ``./.dataforge/config.toml`` unless one of
them moves it. That file may itself set ``workspace``, which then wins over
the default. Every other setting is read from the same file. Relative paths,
in any source, are resolved from the working directory.

Workspace Settings are guardrails, not a boundary: in a shell the Agent can
type any flag. A bad value in any source is ``invalid_setting``, naming the
setting and the source, so a typo never falls back silently.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..records import DataForgeError

#: The Workspace when nothing sets one, relative to the working directory.
DEFAULT_WORKSPACE = ".dataforge"
#: The settings file's name, inside the Workspace.
CONFIG_FILE = "config.toml"
#: The Literal Policy values; ``allow`` is the default (ADR-0107).
LITERAL_POLICIES: tuple[str, ...] = ("allow", "deny")

#: Env var per setting. Declared Inputs are ``os.pathsep``-separated.
WORKSPACE_ENV = "DATAFORGE_ML_WORKSPACE"
LITERAL_POLICY_ENV = "DATAFORGE_ML_LITERAL_POLICY"
DECLARED_INPUTS_ENV = "DATAFORGE_ML_DECLARED_INPUTS"
CORE_BUDGET_ENV = "DATAFORGE_ML_CORE_BUDGET"

#: Setting name to env var. The names are the ``config.toml`` keys, and the
#: CLI flags are the same names in kebab-case.
ENV_VARS: dict[str, str] = {
    "workspace": WORKSPACE_ENV,
    "literal_policy": LITERAL_POLICY_ENV,
    "declared_inputs": DECLARED_INPUTS_ENV,
    "core_budget": CORE_BUDGET_ENV,
}
SETTING_NAMES: tuple[str, ...] = tuple(ENV_VARS)


@dataclass(frozen=True)
class WorkspaceSettings:
    """The four Workspace Settings, resolved.

    ``declared_inputs`` is empty when none are declared, which means ``load``
    reads any path.
    """

    workspace: Path
    literal_policy: str
    declared_inputs: tuple[Path, ...]
    core_budget: int


# Flag values the CLI parsed, as raw text by setting name. Code Mode never
# sets this, so it reads env vars, the file and the defaults only.
_FLAGS: ContextVar[Mapping[str, str] | None] = ContextVar(
    "dataforge_ml_setting_flags", default=None
)


@contextmanager
def flag_overrides(flags: Mapping[str, str]) -> Generator[None]:
    """Use CLI flag values as the highest-precedence source in the block.

    Parameters
    ----------
    flags : mapping of str to str
        Raw flag text by setting name, only for the flags that were passed.
    """
    token = _FLAGS.set(dict(flags))
    try:
        yield
    finally:
        _FLAGS.reset(token)


def _invalid(name: str, source: str, problem: str) -> DataForgeError:
    return DataForgeError.from_code(
        "invalid_setting",
        f"The Workspace Setting `{name}` from {source} is invalid: {problem}",
        param=name,
    )


def _source_name(name: str, layer: str, config_path: Path | None) -> str:
    if layer == "flag":
        return f"the flag `--{name.replace('_', '-')}`"
    if layer == "env":
        return f"the env var `{ENV_VARS[name]}`"
    return f"`{config_path}`"


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DataForgeError.from_code(
            "invalid_setting",
            f"The settings file `{path}` cannot be read as TOML: {exc}",
        ) from None
    unknown = sorted(set(data) - set(SETTING_NAMES))
    if unknown:
        raise DataForgeError.from_code(
            "invalid_setting",
            f"The settings file `{path}` has unknown keys "
            f"{', '.join(f'`{k}`' for k in unknown)}; the keys are "
            f"{', '.join(f'`{k}`' for k in SETTING_NAMES)}.",
            param=unknown[0],
            allowed=list(SETTING_NAMES),
        )
    return data


def _pick(
    name: str, config: Mapping[str, Any]
) -> tuple[str, Any] | None:
    flags = _FLAGS.get() or {}
    if name in flags:
        return "flag", flags[name]
    raw = os.environ.get(ENV_VARS[name])
    if raw is not None:
        return "env", raw
    if name in config:
        return "config", config[name]
    return None


def _workspace_before_config() -> Path:
    flags = _FLAGS.get() or {}
    raw = flags.get("workspace", os.environ.get(WORKSPACE_ENV))
    if raw is not None and not raw.strip():
        source = _source_name("workspace", "flag" if "workspace" in flags else "env", None)
        raise _invalid("workspace", source, "it is empty.")
    return Path(raw if raw is not None else DEFAULT_WORKSPACE).resolve()


def _default_core_budget() -> int:
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


def resolve_settings() -> WorkspaceSettings:
    """Resolve every Workspace Setting from flag, env var, ``config.toml`` and default.

    Returns
    -------
    WorkspaceSettings
        The four settings. Nothing is created on disk.

    Raises
    ------
    DataForgeError
        ``invalid_setting`` when any source holds a value of the wrong type
        or out of range, or the settings file is not valid TOML or has an
        unknown key.
    """
    config_path = _workspace_before_config() / CONFIG_FILE
    config = _read_config(config_path)

    def source(name: str, layer: str) -> str:
        return _source_name(name, layer, config_path)

    workspace = _workspace_before_config()
    picked = _pick("workspace", config)
    if picked is not None and picked[0] == "config":
        value = picked[1]
        if not isinstance(value, str) or not value.strip():
            raise _invalid("workspace", source("workspace", "config"), "it must be a non-empty string.")
        workspace = Path(value).resolve()

    literal_policy = "allow"
    picked = _pick("literal_policy", config)
    if picked is not None:
        layer, value = picked
        if value not in LITERAL_POLICIES:
            raise DataForgeError.from_code(
                "invalid_setting",
                f"The Workspace Setting `literal_policy` from {source('literal_policy', layer)} "
                f"is invalid: it must be `allow` or `deny`.",
                param="literal_policy",
                allowed=list(LITERAL_POLICIES),
            )
        literal_policy = value

    declared: tuple[Path, ...] = ()
    picked = _pick("declared_inputs", config)
    if picked is not None:
        layer, value = picked
        if layer == "config":
            if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
                raise _invalid(
                    "declared_inputs",
                    source("declared_inputs", layer),
                    "it must be a list of path strings.",
                )
            paths = value
        else:
            paths = value.split(os.pathsep)
        declared = tuple(Path(p).resolve() for p in paths if p)

    core_budget = _default_core_budget()
    picked = _pick("core_budget", config)
    if picked is not None:
        layer, value = picked
        if layer != "config":
            try:
                value = int(value.strip())
            except ValueError:
                value = None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _invalid(
                "core_budget", source("core_budget", layer), "it must be a whole number of at least 1."
            )
        core_budget = value

    return WorkspaceSettings(
        workspace=workspace,
        literal_policy=literal_policy,
        declared_inputs=declared,
        core_budget=core_budget,
    )
