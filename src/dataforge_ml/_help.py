"""Read the generated, committed help the ``dataforge`` CLI prints (ADR-0112).

This module imports nothing but the standard library, so ``dataforge --help``
and ``dataforge schema`` answer without importing pandas, polars or
scikit-learn. ``_boundary/_help_build.py`` writes the file it reads.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

#: The generated help, shipped inside the package.
HELP_FILE = Path(__file__).with_name("_help.json")
#: The CLI command that prints one argument's schema layer.
SCHEMA_COMMAND = "schema"

SCHEMA_USAGE = """\
usage: dataforge schema <command> <argument> [path]

Print one argument's schema, one level at a time. A nested field names the
command that opens it. `path` is dotted for deeper levels, such as
`dataforge schema <command> <argument> <field>.<field>`.
`dataforge <command> --help` lists a command's arguments.
"""


@functools.cache
def _data() -> dict[str, Any]:
    return json.loads(HELP_FILE.read_text("utf-8"))


def commands() -> list[str]:
    """Return every command the help covers, in kebab-case."""
    return list(_data()["commands"])


def overview() -> str:
    """Return the ``dataforge --help`` text."""
    return _data()["overview"]


def command_page(command: str) -> str | None:
    """Return the ``dataforge <command> --help`` text, or None for no such command."""
    return _data()["commands"].get(command)


def schema_paths(command: str) -> list[str]:
    """Return every argument path ``dataforge schema`` opens for ``command``, dotted."""
    return list(_data()["schemas"].get(command, {}))


def schema_page(command: str, path: str) -> str | None:
    """Return the schema layer of a dotted argument path, or None when it has none."""
    return _data()["schemas"].get(command, {}).get(path)


def schema_pointer(command: str, path: str) -> str:
    """Return the ``dataforge schema`` command that opens a dotted argument path.

    Parameters
    ----------
    command : str
        The command, in kebab-case.
    path : str
        The argument name, then any field names below it, joined by dots.

    Returns
    -------
    str
        Such as ``dataforge schema profile thresholds numeric``.
    """
    argument, _, rest = path.partition(".")
    return " ".join(filter(None, ["dataforge", SCHEMA_COMMAND, command, argument, rest]))
