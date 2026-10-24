"""Build the layered help the ``dataforge`` CLI prints (ADR-0112).

The help is generated ahead of time from each Operation's seven-part
docstring and its Strict Subset schema, and committed as ``_help.json``. The
CLI then answers ``--help`` and ``dataforge schema`` by reading that file,
without importing the heavy dependencies. A contract test fails when the file
is stale. Rebuild it after changing an Operation with::

    uv run python -m dataforge_ml._boundary._help_build

There are three layers, and each names the one below it:

- the overview (``dataforge --help``): commands by phase, one line each;
- one page per command (``dataforge <command> --help``);
- one page per argument path (``dataforge schema <command> <argument> [path]``),
  one level of a nested argument at a time.

A layer over :data:`LAYER_BUDGET_TOKENS` fails a test. The fix is a narrower
layer, never a shorter text.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._cli import EXIT_CODES, kebab, takes_json
from .._help import HELP_FILE, SCHEMA_COMMAND, schema_pointer
from .._skill_install import INSTALL_COMMAND
from ._error_codes import REGISTRY
from ._operation import OperationSpec, operations

#: Most ``o200k_base`` tokens one help layer may have (ADR-0108, ADR-0112).
LAYER_BUDGET_TOKENS = 1_500
#: The order phases are listed in: the workflow's, with the Workspace last.
PHASE_ORDER: tuple[str, ...] = ("load", "profile", "split", "impute", "evaluate", "workspace")

_WIDTH = 88
_SHARED_FIELDS = ("warnings", "omitted")


def _plain(text: str) -> str:
    # RST literals read as Markdown ones in a terminal.
    return text.replace("``", "`")


def _wrap(text: str, indent: str) -> list[str]:
    return textwrap.wrap(
        _plain(text), _WIDTH, initial_indent=indent, subsequent_indent=indent
    ) or [indent.rstrip()]


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?\.)(\s|$)", text.strip(), re.DOTALL)
    return match.group(1) if match else text.strip()


# --- Schema nodes ------------------------------------------------------------


def _unwrap(node: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the node without its ``null`` branch, and whether it had one."""
    branches = node.get("anyOf")
    if branches:
        others = [b for b in branches if b.get("type") != "null"]
        if len(others) < len(branches) and len(others) == 1:
            inner = {**others[0]}
            if node.get("description"):
                inner["description"] = node["description"]
            return inner, True
    return dict(node), False


def _is_table(node: Mapping[str, Any]) -> bool:
    return node.get("type") == "object" and list(node.get("properties", {})) == [
        "fields",
        "rows",
    ]


def _opens(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The object nodes a schema layer lists the fields of, one per variant."""
    node, _ = _unwrap(node)
    if node.get("type") == "object":
        return [node]
    if node.get("type") == "array":
        return _opens(node["items"])
    return [o for branch in node.get("anyOf", []) for o in _opens(branch)]


def _type_text(node: Mapping[str, Any]) -> str:
    node, _ = _unwrap(node)
    if "enum" in node:
        return "one of " + "|".join(map(str, node["enum"]))
    kind = node.get("type")
    if _is_table(node):
        names = node["properties"]["fields"]["items"]["enum"]
        return "table of (" + ", ".join(names) + ")"
    if kind == "array":
        return "list of " + _type_text(node["items"])
    if kind == "boolean":
        return "true|false"
    if kind == "object":
        return "object"
    if kind:
        return str(kind)
    return " or ".join(_type_text(b) for b in node.get("anyOf", []))


# --- Layer 1: the overview -----------------------------------------------------


def _overview(specs: Mapping[str, OperationSpec]) -> str:
    width = max(len(kebab(name)) for name in specs)
    lines = ["usage: dataforge <command> [--flag value ...]", "", "Commands, by phase:"]
    for phase in PHASE_ORDER:
        names = sorted(name for name, spec in specs.items() if spec.phase == phase)
        if not names:
            continue
        lines.append(f"  {phase}:")
        for name in names:
            lines.append(f"    {kebab(name).ljust(width)}  {specs[name].docstring.summary}")
    lines += [
        "",
        "Go deeper:",
        "  dataforge <command> --help    one command: flags, result, errors, an example",
        f"  dataforge {SCHEMA_COMMAND} <command> <argument> [path]",
        "                                one nested argument's schema, one level at a time",
        "  The `dataforge` Skill (SKILL.md) teaches the workflow and what each exit code asks.",
        f"  dataforge {INSTALL_COMMAND} --skills [--global]",
        "                                install the Skill for Claude Code, Codex, Gemini CLI, Cursor",
        "",
        "Every command prints one JSON value on stdout, errors too. Exit codes: 0 success,",
        "2 InvalidInput, 3 UnknownName, 4 Conflict, 5 Infeasible, 6 IncompatibleArtifact,",
        "1 Unexpected. A list of records prints as {\"fields\": [...], \"rows\": [[...], ...]}.",
        "",
        "Workspace Settings, set by the Operator; leave them as they are:",
        "  --workspace, --literal-policy, --declared-inputs, --core-budget",
        "--progress prints one line per Pipeline Event to stderr.",
    ]
    return "\n".join(lines)


# --- Layer 2: one page per command ---------------------------------------------


def _usage(command: str, spec: OperationSpec, schema: Mapping[str, Any]) -> str:
    parts = [f"usage: dataforge {command}"]
    for name, field in spec.args_model.model_fields.items():
        node, _ = _unwrap(schema["properties"][name])
        if takes_json(spec, name):
            value = "<json>"
        elif "enum" in node:
            value = f"<{name}>"
        else:
            value = f"<{node.get('type', 'value')}>"
        flag = f"--{kebab(name)} {value}"
        parts.append(flag if field.is_required() else f"[{flag}]")
    return " ".join(parts)


def _flags(command: str, spec: OperationSpec, schema: Mapping[str, Any]) -> list[str]:
    lines = ["Flags (a flag left out is null, which means its default):"]
    for name, field in spec.args_model.model_fields.items():
        node = schema["properties"][name]
        need = "required" if field.is_required() else "optional"
        description = node.get("description", "")
        if takes_json(spec, name):
            lines.append(f"  --{kebab(name)} <JSON, inline or @file.json>, {need}")
            lines += _wrap(_first_sentence(description), "      ")
            lines.append(f"      Fields: {schema_pointer(command, name)}")
        else:
            lines.append(f"  --{kebab(name)} <{_type_text(node)}>, {need}")
            lines += _wrap(description, "      ")
    return lines


def _returns(spec: OperationSpec, schema: Mapping[str, Any]) -> list[str]:
    lines = [f"Returns {spec.result_model.__name__}, as one JSON object:"]
    for name, node in schema["properties"].items():
        if name in _SHARED_FIELDS:
            continue
        lines.append(f"  {name}: {_type_text(node)}")
        lines += _wrap(node.get("description", ""), "      ")
    lines += _wrap(
        "Every result also has `warnings` (caveats; read them first) and `omitted` "
        "(cut lists, each naming the command that returns the rest).",
        "  ",
    )
    return lines


def _errors(spec: OperationSpec) -> list[str]:
    lines = ["Errors (Error Code, exit code):"]
    for code, text in spec.docstring.entries("Errors").items():
        lines.append(f"  {code} ({EXIT_CODES[REGISTRY[code].category]})")
        lines += _wrap(text, "      ")
    return lines


def _see_also(spec: OperationSpec) -> list[str]:
    lines = ["See Also:"]
    for name, text in sorted(spec.docstring.see_also().items()):
        lines.append(f"  {kebab(name)}: {_plain(text)}")
    return lines


def _cost(spec: OperationSpec) -> list[str]:
    if spec.scales_with_data:
        return _wrap(
            "Cost: grows with the data. On large data, run it in the background or "
            "with a longer timeout.",
            "",
        )
    return ["Cost: reads only stored records; it does not grow with the data."]


def _command_page(command: str, spec: OperationSpec) -> str:
    inputs = spec.input_schema()
    lines = [_usage(command, spec, inputs), "", spec.docstring.summary, "", "Use when:"]
    lines += [
        line
        for paragraph in spec.docstring.sections["Use when"].split("\n\n")
        for line in [*_wrap(" ".join(paragraph.split()), "  "), ""]
    ]
    for block in (
        _flags(command, spec, inputs),
        _returns(spec, spec.output_schema()),
        _errors(spec),
        _see_also(spec),
        _cost(spec),
        ["Example:", *("  " + line for line in example_text(spec).splitlines())],
    ):
        lines += [*block, ""]
    return "\n".join(lines).rstrip() + "\n"


# --- Layer 3: one page per argument path ----------------------------------------


def _field_lines(command: str, path: str, node: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, prop in node.get("properties", {}).items():
        need = ", optional" if _unwrap(prop)[1] else ""
        lines.append(f"  {name}: {_type_text(prop)}{need}")
        if _opens(prop):
            lines += _wrap(_first_sentence(prop.get("description", "")), "      ")
            lines.append(f"      Fields: {schema_pointer(command, f'{path}.{name}')}")
        else:
            lines += _wrap(prop.get("description", ""), "      ")
    return lines


def _schema_pages(command: str, name: str, node: Mapping[str, Any]) -> dict[str, str]:
    """The page for argument path ``name`` and, recursively, every path below it."""
    pages: dict[str, str] = {}
    _, nullable = _unwrap(node)
    need = "optional" if nullable else "required"
    lines = [
        f"{name}: {_type_text(node)}, {need}",
        *_wrap(node.get("description", ""), "  "),
    ]
    variants = _opens(node)
    for i, variant in enumerate(variants):
        heading = "Fields:" if len(variants) == 1 else f"Variant {i + 1} fields:"
        lines += ["", heading, *_field_lines(command, name, variant)]
    pages[name] = "\n".join(lines) + "\n"
    children: dict[str, list[Mapping[str, Any]]] = {}
    for variant in variants:
        for child, prop in variant.get("properties", {}).items():
            if _opens(prop):
                children.setdefault(child, []).append(prop)
    for child, props in children.items():
        # A field in several variants opens the first variant's schema.
        pages.update(_schema_pages(command, f"{name}.{child}", props[0]))
    return pages


# --- Examples -----------------------------------------------------------------


def example_text(spec: OperationSpec) -> str:
    """Return an Operation's ``Example`` section as the shell session it shows.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.

    Returns
    -------
    str
        The section, dedented: ``$ dataforge ...`` command lines, each
        followed by the JSON it prints.
    """
    return textwrap.dedent(spec.docstring.sections["Example"]).strip()


def example_steps(spec: OperationSpec) -> list[tuple[str, str]]:
    """Split an Operation's ``Example`` into commands and their expected output.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.

    Returns
    -------
    list of (str, str)
        Each command, without its ``$ ``, and the one line of JSON it prints.
        The docstring may wrap that JSON over several lines, which are joined
        back without spaces. ``...`` in the JSON matches any text.
    """
    steps: list[tuple[str, list[str]]] = []
    for line in example_text(spec).splitlines():
        if line.startswith("$ "):
            steps.append((line[2:].strip(), []))
        elif steps and line.strip():
            steps[-1][1].append(line.strip())
    return [(command, "".join(output)) for command, output in steps]


# --- The whole file -------------------------------------------------------------


def build_help(specs: Mapping[str, OperationSpec] | None = None) -> dict[str, Any]:
    """Build every help layer from the Operations' docstrings and schemas.

    Parameters
    ----------
    specs : mapping of str to OperationSpec, optional
        The Operations, by name. Every registered Operation by default.

    Returns
    -------
    dict
        ``overview``: the ``dataforge --help`` text. ``commands``: each
        command's ``--help`` text, by command. ``schemas``: by command, each
        argument path's ``dataforge schema`` text, keyed by its dotted path.
    """
    specs = dict(operations() if specs is None else specs)
    commands: dict[str, str] = {}
    schemas: dict[str, dict[str, str]] = {}
    for name in sorted(specs):
        spec, command = specs[name], kebab(name)
        commands[command] = _command_page(command, spec)
        pages: dict[str, str] = {}
        for argument, node in spec.input_schema()["properties"].items():
            pages.update(_schema_pages(command, argument, node))
        schemas[command] = pages
    return {"overview": _overview(specs), "commands": commands, "schemas": schemas}


def help_json(specs: Mapping[str, OperationSpec] | None = None) -> str:
    """Return the text of ``_help.json`` as :func:`build_help` builds it.

    Parameters
    ----------
    specs : mapping of str to OperationSpec, optional
        The Operations, by name. Every registered Operation by default.

    Returns
    -------
    str
        Indented JSON, ending in a newline.
    """
    return json.dumps(build_help(specs), indent=2, ensure_ascii=False) + "\n"


def write_help(path: Path = HELP_FILE) -> None:
    """Write the generated help to ``path``.

    Parameters
    ----------
    path : Path, optional
        Where to write; the packaged ``_help.json`` by default.
    """
    path.write_text(help_json(), "utf-8")


if __name__ == "__main__":
    write_help()
