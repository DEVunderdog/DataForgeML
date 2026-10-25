"""The ``dataforge`` command: one Operation per process (ADR-0112).

``dataforge <command> [--flag value ...]`` runs one Operation. The command is
the Operation name in kebab-case and each flag is a parameter name in
kebab-case. Scalars are typed by the parameter (``--limit 5`` is an integer),
a simple list is comma-separated (``--exclude-columns a,b``), and a nested
argument takes inline JSON or ``@file.json``. The arguments are then validated
by the same args model as Code Mode, so ``param`` paths are the same.

stdout carries exactly one JSON value: the Wire Record on success, the Library
Error on failure, byte-stable and identical to Code Mode's ``to_dict()``. The
exit code names the error's category. stderr stays empty unless
``--progress`` is passed, which prints one line per Pipeline Event, or an
installed Skill is stamped with another library version, which warns once.

``dataforge install --skills [--global]`` installs the Skill
(``_skill_install.py``). It is the one command that is no Operation, so its
JSON is a plain object of the library version and the installed folders.

The Workspace Setting flags (``--workspace``, ``--literal-policy``,
``--declared-inputs``, ``--core-budget``) and ``--progress`` go before or
after the command.

``--help`` and ``dataforge schema`` print the generated, committed help
(``_help.py``), so they never import the Operations or the heavy
dependencies behind them. Only running a command does.
"""

from __future__ import annotations

import json
import sys
import types
import typing
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from . import _help, _skill_install
from ._boundary._settings import SETTING_NAMES, flag_overrides
from ._version import library_version
from .records import DataForgeError, canonical_json

if TYPE_CHECKING:
    from ._boundary._operation import OperationSpec

#: Exit code per Library Error category. ``0`` is success, even with warnings.
EXIT_CODES: dict[str, int] = {
    "InvalidInput": 2,
    "UnknownName": 3,
    "Conflict": 4,
    "Infeasible": 5,
    "IncompatibleArtifact": 6,
    "Unexpected": 1,
}

_SETTING_FLAGS = {f"--{name.replace('_', '-')}": name for name in SETTING_NAMES}
_PROGRESS_FLAG = "--progress"
_HELP_FLAGS = ("--help", "-h")
_SCALARS = (str, int, float, bool)


def kebab(name: str) -> str:
    """Return the CLI spelling of an Operation or parameter name.

    Parameters
    ----------
    name : str
        A snake_case name.

    Returns
    -------
    str
        The same name in kebab-case.
    """
    return name.replace("_", "-")


def _strip(annotation: Any) -> Any:
    # Annotated[X, ...] -> X; X | None -> X.
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        elif origin in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(members) != 1:
                return annotation
            annotation = members[0]
        else:
            return annotation


def _scalar(annotation: Any) -> type | None:
    annotation = _strip(annotation)
    if typing.get_origin(annotation) is typing.Literal:
        kinds = {type(v) for v in typing.get_args(annotation)}
        return kinds.pop() if len(kinds) == 1 and kinds <= set(_SCALARS) else None
    return annotation if annotation in _SCALARS else None


def _list_item(annotation: Any) -> type | None:
    annotation = _strip(annotation)
    if typing.get_origin(annotation) in (list, tuple):
        items = typing.get_args(annotation)
        if len(items) == 1 or (len(items) == 2 and items[1] is Ellipsis):
            return _scalar(items[0])
    return None


def takes_json(spec: OperationSpec, name: str) -> bool:
    """Return whether a parameter's flag takes JSON rather than a plain value.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.
    name : str
        A parameter name, snake_case.

    Returns
    -------
    bool
        True for a nested argument: anything but a scalar or a list of them.
    """
    annotation = spec.args_model.model_fields[name].annotation
    return _scalar(annotation) is None and _list_item(annotation) is None


def _convert(text: str, kind: type) -> Any:
    # A value that does not parse stays text, so the args model reports it
    # as `invalid_argument` with its param path, like any other bad value.
    if kind is int:
        try:
            return int(text)
        except ValueError:
            return text
    if kind is float:
        for parse in (int, float):
            try:
                return parse(text)
            except ValueError:
                continue
        return text
    if kind is bool:
        return {"true": True, "false": False}.get(text, text)
    return text


def _bad_flag(message: str, param: str | None = None) -> DataForgeError:
    return DataForgeError.from_code(
        "invalid_argument",
        message,
        param=param,
        params=[param] if param else None,
        issues=[{"param": param or "arguments", "problem": message}],
    )


def _json_value(text: str, param: str) -> Any:
    source = text
    if text.startswith("@"):
        try:
            source = Path(text[1:]).read_text("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise _bad_flag(
                f"`--{kebab(param)}` names a file that cannot be read: {exc}", param
            ) from None
    try:
        return json.loads(source)
    except ValueError as exc:
        raise _bad_flag(
            f"`--{kebab(param)}` takes JSON, inline or as `@file.json`, and this "
            f"is not valid JSON: {exc}",
            param,
        ) from None


def _value(spec: OperationSpec, name: str, text: str) -> Any:
    if name not in spec.args_model.model_fields:
        return text  # an unknown argument: the args model names it
    annotation = spec.args_model.model_fields[name].annotation
    kind = _scalar(annotation)
    if kind is not None:
        return _convert(text, kind)
    item = _list_item(annotation)
    if item is not None:
        return [_convert(part, item) for part in text.split(",")] if text else []
    return _json_value(text, name)


def build_arguments(spec: OperationSpec, tokens: Sequence[str]) -> dict[str, Any]:
    """Turn an Operation's flags into its JSON arguments object.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.
    tokens : sequence of str
        The flags after the command, Workspace Setting flags already removed.

    Returns
    -------
    dict
        Arguments by snake_case name, ready for the args model.

    Raises
    ------
    DataForgeError
        ``invalid_argument`` for a bare value, a flag passed twice, a flag
        with no value, or malformed JSON in a nested flag.
    """
    arguments: dict[str, Any] = {}
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if not token.startswith("--") or token == "--":
            raise _bad_flag(
                f"`{token}` is not a flag. Pass every argument as `--name value`."
            )
        flag, has_value, text = token.partition("=")
        name = flag[2:].replace("-", "_")
        if not has_value:
            if position + 1 >= len(tokens):
                raise _bad_flag(f"`{flag}` needs a value.", name)
            position += 1
            text = tokens[position]
        if name in arguments:
            raise _bad_flag(f"`{flag}` is passed more than once.", name)
        arguments[name] = _value(spec, name, text)
        position += 1
    return arguments


def _split_globals(argv: Sequence[str]) -> tuple[dict[str, str], bool, bool, list[str]]:
    settings: dict[str, str] = {}
    progress = help_asked = False
    rest: list[str] = []
    position = 0
    while position < len(argv):
        token = argv[position]
        flag, has_value, text = token.partition("=")
        if flag in _SETTING_FLAGS:
            name = _SETTING_FLAGS[flag]
            if not has_value:
                if position + 1 >= len(argv):
                    raise _bad_flag(f"`{flag}` needs a value.", name)
                position += 1
                text = argv[position]
            if name in settings:
                raise _bad_flag(f"`{flag}` is passed more than once.", name)
            settings[name] = text
        elif token == _PROGRESS_FLAG:
            progress = True
        elif token in _HELP_FLAGS:
            help_asked = True
        else:
            rest.append(token)
        position += 1
    return settings, progress, help_asked, rest


def _commands() -> dict[str, OperationSpec]:
    from ._boundary._operation import operations

    return {kebab(name): spec for name, spec in sorted(operations().items())}


def _write(stream: TextIO, text: str) -> None:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write((text + "\n").encode("utf-8"))
    else:
        stream.write(text + "\n")
    stream.flush()


def _unknown_command(command: str) -> DataForgeError:
    import difflib

    names = _help.commands()
    return DataForgeError.from_code(
        "unknown_operation",
        f"No command is named `{command}`.",
        candidates=difflib.get_close_matches(command, names, n=5, cutoff=0.5) or names[:5],
    )


def _schema(tokens: Sequence[str], help_asked: bool, stdout: TextIO) -> int:
    # dataforge schema <command> <argument> [path]: one level of one argument.
    if help_asked:
        _write(stdout, _help.SCHEMA_USAGE.rstrip("\n"))
        return 0
    if not tokens:
        raise DataForgeError.from_code(
            "invalid_argument",
            "Name a command and one of its arguments, such as "
            "`dataforge schema load path`.",
            allowed=_help.commands(),
        )
    command, names = tokens[0], tokens[1:]
    if _help.command_page(command) is None:
        raise _unknown_command(command)
    paths = _help.schema_paths(command)
    path = ".".join(names)
    page = _help.schema_page(command, path) if 1 <= len(names) <= 2 else None
    if page is None:
        parent = path
        while parent and parent not in paths:
            parent = parent.rpartition(".")[0]
        prefix = f"{parent}." if parent else ""
        children = [
            p[len(prefix):] for p in paths if p.startswith(prefix) and "." not in p[len(prefix):]
        ]
        message = (
            f"`{command}` has no argument path `{path}` to open. Pass the "
            "argument, then at most one dotted path of nested fields."
            if path
            else f"Name one of `{command}`'s arguments after the command."
        )
        raise DataForgeError.from_code(
            "invalid_argument",
            message,
            param=path or None,
            allowed=[prefix + c for c in children],
        )
    _write(stdout, page.rstrip("\n"))
    return 0


def _install(tokens: Sequence[str], help_asked: bool, stdout: TextIO) -> int:
    # dataforge install --skills [--global]: not an Operation, so no Handle.
    if help_asked:
        _write(stdout, _skill_install.INSTALL_USAGE.rstrip("\n"))
        return 0
    allowed = list(_skill_install.INSTALL_FLAGS)
    unknown = [token for token in tokens if token not in allowed]
    if unknown:
        raise DataForgeError.from_code(
            "invalid_argument",
            f"`dataforge install` takes no `{unknown[0]}`.",
            allowed=allowed,
        )
    if "--skills" not in tokens:
        raise DataForgeError.from_code(
            "invalid_argument",
            "Name what to install, as in `dataforge install --skills`.",
            allowed=allowed,
        )
    root = Path.home() if "--global" in tokens else Path.cwd()
    installed = _skill_install.install_skills(root)
    result = {
        "library_version": library_version(),
        "installed": [str(folder) for folder in installed],
    }
    _write(stdout, canonical_json(result))
    return 0


def _dispatch(argv: Sequence[str], stdout: TextIO, stderr: TextIO) -> int:
    settings, progress, help_asked, rest = _split_globals(argv)
    if rest and rest[0] == _skill_install.INSTALL_COMMAND:
        return _install(rest[1:], help_asked, stdout)
    warning = _skill_install.stale_skill_warning_once()
    if warning is not None:
        _write(stderr, warning)
    if not rest:
        if help_asked:
            _write(stdout, _help.overview())
            return 0
        raise DataForgeError.from_code(
            "invalid_argument",
            "Name a command, such as `dataforge load --path train.csv`. "
            "Run `dataforge --help` to list them.",
            allowed=_help.commands(),
        )
    command, tokens = rest[0], rest[1:]
    if command == _help.SCHEMA_COMMAND:
        return _schema(tokens, help_asked, stdout)
    if help_asked:
        page = _help.command_page(command)
        if page is None:
            raise _unknown_command(command)
        _write(stdout, page.rstrip("\n"))
        return 0
    spec = _commands().get(command)
    if spec is None:
        raise _unknown_command(command)
    from ._boundary._operation import invoke_json, progress_to
    from ._observability import stderr_observer

    arguments = build_arguments(spec, tokens)
    with flag_overrides(settings), progress_to(stderr_observer if progress else None):
        record = invoke_json(spec, json.dumps(arguments))
    _write(stdout, canonical_json(record.to_dict()))
    return 0


def run(argv: Sequence[str], stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Run one ``dataforge`` command and return its exit code.

    Parameters
    ----------
    argv : sequence of str
        The arguments after ``dataforge``.
    stdout : TextIO, optional
        Where the JSON goes; ``sys.stdout`` by default.
    stderr : TextIO, optional
        Where the stale-Skill warning goes; ``sys.stderr`` by default.

    Returns
    -------
    int
        ``0`` on success, else the exit code of the Library Error's category
        (:data:`EXIT_CODES`). ``MemoryError`` and ``KeyboardInterrupt``
        propagate.
    """
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    try:
        with warnings.catch_warnings():
            # stderr carries progress and the stale-Skill line only; library
            # caveats travel in the JSON.
            warnings.simplefilter("ignore")
            return _dispatch(argv, stdout, stderr)
    except DataForgeError as err:
        _write(stdout, canonical_json(err.to_dict()))
        return EXIT_CODES[err.category]
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001 - the process boundary: every failure prints JSON
        from ._boundary._operation import _unexpected

        err = _unexpected(exc)
        _write(stdout, canonical_json(err.to_dict()))
        return EXIT_CODES[err.category]


def main() -> None:
    """Entry point of the ``dataforge`` console script.

    Raises
    ------
    SystemExit
        Always, with the exit code from :func:`run`.
    """
    raise SystemExit(run(sys.argv[1:]))
