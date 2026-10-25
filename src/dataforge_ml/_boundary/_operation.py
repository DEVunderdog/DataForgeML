"""The Operation boundary: registration, argument checks and the catch-all.

``@operation`` turns a thin Operation function into both deliveries' entry
point (ADR-0099). At decoration time it builds the args model from the
signature, with parameter descriptions parsed from the docstring (ADR-0101),
and registers the Operation. At call time, in either delivery, it:

1. validates every argument strictly before any work, reporting all problems
   in one ``invalid_argument`` error (ADR-0100);
2. resolves the Workspace Settings once, then runs the Operation inside a
   call context that supplies them and the Lineage to every Handle it writes
   (ADR-0103, ADR-0112);
3. under the ``deny`` Literal Policy, nulls every Literal Field of the
   result and reports it in ``omitted`` and one warning (ADR-0107);
4. turns any foreign exception into ``unexpected``, letting ``MemoryError``
   and ``KeyboardInterrupt`` through.

Each call emits a ``stage_start`` and a ``stage_end`` Pipeline Event to the
progress observer, when one is set. The CLI sets one only for ``--progress``.
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .._observability import EventType, Observer, PipelineEvent, _emit
from ..records import (  # noqa: F401 - re-exported for the Operation modules
    DataForgeError,
    OutputRecord,
    canonical_json,
)
from . import _docstring
from ._literal import withhold_literals
from ._schema import to_strict_subset
from ._settings import resolve_settings
from ._workspace import _CALL, CallContext

#: Default Result Budget, in ``o200k_base`` tokens (ADR-0102, ADR-0112).
DEFAULT_RESULT_BUDGET = 10_000
#: Most characters a result's JSON may have, whatever its token count (ADR-0112).
RESULT_BUDGET_CHARACTERS = 25_000
#: Most characters a Library Error's JSON may have (ADR-0112).
ERROR_BUDGET_CHARACTERS = 8_000
#: The catalogue's closed ``effect`` and ``phase`` vocabularies (ADR-0109).
EFFECTS = ("read", "create", "export")
PHASES = ("workspace", "load", "profile", "split", "impute", "evaluate")

_FOREIGN_MESSAGE_CAP = 300
_PASS_THROUGH = (MemoryError, KeyboardInterrupt)


class _Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


@dataclass(frozen=True)
class OperationSpec:
    """Everything the deliveries and the contract tests know about one Operation."""

    name: str
    function: Callable[..., OutputRecord]
    args_model: type[BaseModel]
    result_model: type[OutputRecord]
    effect: str
    phase: str
    scales_with_data: bool
    result_budget_tokens: int
    docstring: _docstring.ParsedDocstring
    module: str

    def input_schema(self) -> dict[str, Any]:
        """Return the arguments schema, in the Strict Subset."""
        return to_strict_subset(self.args_model.model_json_schema())

    def output_schema(self) -> dict[str, Any]:
        """Return the result schema, in the Strict Subset."""
        return to_strict_subset(self.result_model.model_json_schema(mode="serialization"))

    def errors(self) -> list[str]:
        """Return the Error Codes the docstring's ``Errors`` section lists."""
        return list(self.docstring.entries("Errors"))


_OPERATIONS: dict[str, OperationSpec] = {}
_PROGRESS: ContextVar[Observer | None] = ContextVar("dataforge_ml_progress", default=None)


@contextmanager
def progress_to(observer: Observer | None) -> Generator[None]:
    """Send every Pipeline Event of the Operations called in the block to ``observer``.

    Parameters
    ----------
    observer : callable or None
        Called with each :class:`~dataforge_ml._observability.PipelineEvent`.
        None turns progress off.
    """
    token = _PROGRESS.set(observer)
    try:
        yield
    finally:
        _PROGRESS.reset(token)


def progress_observer() -> Observer | None:
    """Return the progress observer of the running call, or None."""
    return _PROGRESS.get()


def _stage(spec: OperationSpec, event_type: EventType, message: str) -> None:
    _emit(
        PipelineEvent(
            event_type=event_type, phase=spec.phase, stage=spec.name, message=message
        ),
        _PROGRESS.get(),
    )


def operations() -> dict[str, OperationSpec]:
    """Return every registered Operation, by name, after importing them all."""
    from .. import operations as _  # noqa: F401  (registers every Operation)

    return dict(_OPERATIONS)


def _param_path(loc: tuple[Any, ...]) -> str:
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else str(part)
    return path or "arguments"


def _invalid_argument(exc: ValidationError, order: list[str]) -> DataForgeError:
    # Signature order first, then unknown arguments by name: the same error
    # in both deliveries, whatever order the arguments arrived in.
    def rank(err: Any) -> tuple[int, str]:
        head = err["loc"][0] if err["loc"] else ""
        return (order.index(head) if head in order else len(order), _param_path(err["loc"]))

    issues, params, allowed = [], [], []
    for err in sorted(exc.errors(include_input=False, include_url=False), key=rank):
        path = _param_path(err["loc"])
        issues.append({"param": path, "problem": err["msg"]})
        if path not in params:
            params.append(path)
        if err["type"] in ("enum", "literal_error"):
            allowed += re.findall(r"'([^']*)'", str(err.get("ctx", {}).get("expected", "")))
    names = ", ".join(f"`{p}`" for p in params[:5])
    more = f" and {len(params) - 5} more" if len(params) > 5 else ""
    return DataForgeError.from_code(
        "invalid_argument",
        f"{len(params)} argument(s) are invalid: {names}{more}. See `issues`.",
        param=params[0] if len(params) == 1 else None,
        params=params,
        allowed=allowed or None,
        issues=issues,
    )


def _unexpected(exc: Exception) -> DataForgeError:
    text = str(exc).splitlines()[0] if str(exc) else ""
    if len(text) > _FOREIGN_MESSAGE_CAP:
        text = text[: _FOREIGN_MESSAGE_CAP - 1] + "…"
    name = type(exc).__name__
    return DataForgeError.from_code("unexpected", f"{name}: {text}" if text else name)


def _loosen(value: Any) -> Any:
    # Shapes are loose (ADR-0101): a tuple is accepted for an array.
    if isinstance(value, tuple):
        return [_loosen(v) for v in value]
    if isinstance(value, list):
        return [_loosen(v) for v in value]
    if isinstance(value, dict):
        return {k: _loosen(v) for k, v in value.items()}
    return value


def _run(spec: OperationSpec, validate: Callable[[], BaseModel]) -> OutputRecord:
    try:
        try:
            args = validate()
        except ValidationError as exc:
            raise _invalid_argument(exc, list(spec.args_model.model_fields)) from None
        call = CallContext(
            operation=spec.name,
            arguments=args.model_dump(mode="json", exclude_unset=True),
            settings=resolve_settings(),
        )
        token = _CALL.set(call)
        try:
            _stage(spec, EventType.stage_start, f"{spec.name}: started")
            result = spec.function(**{k: getattr(args, k) for k in type(args).model_fields})
            _stage(spec, EventType.stage_end, f"{spec.name}: finished")
        finally:
            _CALL.reset(token)
        if not isinstance(result, spec.result_model):
            raise DataForgeError.from_code(
                "unexpected",
                f"Operation {spec.name!r} returned {type(result).__name__}, "
                f"not {spec.result_model.__name__}.",
            )
        if call.settings.literal_policy == "deny":
            result = withhold_literals(result)
        return result
    except DataForgeError:
        raise
    except _PASS_THROUGH:
        raise
    except Exception as exc:
        raise _unexpected(exc) from exc


def invoke_python(spec: OperationSpec, arguments: dict[str, Any]) -> OutputRecord:
    """Run an Operation from Code Mode keyword arguments.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.
    arguments : dict
        Keyword arguments as the Agent's code passed them.

    Returns
    -------
    OutputRecord
        The Operation's Wire Record.

    Raises
    ------
    DataForgeError
        Every failure, foreign ones mapped to ``unexpected``.
    """
    return _run(spec, lambda: spec.args_model.model_validate(_loosen(arguments)))


def invoke_json(spec: OperationSpec, arguments_json: str) -> OutputRecord:
    """Run an Operation from its arguments as one JSON object, as the CLI builds them.

    Parameters
    ----------
    spec : OperationSpec
        The Operation.
    arguments_json : str
        The arguments, as JSON object text.

    Returns
    -------
    OutputRecord
        The Operation's Wire Record.

    Raises
    ------
    DataForgeError
        Every failure, foreign ones mapped to ``unexpected``.
    """
    return _run(spec, lambda: spec.args_model.model_validate_json(arguments_json))


def _build_spec(
    func: Callable[..., OutputRecord],
    *,
    effect: str,
    phase: str,
    scales_with_data: bool,
    result_budget_tokens: int,
) -> OperationSpec:
    name = func.__name__
    if effect not in EFFECTS or phase not in PHASES:
        raise ValueError(f"{name}: unknown effect {effect!r} or phase {phase!r}")
    if result_budget_tokens > DEFAULT_RESULT_BUDGET:
        raise ValueError(f"{name}: a Result Budget may be lower than the default, never higher")
    doc = _docstring.parse(func.__doc__)
    descriptions = doc.entries("Parameters")
    hints = typing.get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{name}: *args and **kwargs are outside the Strict Subset")
        if param.name not in hints:
            raise TypeError(f"{name}: parameter {param.name!r} has no annotation")
        if param.default is param.empty:
            default: Any = ...
        elif param.default is None:
            default = None
        else:
            raise TypeError(f"{name}: {param.name!r} defaults to non-null; null means default")
        fields[param.name] = (
            hints[param.name],
            Field(default, description=descriptions.get(param.name)),
        )
    result_model = hints.get("return")
    if not (isinstance(result_model, type) and issubclass(result_model, OutputRecord)):
        raise TypeError(f"{name}: must return an OutputRecord subclass")
    args_model = create_model(f"{name}_arguments", __base__=_Arguments, **fields)
    return OperationSpec(
        name=name,
        function=func,
        args_model=args_model,
        result_model=result_model,
        effect=effect,
        phase=phase,
        scales_with_data=scales_with_data,
        result_budget_tokens=result_budget_tokens,
        docstring=doc,
        module=func.__module__,
    )


def operation(
    *,
    effect: str,
    phase: str,
    scales_with_data: bool,
    result_budget_tokens: int = DEFAULT_RESULT_BUDGET,
) -> Callable[[Callable[..., OutputRecord]], Callable[..., OutputRecord]]:
    """Declare a function as an Operation and register it.

    Parameters
    ----------
    effect : str
        ``read``, ``create`` or ``export`` (ADR-0109).
    phase : str
        One of :data:`PHASES`.
    scales_with_data : bool
        True when the Operation reads dataset rows.
    result_budget_tokens : int, optional
        The declared Result Budget; at most the 10k default.

    Returns
    -------
    callable
        A decorator returning the Code Mode entry point, whose signature and
        docstring are the Operation's own.
    """

    def decorate(func: Callable[..., OutputRecord]) -> Callable[..., OutputRecord]:
        spec = _build_spec(
            func,
            effect=effect,
            phase=phase,
            scales_with_data=scales_with_data,
            result_budget_tokens=result_budget_tokens,
        )
        names = list(spec.args_model.model_fields)

        @functools.wraps(func)
        def code_mode(*args: Any, **kwargs: Any) -> OutputRecord:
            if len(args) > len(names):
                raise DataForgeError.from_code(
                    "invalid_argument",
                    f"`{spec.name}` takes at most {len(names)} positional "
                    f"argument(s); pass arguments by keyword.",
                )
            arguments = dict(zip(names, args))
            duplicated = sorted(set(arguments) & set(kwargs))
            if duplicated:
                raise DataForgeError.from_code(
                    "invalid_argument",
                    "An argument was passed both by position and by keyword.",
                    params=duplicated,
                )
            arguments.update(kwargs)
            return invoke_python(spec, arguments)

        code_mode.__operation__ = spec  # type: ignore[attr-defined]
        if spec.name in _OPERATIONS:
            raise ValueError(f"Operation {spec.name!r} is registered twice")
        _OPERATIONS[spec.name] = spec
        return code_mode

    return decorate
