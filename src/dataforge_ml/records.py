"""Shared records: the Library Error, Omission, Dial Pointer and Result Warning.

Every Operation's own file holds the Wire Records only it returns. The records
in this file are shared by all Operations, so they are documented once here.

Wire Record
    A frozen record that rejects unknown fields. In Code Mode an Operation
    returns one, and ``record.to_dict()`` gives exactly the JSON the
    ``dataforge`` CLI prints. Enums travel as their string values.

Library Error
    Every failure raises a :class:`DataForgeError`. Branch on ``err.code``,
    never on the message. ``err.to_dict()`` is the same JSON the CLI
    prints, and the CLI's exit code names the category. The exception class is one of six categories,
    and the category says what kind of fix is needed:

    - :class:`InvalidInput` and :class:`UnknownName`: fix the call.
    - :class:`Conflict`: change one of the two declarations.
    - :class:`Infeasible`: change the plan or the data.
    - :class:`IncompatibleArtifact`: run the Operation that made the Handle
      again.
    - :class:`Unexpected`: stop and report.

    Errors name arguments, columns and dials, never cell values. Every list
    field holds at most 20 items, ``candidates`` at most 5; a cut list is
    counted in ``omitted``.

Table Layout
    Every list of records in a result is a :class:`Table`. In Code Mode it is
    a read-only sequence of records, so ``result.warnings[0].code`` works as
    on a list. In JSON it is ``{"fields": [...], "rows": [[...], ...]}``: the
    field names once, then each record's values in that order. A row holds
    at most 12 fields; a wider record nests its extra fields.

Literal Field
    A result field that can hold a value from a cell of the data: a label,
    a bound, a percentile, the median or the mode. Every other field is free
    of cell values. A Literal Field is declared as
    ``Annotated[<type> | None, LITERAL]``. Its output schema node carries
    ``"x-literal": true``, so a Harness finds it by machine, and
    :func:`literal_field_paths` lists them in Python. Under the Operator's
    ``deny`` Literal Policy every Literal Field is ``null``, listed in
    ``omitted`` with reason ``literal_policy``, under one ``literal_policy``
    Result Warning. Counts, shares and values you supplied stay.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, Generic, Literal, TypeVar, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from ._boundary._error_codes import entry_for

#: Most items any list field of a Library Error holds.
ERROR_LIST_CAP = 20
#: Most close-match candidates an ``UnknownName`` error offers.
CANDIDATES_CAP = 5
#: Longest message a Library Error carries, in characters.
MESSAGE_CAP = 500
#: Most fields a row in Table Layout holds (ADR-0112).
MAX_ROW_FIELDS = 12


def canonical_json(value: Any) -> str:
    """Serialise a JSON value byte-stably: insertion key order, no spaces, no NaN.

    Parameters
    ----------
    value : Any
        A JSON value, such as a record's ``to_dict()``.

    Returns
    -------
    str
        The exact text the CLI prints.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class WireRecord(BaseModel):
    """Base of every Wire Record: frozen, strict, and closed to extra fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        use_attribute_docstrings=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as the exact JSON object the CLI prints.

        Returns
        -------
        dict
            Keys in schema order, enums as their values, floats unrounded.
        """
        return self.model_dump(mode="json")


#: The output schema keyword that marks a Literal Field (ADR-0107). An
#: ``x-`` keyword is an annotation every JSON Schema reader ignores, so the
#: mark changes no validation and carries no data.
LITERAL_KEYWORD = "x-literal"


class LiteralFieldMarker:
    """Marks a Wire Record field as a Literal Field: it can hold a cell value.

    Use the single instance :data:`LITERAL` in ``Annotated`` metadata. The
    mark is read by :func:`literal_field_paths`, and it writes
    ``"x-literal": true`` (:data:`LITERAL_KEYWORD`) on the field's schema
    node. The field's type must admit ``None``, which is what the ``deny``
    Literal Policy returns.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "LITERAL"

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {**handler(schema), LITERAL_KEYWORD: True}


#: The Literal Field mark: ``value: Annotated[float | None, LITERAL]``.
LITERAL = LiteralFieldMarker()


def literal_field_paths(model: type[BaseModel]) -> list[str]:
    """List every Literal Field a record can carry, as dotted paths.

    Nested records, Tables, lists and unions are walked; a list or Table
    step is written ``[]``, such as ``top_values[].value``.

    Parameters
    ----------
    model : type of BaseModel
        A Wire Record class, such as an Operation's result.

    Returns
    -------
    list of str
        The paths, in field declaration order.
    """
    paths: list[str] = []

    def records_in(annotation: Any) -> list[tuple[type[BaseModel], str]]:
        # Every record type reachable through unions, lists and Tables, with
        # the path suffix the container adds.
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Annotated:
            return records_in(args[0])
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [(annotation, "")]
        if origin in (list, tuple, Table) or (
            isinstance(origin, type) and issubclass(origin, Table)
        ):
            return [(m, "[]" + s) for a in args for m, s in records_in(a)]
        return [found for a in args for found in records_in(a)]

    def walk(record: type[BaseModel], prefix: str, seen: tuple[type, ...]) -> None:
        if record in seen:
            return
        for name, info in record.model_fields.items():
            path = f"{prefix}{name}"
            if any(isinstance(m, LiteralFieldMarker) for m in info.metadata):
                paths.append(path)
            for nested, suffix in records_in(info.annotation):
                walk(nested, f"{path}{suffix}.", (*seen, record))

    walk(model, "", ())
    return list(dict.fromkeys(paths))


R = TypeVar("R", bound=WireRecord)


class Table(Sequence[R], Generic[R]):
    """A list of records that travels in Table Layout.

    In Code Mode a Table reads like a tuple of records: index it, iterate it,
    take its ``len``. Only its JSON differs from a list's: ``fields`` names
    the record's fields once, in declaration order, and each entry of
    ``rows`` holds one record's values in that order. Declare a result field
    as ``Table[SomeRecord]``; the record may have at most 12 fields.

    Parameters
    ----------
    items : iterable of WireRecord, optional
        The records, in order. Empty when omitted.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[R] = ()) -> None:
        self._items: tuple[R, ...] = tuple(items)

    @overload
    def __getitem__(self, index: int) -> R: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[R]: ...
    def __getitem__(self, index: int | slice) -> R | Sequence[R]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[R]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Table):
            return self._items == other._items
        if isinstance(other, (list, tuple)):
            return self._items == tuple(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # unhashable, like the list it replaces

    def __repr__(self) -> str:
        return f"Table({list(self._items)!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        args = getattr(source, "__args__", ())
        if len(args) != 1 or not (isinstance(args[0], type) and issubclass(args[0], WireRecord)):
            raise TypeError("declare a Table as Table[SomeWireRecord]")
        record: type[WireRecord] = args[0]
        fields = list(record.model_fields)
        if len(fields) > MAX_ROW_FIELDS:
            raise TypeError(
                f"{record.__name__} has {len(fields)} fields; a row in Table Layout "
                f"holds at most {MAX_ROW_FIELDS}, so nest the extra fields"
            )

        def from_wire(value: Any) -> Any:
            if isinstance(value, Table):
                return list(value)
            if not isinstance(value, dict):
                return value  # a list of records, as the library builds them
            if list(value) != ["fields", "rows"] or value["fields"] != fields:
                raise ValueError(f"Table Layout must be {{fields: {fields}, rows: [...]}}")
            rows = value["rows"]
            if not isinstance(rows, list) or not all(
                isinstance(row, list) and len(row) == len(fields) for row in rows
            ):
                raise ValueError(f"every row must be a list of {len(fields)} values")
            return [dict(zip(fields, row)) for row in rows]

        def to_wire(table: Table[Any], info: core_schema.SerializationInfo) -> dict[str, Any]:
            return {
                "fields": list(fields),
                "rows": [list(item.model_dump(mode=info.mode).values()) for item in table],
            }

        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.no_info_before_validator_function(
                from_wire, core_schema.list_schema(handler.generate_schema(record))
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                to_wire, info_arg=True
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # after-validator -> before-validator -> list -> the record's schema.
        record = handler.resolve_ref_schema(handler(schema["schema"]["schema"]["items_schema"]))
        properties: dict[str, Any] = record["properties"]
        names = list(properties)
        positions = [
            {
                **prop,
                "description": f"Position {i}, `{name}`: {prop.get('description', '')}".rstrip(),
            }
            for i, (name, prop) in enumerate(properties.items())
        ]
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "enum": names},
                    "description": (
                        "Always exactly " + ", ".join(f"`{n}`" for n in names) + ", in "
                        "this order. A row's value at position i is the field named here "
                        "at i."
                    ),
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "array", "items": {"anyOf": positions}},
                    "description": (
                        f"{record.get('description', '')} One row per record, its "
                        "values in `fields` order. The i-th schema below describes "
                        "position i."
                    ).lstrip(),
                },
            },
            "required": ["fields", "rows"],
            "additionalProperties": False,
        }


class Omission(WireRecord):
    """Something left out of a result: a capped list was cut, or a Literal Field withheld."""

    field: str
    """Name of the list field that was cut, or the path of the Literal Field withheld, such as ``top_values[].value``."""
    shown: int
    """How many items the list holds; 0 for a withheld Literal Field."""
    total: int
    """How many items exist; for a withheld Literal Field, how many values at that path are null."""
    operation: str | None
    """The Operation that returns the rest, or null when no Operation does."""
    reason: Literal["cap", "literal_policy"] = "cap"
    """``cap`` when a list was cut to its cap; ``literal_policy`` when the Operator's Literal Policy is ``deny``."""


class ResultWarning(WireRecord):
    """A caveat on a successful result: it worked, but read this first."""

    code: str
    """The warning's Error Code. Branch on this, not on ``message``."""
    message: str
    """What happened and what to do, in one or two sentences."""
    columns: list[str] | None = None
    """Columns the warning concerns, or null."""
    param: str | None = None
    """The argument the warning concerns, or null."""
    hint: str | None = None
    """Library-written advice, or null."""


class DialPointer(WireRecord):
    """The dial whose value decided a threshold outcome. Turn it to change the outcome."""

    operation: str
    """The Operation that takes the dial."""
    dial: str
    """The dial's argument path, such as ``thresholds.missingness.high_null_ratio``."""
    value: bool | int | float | str | None
    """The dial's resolved value."""
    source: Literal["default", "argument"]
    """``default`` when the library default decided, ``argument`` when the caller set it."""


class OutputRecord(WireRecord):
    """Base of every Operation result: warnings and omissions in the same place."""

    warnings: Table[ResultWarning] = Field(default_factory=Table)
    """Caveats on this result, most actionable first. Empty when there are none."""
    omitted: Table[Omission] = Field(default_factory=Table)
    """One entry per list that was cut, naming the Operation that returns the rest. Empty when nothing was cut."""


class ColumnKeyedResult(OutputRecord):
    """A result whose ``columns`` list is keyed by column name."""

    def get(self, column: str) -> Any:
        """Return the ``columns`` entry for ``column``, or None.

        Parameters
        ----------
        column : str
            Column name, exactly as in the data.

        Returns
        -------
        record or None
            The entry whose ``column`` equals ``column``, or None when it is
            absent or was cut (check ``omitted``).
        """
        for item in self.columns:
            if item.column == column:
                return item
        return None


class ArgumentIssue(WireRecord):
    """One problem with one argument, inside an ``invalid_argument`` error."""

    param: str
    """Argument path, such as ``path`` or ``strategies[2].column``."""
    problem: str
    """What is wrong with it."""


class LibraryError(WireRecord):
    """The JSON shape of every failure. Fields that do not apply are absent."""

    code: str
    """The stable Error Code. Branch on this, not on ``message``."""
    message: str
    """What went wrong and what to do, in one or two sentences."""
    param: str | None = None
    """The argument at fault, as a path such as ``strategies[2].column``."""
    params: list[str] | None = None
    """Every argument at fault, when there are several."""
    columns: list[str] | None = None
    """Columns at fault."""
    allowed: list[str] | None = None
    """The values or kinds that would have been accepted."""
    candidates: list[str] | None = None
    """Close matches for an unknown name, best first."""
    hint: str | None = None
    """Library-written advice for this code."""
    issues: list[ArgumentIssue] | None = None
    """Per-argument problems, for ``invalid_argument``."""
    omitted: list[Omission] | None = None
    """Lists in this error that were cut."""

    def to_dict(self) -> dict[str, Any]:
        """Return the error as the exact JSON the CLI prints on failure.

        Returns
        -------
        dict
            ``code`` and ``message`` plus only the fields this error fills.
        """
        return {k: v for k, v in self.model_dump(mode="json").items() if v is not None}


def _cut(text: str, limit: int = MESSAGE_CAP) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class DataForgeError(Exception):
    """Base of every Library Error. Catch a category, then branch on ``code``.

    Raise one through :meth:`from_code`, which picks the category class from
    the Error Code registry. Constructing a category class directly checks
    that the code belongs to it.

    Parameters
    ----------
    code : str
        A registered Error Code.
    message : str
        What went wrong and what to do. Cut at 500 characters.
    **fields
        Optional record fields: ``param``, ``params``, ``columns``,
        ``allowed``, ``candidates``, ``hint`` and ``issues``. List fields are
        capped and the cut is counted in ``omitted``. ``hint`` defaults to the
        registry's hint.

    Raises
    ------
    TypeError
        If the base class is constructed directly, or ``code``'s registered
        category is not this class.
    """

    category: ClassVar[str] = ""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        param: str | None = None,
        params: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        allowed: Sequence[str] | None = None,
        candidates: Sequence[str] | None = None,
        hint: str | None = None,
        issues: Sequence[ArgumentIssue | dict[str, str]] | None = None,
    ) -> None:
        entry = entry_for(code)
        if type(self) is DataForgeError or entry.category != type(self).category:
            raise TypeError(
                f"Error Code {code!r} is registered under {entry.category!r}; "
                f"raise it through DataForgeError.from_code"
            )
        omitted: list[Omission] = []

        def capped(name: str, values: Iterable[Any] | None, cap: int) -> list[Any] | None:
            if values is None:
                return None
            values = list(values)
            if len(values) > cap:
                omitted.append(
                    Omission(field=name, shown=cap, total=len(values), operation=None)
                )
                values = values[:cap]
            return values

        self.record = LibraryError(
            code=code,
            message=_cut(message),
            param=param,
            params=capped("params", params, ERROR_LIST_CAP),
            columns=capped("columns", columns, ERROR_LIST_CAP),
            allowed=capped("allowed", allowed, ERROR_LIST_CAP),
            candidates=capped("candidates", candidates, CANDIDATES_CAP),
            hint=hint if hint is not None else entry.hint,
            issues=capped(
                "issues",
                None
                if issues is None
                else [
                    i if isinstance(i, ArgumentIssue) else ArgumentIssue(**i)
                    for i in issues
                ],
                ERROR_LIST_CAP,
            ),
            omitted=omitted or None,
        )
        super().__init__(f"{code}: {self.record.message}")

    @classmethod
    def from_code(cls, code: str, message: str, **fields: Any) -> DataForgeError:
        """Build the Library Error for ``code`` in its registered category.

        Parameters
        ----------
        code : str
            A registered Error Code.
        message : str
            What went wrong and what to do.
        **fields
            Optional record fields, as for the class constructor.

        Returns
        -------
        DataForgeError
            An instance of the category class the registry names for
            ``code``.

        Raises
        ------
        TypeError
            If ``code`` is registered as a ``warning``, not an error.
        """
        category = entry_for(code).category
        if category not in _CATEGORY_CLASSES:
            raise TypeError(f"Error Code {code!r} is a warning, not an error")
        return _CATEGORY_CLASSES[category](code, message, **fields)

    @property
    def code(self) -> str:
        """The stable Error Code."""
        return self.record.code

    def __getattr__(self, name: str) -> Any:
        # err.columns, err.param, ... read straight off the record.
        if name in LibraryError.model_fields:
            return getattr(self.record, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return the error as the exact JSON the CLI prints on failure.

        Returns
        -------
        dict
            ``code`` and ``message`` plus only the fields this error fills.
        """
        return self.record.to_dict()


class InvalidInput(DataForgeError):
    """An argument is malformed or not allowed. Fix the call."""

    category = "InvalidInput"


class UnknownName(DataForgeError):
    """A supplied name does not exist. Pick one of ``candidates``."""

    category = "UnknownName"


class Conflict(DataForgeError):
    """Two declarations contradict each other. Change one of them."""

    category = "Conflict"


class Infeasible(DataForgeError):
    """The call is well-formed but cannot run on this data. Change the plan or the data."""

    category = "Infeasible"


class IncompatibleArtifact(DataForgeError):
    """A stored Handle cannot be read by this library version. Make it again."""

    category = "IncompatibleArtifact"


class Unexpected(DataForgeError):
    """An internal failure. Stop and report the code and message."""

    category = "Unexpected"


_CATEGORY_CLASSES: dict[str, type[DataForgeError]] = {
    c.category: c
    for c in (InvalidInput, UnknownName, Conflict, Infeasible, IncompatibleArtifact, Unexpected)
}

__all__ = [
    "LITERAL",
    "LITERAL_KEYWORD",
    "ArgumentIssue",
    "ColumnKeyedResult",
    "Conflict",
    "DataForgeError",
    "DialPointer",
    "IncompatibleArtifact",
    "Infeasible",
    "InvalidInput",
    "LibraryError",
    "LiteralFieldMarker",
    "Omission",
    "OutputRecord",
    "ResultWarning",
    "Table",
    "Unexpected",
    "UnknownName",
    "WireRecord",
    "literal_field_paths",
]
