"""Shared records: the Library Error, Omission, Dial Pointer and Result Warning.

Every Operation's own file holds the Wire Records only it returns. The records
in this file are shared by all Operations, so they are documented once here.

Wire Record
    A frozen record that rejects unknown fields. In Code Mode an Operation
    returns one, and ``record.to_dict()`` gives exactly the JSON a Tool Call
    returns. Enums travel as their string values.

Library Error
    Every failure raises a :class:`DataForgeError`. Branch on ``err.code``,
    never on the message. ``err.to_dict()`` is the same JSON a Tool Call
    returns with ``is_error``. The exception class is one of six categories,
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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ._boundary._error_codes import entry_for

#: Most items any list field of a Library Error holds.
ERROR_LIST_CAP = 20
#: Most close-match candidates an ``UnknownName`` error offers.
CANDIDATES_CAP = 5
#: Longest message a Library Error carries, in characters.
MESSAGE_CAP = 500


class WireRecord(BaseModel):
    """Base of every Wire Record: frozen, strict, and closed to extra fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        use_attribute_docstrings=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as the exact JSON object a Tool Call returns.

        Returns
        -------
        dict
            Keys in schema order, enums as their values, floats unrounded.
        """
        return self.model_dump(mode="json")


class Omission(WireRecord):
    """A capped list was cut. Call ``operation`` to read the rest."""

    field: str
    """Name of the list field that was cut."""
    shown: int
    """How many items the list holds."""
    total: int
    """How many items exist."""
    operation: str | None
    """The Operation that returns the rest, or null when no Operation does."""


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

    warnings: list[ResultWarning] = Field(default_factory=list)
    """Caveats on this result, most actionable first. Empty when there are none."""
    omitted: list[Omission] = Field(default_factory=list)
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
        """Return the error as the exact JSON a Tool Call returns with ``is_error``.

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
        """Return the error as the exact JSON a Tool Call returns with ``is_error``.

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
    "ArgumentIssue",
    "ColumnKeyedResult",
    "Conflict",
    "DataForgeError",
    "DialPointer",
    "IncompatibleArtifact",
    "Infeasible",
    "InvalidInput",
    "LibraryError",
    "Omission",
    "OutputRecord",
    "ResultWarning",
    "Unexpected",
    "UnknownName",
    "WireRecord",
]
