"""The ``list_handles`` Operation: find the Workspace's Handles again, with their Lineage."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._list_handles import PAGE_CAP, list_workspace_handles
from .._boundary._operation import operation
from ..records import OutputRecord, Table, WireRecord

#: The Handle kinds, as ``list_handles`` takes them (ADR-0110).
HandleKind = Literal["frame", "profile", "routing", "recipe", "unit", "imputer", "c2st"]


class LineageArgument(WireRecord):
    """One argument the Operation that made the Handle was called with."""

    name: str
    """Argument name, as in that Operation's signature."""
    value: str
    """The value as passed, as JSON text. Cut at 80 characters, ending in ``…``, when ``value_cut`` is true."""
    value_cut: bool
    """True when ``value`` was cut and is not valid JSON."""


class SizeMeasure(WireRecord):
    """One measure of a Handle's size."""

    measure: str
    """What is counted, such as ``rows`` or ``columns``."""
    value: int
    """The count."""


class HandleSummary(WireRecord):
    """One Handle and its Lineage: what made it, from what, with which arguments."""

    handle: str
    """The Handle, such as ``frame_7f3a9c2e``. Pass it to any Operation that takes this kind."""
    kind: str
    """The Handle's kind: the prefix before the underscore."""
    operation: str
    """The Operation that made the Handle."""
    arguments: Table[LineageArgument]
    """The arguments that Operation was called with, as passed. Arguments left at their default are absent."""
    inputs: list[str]
    """The Handles that Operation read, in the order it read them. At most 10; past 10, ``omitted`` counts the rest."""
    size: Table[SizeMeasure]
    """A short size summary, such as rows and columns for a frame."""


class ListHandlesResult(OutputRecord):
    """One page of Handles, newest first, with their Lineage."""

    total: int
    """How many Handles of the asked kind exist in the Workspace."""
    offset: int
    """How many of the newest Handles this page skipped."""
    handles: Table[HandleSummary]
    """Handles, newest first. At most 12; when more remain, ``omitted`` counts them and names ``list_handles``."""


@operation(effect="read", phase="workspace", scales_with_data=False)
def list_handles(
    kind: HandleKind | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=PAGE_CAP)] | None = None,
) -> ListHandlesResult:
    """List the Workspace's Handles newest first, each with its Lineage and size.

    Use when
    --------
    Call this to find Handles made earlier: after a restart, as a sub-agent
    sharing the Workspace, or when you have lost track of a Handle string.
    Each Handle comes with its Lineage, which is the Operation that made it,
    the arguments it was passed, and the Handles it read. Read the Lineage
    to tell two Handles of the same kind apart. Handles never expire and are
    never deleted, so an old Handle stays usable. This reads only stored
    records, never dataset rows.

    Parameters
    ----------
    kind : str or None
        Only Handles of this kind: ``frame``, ``profile``, ``routing``,
        ``recipe``, ``unit``, ``imputer`` or ``c2st``. Null lists every kind.
    offset : int or None
        How many of the newest Handles to skip, for the next page. Null
        means 0.
    limit : int or None
        Most Handles to return, from 1 to 12. Null means 12.

    Returns
    -------
    ListHandlesResult
        ``total`` Handles of the kind, and ``handles``: this page, newest
        first. When more remain, ``omitted`` counts them; call again with
        ``offset`` raised by the page size.

    Errors
    ------
    invalid_argument
        ``kind`` is not a Handle kind, or ``offset`` or ``limit`` is out of
        range.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    load : Make a frame Handle from a CSV file.
    export_frame : Write a frame Handle out as CSV.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge list-handles --kind frame
    {"warnings":...,"omitted":...,"total":1,"offset":0,
    "handles":{"fields":["handle","kind","operation","arguments","inputs","size"],
    "rows":[["frame_7f3a9c2e","frame","load",
    {"fields":["name","value","value_cut"],"rows":[["path","\\"train.csv\\"",false]]},
    [],{"fields":["measure","value"],"rows":[["rows",6],["columns",3]]}]]}}
    """
    return ListHandlesResult.model_validate(list_workspace_handles(kind, offset, limit))
