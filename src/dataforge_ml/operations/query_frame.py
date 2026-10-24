"""The ``query_frame`` Operation: read raw cells from any frame Handle."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .._boundary._operation import operation
from .._boundary._query_frame import (
    AND_CAP,
    IN_CAP,
    LIST_OPS,
    OR_CAP,
    ROWS_CAP,
    SCALAR_OPS,
    query_rows,
)
from ..records import LITERAL, OutputRecord, Table, WireRecord
from .describe_target import CellValue

#: A filter value: a number, a string or a bool.
FilterScalar = bool | int | float | str


class Predicate(WireRecord):
    """One test on one column's raw cells. A null or NaN cell passes only ``is_null``."""

    column: str
    """The column the test reads."""
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_null", "not_null"]
    """The test. ``eq`` to ``ge`` compare each cell with ``value``; ``in`` and ``not_in`` check it against a list; ``is_null`` and ``not_null`` read no value. No string matching or regex."""
    value: FilterScalar | list[FilterScalar] | None = None
    """What the cells are tested against. One value for ``eq`` to ``ge``, a list of up to 100 for ``in`` and ``not_in``, null for ``is_null`` and ``not_null``. A numeric column takes numbers, a text column strings, a bool column bools; other columns take only ``is_null`` and ``not_null``."""

    @model_validator(mode="after")
    def _value_fits_op(self) -> Predicate:
        # The list cap is checked here, not on the union, so a long list is
        # one issue that names the cap rather than one per union branch.
        if self.op in SCALAR_OPS:
            if self.value is None or isinstance(self.value, list):
                raise ValueError(f"`{self.op}` takes one value, not a list or null")
        elif self.op in LIST_OPS:
            if not isinstance(self.value, list):
                raise ValueError(f"`{self.op}` takes a list of values")
            if len(self.value) > IN_CAP:
                raise ValueError(
                    f"`{self.op}` lists at most {IN_CAP} values, not {len(self.value)}"
                )
        elif self.value is not None:
            raise ValueError(f"`{self.op}` takes no value; set `value` to null")
        return self


#: Predicates that must all pass.
AndList = Annotated[list[Predicate], Field(min_length=1, max_length=AND_CAP)]


class DatasetOrder(WireRecord):
    """The first rows, in the order the frame holds them."""

    kind: Literal["dataset"]
    """Always ``dataset``."""


class RandomOrder(WireRecord):
    """A seeded random sample of the matching rows, in drawn order."""

    kind: Literal["random"]
    """Always ``random``."""
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None
    """Seed of the draw, from 0 to 4294967295. The same seed on the same frame and filter draws the same rows. Null means a fixed default."""


class SortOrder(WireRecord):
    """The matching rows sorted by one column. Null and NaN cells come last; ties keep dataset order."""

    kind: Literal["sort"]
    """Always ``sort``."""
    column: str
    """The column to sort by."""
    direction: Literal["ascending", "descending"] | None = None
    """``ascending`` puts the smallest first, ``descending`` the largest. Null means ``ascending``."""


#: Which matching rows rows mode returns.
Order = Annotated[DatasetOrder | RandomOrder | SortOrder, Field(discriminator="kind")]


class RowsMode(WireRecord):
    """Return rows: each chosen column's cells, one per returned row."""

    kind: Literal["rows"]
    """Always ``rows``."""
    columns: list[str] | None = None
    """The columns to return, in this order. Null means every column. At most 10 are returned, the first 10 named; ``omitted`` counts the rest."""
    limit: Annotated[int, Field(ge=1, le=ROWS_CAP)] | None = None
    """Most rows to return, from 1 to 50. Null means 20."""
    order: Order | None = None
    """Which matching rows to return: ``dataset`` takes the first, ``random`` a seeded sample, ``sort`` the top by one column. Null means ``dataset``."""


class CellsColumn(WireRecord):
    """One column's raw cells, one per returned row."""

    kind: Literal["cells"]
    """Always ``cells``: raw cells of a column."""
    name: str
    """Column name."""
    dtype: str
    """The column's storage type, such as ``Int64``, ``Float64`` or ``String``."""
    values: Annotated[list[CellValue] | None, LITERAL]
    """The cells as stored, one per entry of ``row_positions``, in that order. Missing cells and NaN are null; datetimes are ISO text; text over 64 characters is cut and counted in ``omitted``. A Literal Field."""


class QueryFrameResult(OutputRecord):
    """Raw cells from a frame, laid out by column. Read row ``i`` as ``values[i]`` of every column."""

    frame: str
    """The frame Handle read."""
    mode: Literal["rows"]
    """The mode that ran."""
    total_rows: int
    """Rows in the frame."""
    matched_rows: int
    """Rows that match the filter, counted exactly over every row, past any cap. With no filter, every row matches."""
    row_positions: list[int]
    """Each returned row's position in the frame, from 0, in returned order. When more rows match, ``omitted`` counts them."""
    columns: Table[CellsColumn]
    """The chosen columns, in the order asked, or in dataset order. At most 10; ``omitted`` counts the rest."""


@operation(effect="read", phase="workspace", scales_with_data=True)
def query_frame(
    frame: str,
    filter: Annotated[list[AndList], Field(min_length=1, max_length=OR_CAP)] | None = None,
    mode: RowsMode | None = None,
) -> QueryFrameResult:
    """Read raw cells from any frame Handle: a loaded frame, a split fold or an imputed frame.

    Use when
    --------
    Call this to look at actual rows: to check that a sentinel such as
    ``-999`` is where you think, to see what an imputation filled, or to
    eyeball a fold. Cells come back exactly as stored. Declared sentinels
    are not read as missing, so ``-999`` shows as ``-999``; ``missingness``
    gives the profile's view of what is missing. ``filter`` keeps the rows
    that pass any one of its AND-lists, such as every row where ``amount``
    is ``-999``. Rows mode returns up to ``limit`` matching rows, each
    with its position in the frame: the first ones in dataset order, a
    seeded ``random`` sample, or the extremes of one column in ``sort``
    order.
    The cells are in Literal Fields. It never makes a Handle and never
    changes the frame.

    Parameters
    ----------
    frame : str
        A frame Handle, such as ``frame_7f3a9c2e``.
    filter : list of list of Predicate or None
        An OR of up to 5 AND-lists, each of 1 to 10
        ``{"column": ..., "op": ..., "value": ...}`` predicates. A row
        matches when every predicate of any one list passes. The ops are
        ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``, ``in`` (up to 100
        values), ``not_in``, ``is_null`` and ``not_null``. A null or NaN
        cell passes only ``is_null``. A declared sentinel is a value like
        any other, so ``is_null`` can disagree with the profile's null
        ratio. Null keeps every row.
    mode : RowsMode or None
        ``{"kind": "rows", "columns": [...], "limit": n, "order": {...}}``.
        ``columns`` null means every column, at most 10 returned; ``limit``
        null means 20, at most 50. ``order`` is ``{"kind": "dataset"}``
        (the default), ``{"kind": "random", "seed": n}`` (null seed means
        a fixed default, so the same call draws the same rows) or
        ``{"kind": "sort", "column": ..., "direction": "descending"}``
        (null direction means ascending; null and NaN cells last, ties in
        dataset order). Null means rows mode with every default.

    Returns
    -------
    QueryFrameResult
        ``total_rows`` and ``matched_rows``, the exact count of rows the
        filter keeps, ``row_positions`` of the rows
        returned, and ``columns``: per column its ``name``, ``dtype`` and
        ``values``, one per row position. Row ``i`` is ``values[i]`` of
        every column. ``omitted`` counts rows and columns past the caps and
        text cells cut at 64 characters.

    Errors
    ------
    unknown_handle
        ``frame`` names no frame Handle in this Workspace. ``candidates``
        lists up to 5 frame Handles, newest first.
    wrong_handle_kind
        ``frame`` is a Handle of another kind. ``allowed`` names ``frame``.
    unknown_column
        A name in ``filter``, ``mode.columns`` or ``mode.order.column`` is
        not a column of the frame.
        ``candidates`` lists close matches; ``issues`` lists every unknown
        name.
    invalid_argument
        An argument has the wrong type or is out of its range: a filter
        past its caps, a value that does not fit its op, or a value whose
        type does not fit its column. ``issues`` lists every one.
    incompatible_artifact
        Another library format version wrote ``frame``. Load it again.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    list_handles : Find a frame Handle to read.
    value_summary : Read one column's top values and bounds off a profile.
    export_frame : Write the whole frame out as CSV.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge query-frame --frame frame_7f3a9c2e --filter '[[{"column":"age","op":"gt","value":40}]]' --mode '{"kind":"rows","columns":["age","city"],"limit":2}'
    {...,"frame":"frame_7f3a9c2e","mode":"rows","total_rows":6,"matched_rows":3,"row_positions":[1,3],"columns":{"fields":["kind","name","dtype","values"],"rows":[["cells","age","Int64",[41,55]],["cells","city","String",["south","east"]]]}}
    """
    mode = mode or RowsMode(kind="rows")
    return QueryFrameResult.model_validate(
        query_rows(
            frame=frame, filter=filter, columns=mode.columns, limit=mode.limit, order=mode.order
        )
    )
