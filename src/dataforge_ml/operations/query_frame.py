"""The ``query_frame`` Operation: read raw cells from any frame Handle."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._query_frame import ROWS_CAP, query_rows
from ..records import LITERAL, OutputRecord, Table, WireRecord
from .describe_target import CellValue


class DatasetOrder(WireRecord):
    """The first rows, in the order the frame holds them."""

    kind: Literal["dataset"]
    """Always ``dataset``."""


class RowsMode(WireRecord):
    """Return rows: each chosen column's cells, one per returned row."""

    kind: Literal["rows"]
    """Always ``rows``."""
    columns: list[str] | None = None
    """The columns to return, in this order. Null means every column. At most 10 are returned, the first 10 named; ``omitted`` counts the rest."""
    limit: Annotated[int, Field(ge=1, le=ROWS_CAP)] | None = None
    """Most rows to return, from 1 to 50. Null means 20."""
    order: DatasetOrder | None = None
    """Which rows to return. Null means ``dataset``."""


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
    """Rows that match, counted over every row. With no filter, every row matches."""
    row_positions: list[int]
    """Each returned row's position in the frame, from 0, in returned order. When more rows match, ``omitted`` counts them."""
    columns: Table[CellsColumn]
    """The chosen columns, in the order asked, or in dataset order. At most 10; ``omitted`` counts the rest."""


@operation(effect="read", phase="workspace", scales_with_data=True)
def query_frame(frame: str, mode: RowsMode | None = None) -> QueryFrameResult:
    """Read raw cells from any frame Handle: a loaded frame, a split fold or an imputed frame.

    Use when
    --------
    Call this to look at actual rows: to check that a sentinel such as
    ``-999`` is where you think, to see what an imputation filled, or to
    eyeball a fold. Cells come back exactly as stored. Declared sentinels
    are not read as missing, so ``-999`` shows as ``-999``; ``missingness``
    gives the profile's view of what is missing. Rows mode returns the first
    ``limit`` rows in dataset order, each with its position in the frame.
    The cells are in Literal Fields. It never makes a Handle and never
    changes the frame.

    Parameters
    ----------
    frame : str
        A frame Handle, such as ``frame_7f3a9c2e``.
    mode : RowsMode or None
        ``{"kind": "rows", "columns": [...], "limit": n}``. ``columns`` null
        means every column, at most 10 returned; ``limit`` null means 20, at
        most 50. Null means rows mode with every default.

    Returns
    -------
    QueryFrameResult
        ``total_rows`` and ``matched_rows``, ``row_positions`` of the rows
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
        A name in ``mode.columns`` is not a column of the frame.
        ``candidates`` lists close matches; ``issues`` lists every unknown
        name.
    invalid_argument
        An argument has the wrong type or is out of its range.
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
    $ dataforge query-frame --frame frame_7f3a9c2e --mode '{"kind":"rows","columns":["age","city"],"limit":2}'
    {...,"frame":"frame_7f3a9c2e","mode":"rows","total_rows":6,"matched_rows":6,"row_positions":[0,1],"columns":{"fields":["kind","name","dtype","values"],"rows":[["cells","age","Int64",[34,41]],["cells","city","String",["north","south"]]]}}
    """
    mode = mode or RowsMode(kind="rows")
    return QueryFrameResult.model_validate(
        query_rows(frame=frame, columns=mode.columns, limit=mode.limit)
    )
