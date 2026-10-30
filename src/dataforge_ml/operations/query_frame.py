"""The ``query_frame`` Operation: read raw cells from any frame Handle."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .._boundary._operation import operation
from .._boundary._query_frame import (
    AND_CAP,
    GROUP_BY_CAP,
    IN_CAP,
    LIST_OPS,
    MEASURES_CAP,
    OR_CAP,
    ROWS_CAP,
    SCALAR_OPS,
    query_aggregate,
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


class Measure(WireRecord):
    """One measure of one column, taken over each group's rows."""

    column: str
    """The column measured."""
    fn: Literal["count", "null_count", "nunique", "sum", "mean", "std", "min", "max", "median"]
    """``count`` counts the cells that are not null, ``null_count`` those that are, ``nunique`` the distinct values that are not null. ``sum``, ``mean``, ``std`` (sample), ``min``, ``max`` and ``median`` read numeric columns only and skip null cells. NaN counts as null."""


class AggregateMode(WireRecord):
    """Return measures of the matching rows, one value per group."""

    kind: Literal["aggregate"]
    """Always ``aggregate``."""
    group_by: Annotated[list[str], Field(max_length=GROUP_BY_CAP)] | None = None
    """Up to 3 key columns; each distinct combination of their values is a group, a null key included. Null or empty means the matching rows are one group."""
    measures: Annotated[list[Measure], Field(max_length=MEASURES_CAP)] | None = None
    """Up to 5 measures, returned in this order within ``measures`` and ``literal_measures``. Null means none: only the group sizes."""


#: How ``query_frame`` reads the matching rows: as rows or as grouped measures.
Mode = Annotated[RowsMode | AggregateMode, Field(discriminator="kind")]


class CellsColumn(WireRecord):
    """One column's raw cells: one per returned row, or one group key per group."""

    kind: Literal["cells"]
    """Always ``cells``: raw cells of a column."""
    name: str
    """Column name."""
    dtype: str
    """The column's storage type, such as ``Int64``, ``Float64`` or ``String``."""
    values: Annotated[list[CellValue] | None, LITERAL]
    """The cells as stored, one per entry of ``row_positions`` or ``group_sizes``, in that order. Missing cells and NaN are null; datetimes are ISO text; text over 64 characters is cut and counted in ``omitted``. A Literal Field."""


class MeasureColumn(WireRecord):
    """One measure that is a count or a statistic no cell need equal, one value per group."""

    kind: Literal["measure"]
    """Always ``measure``: ``count``, ``null_count``, ``nunique``, ``sum``, ``mean`` or ``std``."""
    column: str
    """The column measured."""
    fn: Literal["count", "null_count", "nunique", "sum", "mean", "std"]
    """The measure."""
    values: list[int | float | None]
    """One value per entry of ``group_sizes``, in that order. Null when the group has no value to measure, such as ``std`` of one cell."""


class LiteralMeasureColumn(WireRecord):
    """One measure whose value can equal a cell, one value per group."""

    kind: Literal["literal_measure"]
    """Always ``literal_measure``: ``min``, ``max`` or ``median``."""
    column: str
    """The column measured."""
    fn: Literal["min", "max", "median"]
    """The measure."""
    values: Annotated[list[CellValue] | None, LITERAL]
    """One value per entry of ``group_sizes``, in that order. Null when the group has no cell to measure. A Literal Field."""


class QueryFrameResult(OutputRecord):
    """Raw cells or grouped measures from a frame, laid out by column. Read row or group ``i`` as ``values[i]`` of every column."""

    frame: str
    """The frame Handle read."""
    mode: Literal["rows", "aggregate"]
    """The mode that ran."""
    total_rows: int
    """Rows in the frame."""
    matched_rows: int
    """Rows that match the filter, counted exactly over every row, past any cap. With no filter, every row matches."""
    row_positions: list[int] | None
    """Rows mode: each returned row's position in the frame, from 0, in returned order. When more rows match, ``omitted`` counts them. Null in aggregate mode."""
    group_count: int | None
    """Aggregate mode: how many groups the matching rows form, counted exactly, past the cap of 100. Null in rows mode."""
    group_sizes: list[int] | None
    """Aggregate mode: each returned group's row count, largest first, ties in key order with null keys last. At most 100; ``omitted`` counts the rest. Null in rows mode."""
    columns: Table[CellsColumn]
    """Rows mode: the chosen columns, in the order asked, or in dataset order, at most 10; ``omitted`` counts the rest. Aggregate mode: the ``group_by`` keys, one value per group."""
    measures: Table[MeasureColumn]
    """Aggregate mode: the counts and statistics asked for, in the order asked. Empty in rows mode."""
    literal_measures: Table[LiteralMeasureColumn]
    """Aggregate mode: the ``min``, ``max`` and ``median`` measures asked for, in the order asked. Empty in rows mode."""


@operation(effect="read", phase="workspace", scales_with_data=True)
def query_frame(
    frame: str,
    filter: Annotated[list[AndList], Field(min_length=1, max_length=OR_CAP)] | None = None,
    mode: Mode | None = None,
) -> QueryFrameResult:
    """Read raw cells, or measures of them per group, from any frame Handle: a loaded frame, a split fold or an imputed frame.

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
    order. Aggregate mode summarises the matching rows instead: counts
    and statistics per group of up to 3 key columns, such as the mean
    ``income`` per ``city``.
    The cells, group keys, ``min``, ``max`` and ``median`` are in Literal
    Fields. It never makes a Handle and never changes the frame.

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
    mode : RowsMode, AggregateMode or None
        Rows mode is
        ``{"kind": "rows", "columns": [...], "limit": n, "order": {...}}``.
        ``columns`` null means every column, at most 10 returned; ``limit``
        null means 20, at most 50. ``order`` is ``{"kind": "dataset"}``
        (the default), ``{"kind": "random", "seed": n}`` (null seed means
        a fixed default, so the same call draws the same rows) or
        ``{"kind": "sort", "column": ..., "direction": "descending"}``
        (null direction means ascending; null and NaN cells last, ties in
        dataset order). Aggregate mode is
        ``{"kind": "aggregate", "group_by": [...], "measures": [{"column": ..., "fn": ...}]}``
        with up to 3 ``group_by`` columns (empty means one group of every
        matching row) and up to 5 measures: ``count``, ``null_count``,
        ``nunique``, and on numeric columns ``sum``, ``mean``, ``std``,
        ``min``, ``max`` and ``median``. NaN counts as null. Null means
        rows mode with every default.

    Returns
    -------
    QueryFrameResult
        ``total_rows`` and ``matched_rows``, the exact count of rows the
        filter keeps. Rows mode adds ``row_positions`` of the rows
        returned, and ``columns``: per column its ``name``, ``dtype`` and
        ``values``, one per row position. Row ``i`` is ``values[i]`` of
        every column. Aggregate mode adds ``group_count``, the exact
        number of groups, and per returned group, largest first, its size
        in ``group_sizes``, its keys in ``columns`` and its measures in
        ``measures`` and ``literal_measures``: group ``i`` is
        ``values[i]`` of each. ``omitted`` counts rows, columns and groups
        past the caps and text cells cut at 64 characters.

    Errors
    ------
    unknown_handle
        ``frame`` names no frame Handle in this Workspace. ``candidates``
        lists up to 5 frame Handles, newest first.
    wrong_handle_kind
        ``frame`` is a Handle of another kind. ``allowed`` names ``frame``.
    unknown_column
        A name in ``filter``, ``mode.columns``, ``mode.order.column``,
        ``mode.group_by`` or ``mode.measures`` is not a column of the frame.
        ``candidates`` lists close matches; ``issues`` lists every unknown
        name.
    invalid_argument
        An argument has the wrong type or is out of its range: a filter,
        ``group_by`` or ``measures`` past its caps, a value that does not
        fit its op, a value whose type does not fit its column, or a
        numeric measure on a column that is not numeric. ``issues`` lists
        every one.
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
    {...,"frame":"frame_7f3a9c2e","mode":"rows","total_rows":6,"matched_rows":3,"row_positions":[1,3],"group_count":null,"group_sizes":null,"columns":{"fields":["kind","name","dtype","values"],"rows":[["cells","age","Int64",[41,55]],["cells","city","String",["south","east"]]]},...}
    $ dataforge query-frame --frame frame_7f3a9c2e --mode '{"kind":"aggregate","group_by":["city"],"measures":[{"column":"income","fn":"mean"}]}'
    {...,"mode":"aggregate","total_rows":6,"matched_rows":6,"row_positions":null,"group_count":4,"group_sizes":[2,2,1,1],"columns":{"fields":["kind","name","dtype","values"],"rows":[["cells","city","String",["north","south","east",null]]]},"measures":{"fields":["kind","column","fn","values"],"rows":[["measure","income","mean",[52000.5,63500.375,73000.25,48000.0]]]},...}
    """
    mode = mode or RowsMode(kind="rows")
    if mode.kind == "aggregate":
        return QueryFrameResult.model_validate(
            query_aggregate(
                frame=frame, filter=filter, group_by=mode.group_by, measures=mode.measures
            )
        )
    return QueryFrameResult.model_validate(
        query_rows(
            frame=frame, filter=filter, columns=mode.columns, limit=mode.limit, order=mode.order
        )
    )
