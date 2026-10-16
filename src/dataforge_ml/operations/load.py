"""The ``load`` Operation: read a CSV into a frame Handle."""

from __future__ import annotations

from .._boundary._load import load_declared_csv
from .._boundary._operation import operation
from ..records import ColumnKeyedResult, Table, WireRecord


class LoadedColumn(WireRecord):
    """One column of the loaded frame."""

    column: str
    """Column name, exactly as in the CSV header."""
    dtype: str
    """Type the CSV reader inferred, such as ``Int64``, ``Float64`` or ``String``."""


class LoadResult(ColumnKeyedResult):
    """The new frame Handle and the frame's shape. ``result.get(name)`` finds one column."""

    frame: str
    """Frame Handle for the loaded data, such as ``frame_7f3a9c2e``."""
    n_rows: int
    """Number of data rows, not counting the header."""
    n_columns: int
    """Number of columns, including any not listed in ``columns``."""
    columns: Table[LoadedColumn]
    """Columns in CSV header order, at most 100. Past 100, ``omitted`` counts the rest."""


@operation(effect="create", phase="load", scales_with_data=True)
def load(path: str) -> LoadResult:
    """Read a CSV file into a new frame Handle and report its shape.

    Use when
    --------
    Call this first, once per dataset. If the Operator declared inputs, only
    those paths load, and any other path is refused, including an edited copy
    of a declared file. With none declared, any CSV path loads. Every later
    Operation reads the data through the returned frame Handle: a short string
    such as ``frame_7f3a9c2e`` naming the stored frame, which never changes.
    Loading the same path twice makes two Handles over identical data.

    Parameters
    ----------
    path : str
        Path of the CSV file, exactly as you were given it. A relative path
        is resolved from the working directory.

    Returns
    -------
    LoadResult
        ``frame``, ``n_rows``, ``n_columns``, and ``columns`` with each
        column's inferred dtype. ``columns`` lists at most 100; past that,
        ``omitted`` counts the rest.

    Errors
    ------
    input_not_declared
        The Operator declared inputs and ``path`` is not one of them. Nothing
        is loaded or written.
    input_not_found
        No file exists at ``path``.
    invalid_argument
        ``path`` is missing or not a string.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.
    unexpected
        The file could not be read as CSV. Stop and report it.

    See Also
    --------
    list_handles : Find frame Handles made earlier in this Workspace.
    export_frame : Write a frame Handle out as CSV.

    Example
    -------
    $ dataforge load --path train.csv
    {"warnings":{"fields":["code","message","columns","param","hint"],"rows":[]},
    "omitted":{"fields":["field","shown","total","operation","reason"],"rows":[]},
    "frame":"frame_7f3a9c2e","n_rows":6,"n_columns":3,
    "columns":{"fields":["column","dtype"],
    "rows":[["age","Int64"],["city","String"],["income","Float64"]]}}
    """
    return LoadResult.model_validate(load_declared_csv(path))
