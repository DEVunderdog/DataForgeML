"""The ``load`` Operation: read a Declared Input CSV into a frame Handle."""

from __future__ import annotations

from .._boundary._load import load_declared_csv
from .._boundary._operation import operation
from ..records import ColumnKeyedResult, WireRecord


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
    columns: list[LoadedColumn]
    """Columns in CSV header order, at most 100. Past 100, ``omitted`` counts the rest."""


@operation(effect="create", phase="load", scales_with_data=True)
def load(path: str) -> LoadResult:
    """Read a Declared Input CSV into a new frame Handle and report its shape.

    Use when
    --------
    Call this first, once per dataset, on a path the Harness declared. A
    Declared Input is a CSV path the Harness lets the library read; any other
    path is refused, including an edited copy of a declared file. Every later
    Operation reads the data through the returned frame Handle: a short string
    such as ``frame_7f3a9c2e`` naming the stored frame, which never changes.
    Loading the same path twice makes two Handles over identical data.

    Parameters
    ----------
    path : str
        A Declared Input path, exactly as the Harness gave it. A relative
        path is resolved from the working directory.

    Returns
    -------
    LoadResult
        ``frame``, ``n_rows``, ``n_columns``, and ``columns`` with each
        column's inferred dtype. ``columns`` lists at most 100; past that,
        ``omitted`` counts the rest and names ``describe_columns``.

    Errors
    ------
    input_not_declared
        ``path`` is not a Declared Input. Nothing is loaded or written.
    invalid_argument
        ``path`` is missing or not a string.
    workspace_not_set
        The Harness set no Workspace. Stop and report it.
    unexpected
        The file could not be read as CSV. Stop and report it.

    See Also
    --------
    profile : Profile the frame Handle. The next step.
    list_handles : Find frame Handles made earlier in this Workspace.

    Example
    -------
    >>> from dataforge_ml import load
    >>> result = load(path="train.csv")
    >>> result.frame.startswith("frame_")
    True
    >>> result.n_rows, result.n_columns
    (6, 3)
    >>> result.get("age").dtype
    'Int64'
    """
    return LoadResult.model_validate(load_declared_csv(path))
