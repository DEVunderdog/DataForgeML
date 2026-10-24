"""Internal code behind the ``query_frame`` Operation (ADR-0111).

It reads any frame Handle and returns raw cells, column by column, exactly as
stored: declared sentinels are not read as missing. It never writes a Handle.
Every output cap cuts the result and is listed in ``omitted``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._detail import page, unknown_columns
from ._profile_detail import _scalar
from ._workspace import read_frame

#: Rows a rows-mode call returns when ``limit`` is null.
ROWS_DEFAULT = 20
#: Most rows a rows-mode call returns.
ROWS_CAP = 50
#: Most columns a rows-mode call returns.
COLUMNS_CAP = 10
#: Longest string cell returned, in characters, before it is cut.
CELL_CAP = 64


def _cell(value: Any) -> tuple[Any, bool]:
    # A stored cell as JSON can carry it, and whether it was cut. NaN and the
    # infinities have no JSON form, so they read as null.
    value = _scalar(value)
    if isinstance(value, str) and len(value) > CELL_CAP:
        return value[: CELL_CAP - 1] + "…", True
    return value, False


def query_rows(
    *,
    frame: str,
    columns: Sequence[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return the first rows of a frame Handle, in dataset order, laid out by column.

    Parameters
    ----------
    frame : str
        The frame Handle.
    columns : sequence of str or None
        The columns to return, in this order; None means every column.
        Either way at most :data:`COLUMNS_CAP` are returned.
    limit : int or None
        Most rows to return; None means :data:`ROWS_DEFAULT`.

    Returns
    -------
    dict
        The fields of the ``query_frame`` result in rows mode.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every name not in the frame, plus the frame
        Handle's own errors.
    """
    data, _ = read_frame(frame, param="frame")
    names = list(data.columns)
    if columns is None:
        chosen = names
    else:
        chosen = list(dict.fromkeys(columns))
        unknown = [(f"mode.columns[{i}]", c) for i, c in enumerate(columns) if c not in names]
        if unknown:
            raise unknown_columns(unknown, pool=names, excluded=(), holder="frame")

    matched = range(data.height)  # no filter yet: every row matches
    # No offset and a hard row cap, so no Operation returns the rows cut here.
    positions, _, omitted = page(
        list(matched),
        None,
        ROWS_DEFAULT if limit is None else limit,
        ROWS_CAP,
        field="row_positions",
        operation=None,
    )
    chosen, _, cut_columns = page(
        chosen, None, None, COLUMNS_CAP, field="columns", operation="query_frame"
    )
    omitted += cut_columns

    head = data.head(len(positions))
    cut = 0
    records = []
    for name in chosen:
        values = []
        for value in head.get_column(name).to_list():
            value, was_cut = _cell(value)
            values.append(value)
            cut += was_cut
        records.append(
            {"kind": "cells", "name": name, "dtype": str(data.schema[name]), "values": values}
        )
    if cut:
        omitted.append(
            {
                "field": "columns[].values",
                "shown": 0,
                "total": cut,
                "operation": None,
                "reason": "truncated",
            }
        )
    return {
        "warnings": [],
        "omitted": omitted,
        "frame": frame,
        "mode": "rows",
        "total_rows": data.height,
        "matched_rows": len(matched),
        "row_positions": positions,
        "columns": records,
    }
