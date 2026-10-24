"""Internal code behind the ``query_frame`` Operation (ADR-0111).

It reads any frame Handle and returns raw cells, column by column, exactly as
stored: declared sentinels are not read as missing. It never writes a Handle.
Every output cap cuts the result and is listed in ``omitted``; every input cap
refuses the call before any work.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from ..records import DataForgeError
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
#: Most AND-lists a filter ORs together.
OR_CAP = 5
#: Most predicates in one AND-list.
AND_CAP = 10
#: Most values an ``in`` or ``not_in`` predicate lists.
IN_CAP = 100
#: The ops that compare each cell with one value.
SCALAR_OPS = ("eq", "ne", "lt", "le", "gt", "ge")
#: The ops that check each cell against a list of values.
LIST_OPS = ("in", "not_in")
#: The ops that read no value.
NULL_OPS = ("is_null", "not_null")


def _accepts(dtype: pl.DataType) -> tuple[str, tuple[type, ...]] | None:
    # What kind of filter value a column of this dtype compares with, or None
    # when it takes only the null ops.
    if dtype == pl.Boolean:
        return "bools", (bool,)
    if dtype.is_numeric():
        return "numbers", (int, float)
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        return "strings", (str,)
    return None


def _value_problem(predicate: Any, dtype: pl.DataType) -> str | None:
    if predicate.op in NULL_OPS:
        return None
    accepts = _accepts(dtype)
    if accepts is None:
        return f"`{predicate.column}` is {dtype}, which takes only `is_null` and `not_null`"
    kind, types = accepts
    values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
    # bool is an int in Python, so a bool must not pass as a number.
    if all(isinstance(v, types) and (bool in types or not isinstance(v, bool)) for v in values):
        return None
    return f"`{predicate.column}` is {dtype}, so it takes {kind}"


def _predicate_expr(predicate: Any, dtype: pl.DataType) -> pl.Expr:
    # One predicate as a polars expression that is never null. NaN reads as
    # null (as the cells do), so a null or NaN cell passes only ``is_null``.
    cell = pl.col(predicate.column)
    nan = dtype.is_float()
    if predicate.op in NULL_OPS:
        missing = (cell.is_null() | cell.is_nan()) if nan else cell.is_null()
        return missing if predicate.op == "is_null" else ~missing
    if predicate.op in LIST_OPS:
        hit = (
            pl.any_horizontal(cell == pl.lit(v) for v in predicate.value)
            if predicate.value
            else pl.lit(False)
        )
        test = hit if predicate.op == "in" else ~hit
    else:
        value = pl.lit(predicate.value)
        test = {
            "eq": cell == value,
            "ne": cell != value,
            "lt": cell < value,
            "le": cell <= value,
            "gt": cell > value,
            "ge": cell >= value,
        }[predicate.op]
    if nan:
        test = test & cell.is_not_nan()
    return (test & cell.is_not_null()).fill_null(False)


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
    filter: Sequence[Sequence[Any]] | None,
    columns: Sequence[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return the first matching rows of a frame Handle, in dataset order, laid out by column.

    Parameters
    ----------
    frame : str
        The frame Handle.
    filter : sequence of sequence of Predicate or None
        An OR of AND-lists of predicates, each with ``column``, ``op`` and
        ``value``, already checked against the input caps and their ops.
        None keeps every row.
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
        ``unknown_column`` naming every name in ``filter`` or ``columns`` not
        in the frame; ``invalid_argument`` naming every filter value whose
        type does not fit its column; plus the frame Handle's own errors.
        Both are raised before any row is read.
    """
    data, _ = read_frame(frame, param="frame")
    names = list(data.columns)
    branches = [
        [(f"filter[{i}][{j}]", p) for j, p in enumerate(branch)]
        for i, branch in enumerate(filter or ())
    ]
    unknown = [
        (f"{at}.column", p.column)
        for branch in branches
        for at, p in branch
        if p.column not in names
    ]
    if columns is None:
        chosen = names
    else:
        chosen = list(dict.fromkeys(columns))
        unknown += [(f"mode.columns[{i}]", c) for i, c in enumerate(columns) if c not in names]
    if unknown:
        raise unknown_columns(unknown, pool=names, excluded=(), holder="frame")
    issues = [
        {"param": f"{at}.value", "problem": problem}
        for branch in branches
        for at, p in branch
        if (problem := _value_problem(p, data.schema[p.column])) is not None
    ]
    if issues:
        params = [i["param"] for i in issues]
        raise DataForgeError.from_code(
            "invalid_argument",
            f"{len(issues)} filter value(s) do not fit their column, first "
            f"`{params[0]}`: {issues[0]['problem']}. See `issues`.",
            param=params[0] if len(params) == 1 else None,
            params=params,
            issues=issues,
        )

    total_rows = data.height
    if branches:
        keep = pl.any_horizontal(
            pl.all_horizontal(_predicate_expr(p, data.schema[p.column]) for _, p in branch)
            for branch in branches
        )
        mask = data.select(keep.alias("keep")).to_series()
        matched = mask.arg_true().to_list()  # counted over every row, before any cap
        data = data.filter(mask)
    else:
        matched = list(range(total_rows))
    # No offset and a hard row cap, so no Operation returns the rows cut here.
    positions, _, omitted = page(
        matched,
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
        "total_rows": total_rows,
        "matched_rows": len(matched),
        "row_positions": positions,
        "columns": records,
    }
