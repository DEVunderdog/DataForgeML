"""Internal code behind the ``query_frame`` Operation (ADR-0111).

It reads any frame Handle and returns raw cells, column by column, exactly as
stored, or measures of them per group: declared sentinels are not read as
missing. It never writes a Handle.
Every output cap cuts the result and is listed in ``omitted``; every input cap
refuses the call before any work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
#: Most key columns an aggregate-mode call groups by.
GROUP_BY_CAP = 3
#: Most measures an aggregate-mode call takes.
MEASURES_CAP = 5
#: Most groups an aggregate-mode call returns, largest first.
GROUPS_CAP = 100
#: The measures that read only numeric columns.
NUMERIC_MEASURES = ("sum", "mean", "std", "min", "max", "median")
#: The measures whose value can equal a cell, so they go out in a Literal Field.
LITERAL_MEASURES = ("min", "max", "median")


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


def _ordered(data: pl.DataFrame, matched: list[int], order: Any) -> list[int]:
    # The matching positions in the order the rows are returned in. ``random``
    # draws a permutation of them; ``sort`` is stable, so ties keep dataset
    # order, and a null or NaN key comes last either way.
    if order is None or order.kind == "dataset":
        return matched
    if order.kind == "random":
        import numpy as np

        from .._seed import DEFAULT_SEED

        seed = DEFAULT_SEED if order.seed is None else order.seed
        return [matched[i] for i in np.random.default_rng(seed).permutation(len(matched))]
    dtype = data.schema[order.column]
    key = pl.col(order.column)
    if dtype.is_float():
        key = pl.when(key.is_nan()).then(None).otherwise(key)
    elif dtype == pl.Categorical:
        key = key.cast(pl.String)  # by label, not by when the label was first seen
    at = pl.Series("position", matched, dtype=pl.UInt32)
    keys = data.select(key.alias("key"))[at].with_columns(at)
    return (
        keys.sort(
            "key",
            descending=order.direction == "descending",
            nulls_last=True,
            maintain_order=True,
        )
        .get_column("position")
        .to_list()
    )


def _filtered(
    frame: str,
    filter: Sequence[Sequence[Any]] | None,
    named: list[tuple[str, str]],
    problems: Callable[[pl.Schema], list[dict[str, str]]] = lambda schema: [],
) -> tuple[pl.DataFrame, list[int]]:
    # Read the frame, check every name and value the call gives, then return
    # the frame and the positions of the rows the filter keeps, in dataset
    # order. ``named`` holds the mode's (param, column) pairs; ``problems``
    # adds the mode's own invalid arguments, reported with the filter's.
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
    unknown += [(at, c) for at, c in named if c not in names]
    if unknown:
        raise unknown_columns(unknown, pool=names, excluded=(), holder="frame")
    issues = [
        {"param": f"{at}.value", "problem": problem}
        for branch in branches
        for at, p in branch
        if (problem := _value_problem(p, data.schema[p.column])) is not None
    ]
    issues += problems(data.schema)
    if issues:
        params = [i["param"] for i in issues]
        raise DataForgeError.from_code(
            "invalid_argument",
            f"{len(issues)} argument(s) do not fit their column, first "
            f"`{params[0]}`: {issues[0]['problem']}. See `issues`.",
            param=params[0] if len(params) == 1 else None,
            params=params,
            issues=issues,
        )
    if not branches:
        return data, list(range(data.height))
    keep = pl.any_horizontal(
        pl.all_horizontal(_predicate_expr(p, data.schema[p.column]) for _, p in branch)
        for branch in branches
    )
    mask = data.select(keep.alias("keep")).to_series()
    return data, mask.arg_true().to_list()  # counted over every row, before any cap


def _cells_columns(
    columns: list[tuple[str, pl.DataType, list[Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # The ``cells`` column records for (name, dtype, values), and the
    # ``omitted`` entry counting the text cells cut, if any were.
    records, cut = [], 0
    for name, dtype, values in columns:
        cells = [_cell(v) for v in values]
        records.append(
            {"kind": "cells", "name": name, "dtype": str(dtype), "values": [v for v, _ in cells]}
        )
        cut += sum(was_cut for _, was_cut in cells)
    if not cut:
        return records, []
    return records, [
        {
            "field": "columns[].values",
            "shown": 0,
            "total": cut,
            "operation": None,
            "reason": "truncated",
        }
    ]


def query_rows(
    *,
    frame: str,
    filter: Sequence[Sequence[Any]] | None,
    columns: Sequence[str] | None,
    limit: int | None,
    order: Any = None,
) -> dict[str, Any]:
    """Return matching rows of a frame Handle in the asked order, laid out by column.

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
    order : DatasetOrder, RandomOrder, SortOrder or None
        Which matching rows to return: the first (None or ``dataset``), a
        permutation seeded by ``seed`` (``random``), or sorted by ``column``
        in ``direction`` (``sort``). The limit cuts after the order.

    Returns
    -------
    dict
        The fields of the ``query_frame`` result in rows mode.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every name in ``filter``, ``columns`` or
        ``order`` not in the frame; ``invalid_argument`` naming every filter value whose
        type does not fit its column; plus the frame Handle's own errors.
        Both are raised before any row is read.
    """
    named = [(f"mode.columns[{i}]", c) for i, c in enumerate(columns or ())]
    if order is not None and order.kind == "sort":
        named.append(("mode.order.column", order.column))
    data, matched = _filtered(frame, filter, named)
    chosen = list(data.columns) if columns is None else list(dict.fromkeys(columns))
    total_rows = data.height
    # No offset and a hard row cap, so no Operation returns the rows cut here.
    positions, _, omitted = page(
        _ordered(data, matched, order),
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

    head = data.select(chosen)[pl.Series(positions, dtype=pl.UInt32)]
    records, cut_cells = _cells_columns(
        [(name, data.schema[name], head.get_column(name).to_list()) for name in chosen]
    )
    omitted += cut_cells
    return {
        "warnings": [],
        "omitted": omitted,
        "frame": frame,
        "mode": "rows",
        "total_rows": total_rows,
        "matched_rows": len(matched),
        "row_positions": positions,
        "group_count": None,
        "group_sizes": None,
        "columns": records,
        "measures": [],
        "literal_measures": [],
    }


def _measure_expr(name: str, fn: str) -> pl.Expr:
    cell = pl.col(name)
    if fn == "count":
        return cell.count()
    if fn == "null_count":
        return cell.null_count()
    if fn == "nunique":
        return cell.drop_nulls().n_unique()
    return getattr(cell, fn)()  # sum, mean, std, min, max, median


def query_aggregate(
    *,
    frame: str,
    filter: Sequence[Sequence[Any]] | None,
    group_by: Sequence[str] | None,
    measures: Sequence[Any] | None,
) -> dict[str, Any]:
    """Return measures of the matching rows of a frame Handle, per group.

    NaN reads as null, as it does in the filter, so ``null_count`` counts it
    and the other measures skip it. Groups come largest first, ties in key
    order with a null key last; at most :data:`GROUPS_CAP` are returned.

    Parameters
    ----------
    frame : str
        The frame Handle.
    filter : sequence of sequence of Predicate or None
        As in :func:`query_rows`. Only the rows it keeps are measured.
    group_by : sequence of str or None
        The key columns, at most :data:`GROUP_BY_CAP`; None or empty means
        the matching rows are one group.
    measures : sequence of Measure or None
        Each with ``column`` and ``fn``, at most :data:`MEASURES_CAP`, in
        the order returned. None means none: only the group sizes.

    Returns
    -------
    dict
        The fields of the ``query_frame`` result in aggregate mode.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every name in ``filter``, ``group_by`` or
        ``measures`` not in the frame; ``invalid_argument`` naming every
        filter value whose type does not fit its column and every numeric
        measure on a column that is not numeric; plus the frame Handle's
        own errors. All are raised before any row is read.
    """
    keys = list(dict.fromkeys(group_by or ()))
    measures = list(measures or ())
    named = [(f"mode.group_by[{i}]", c) for i, c in enumerate(group_by or ())]
    named += [(f"mode.measures[{i}].column", m.column) for i, m in enumerate(measures)]

    def problems(schema: pl.Schema) -> list[dict[str, str]]:
        return [
            {
                "param": f"mode.measures[{i}].fn",
                "problem": f"`{m.fn}` reads numbers, and `{m.column}` is {schema[m.column]}",
            }
            for i, m in enumerate(measures)
            if m.fn in NUMERIC_MEASURES and not schema[m.column].is_numeric()
        ]

    data, matched = _filtered(frame, filter, named, problems)
    schema = data.schema
    used = list(dict.fromkeys([*keys, *(m.column for m in measures)]))
    clean = []
    for name in used:
        cell = pl.col(name)
        if schema[name].is_float():
            cell = pl.when(cell.is_nan()).then(None).otherwise(cell)
        elif name in keys and (schema[name] == pl.Categorical or isinstance(schema[name], pl.Enum)):
            cell = cell.cast(pl.String)  # sorted by label, not by when it was first seen
        clean.append(cell.alias(name))
    rows = data.select(clean)[pl.Series(matched, dtype=pl.UInt32)] if used else data.clear()
    aggregates = [
        pl.len().alias("__size"),
        *(_measure_expr(m.column, m.fn).alias(f"__m{i}") for i, m in enumerate(measures)),
    ]
    if keys:
        # Aliased so no key can share a name with a size or measure column.
        rows = rows.select(
            *(pl.col(k).alias(f"__k{i}") for i, k in enumerate(keys)),
            *(pl.col(c) for c in used),
        )
        key_aliases = [f"__k{i}" for i in range(len(keys))]
        groups = rows.group_by(key_aliases).agg(aggregates).sort(
            ["__size", *key_aliases],
            descending=[True, *(False for _ in keys)],
            nulls_last=True,
        )
    else:
        # Sized by hand: with no measure, ``rows`` has no columns, so no height.
        groups = pl.DataFrame({"__size": [len(matched)]})
        if measures:
            groups = groups.hstack(rows.select(aggregates[1:]))
        key_aliases = []
    group_count = groups.height
    sizes, _, omitted = page(
        groups.get_column("__size").to_list(),
        None,
        None,
        GROUPS_CAP,
        field="group_sizes",
        operation=None,
    )
    groups = groups.head(GROUPS_CAP)

    records, cut_cells = _cells_columns(
        [
            (name, schema[name], groups.get_column(alias).to_list())
            for name, alias in zip(keys, key_aliases)
        ]
    )
    omitted += cut_cells
    plain, literal = [], []
    for i, m in enumerate(measures):
        literal_fn = m.fn in LITERAL_MEASURES
        values = [_scalar(v) for v in groups.get_column(f"__m{i}").to_list()]
        (literal if literal_fn else plain).append(
            {
                "kind": "literal_measure" if literal_fn else "measure",
                "column": m.column,
                "fn": m.fn,
                "values": values,
            }
        )
    return {
        "warnings": [],
        "omitted": omitted,
        "frame": frame,
        "mode": "aggregate",
        "total_rows": data.height,
        "matched_rows": len(matched),
        "row_positions": None,
        "group_count": group_count,
        "group_sizes": sizes,
        "columns": records,
        "measures": plain,
        "literal_measures": literal,
    }
