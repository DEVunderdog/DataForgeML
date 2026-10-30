"""``query_frame``: raw cells of any frame Handle, filtered, as rows or grouped measures (ADR-0111)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataforge_ml._boundary._settings import LITERAL_POLICY_ENV

from .conftest import AMES, Harness, assert_within_budget, records


def _load(harness: Harness, path: str = "train.csv") -> str:
    return harness.code_mode("load", path=path).json["frame"]


def _csv(harness: Harness, name: str, lines: list[str]) -> str:
    path = harness.inputs / name
    path.write_text("\n".join(lines) + "\n")
    harness.declare(path)
    return _load(harness, str(path))


def _columns(json: dict) -> dict[str, list]:
    return {c["name"]: c["values"] for c in records(json["columns"])}


def test_rows_mode_returns_every_column_of_a_source_frame_in_dataset_order(
    harness: Harness,
) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame)

    assert out.json["frame"] == frame and out.json["mode"] == "rows"
    assert out.json["total_rows"] == 6 and out.json["matched_rows"] == 6
    assert out.json["row_positions"] == [0, 1, 2, 3, 4, 5]
    assert records(out.json["columns"]) == [
        {"kind": "cells", "name": "age", "dtype": "Int64", "values": [34, 41, 29, 55, 38, 47]},
        {
            "kind": "cells",
            "name": "city",
            "dtype": "String",
            "values": ["north", "south", "north", "east", None, "south"],
        },
        {
            "kind": "cells",
            "name": "income",
            "dtype": "Float64",
            "values": [52000.5, 61000.0, None, 73000.25, 48000.0, 66000.75],
        },
    ]
    assert records(out.json["omitted"]) == [] and records(out.json["warnings"]) == []


def test_rows_mode_reads_folds_and_imputed_frames(harness: Harness) -> None:
    frame = _load(harness)
    profile = harness.code_mode("profile", frame=frame).json["profile"]
    split = harness.code_mode(
        "split", profile=profile, scheme={"kind": "random", "test_size": 0.5}
    ).json
    train = records(split["folds"])[0]["train"]
    routing = harness.code_mode("route", profile=profile, default_strategy="median").json[
        "routing"
    ]
    resolved = harness.code_mode("resolve_recipe", routing=routing).json
    units = [
        harness.code_mode(
            "fit_unit", recipe=resolved["recipe"], unit_id=u["unit_id"]
        ).json["unit"]
        for u in records(resolved["units"])
    ]
    imputer = harness.code_mode(
        "compose_imputer", recipe=resolved["recipe"], units=units
    ).json["imputer"]
    imputed = harness.code_mode("impute", imputer=imputer, frame=frame).json["frame"]

    fold = harness.call("query_frame", frame=train, mode={"kind": "rows", "columns": ["age"]})
    test = harness.call("query_frame", frame=split["test"])
    filled = harness.call("query_frame", frame=imputed, mode={"kind": "rows", "columns": ["income"]})

    assert fold.json["total_rows"] == 3 and len(_columns(fold.json)["age"]) == 3
    assert set(_columns(fold.json)["age"]) < {34, 41, 29, 55, 38, 47}
    assert test.json["total_rows"] == 3 and len(test.json["row_positions"]) == 3
    assert None not in _columns(filled.json)["income"]
    assert _columns(filled.json)["income"][:2] == [52000.5, 61000.0]


def test_limit_defaults_to_20_and_rows_past_it_are_counted_in_omitted(
    harness: Harness,
) -> None:
    frame = _csv(harness, "long.csv", ["n", *map(str, range(60))])

    default = harness.call("query_frame", frame=frame)
    widest = harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 50})
    over = harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 51})

    assert default.json["row_positions"] == list(range(20))
    assert _columns(default.json)["n"] == list(range(20))
    assert default.json["matched_rows"] == 60
    assert records(default.json["omitted"]) == [
        {"field": "row_positions", "shown": 20, "total": 60, "operation": None, "reason": "cap"}
    ]
    assert widest.json["row_positions"] == list(range(50))
    assert over.code == "invalid_argument"


def test_at_most_10_columns_are_returned_and_the_rest_counted(harness: Harness) -> None:
    names = [f"c{i:02d}" for i in range(12)]
    frame = _csv(harness, "wide.csv", [",".join(names), ",".join("1" * 12)])

    every = harness.call("query_frame", frame=frame)
    named = harness.call(
        "query_frame", frame=frame, mode={"kind": "rows", "columns": names[::-1][:11]}
    )

    assert [c["name"] for c in records(every.json["columns"])] == names[:10]
    assert records(every.json["omitted"]) == [
        {"field": "columns", "shown": 10, "total": 12, "operation": "query_frame", "reason": "cap"}
    ]
    assert [c["name"] for c in records(named.json["columns"])] == names[::-1][:10]
    assert records(named.json["omitted"])[0]["total"] == 11


def test_text_cells_are_cut_at_64_characters_and_counted(harness: Harness) -> None:
    frame = _csv(harness, "text.csv", ["note", "x" * 100, "short", "y" * 64])

    out = harness.call("query_frame", frame=frame)

    cut, short, exact = _columns(out.json)["note"]
    assert cut == "x" * 63 + "…" and short == "short" and exact == "y" * 64
    assert records(out.json["omitted"]) == [
        {"field": "columns[].values", "shown": 0, "total": 1, "operation": None, "reason": "truncated"}
    ]


def test_sentinel_cells_appear_as_stored(harness: Harness) -> None:
    frame = _csv(harness, "sentinel.csv", ["amount", "1.5", "-999.0", "2.5", "-999.0"])
    profile = harness.code_mode(
        "profile", frame=frame, sentinels=[{"column": "amount", "values": [-999.0]}]
    ).json["profile"]
    split = harness.code_mode(
        "split", profile=profile, scheme={"kind": "kfold", "k": 2}
    ).json
    folds = [f["train"] for f in records(split["folds"])]

    source = harness.call("query_frame", frame=frame)
    in_folds = [_columns(harness.call("query_frame", frame=f).json)["amount"] for f in folds]

    assert _columns(source.json)["amount"] == [1.5, -999.0, 2.5, -999.0]
    assert sum(v.count(-999.0) for v in in_folds) == 2


def test_an_unknown_column_names_every_one_and_offers_close_matches(
    harness: Harness,
) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame", frame=frame, mode={"kind": "rows", "columns": ["age", "citty", "incme"]}
    )

    assert out.code == "unknown_column"
    assert out.json["params"] == ["mode.columns[1]", "mode.columns[2]"]
    assert "city" in out.json["candidates"] and "income" in out.json["candidates"]


def test_a_wrong_handle_kind_is_refused(harness: Harness) -> None:
    frame = _load(harness)
    profile = harness.code_mode("profile", frame=frame).json["profile"]

    assert harness.call("query_frame", frame=profile).code == "wrong_handle_kind"


def test_deny_withholds_the_cells_and_keeps_counts_and_positions(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _load(harness)
    monkeypatch.setenv(LITERAL_POLICY_ENV, "deny")

    out = harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 4})

    assert out.json["total_rows"] == 6 and out.json["matched_rows"] == 6
    assert out.json["row_positions"] == [0, 1, 2, 3]
    assert [(c["name"], c["dtype"], c["values"]) for c in records(out.json["columns"])] == [
        ("age", "Int64", None),
        ("city", "String", None),
        ("income", "Float64", None),
    ]
    assert records(out.json["omitted"]) == [
        {"field": "row_positions", "shown": 4, "total": 6, "operation": None, "reason": "cap"},
        {
            "field": "columns[].values",
            "shown": 0,
            "total": 3,
            "operation": None,
            "reason": "literal_policy",
        },
    ]
    assert [w["code"] for w in records(out.json["warnings"])] == ["literal_policy"]


# --- The filter ------------------------------------------------------------------


def _where(column: str, op: str, value: object = None) -> dict:
    return {"column": column, "op": op, "value": value}


@pytest.mark.parametrize(
    ("predicate", "positions"),
    [
        (_where("age", "eq", 41), [1]),
        (_where("city", "ne", "north"), [1, 3, 5]),
        (_where("age", "lt", 38), [0, 2]),
        (_where("age", "le", 38), [0, 2, 4]),
        (_where("income", "gt", 61000.0), [3, 5]),
        (_where("income", "ge", 61000), [1, 3, 5]),
        (_where("city", "in", ["east", "south"]), [1, 3, 5]),
        (_where("city", "not_in", ["north"]), [1, 3, 5]),
        (_where("income", "is_null"), [2]),
        (_where("city", "not_null"), [0, 1, 2, 3, 5]),
    ],
)
def test_every_op_filters_the_rows(
    harness: Harness, predicate: dict, positions: list[int]
) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, filter=[[predicate]])

    assert out.json["row_positions"] == positions
    assert out.json["matched_rows"] == len(positions) and out.json["total_rows"] == 6
    ages = [34, 41, 29, 55, 38, 47]
    assert _columns(out.json)["age"] == [ages[p] for p in positions]


def test_the_filter_is_an_or_of_and_lists(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[
            [_where("age", "gt", 40), _where("city", "eq", "south")],
            [_where("income", "is_null")],
        ],
    )

    assert out.json["row_positions"] == [1, 2, 5]
    assert _columns(out.json)["age"] == [41, 29, 47]


def test_nan_and_booleans_filter_like_the_cells_read(harness: Harness) -> None:
    frame = _csv(harness, "nan.csv", ["x,flag", "1.0,true", "NaN,false", ",true", "3.0,false"])

    missing = harness.call("query_frame", frame=frame, filter=[[_where("x", "is_null")]])
    above = harness.call("query_frame", frame=frame, filter=[[_where("x", "ge", 0)]])
    flagged = harness.call("query_frame", frame=frame, filter=[[_where("flag", "eq", True)]])

    assert missing.json["row_positions"] == [1, 2]
    assert above.json["row_positions"] == [0, 3]
    assert flagged.json["row_positions"] == [0, 2]


def test_matched_rows_is_exact_when_the_rows_are_capped(harness: Harness) -> None:
    frame = _csv(harness, "long.csv", ["n", *map(str, range(60))])

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("n", "ge", 10)]],
        mode={"kind": "rows", "limit": 5},
    )

    assert out.json["matched_rows"] == 50 and out.json["total_rows"] == 60
    assert out.json["row_positions"] == [10, 11, 12, 13, 14]
    assert _columns(out.json)["n"] == [10, 11, 12, 13, 14]
    assert records(out.json["omitted"]) == [
        {"field": "row_positions", "shown": 5, "total": 50, "operation": None, "reason": "cap"}
    ]


def test_a_filter_matching_nothing_returns_no_rows(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, filter=[[_where("age", "gt", 100)]])

    assert out.json["matched_rows"] == 0 and out.json["row_positions"] == []
    assert _columns(out.json)["age"] == []


@pytest.mark.parametrize(
    ("filter", "param"),
    [
        ([[_where("age", "eq", 1)]] * 6, "filter"),
        ([[_where("age", "eq", 1)] * 11], "filter[0]"),
        ([[_where("age", "in", list(range(101)))]], "filter[0][0]"),
    ],
)
def test_an_over_cap_filter_is_refused_before_any_work(
    harness: Harness, filter: list, param: str
) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, filter=filter)

    assert out.code == "invalid_argument"
    assert out.json["params"] == [param]
    assert any(cap in out.json["issues"][0]["problem"] for cap in ("5", "10", "100"))


@pytest.mark.parametrize(
    "predicate",
    [
        _where("age", "in", 41),
        _where("age", "eq", [41]),
        _where("age", "eq"),
        _where("age", "is_null", 41),
    ],
)
def test_a_value_that_does_not_fit_its_op_is_refused(harness: Harness, predicate: dict) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, filter=[[predicate]])

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["filter[0][0]"]


def test_a_value_that_does_not_fit_its_column_names_every_one(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[
            [_where("age", "eq", "old"), _where("city", "lt", 3)],
            [_where("income", "in", [1.5, True])],
        ],
    )

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["filter[0][0].value", "filter[0][1].value", "filter[1][0].value"]


def test_unknown_filter_columns_are_named_with_the_mode_columns(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("agee", "eq", 1)], [_where("city", "is_null")]],
        mode={"kind": "rows", "columns": ["incme"]},
    )

    assert out.code == "unknown_column"
    assert out.json["params"] == ["filter[0][0].column", "mode.columns[0]"]
    assert "age" in out.json["candidates"] and "income" in out.json["candidates"]


# --- The order -------------------------------------------------------------------


def _order(**order: object) -> dict:
    return {"kind": "rows", "order": order}


def test_seeded_random_order_is_repeatable_and_samples_the_matches(harness: Harness) -> None:
    frame = _csv(harness, "long.csv", ["n", *map(str, range(60))])
    matches = [[_where("n", "ge", 10)]]

    first = harness.call(
        "query_frame", frame=frame, filter=matches, mode=_order(kind="random", seed=7) | {"limit": 8}
    )
    again = harness.call(
        "query_frame", frame=frame, filter=matches, mode=_order(kind="random", seed=7) | {"limit": 8}
    )
    other = harness.call(
        "query_frame", frame=frame, filter=matches, mode=_order(kind="random", seed=8) | {"limit": 8}
    )
    default = harness.call("query_frame", frame=frame, mode=_order(kind="random"))
    default_again = harness.call("query_frame", frame=frame, mode=_order(kind="random"))

    positions = first.json["row_positions"]
    assert positions == again.json["row_positions"] != other.json["row_positions"]
    assert len(set(positions)) == 8 and all(p >= 10 for p in positions)
    assert positions != sorted(positions)
    assert _columns(first.json)["n"] == positions  # cells line up with source positions
    assert first.json["matched_rows"] == 50
    assert records(first.json["omitted"]) == [
        {"field": "row_positions", "shown": 8, "total": 50, "operation": None, "reason": "cap"}
    ]
    assert default.json["row_positions"] == default_again.json["row_positions"]
    assert len(default.json["row_positions"]) == 20


def test_random_order_returns_every_match_when_fewer_than_the_limit(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, mode=_order(kind="random", seed=1))

    assert sorted(out.json["row_positions"]) == [0, 1, 2, 3, 4, 5]
    ages = [34, 41, 29, 55, 38, 47]
    assert _columns(out.json)["age"] == [ages[p] for p in out.json["row_positions"]]


@pytest.mark.parametrize(
    ("direction", "positions"),
    [
        (None, [2, 0, 4, 1, 5, 3]),
        ("ascending", [2, 0, 4, 1, 5, 3]),
        ("descending", [3, 5, 1, 4, 0, 2]),
    ],
)
def test_sort_orders_by_the_named_column_and_direction(
    harness: Harness, direction: str | None, positions: list[int]
) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame", frame=frame, mode=_order(kind="sort", column="age", direction=direction)
    )

    assert out.json["row_positions"] == positions
    ages = [34, 41, 29, 55, 38, 47]
    assert _columns(out.json)["age"] == [ages[p] for p in positions]


def test_sort_puts_null_and_nan_last_and_keeps_ties_in_dataset_order(harness: Harness) -> None:
    frame = _csv(harness, "ties.csv", ["x,tag", "2.0,a", "NaN,b", "1.0,c", ",d", "2.0,e", "1.0,f"])

    up = harness.call("query_frame", frame=frame, mode=_order(kind="sort", column="x"))
    down = harness.call(
        "query_frame", frame=frame, mode=_order(kind="sort", column="x", direction="descending")
    )

    assert up.json["row_positions"] == [2, 5, 0, 4, 1, 3]
    assert down.json["row_positions"] == [0, 4, 2, 5, 1, 3]
    assert _columns(up.json)["tag"] == ["c", "f", "a", "e", "b", "d"]


def test_sort_applies_after_the_filter_and_before_the_limit(harness: Harness) -> None:
    frame = _csv(harness, "long.csv", ["n", *map(str, range(60))])

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("n", "lt", 30)]],
        mode=_order(kind="sort", column="n", direction="descending") | {"limit": 3},
    )

    assert out.json["row_positions"] == [29, 28, 27]
    assert _columns(out.json)["n"] == [29, 28, 27]
    assert out.json["matched_rows"] == 30


def test_an_unknown_sort_column_is_named(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, mode=_order(kind="sort", column="agee"))

    assert out.code == "unknown_column"
    assert out.json["params"] == ["mode.order.column"]
    assert "age" in out.json["candidates"]


# --- Aggregate mode --------------------------------------------------------------


def _aggregate(group_by: list[str] | None = None, *measures: tuple[str, str]) -> dict:
    return {
        "kind": "aggregate",
        "group_by": group_by,
        "measures": [{"column": c, "fn": fn} for c, fn in measures],
    }


def _measures(json: dict) -> dict[tuple[str, str], list]:
    every = [*records(json["measures"]), *records(json["literal_measures"])]
    return {(m["column"], m["fn"]): m["values"] for m in every}


def test_aggregate_mode_measures_each_group_largest_first(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        mode=_aggregate(
            ["city"],
            ("income", "count"),
            ("income", "null_count"),
            ("income", "mean"),
            ("age", "max"),
            ("age", "median"),
        ),
    )

    assert out.json["mode"] == "aggregate" and out.json["row_positions"] is None
    assert out.json["total_rows"] == 6 and out.json["matched_rows"] == 6
    assert out.json["group_count"] == 4 and out.json["group_sizes"] == [2, 2, 1, 1]
    assert records(out.json["columns"]) == [
        {"kind": "cells", "name": "city", "dtype": "String", "values": ["north", "south", "east", None]}
    ]
    assert [m["kind"] for m in records(out.json["measures"])] == ["measure"] * 3
    assert [m["kind"] for m in records(out.json["literal_measures"])] == ["literal_measure"] * 2
    assert _measures(out.json) == {
        ("income", "count"): [1, 2, 1, 1],
        ("income", "null_count"): [1, 0, 0, 0],
        ("income", "mean"): [52000.5, 63500.375, 73000.25, 48000.0],
        ("age", "max"): [34, 47, 55, 38],
        ("age", "median"): [31.5, 44.0, 55.0, 38.0],
    }
    assert records(out.json["omitted"]) == []


def test_every_measure_is_computed_over_the_whole_frame_with_no_group_by(
    harness: Harness,
) -> None:
    frame = _load(harness)
    fns = ("count", "null_count", "nunique", "sum", "mean", "std", "min", "max", "median")

    outs = [
        harness.call(
            "query_frame",
            frame=frame,
            mode=_aggregate([], *(("age", fn) for fn in fns[i : i + 5])),
        )
        for i in (0, 5)
    ]

    assert all(o.json["group_count"] == 1 and o.json["group_sizes"] == [6] for o in outs)
    assert all(records(o.json["columns"]) == [] for o in outs)
    measured = _measures(outs[0].json) | _measures(outs[1].json)
    ages = [34, 41, 29, 55, 38, 47]
    assert measured[("age", "count")] == [6] and measured[("age", "null_count")] == [0]
    assert measured[("age", "nunique")] == [6] and measured[("age", "sum")] == [sum(ages)]
    assert measured[("age", "mean")] == [pytest.approx(sum(ages) / 6)]
    assert measured[("age", "std")] == [pytest.approx(9.3095, abs=1e-4)]
    assert measured[("age", "min")] == [29] and measured[("age", "max")] == [55]
    assert measured[("age", "median")] == [39.5]
    assert harness.call(
        "query_frame", frame=frame, mode={"kind": "aggregate"}
    ).json["group_sizes"] == [6]


def test_aggregate_mode_measures_only_the_rows_the_filter_keeps(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("age", "gt", 35)]],
        mode=_aggregate(["city"], ("age", "sum"), ("age", "min")),
    )
    nothing = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("age", "gt", 100)]],
        mode=_aggregate([], ("age", "count"), ("age", "mean")),
    )

    assert out.json["matched_rows"] == 4 and out.json["group_count"] == 3
    assert out.json["group_sizes"] == [2, 1, 1]
    assert _columns(out.json)["city"] == ["south", "east", None]
    assert _measures(out.json) == {("age", "sum"): [88, 55, 38], ("age", "min"): [41, 55, 38]}
    assert nothing.json["matched_rows"] == 0 and nothing.json["group_sizes"] == [0]
    assert _measures(nothing.json) == {("age", "count"): [0], ("age", "mean"): [None]}


def test_nan_counts_as_null_in_measures_and_keys(harness: Harness) -> None:
    frame = _csv(harness, "nan.csv", ["x,k", "1.0,1.5", "NaN,NaN", ",", "3.0,1.5", "NaN,2.5"])

    out = harness.call(
        "query_frame",
        frame=frame,
        mode=_aggregate(["k"], ("x", "null_count"), ("x", "nunique"), ("x", "mean")),
    )

    assert out.json["group_sizes"] == [2, 2, 1]
    assert _columns(out.json)["k"] == [1.5, None, 2.5]
    assert _measures(out.json) == {
        ("x", "null_count"): [0, 2, 1],
        ("x", "nunique"): [2, 0, 0],
        ("x", "mean"): [2.0, None, None],
    }


def test_groups_over_several_keys_tie_in_key_order(harness: Harness) -> None:
    frame = _csv(
        harness, "keys.csv", ["a,b,n", "y,2,1", "x,2,1", "y,1,1", "x,1,1", "x,1,1", ",1,1"]
    )

    out = harness.call("query_frame", frame=frame, mode=_aggregate(["a", "b"], ("n", "sum")))

    assert out.json["group_sizes"] == [2, 1, 1, 1, 1]
    assert _columns(out.json) == {"a": ["x", "x", "y", "y", None], "b": [1, 2, 1, 2, 1]}
    assert _measures(out.json) == {("n", "sum"): [2, 1, 1, 1, 1]}


def test_at_most_100_groups_are_returned_and_group_count_is_exact(harness: Harness) -> None:
    lines = ["g,v"] + [f"{i},{i}" for i in range(130)] + [f"{i},{i}" for i in range(5)]
    frame = _csv(harness, "groups.csv", lines)

    out = harness.call("query_frame", frame=frame, mode=_aggregate(["g"], ("v", "count")))

    assert out.json["group_count"] == 130
    assert out.json["group_sizes"] == [2] * 5 + [1] * 95
    assert _columns(out.json)["g"] == list(range(100))
    assert _measures(out.json)[("v", "count")] == [2] * 5 + [1] * 95
    assert records(out.json["omitted"]) == [
        {"field": "group_sizes", "shown": 100, "total": 130, "operation": None, "reason": "cap"}
    ]


def test_long_text_group_keys_are_cut_at_64_characters(harness: Harness) -> None:
    frame = _csv(harness, "keys.csv", ["k", "x" * 100, "x" * 100, "short"])

    out = harness.call("query_frame", frame=frame, mode=_aggregate(["k"]))

    assert _columns(out.json)["k"] == ["x" * 63 + "…", "short"]
    assert records(out.json["omitted"]) == [
        {"field": "columns[].values", "shown": 0, "total": 1, "operation": None, "reason": "truncated"}
    ]


@pytest.mark.parametrize(
    ("mode", "param"),
    [
        (_aggregate(["age", "city", "income", "age"]), "mode.aggregate.group_by"),
        (_aggregate([], *[("age", "count")] * 6), "mode.aggregate.measures"),
        (_aggregate([], ("age", "mode")), "mode.aggregate.measures[0].fn"),
    ],
)
def test_an_over_cap_or_unknown_aggregate_argument_is_refused(
    harness: Harness, mode: dict, param: str
) -> None:
    frame = _load(harness)

    out = harness.call("query_frame", frame=frame, mode=mode)

    assert out.code == "invalid_argument"
    assert out.json["params"] == [param]


def test_a_numeric_measure_on_a_text_column_names_every_one(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        mode=_aggregate([], ("city", "nunique"), ("city", "mean"), ("city", "max")),
    )

    assert out.code == "invalid_argument"
    assert out.json["params"] == ["mode.measures[1].fn", "mode.measures[2].fn"]


def test_unknown_group_and_measure_columns_are_named(harness: Harness) -> None:
    frame = _load(harness)

    out = harness.call(
        "query_frame",
        frame=frame,
        filter=[[_where("agee", "eq", 1)]],
        mode=_aggregate(["citty"], ("incme", "sum")),
    )

    assert out.code == "unknown_column"
    assert out.json["params"] == [
        "filter[0][0].column",
        "mode.group_by[0]",
        "mode.measures[0].column",
    ]


def test_deny_withholds_group_keys_and_literal_measures_and_keeps_counts(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _load(harness)
    monkeypatch.setenv(LITERAL_POLICY_ENV, "deny")

    out = harness.call(
        "query_frame",
        frame=frame,
        mode=_aggregate(["city"], ("income", "mean"), ("age", "min")),
    )

    assert out.json["group_count"] == 4 and out.json["group_sizes"] == [2, 2, 1, 1]
    assert records(out.json["columns"])[0]["values"] is None
    assert _measures(out.json) == {
        ("income", "mean"): [52000.5, 63500.375, 73000.25, 48000.0],
        ("age", "min"): None,
    }
    assert [o["field"] for o in records(out.json["omitted"])] == [
        "columns[].values",
        "literal_measures[].values",
    ]


def test_aggregate_worst_case_is_under_the_result_budget(harness: Harness) -> None:
    # The largest aggregate result on Ames, the supported fixture rows mode's
    # worst case also uses: 100 groups of three text keys, and five measures
    # of long-fractioned numbers. Keys the full 64 characters long would
    # overrun the character budget, as 50 rows of ten such cells would.
    harness.declare(AMES)
    frame = _load(harness, str(AMES))
    keys = ["Neighborhood", "Exterior 1st", "Exterior 2nd"]
    fns = ("mean", "std", "median", "min", "max")

    out = harness.call(
        "query_frame", frame=frame, mode=_aggregate(keys, *(("SalePrice", f) for f in fns))
    )

    assert out.json["group_count"] > 100 and len(out.json["group_sizes"]) == 100
    assert_within_budget("query_frame", out.json, is_error=False)


def test_query_frame_never_writes_a_handle(harness: Harness, tmp_path: Path) -> None:
    frame = _load(harness)
    before = harness.handles()

    harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 3})

    assert harness.handles() == before
