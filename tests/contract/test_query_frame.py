"""``query_frame`` in rows mode: raw cells of any frame Handle, filtered, in dataset order (ADR-0111)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataforge_ml._boundary._settings import LITERAL_POLICY_ENV

from .conftest import Harness, records


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


def test_query_frame_never_writes_a_handle(harness: Harness, tmp_path: Path) -> None:
    frame = _load(harness)
    before = harness.handles()

    harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 3})

    assert harness.handles() == before
