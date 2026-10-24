"""``query_frame`` in rows mode: raw cells of any frame Handle, in dataset order (ADR-0111)."""

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


def test_query_frame_never_writes_a_handle(harness: Harness, tmp_path: Path) -> None:
    frame = _load(harness)
    before = harness.handles()

    harness.call("query_frame", frame=frame, mode={"kind": "rows", "limit": 3})

    assert harness.handles() == before
