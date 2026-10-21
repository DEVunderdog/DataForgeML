"""``impute`` end to end, in both deliveries: a model-ready frame Handle and per-column records."""

from __future__ import annotations

from typing import Any

from dataforge_ml._boundary._workspace import read_frame, read_record

from .conftest import Harness, records
from .test_route import N_ROWS, _profile, _routing_csv

#: ``income`` is missing on every fifth row of the routing fixture.
INCOME_MISSING = N_ROWS // 5


def _imputer(harness: Harness, profile: str, **route: Any) -> str:
    routing = harness.code_mode("route", profile=profile, **route).json["routing"]
    resolved = harness.code_mode("resolve_recipe", routing=routing).json
    recipe = resolved["recipe"]
    units = [
        harness.code_mode("fit_unit", recipe=recipe, unit_id=unit["unit_id"]).json["unit"]
        for unit in records(resolved["units"])
    ]
    return harness.code_mode("compose_imputer", recipe=recipe, units=units).json["imputer"]


def _source(profile: str) -> str:
    return read_record(profile, "profile", param="profile")["frame"]


def _by_column(out: Any) -> dict[str, dict[str, Any]]:
    return {row["column"]: row for row in records(out.json["columns"])}


# --- The frame ---------------------------------------------------------------------


def test_it_writes_a_filled_frame_handle_and_a_record_per_column(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(
        harness, profile, strategies=[{"column": "income", "strategy": "median"}]
    )
    source = _source(profile)

    out = harness.call("impute", imputer=imputer, frame=source)

    assert not out.is_error, out.json
    assert out.json["imputer"] == imputer
    assert out.json["source_frame"] == source
    assert out.json["rows"] == N_ROWS
    filled, _ = read_frame(out.json["frame"], param="frame")
    data, _ = read_frame(source, param="frame")
    assert filled.columns == ["age", "city", "income"]
    assert filled["income"].null_count() == 0
    median = data["income"].median()
    rows = _by_column(out)
    assert list(rows) == ["income", "age", "city"]  # fills first, then passthrough
    assert rows["income"] == {
        "column": "income", "strategy": "median", "missing": INCOME_MISSING,
        "filled": INCOME_MISSING, "fill_value": median,
    }
    assert rows["city"]["strategy"] == "passthrough"
    assert rows["city"]["filled"] == 0 and rows["city"]["missing"] > 0
    assert filled["city"].null_count() == rows["city"]["missing"]


def test_the_source_frame_stays_as_it_was(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(harness, profile, default_strategy="median")
    source = _source(profile)

    harness.code_mode("impute", imputer=imputer, frame=source)

    data, _ = read_frame(source, param="frame")
    assert data["income"].null_count() == INCOME_MISSING


def test_the_frame_handle_records_its_lineage(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(harness, profile, default_strategy="median")
    source = _source(profile)

    frame = harness.code_mode("impute", imputer=imputer, frame=source).json["frame"]

    record = read_record(frame, "frame", param="frame")
    assert record["lineage"]["operation"] == "impute"
    assert record["lineage"]["inputs"][:1] == [imputer]
    assert source in record["lineage"]["inputs"]
    assert record["size"] == {"rows": N_ROWS, "columns": 3}


# --- Exclusions and Dropped columns -------------------------------------------------


def test_hard_excluded_and_dropped_columns_are_not_in_the_frame(harness: Harness) -> None:
    profile = _profile(harness, exclude_columns=["city"])
    imputer = _imputer(
        harness, profile, strategies=[{"column": "income", "strategy": "dropped"}]
    )

    out = harness.call("impute", imputer=imputer, frame=_source(profile))

    assert not out.is_error, out.json
    filled, _ = read_frame(out.json["frame"], param="frame")
    assert filled.columns == ["age"]
    rows = _by_column(out)
    assert "city" not in rows
    assert rows["income"] == {
        "column": "income", "strategy": "dropped", "missing": INCOME_MISSING,
        "filled": 0, "fill_value": None,
    }


def test_a_column_the_routing_excludes_is_not_in_the_frame(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(
        harness, profile, exclude_columns=["city"],
        strategies=[{"column": "income", "strategy": "mean"}],
    )

    out = harness.call("impute", imputer=imputer, frame=_source(profile))

    assert not out.is_error, out.json
    filled, declarations = read_frame(out.json["frame"], param="frame")
    assert filled.columns == ["age", "income"]
    assert declarations.exclusions == ("city",)


# --- MNAR -------------------------------------------------------------------------


def test_an_mnar_column_keeps_its_nulls_with_its_fill_on_record(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(harness, profile, strategies=[{"column": "income", "strategy": "mnar"}])

    out = harness.call("impute", imputer=imputer, frame=_source(profile))

    assert not out.is_error, out.json
    filled, _ = read_frame(out.json["frame"], param="frame")
    assert filled["income"].null_count() == INCOME_MISSING
    assert filled["income_missing"].sum() == INCOME_MISSING
    rows = _by_column(out)
    assert rows["income"]["strategy"] == "mnar"
    assert rows["income"]["missing"] == INCOME_MISSING
    assert rows["income"]["filled"] == 0
    assert isinstance(rows["income"]["fill_value"], float)
    assert rows["income_missing"] == {
        "column": "income_missing", "strategy": "indicator", "missing": 0,
        "filled": 0, "fill_value": None,
    }


# --- Inherited Declarations ---------------------------------------------------------


def test_a_filled_fold_inherits_the_folds_declarations(harness: Harness) -> None:
    profile = _profile(
        harness, exclude_columns=["city"], sentinels=[{"column": "income", "values": [-1.0]}]
    )
    split = harness.code_mode("split", profile=profile, scheme={"kind": "kfold", "k": 3})
    assert not split.is_error, split.json
    fold = records(split.json["folds"])[0]
    routing = harness.code_mode(
        "route", profile=profile, strategies=[{"column": "income", "strategy": "median"}]
    ).json["routing"]
    recipe = harness.code_mode("resolve_recipe", routing=routing).json["recipe"]
    unit = harness.code_mode(
        "fit_unit", recipe=recipe, unit_id="median:income", frame=fold["train"]
    ).json["unit"]
    imputer = harness.code_mode("compose_imputer", recipe=recipe, units=[unit]).json["imputer"]

    out = harness.call("impute", imputer=imputer, frame=fold["validation"])

    assert not out.is_error, out.json
    assert (
        read_record(out.json["frame"], "frame", param="frame")["declarations"]
        == read_record(fold["validation"], "frame", param="frame")["declarations"]
    )
    filled, declarations = read_frame(out.json["frame"], param="frame")
    assert declarations.exclusions == ("city",)
    assert declarations.sentinels
    assert "city" not in filled.columns


def test_a_loaded_frame_takes_the_declarations_of_the_imputers_chain(harness: Harness) -> None:
    profile = _profile(harness, targets=["age"], exclude_columns=["city"])
    imputer = _imputer(harness, profile, default_strategy="median")
    loaded = harness.code_mode("load", path=str(_routing_csv(harness))).json["frame"]

    frame = harness.code_mode("impute", imputer=imputer, frame=loaded).json["frame"]

    _, declarations = read_frame(frame, param="frame")
    assert declarations.targets == ("age",)
    assert declarations.exclusions == ("city",)


# --- Errors -----------------------------------------------------------------------


def test_a_frame_with_a_column_the_imputer_never_saw_is_refused(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(harness, profile, default_strategy="median")
    wider = harness.inputs / "wider.csv"
    lines = _routing_csv(harness).read_text().splitlines()
    wider.write_text("\n".join([lines[0] + ",extra"] + [f"{l},1" for l in lines[1:]]) + "\n")
    harness.declare(wider)
    frame = harness.code_mode("load", path=str(wider)).json["frame"]
    before = set(harness.handles())

    out = harness.call("impute", imputer=imputer, frame=frame)

    assert out.code == "unseen_column"
    assert out.json["columns"] == ["extra"]
    assert set(harness.handles()) == before


def test_a_frame_lacking_a_fitted_column_is_refused(harness: Harness) -> None:
    profile = _profile(harness)
    imputer = _imputer(harness, profile, default_strategy="median")
    narrow = harness.inputs / "narrow.csv"
    rows = [line.rsplit(",", 1)[0] for line in _routing_csv(harness).read_text().splitlines()]
    narrow.write_text("\n".join(rows) + "\n")
    harness.declare(narrow)
    frame = harness.code_mode("load", path=str(narrow)).json["frame"]

    out = harness.call("impute", imputer=imputer, frame=frame)

    assert out.code == "missing_fitted_columns"
    assert out.json["columns"] == ["income"]


def test_an_imputer_argument_of_another_kind_is_refused(harness: Harness) -> None:
    profile = _profile(harness)
    source = _source(profile)

    out = harness.call("impute", imputer=source, frame=source)

    assert out.code == "wrong_handle_kind"
    assert out.json["param"] == "imputer"
