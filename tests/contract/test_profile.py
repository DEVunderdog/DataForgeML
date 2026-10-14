"""``profile`` end to end, in both deliveries: the Dataset Digest and its rules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataforge_ml._boundary._profile import FLAG_DIALS, dial_defaults
from dataforge_ml._boundary._settings import resolve_settings
from dataforge_ml._boundary._workspace import (
    _CALL,
    CallContext,
    InheritedDeclarations,
    write_handle,
)
from dataforge_ml.operations.profile import ProfileThresholds
from dataforge_ml.records import Conflict, InvalidInput, UnknownName

from .conftest import AMES, Harness, normalise_handles, records, write_flagged_wide_csv


def _frame(harness: Harness, path: str = "train.csv") -> str:
    return harness.code_mode("load", path=path).json["frame"]


def _profile_handles(harness: Harness) -> list[str]:
    return [h for h in harness.handles() if h.startswith("profile_")]


def _write_seeded_csv(path: Path) -> Path:
    # Enough rows for every stochastic step: the bimodal mixture fit, mutual
    # information and the random-forest nonlinearity probe.
    rng = np.random.default_rng(7)
    n = 240
    x = rng.uniform(0, 6, n)
    frame = pl.DataFrame(
        {
            "x": x,
            "wave": np.sin(x) * 3 + rng.normal(0, 0.2, n),
            "twin_peaks": np.where(rng.random(n) < 0.5, rng.normal(0, 1, n), rng.normal(9, 1, n)),
            "noise": rng.normal(0, 1, n),
            "price": x * 10 + rng.normal(0, 1, n),
        }
    ).with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("noise")).alias("noise")
    )
    frame.write_csv(path)
    return path


def _frame_with_declarations(harness: Harness, declarations: InheritedDeclarations) -> str:
    # No Operation makes a frame with declarations yet (split and impute will),
    # so write one the way they will: a frame Handle carrying its record.
    data = pl.read_csv(harness.inputs / "train.csv")
    call = CallContext(operation="split", arguments={}, settings=resolve_settings())
    token = _CALL.set(call)
    try:
        return write_handle(
            "frame",
            size={"rows": data.height, "columns": data.width},
            payload_name="frame.parquet",
            write_payload=lambda tmp: data.write_parquet(tmp),
            declarations=declarations,
        )
    finally:
        _CALL.reset(token)


# --- The digest -------------------------------------------------------------------


def test_profile_returns_a_profile_handle_and_the_digest(harness: Harness) -> None:
    frame = _frame(harness)

    out = harness.call("profile", frame=frame, targets=["income"])

    assert not out.is_error
    assert out.json["profile"].startswith("profile_") and out.json["frame"] == frame
    assert (out.json["n_rows"], out.json["n_columns"]) == (6, 3)
    assert out.json["duplicate_rows"] == 0
    assert records(out.json["targets"]) == [
        {"column": "income", "problem_type": "regression", "missing_ratio": 1 / 6}
    ]
    assert [r["column"] for r in records(out.json["columns"])] == ["age", "city", "income"]
    income = out.record.get("income")
    assert (income.role, income.semantic_type, income.cardinality) == ("target", "numeric", 5)
    assert out.record.get("city").null_ratio == 1 / 6
    assert out.json["profile"] in _profile_handles(harness)


def test_ames_digest_fits_the_result_budget_whole(harness: Harness) -> None:
    harness.declare(AMES)
    frame = _frame(harness, str(AMES))

    out = harness.call("profile", frame=frame, targets=["SalePrice"])  # budget checked inside

    columns = records(out.json["columns"])
    assert out.json["n_columns"] == 82 and len(columns) == 82
    assert records(out.json["omitted"]) == []
    assert out.record.get("SalePrice").role == "target"
    flagged = [bool(c["flags"]) for c in columns]
    assert flagged == sorted(flagged, reverse=True), "flagged columns come first"
    assert any(flagged)


def test_wide_digest_is_capped_at_100_flagged_first_naming_describe_columns(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", 150)
    harness.declare(wide)

    out = harness.call("profile", frame=_frame(harness, str(wide)))

    columns = records(out.json["columns"])
    assert len(columns) == 100 and columns[0]["column"].startswith("c0000_")
    assert all(c["flags"] for c in columns)
    assert records(out.json["omitted"])[0] == {
        "field": "columns", "shown": 100, "total": 150, "operation": "describe_columns",
    }


def test_flagged_columns_come_first_then_dataset_order(harness: Harness, tmp_path: Path) -> None:
    csv = tmp_path / "mixed.csv"
    rows = ["clean,sparse,steady"] + [
        f"{i},{'' if i % 4 else i},{i % 3}" for i in range(40)
    ]
    csv.write_text("\n".join(rows) + "\n")
    harness.declare(csv)

    out = harness.call("profile", frame=_frame(harness, str(csv)))

    order = [c["column"] for c in records(out.json["columns"])]
    assert order[0] == "sparse" and order[1:] == ["clean", "steady"]


# --- Dial Pointers ----------------------------------------------------------------


def test_every_threshold_decided_flag_carries_a_dial_pointer(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", 12)
    harness.declare(wide)

    out = harness.call(
        "profile",
        frame=_frame(harness, str(wide)),
        thresholds={"missingness": {"col_drop_threshold": 0.4}},
    )

    raised = {f for c in records(out.json["columns"]) for f in c["flags"]}
    pointers = {e["flag"]: e["dials"] for e in records(out.json["flag_dials"])}
    assert raised == set(pointers) == {"missingness.drop_candidate", "missingness.mar_suspect"}
    assert pointers["missingness.drop_candidate"] == [
        {
            "operation": "profile",
            "dial": "thresholds.missingness.col_drop_threshold",
            "value": 0.4,
            "source": "argument",
        }
    ]
    assert pointers["missingness.mar_suspect"] == [
        {
            "operation": "profile",
            "dial": "thresholds.missingness.mar_correlation_threshold",
            "value": 0.6,
            "source": "default",
        }
    ]


def test_turning_the_pointed_dial_clears_the_flag(harness: Harness, tmp_path: Path) -> None:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", 3)
    harness.declare(wide)
    frame = _frame(harness, str(wide))

    out = harness.call(
        "profile", frame=frame, thresholds={"missingness": {"col_drop_threshold": 0.7}}
    )

    raised = {f for c in records(out.json["columns"]) for f in c["flags"]}
    assert "missingness.drop_candidate" not in raised


def test_every_flag_the_profiler_can_raise_maps_to_real_dials() -> None:
    from dataforge_ml.profiling._boolean_config import BooleanFlag
    from dataforge_ml.profiling._categorical_config import CategoricalFlag
    from dataforge_ml.profiling._datetime_config import DatetimeFlag
    from dataforge_ml.profiling._missingness_config import MissingnessFlag
    from dataforge_ml.profiling._numeric_config import NumericFlag

    raisable = {
        f"{group}.{flag.value}"
        for group, enum in (
            ("missingness", MissingnessFlag),
            ("numeric", NumericFlag),
            ("categorical", CategoricalFlag),
            ("datetime", DatetimeFlag),
            ("boolean", BooleanFlag),
        )
        for flag in enum
    }
    assert set(FLAG_DIALS) == raisable
    assert {d for dials in FLAG_DIALS.values() for d in dials} <= set(dial_defaults())


def test_every_dial_on_the_wire_states_the_library_default() -> None:
    wire: dict[str, str] = {}
    for group, info in ProfileThresholds.model_fields.items():
        record = info.annotation.__args__[0]  # GroupDials | None
        for name, dial in record.model_fields.items():
            wire[f"{group}.{name}"] = dial.description or ""
    defaults = dial_defaults()
    assert set(wire) == set(defaults)
    for path, text in wire.items():
        stated = re.search(r"Null means (-?[0-9.]+)\.$", text)
        assert stated, f"{path}: description must end with its default"
        assert float(stated.group(1)) == pytest.approx(defaults[path]), path


# --- Seeds and fallbacks ----------------------------------------------------------


def test_null_seed_gives_an_identical_result_twice(harness: Harness, tmp_path: Path) -> None:
    csv = _write_seeded_csv(tmp_path / "seeded.csv")
    harness.declare(csv)
    frame = _frame(harness, str(csv))

    first = harness.code_mode("profile", frame=frame, targets=["price"])
    second = harness.call("profile", frame=frame, targets=["price"])

    assert first.json["profile"] != second.json["profile"]
    assert normalise_handles(str({**first.json, "profile": None})) == normalise_handles(
        str({**second.json, "profile": None})
    )
    assert "numeric.bimodal" in second.record.get("twin_peaks").flags


def test_correlation_and_nonlinearity_always_run_and_fall_back_with_codes(
    harness: Harness, tmp_path: Path
) -> None:
    wide = write_flagged_wide_csv(tmp_path / "wide.csv", 30)
    harness.declare(wide)

    out = harness.call("profile", frame=_frame(harness, str(wide)))

    warnings = records(out.json["warnings"])
    assert [w["code"] for w in warnings] == ["nonlinearity_too_few_rows"]
    assert warnings[0]["param"] == "thresholds.nonlinearity.min_rows"
    assert len(warnings[0]["columns"]) == 20
    assert {"field": "warnings[0].columns", "shown": 20, "total": 30, "operation": None} in records(
        out.json["omitted"]
    )


def test_mutual_information_with_the_target_falls_back_with_a_code(harness: Harness) -> None:
    out = harness.call("profile", frame=_frame(harness), targets=["income"])

    warnings = records(out.json["warnings"])
    assert [w["code"] for w in warnings] == ["mutual_information_too_few_rows"]
    assert warnings[0]["param"] == "thresholds.correlation.mi_min_rows"


# --- Declarations -----------------------------------------------------------------


def test_declarations_are_echoed_and_shape_the_profile(harness: Harness) -> None:
    out = harness.call(
        "profile",
        frame=_frame(harness),
        semantic_types=[{"column": "age", "semantic_type": "numeric"}],
        numeric_kinds=[{"column": "age", "numeric_kind": "bounded_discrete"}],
        sentinels=[{"column": "city", "values": ["north"]}],
        targets=["income"],
        exclude_columns=["city"],
    )

    declared = out.json["declarations"]
    assert records(declared["semantic_types"]) == [{"column": "age", "semantic_type": "numeric"}]
    assert records(declared["numeric_kinds"]) == [
        {"column": "age", "numeric_kind": "bounded_discrete"}
    ]
    assert records(declared["sentinels"]) == [{"column": "city", "values": ["north"]}]
    assert (declared["targets"], declared["exclude_columns"]) == (["income"], ["city"])
    age = out.record.get("age")
    assert (age.semantic_type, age.numeric_kind) == ("numeric", "bounded_discrete")
    assert out.record.get("city") is None  # excluded columns are not profiled


def test_skipped_columns_stay_in_the_digest_without_statistics(harness: Harness) -> None:
    out = harness.call("profile", frame=_frame(harness), skip_columns=["city"])

    city = out.record.get("city")
    assert (city.role, city.semantic_type, city.cardinality, city.flags) == (
        "skipped", None, None, []
    )
    assert out.json["declarations"]["skip_columns"] == ["city"]


def test_numeric_sentinels_count_as_missing(harness: Harness) -> None:
    out = harness.call("profile", frame=_frame(harness), sentinels=[{"column": "age", "values": [34]}])

    assert out.record.get("age").null_ratio == 1 / 6
    assert records(out.json["declarations"]["sentinels"]) == [{"column": "age", "values": [34.0]}]


def test_the_profile_inherits_the_frames_declarations(harness: Harness) -> None:
    frame = _frame_with_declarations(
        harness,
        InheritedDeclarations(
            targets=("income",),
            type_overrides=(
                {"column": "age", "semantic_type": "numeric", "numeric_kind": None,
                 "format": None, "epoch_unit": None},
            ),
            exclusions=("city",),
        ),
    )

    out = harness.call("profile", frame=frame, exclude_columns=["age"])

    declared = out.json["declarations"]
    assert declared["targets"] == ["income"]
    assert declared["exclude_columns"] == ["city", "age"]
    assert records(declared["semantic_types"]) == [{"column": "age", "semantic_type": "numeric"}]
    assert [c["column"] for c in records(out.json["columns"])] == ["income"]


def test_contradicting_an_inherited_declaration_is_a_conflict(harness: Harness) -> None:
    frame = _frame_with_declarations(harness, InheritedDeclarations(targets=("income",)))

    out = harness.call(
        "profile", frame=frame, exclude_columns=["income"],
    )

    assert out.code == "declaration_conflict"
    assert isinstance(out.error, Conflict)
    assert out.json["param"] == "exclude_columns[0]" and out.json["columns"] == ["income"]
    assert _profile_handles(harness) == []


# --- Errors -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "value", "param"),
    [
        ("semantic_types", [{"column": "incme", "semantic_type": "numeric"}], "semantic_types[0].column"),
        ("numeric_kinds", [{"column": "incme", "numeric_kind": "continuous"}], "numeric_kinds[0].column"),
        ("sentinels", [{"column": "incme", "values": [-1]}], "sentinels[0].column"),
        ("datetime_formats", [{"column": "incme", "format": "%Y"}], "datetime_formats[0].column"),
        ("targets", ["incme"], "targets[0]"),
        ("exclude_columns", ["incme"], "exclude_columns[0]"),
        ("skip_columns", ["incme"], "skip_columns[0]"),
    ],
)
def test_a_typo_in_any_declaration_is_an_unknown_column_with_candidates(
    harness: Harness, argument: str, value: Any, param: str
) -> None:
    out = harness.call("profile", frame=_frame(harness), **{argument: value})

    assert out.code == "unknown_column"
    assert isinstance(out.error, UnknownName)
    assert (out.json["param"], out.json["columns"]) == (param, ["incme"])
    assert out.json["candidates"][0] == "income"
    assert _profile_handles(harness) == []


def test_every_bad_argument_comes_back_in_one_error(harness: Harness) -> None:
    out = harness.call(
        "profile",
        frame=_frame(harness),
        semantic_types=[
            {"column": "agee", "semantic_type": "numeric"},
            {"column": "city", "semantic_type": "text"},
            {"column": "city", "semantic_type": "text"},
        ],
        targets=["incom"],
        thresholds={"missingness": {"severity_minor": 0.3}},
    )

    assert out.code == "unknown_column"
    assert out.json["params"] == [
        "semantic_types[0].column",
        "semantic_types[2].column",
        "targets[0]",
        "thresholds.missingness.severity_minor",
    ]
    assert [i["param"] for i in out.json["issues"]] == out.json["params"]
    assert out.json["candidates"] == ["age", "income"]
    assert _profile_handles(harness) == []


def test_a_column_listed_twice_is_a_duplicate_key(harness: Harness) -> None:
    out = harness.call(
        "profile",
        frame=_frame(harness),
        sentinels=[{"column": "age", "values": [0]}, {"column": "age", "values": [1]}],
    )

    assert out.code == "duplicate_key"
    assert isinstance(out.error, InvalidInput)
    assert (out.json["param"], out.json["columns"]) == ("sentinels[1].column", ["age"])


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "semantic_types": [{"column": "city", "semantic_type": "categorical"}],
            "numeric_kinds": [{"column": "city", "numeric_kind": "continuous"}],
        },
        {"targets": ["income"], "exclude_columns": ["income"]},
        {"targets": ["income"], "skip_columns": ["income"]},
    ],
)
def test_contradicting_declarations_are_a_conflict(
    harness: Harness, arguments: dict[str, Any]
) -> None:
    out = harness.call("profile", frame=_frame(harness), **arguments)

    assert out.code == "declaration_conflict"
    assert _profile_handles(harness) == []


def test_dials_out_of_order_are_an_invalid_dial_value(harness: Harness) -> None:
    out = harness.call(
        "profile", frame=_frame(harness), thresholds={"numeric": {"skew_moderate": 5.0}}
    )

    assert out.code == "invalid_dial_value"
    assert out.json["param"] == "thresholds.numeric.skew_moderate"


def test_a_dial_out_of_range_names_its_path(harness: Harness) -> None:
    out = harness.call(
        "profile", frame=_frame(harness), thresholds={"numeric": {"skew_high": -1}}
    )

    assert out.code == "invalid_argument"
    assert out.json["param"] == "thresholds.numeric.skew_high"


def test_a_datetime_format_needs_exactly_one_reading(harness: Harness) -> None:
    out = harness.call(
        "profile", frame=_frame(harness), datetime_formats=[{"column": "age"}]
    )

    assert out.code == "invalid_argument"
    assert out.json["param"] == "datetime_formats[0]"


def test_a_frame_argument_of_another_kind_is_refused(harness: Harness) -> None:
    profile = harness.code_mode("profile", frame=_frame(harness)).json["profile"]

    out = harness.call("profile", frame=profile)

    assert out.code == "wrong_handle_kind"
