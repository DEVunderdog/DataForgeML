"""Contract tests every Operation joins automatically.

Each test is parametrised over the Operation registry, so a new Operation file
is picked up with no change here. It must bring three things: a committed
schema snapshot, an ``Example`` that runs as shell commands (checked in
``test_help.py``), and a worst-case entry in :data:`WORST_CASE` below. A missing one fails its test by name.

Refresh snapshots after an intended schema change with
``DATAFORGE_ML_UPDATE_SNAPSHOTS=1 uv run pytest -n 0 tests/contract``.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import dataforge_ml
from dataforge_ml._boundary._docstring import OPERATION_SECTIONS
from dataforge_ml._boundary._error_codes import REGISTRY
from dataforge_ml._boundary._operation import (
    OperationSpec,
    canonical_json,
    operations,
)
from dataforge_ml._boundary._schema import strict_subset_problems
from dataforge_ml.records import MAX_ROW_FIELDS

from .conftest import (
    AMES,
    Harness,
    assert_within_budget,
    records,
    write_flagged_wide_csv,
    write_wide_csv,
)

SNAPSHOTS = Path(__file__).parent / "snapshots" / "schemas"
SOURCE = Path(dataforge_ml.__file__).parent

OPERATIONS = operations()
NAMES = sorted(OPERATIONS)


def _wide_load(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 1,000 columns of 64-character names: the cap and omitted at their largest.
    wide = write_wide_csv(tmp_path / "wide.csv", 1_000)
    harness.declare(wide)
    return harness.call("load", path=str(wide)).json


def _full_page_of_handles(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # A full page whose argument values are cut: long absolute paths, more
    # Handles than one page holds, so ``omitted`` is filled too.
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    narrow = deep / "narrow.csv"
    narrow.write_text("a,b\n1,2\n")
    harness.declare(narrow)
    for _ in range(25):
        harness.code_mode("load", path=str(narrow))
    return harness.call("list_handles").json


def _export(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    frame = harness.code_mode("load", path="train.csv").json["frame"]
    return harness.call("export_frame", frame=frame).json


def _wide_profile(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 150 long-named columns, each flagged twice, past the digest's 100-row
    # cap, with a fallback warning whose column list is at its cap.
    wide = write_flagged_wide_csv(tmp_path / "flagged.csv", 150)
    harness.declare(wide)
    frame = harness.code_mode("load", path=str(wide)).json["frame"]
    return harness.call("profile", frame=frame).json


def _wide_profile_handle(harness: Harness, tmp_path: Path) -> str:
    wide = write_flagged_wide_csv(tmp_path / "flagged.csv", 150)
    harness.declare(wide)
    frame = harness.code_mode("load", path=str(wide)).json["frame"]
    return harness.code_mode("profile", frame=frame).json["profile"]


def _wide_describe_columns(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # A full page of long-named, flagged columns, each co-missing with 149
    # others, so every co-missing list is at its cap.
    return harness.call("describe_columns", profile=_wide_profile_handle(harness, tmp_path)).json


def _wide_top_associations(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 11,175 pairs of long-named columns: a full page of 100.
    return harness.call("top_associations", profile=_wide_profile_handle(harness, tmp_path)).json


def _wide_missingness(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 11,175 co-missing pairs of long-named columns: a full page of 100.
    return harness.call("missingness", profile=_wide_profile_handle(harness, tmp_path)).json


def _ames_profile_handle(harness: Harness) -> str:
    harness.declare(AMES)
    frame = harness.code_mode("load", path=str(AMES)).json["frame"]
    return harness.code_mode(
        "profile", frame=frame, targets=["SalePrice", "Neighborhood"]
    ).json["profile"]


def _ames_describe_target(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # A 28-class target: its class list is at its cap.
    return harness.call(
        "describe_target", profile=_ames_profile_handle(harness), name="Neighborhood"
    ).json


def _long_labels_value_summary(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # A categorical column of 64-character labels, every one of them rare:
    # both lists at their cap of 20.
    csv = tmp_path / "labels.csv"
    labels = [f"label_{i:03d}_".ljust(64, "y") for i in range(60)]
    csv.write_text("tag\n" + "\n".join(labels[i % 60] for i in range(600)) + "\n")
    harness.declare(csv)
    frame = harness.code_mode("load", path=str(csv)).json["frame"]
    profile = harness.code_mode("profile", frame=frame).json["profile"]
    return harness.call("value_summary", profile=profile, column="tag").json


def _wide_route(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 150 long-named columns, past the 100-row cap, with 25 forced past the
    # Feasibility Floor so the warning's column list is at its cap.
    forced = [
        {"column": f"c{i:04d}_".ljust(64, "x"), "strategy": "knn"} for i in range(25)
    ]
    return harness.call(
        "route",
        profile=_wide_profile_handle(harness, tmp_path),
        strategies=forced,
        thresholds={"numeric": {"knn_max_rows": 1}},
    ).json


def _wide_explain_routing(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # A full page of long-named columns, each failing both model checks on a
    # dialled term, so every column carries its longest evidence.
    routing = harness.code_mode(
        "route",
        profile=_wide_profile_handle(harness, tmp_path),
        thresholds={"numeric": {"knn_max_rows": 1, "mice_min_rows_per_predictor": 1e9}},
    ).json["routing"]
    return harness.call("explain_routing", routing=routing).json


def _wide_resolve_recipe(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 150 long-named, mutually correlated columns, each its own
    # ``cluster_conditional`` unit with a hyperparameter record and estimates
    # resolving 149 features: full pages of units, estimates and features.
    routing = harness.code_mode(
        "route",
        profile=_wide_profile_handle(harness, tmp_path),
        default_strategy="cluster_conditional",
    ).json["routing"]
    return harness.call("resolve_recipe", routing=routing).json


def _wide_fit_unit(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # One MICE block over 150 long-named columns, capped at one round so it
    # warns: full pages of the block's columns and of the warning's columns.
    routing = harness.code_mode(
        "route", profile=_wide_profile_handle(harness, tmp_path), default_strategy="mice"
    ).json["routing"]
    recipe = harness.code_mode(
        "resolve_recipe",
        routing=routing,
        hyperparameters=[{"unit_id": "mice", "hyperparameters": {"kind": "mice", "max_iter": 1}}],
    ).json["recipe"]
    return harness.call("fit_unit", recipe=recipe, unit_id="mice").json


def _ames_repeated_kfold(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # The most folds a split may make, 100, each with two frame Handles.
    return harness.call(
        "split",
        profile=_ames_profile_handle(harness),
        scheme={"kind": "repeated_kfold", "k": 10, "repeats": 10, "target": "Neighborhood"},
    ).json


#: Per Operation, a call producing its largest result on the supported fixtures.
WORST_CASE: dict[str, Callable[[Harness, Path], dict[str, Any]]] = {
    "describe_columns": _wide_describe_columns,
    "describe_target": _ames_describe_target,
    "explain_routing": _wide_explain_routing,
    "export_frame": _export,
    "fit_unit": _wide_fit_unit,
    "list_handles": _full_page_of_handles,
    "load": _wide_load,
    "missingness": _wide_missingness,
    "profile": _wide_profile,
    "resolve_recipe": _wide_resolve_recipe,
    "route": _wide_route,
    "split": _ames_repeated_kfold,
    "top_associations": _wide_top_associations,
    "value_summary": _long_labels_value_summary,
}

#: Operations an ``omitted`` entry may already name before they exist, each
#: with the ticket that adds it. Remove an entry when its Operation lands.
PENDING_OPERATIONS: dict[str, str] = {"query_frame": "#603"}


def _schemas(spec: OperationSpec) -> dict[str, Any]:
    return {"input": spec.input_schema(), "output": spec.output_schema()}


# --- Schemas -------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_schema_snapshot(name: str) -> None:
    path = SNAPSHOTS / f"{name}.json"
    current = json.dumps(_schemas(OPERATIONS[name]), indent=2, ensure_ascii=False) + "\n"
    if os.environ.get("DATAFORGE_ML_UPDATE_SNAPSHOTS"):
        path.write_text(current, "utf-8")
    assert path.exists(), f"no committed schema snapshot for {name}; see module docstring"
    assert path.read_text("utf-8") == current, f"schema of {name} changed; review, then refresh"


def test_every_snapshot_belongs_to_an_operation() -> None:
    assert sorted(p.stem for p in SNAPSHOTS.glob("*.json")) == NAMES


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("side", ["input", "output"])
def test_schema_is_in_the_strict_subset(name: str, side: str) -> None:
    schema = _schemas(OPERATIONS[name])[side]
    assert strict_subset_problems(schema) == []


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_and_field_has_a_description(name: str) -> None:
    missing: list[str] = []

    def walk(node: dict[str, Any], path: str) -> None:
        for key, prop in node.get("properties", {}).items():
            if not prop.get("description"):
                missing.append(f"{path}.{key}")
            walk(prop, f"{path}.{key}")
        if "items" in node:
            walk(node["items"], f"{path}[]")
        for branch in node.get("anyOf", []):
            walk(branch, path)

    for side, schema in _schemas(OPERATIONS[name]).items():
        walk(schema, side)
    assert missing == []


def _is_table(node: dict[str, Any]) -> bool:
    return node.get("type") == "object" and list(node.get("properties", {})) == [
        "fields",
        "rows",
    ]


def _table_layout_problems(schema: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    def walk(node: dict[str, Any], path: str) -> None:
        if _is_table(node):
            fields = node["properties"]["fields"]["items"]["enum"]
            if len(fields) > MAX_ROW_FIELDS:
                problems.append(f"{path}: a row of {len(fields)} fields; nest the extras")
        elif node.get("type") == "array" and node["items"].get("type") == "object":
            problems.append(f"{path}: a list of records outside Table Layout")
        for key, prop in node.get("properties", {}).items():
            walk(prop, f"{path}.{key}")
        if "items" in node:
            walk(node["items"], f"{path}[]")
        for i, branch in enumerate(node.get("anyOf", [])):
            walk(branch, f"{path}|{i}")

    walk(schema, "$")
    return problems


@pytest.mark.parametrize("name", NAMES)
def test_every_list_of_records_in_a_result_uses_table_layout(name: str) -> None:
    assert _table_layout_problems(OPERATIONS[name].output_schema()) == []


def test_table_layout_check_catches_what_it_should() -> None:
    record = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert _table_layout_problems({"type": "array", "items": record})
    wide = {
        "type": "object",
        "properties": {
            "fields": {"type": "array", "items": {"enum": [f"f{i}" for i in range(13)]}},
            "rows": {"type": "array", "items": {"type": "array", "items": {}}},
        },
    }
    assert _table_layout_problems(wide)


def test_strict_subset_checker_catches_what_it_should() -> None:
    open_map = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert strict_subset_problems(open_map)
    assert strict_subset_problems({"type": "object", "properties": {"a": {}}})
    assert strict_subset_problems(
        {
            "type": "object",
            "properties": {"a": {"type": "integer", "minimum": 0}},
            "required": [],
            "additionalProperties": False,
        }
    )


# --- The seven-part docstring -----------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_docstring_has_the_seven_parts_in_order(name: str) -> None:
    doc = OPERATIONS[name].docstring
    assert doc.summary.endswith("."), "the summary is one sentence"
    assert list(doc.sections) == list(OPERATION_SECTIONS)
    assert all(doc.sections[s].strip() for s in OPERATION_SECTIONS)


@pytest.mark.parametrize("name", NAMES)
def test_documented_parameters_match_the_signature(name: str) -> None:
    spec = OPERATIONS[name]
    documented = spec.docstring.entries("Parameters")
    assert list(documented) == list(inspect.signature(spec.function).parameters)
    assert all(documented.values())


@pytest.mark.parametrize("name", NAMES)
def test_returns_names_the_wire_record(name: str) -> None:
    spec = OPERATIONS[name]
    assert spec.result_model.__name__ in spec.docstring.sections["Returns"]


@pytest.mark.parametrize("name", NAMES)
def test_every_listed_error_code_is_registered(name: str) -> None:
    codes = OPERATIONS[name].errors()
    assert codes, "an Operation lists at least one Error Code"
    unknown = [c for c in codes if c not in REGISTRY or REGISTRY[c].category == "warning"]
    assert unknown == []


# --- Results --------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_worst_case_result_is_under_the_result_budget(
    name: str, harness: Harness, tmp_path: Path
) -> None:
    assert name in WORST_CASE, f"add a worst-case call for {name} to WORST_CASE"
    assert_within_budget(name, WORST_CASE[name](harness, tmp_path), is_error=False)


@pytest.mark.parametrize("name", NAMES)
def test_every_omitted_operation_is_a_real_command(
    name: str, harness: Harness, tmp_path: Path
) -> None:
    named = [o["operation"] for o in records(WORST_CASE[name](harness, tmp_path)["omitted"])]
    real = set(OPERATIONS) | set(PENDING_OPERATIONS)
    assert [n for n in named if n is not None and n not in real] == []


def test_pending_operations_are_not_yet_registered() -> None:
    # Once a pending Operation lands, its allowance is stale: delete it.
    assert set(PENDING_OPERATIONS) & set(OPERATIONS) == set()


@pytest.mark.parametrize("name", NAMES)
def test_worst_case_error_is_under_the_error_budget(name: str, harness: Harness) -> None:
    # The widest error any call can draw: more bad arguments than the lists
    # hold, each with a long name, so every list is at its cap.
    arguments = {f"argument_{i:02d}_".ljust(64, "x"): "x" * 200 for i in range(25)}
    out = harness.code_mode(name, **arguments)
    assert out.code == "invalid_argument" and out.json["omitted"]
    assert_within_budget(name, out.json, is_error=True)


@pytest.mark.parametrize("name", NAMES)
def test_result_json_is_byte_stable(name: str, harness: Harness, tmp_path: Path) -> None:
    spec = OPERATIONS[name]
    result = WORST_CASE[name](harness, tmp_path)
    record = spec.result_model.model_validate(result)

    first = canonical_json(record.to_dict())
    assert canonical_json(record.to_dict()) == first
    assert canonical_json(spec.result_model.model_validate_json(first).to_dict()) == first

    def keys_in_schema_order(value: Any, schema: dict[str, Any]) -> None:
        branches = schema.get("anyOf", [schema])
        table = next((b for b in branches if _is_table(b)), None)
        if table is not None and isinstance(value, dict):
            assert list(value) == ["fields", "rows"]
            assert value["fields"] == table["properties"]["fields"]["items"]["enum"]
            positions = table["properties"]["rows"]["items"]["items"]["anyOf"]
            for row in value["rows"]:
                assert len(row) == len(positions)
                for cell, position in zip(row, positions):
                    keys_in_schema_order(cell, position)
        elif isinstance(value, dict):
            # In a tagged union, the variant whose fields the value carries.
            objects = [b for b in branches if b.get("type") == "object"]
            node = next(b for b in objects if set(b["properties"]) == set(value))
            assert list(value) == list(node["properties"])
            for key, item in value.items():
                keys_in_schema_order(item, node["properties"][key])
        elif isinstance(value, list):
            node = next(b for b in branches if b.get("type") == "array")
            for item in value:
                keys_in_schema_order(item, node["items"])

    keys_in_schema_order(json.loads(first), spec.output_schema())


def test_every_operation_is_exported_from_the_package() -> None:
    for name in NAMES:
        assert name in dataforge_ml.__all__
        assert getattr(dataforge_ml, name).__operation__ is OPERATIONS[name]
