"""Contract tests every Operation joins automatically.

Each test is parametrised over the Operation registry, so a new Operation file
is picked up with no change here. It must bring three things: a committed
schema snapshot, a runnable ``Example``, and a worst-case entry in
:data:`WORST_CASE` below. A missing one fails its test by name.

Refresh snapshots after an intended schema change with
``DATAFORGE_ML_UPDATE_SNAPSHOTS=1 uv run pytest tests/contract``.
"""

from __future__ import annotations

import copy
import doctest
import inspect
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tiktoken
from anthropic.lib._parse._transform import transform_schema
from openai.lib._pydantic import _ensure_strict_json_schema

import dataforge_ml
from dataforge_ml._boundary._docstring import OPERATION_SECTIONS
from dataforge_ml._boundary._error_codes import CATEGORIES, REGISTRY
from dataforge_ml._boundary._operation import (
    OperationSpec,
    canonical_json,
    capture_calls,
    operations,
)
from dataforge_ml._boundary._schema import strict_subset_problems

from .conftest import Harness, write_wide_csv

SNAPSHOTS = Path(__file__).parent / "snapshots" / "schemas"
SOURCE = Path(dataforge_ml.__file__).parent

OPERATIONS = operations()
NAMES = sorted(OPERATIONS)


def _wide_load(harness: Harness, tmp_path: Path) -> dict[str, Any]:
    # 1,000 columns of 64-character names: the cap and omitted at their largest.
    wide = write_wide_csv(tmp_path / "wide.csv", 1_000)
    harness.declare(wide)
    return harness.call("load", path=str(wide)).json


#: Per Operation, a call producing its largest result on the supported fixtures.
WORST_CASE: dict[str, Callable[[Harness, Path], dict[str, Any]]] = {
    "load": _wide_load,
}


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
@pytest.mark.parametrize("side", ["input", "output"])
def test_vendor_transformers_leave_the_schema_unchanged(name: str, side: str) -> None:
    schema = _schemas(OPERATIONS[name])[side]
    assert transform_schema(copy.deepcopy(schema)) == schema
    openai = copy.deepcopy(schema)
    assert _ensure_strict_json_schema(openai, path=(), root=openai) == schema


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


@pytest.mark.parametrize("name", NAMES)
def test_example_runs_and_its_arguments_are_captured(
    name: str, harness: Harness
) -> None:
    spec = OPERATIONS[name]
    parser = doctest.DocTestParser()
    test = parser.get_doctest(
        spec.docstring.sections["Example"], {}, f"{name}.Example", spec.module, 0
    )
    assert test.examples, "the Example has runnable lines"
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    with capture_calls() as calls:
        runner.run(test)
    assert runner.failures == 0, f"the Example of {name} fails; run it to see the diff"
    examples = [arguments for called, arguments in calls if called == name]
    assert examples, f"the Example never calls {name}"
    for arguments in examples:
        spec.args_model.model_validate_json(json.dumps(arguments))


# --- Results --------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_worst_case_result_is_under_the_result_budget(
    name: str, harness: Harness, tmp_path: Path
) -> None:
    assert name in WORST_CASE, f"add a worst-case call for {name} to WORST_CASE"
    result = WORST_CASE[name](harness, tmp_path)
    tokens = len(tiktoken.get_encoding("o200k_base").encode(canonical_json(result)))
    assert tokens <= OPERATIONS[name].result_budget_tokens, f"{name}: {tokens} tokens"


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
        if isinstance(value, dict):
            node = next(b for b in branches if b.get("type") == "object")
            assert list(value) == list(node["properties"])
            for key, item in value.items():
                keys_in_schema_order(item, node["properties"][key])
        elif isinstance(value, list):
            node = next(b for b in branches if b.get("type") == "array")
            for item in value:
                keys_in_schema_order(item, node["items"])

    keys_in_schema_order(json.loads(first), spec.output_schema())


# --- Error Code registry ----------------------------------------------------


def test_every_raise_site_has_a_registry_entry() -> None:
    raised = set()
    for path in SOURCE.rglob("*.py"):
        raised |= set(re.findall(r'from_code\(\s*"([a-z_]+)"', path.read_text("utf-8")))
    assert raised, "the scan found the raise sites"
    assert sorted(raised - set(REGISTRY)) == []


def test_registry_entries_are_well_formed() -> None:
    for code, entry in REGISTRY.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", code)
        assert entry.category in CATEGORIES
        assert entry.hint.strip()


def test_every_operation_is_exported_from_the_package() -> None:
    for name in NAMES:
        assert name in dataforge_ml.__all__
        assert getattr(dataforge_ml, name).__operation__ is OPERATIONS[name]
