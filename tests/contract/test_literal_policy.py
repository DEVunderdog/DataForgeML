"""The Literal Policy and the canary containment test (ADR-0107, ADR-0112).

A fixture is seeded with canary values: every label carries ``cnry``, every
date is in 2093, and every number is a value no aggregate lands on. Each
Operation is called in both deliveries (Code Mode, and a ``dataforge``
process with ``--progress``) under both policies, with calls that succeed and
calls that fail. Then:

- under ``allow``, no canary appears outside a Literal Field, found by the
  ``x-literal`` keyword in the output schema, as a Harness would find it;
- under ``deny``, no canary appears anywhere: stdout, warnings, errors or
  stderr.

Every Operation must have an entry in :data:`CALLS`; a new Operation fails
:func:`test_every_operation_has_canary_calls` until it is added.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dataforge_ml._boundary._operation import canonical_json, operations
from dataforge_ml._boundary._settings import LITERAL_POLICY_ENV, SETTING_NAMES
from dataforge_ml.records import LITERAL_KEYWORD, literal_field_paths

from .conftest import Harness, normalise_handles, run_dataforge, to_argv

OPERATIONS = operations()
NAMES = sorted(OPERATIONS)
POLICIES = ("allow", "deny")

#: The sentinel the Agent declares. It is the Agent's own value, so it may
#: be echoed under ``deny``; it is deliberately not a canary.
SENTINEL = -999.0
_ROWS = 40
_STRING_CANARIES = ("cnry", "2093-")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")


@dataclass(frozen=True)
class Canaries:
    """The canary fixture's path and every numeric cell value in it."""

    path: Path
    numbers: frozenset[float]

    def found_in(self, text: str) -> list[str]:
        """Every canary in ``text``: a canary string, or a cell's number as a whole token."""
        found = [c for c in _STRING_CANARIES if c in text]
        found += [n for n in _NUMBER.findall(text) if float(n) in self.numbers]
        return found


def write_canaries(directory: Path) -> Canaries:
    """Write the canary CSV: column names are plain, every cell is a canary."""
    rng = random.Random(595)
    amounts = [round(rng.uniform(100, 900), 4) for _ in range(_ROWS)]
    pool = rng.sample(range(860_000, 870_000), 12)
    counts = [pool[i % 12] for i in range(_ROWS)]  # repeats, so not an identifier
    cities = [f"cnry_city_{k}" for k in "aaaaaaaaaabbbbbbbbbbccccccccccddddddeeef"]
    labels = [f"cnry_cls_{'ab'[i % 3 == 0]}" for i in range(_ROWS)]
    lines = ["amount,visits,city,label,when,ident"]
    for i in range(_ROWS):
        amount = "" if i % 7 == 3 else str(SENTINEL) if i % 11 == 5 else repr(amounts[i])
        city = "" if i % 9 == 4 else cities[i]
        lines.append(
            f"{amount},{counts[i]},{city},{labels[i]},2093-{1 + i % 12:02d}-{1 + i % 28:02d},"
            f"cnry_id_{i:03d}"
        )
    path = directory / "canary.csv"
    path.write_text("\n".join(lines) + "\n")
    return Canaries(path=path, numbers=frozenset(map(float, [*amounts, *counts])))


# --- Calls ------------------------------------------------------------------------


@dataclass(frozen=True)
class Handles:
    """The Handles every call can read, made once per test in Code Mode."""

    path: str
    frame: str
    profile: str
    routing: str


Call = tuple[str, dict[str, Any]]

#: Per Operation, the calls to make: some succeed, some fail. Each call is
#: ``(operation, arguments)``. A new Operation adds its entry here.
CALLS: dict[str, Callable[[Handles], list[Call]]] = {
    "load": lambda h: [
        ("load", {"path": h.path}),
        ("load", {"path": "undeclared.csv"}),
    ],
    "profile": lambda h: [
        (
            "profile",
            {
                "frame": h.frame,
                "targets": ["label"],
                "sentinels": [{"column": "amount", "values": [SENTINEL]}],
            },
        ),
        ("profile", {"frame": h.frame, "targets": ["nope"]}),
    ],
    "describe_columns": lambda h: [
        ("describe_columns", {"profile": h.profile}),
        ("describe_columns", {"profile": h.profile, "names": ["citty"]}),
    ],
    "describe_target": lambda h: [
        ("describe_target", {"profile": h.profile, "name": "label"}),
        ("describe_target", {"profile": h.profile, "name": "city"}),
    ],
    "missingness": lambda h: [
        ("missingness", {"profile": h.profile}),
        ("missingness", {"profile": h.profile, "column": "amout"}),
    ],
    "top_associations": lambda h: [
        ("top_associations", {"profile": h.profile}),
        ("top_associations", {"profile": h.profile, "column": "lable"}),
    ],
    "value_summary": lambda h: [
        *(
            ("value_summary", {"profile": h.profile, "column": c})
            for c in ("amount", "visits", "city", "label", "when", "ident")
        ),
        ("value_summary", {"profile": h.profile, "column": "citi"}),
    ],
    "route": lambda h: [
        ("route", {"profile": h.profile}),
        ("route", {"profile": h.profile, "strategies": [{"column": "amount", "strategy": "constant"}]}),
    ],
    "explain_routing": lambda h: [
        ("explain_routing", {"routing": h.routing}),
        ("explain_routing", {"routing": h.routing, "columns": ["citty"]}),
    ],
    "resolve_recipe": lambda h: [
        ("resolve_recipe", {"routing": h.routing}),
        ("resolve_recipe", {"routing": h.profile}),
    ],
    "split": lambda h: [
        ("split", {"profile": h.profile, "scheme": {"kind": "profile_stratified_kfold", "k": 3}}),
        ("split", {"profile": h.profile, "scheme": {"kind": "time", "time_column": "when"}}),
        ("split", {"profile": h.profile, "scheme": {"kind": "kfold", "k": 40}}),
        ("split", {"profile": h.profile, "scheme": {"kind": "group", "group_column": "citty"}}),
    ],
    "export_frame": lambda h: [
        ("export_frame", {"frame": h.frame}),
        ("export_frame", {"frame": h.profile}),
    ],
    "list_handles": lambda h: [
        ("list_handles", {}),
        ("list_handles", {"kind": "routing", "limit": 0}),
    ],
}


def _handles(harness: Harness, canaries: Canaries) -> Handles:
    harness.declare(canaries.path)
    path = str(canaries.path)
    frame = harness.code_mode("load", path=path).json["frame"]
    profile = harness.code_mode(
        "profile",
        frame=frame,
        targets=["label"],
        sentinels=[{"column": "amount", "values": [SENTINEL]}],
    ).json["profile"]
    routing = harness.code_mode("route", profile=profile).json["routing"]
    return Handles(path=path, frame=frame, profile=profile, routing=routing)


# --- Reading Literal Fields by their schema mark ---------------------------------------


def _is_table(node: dict[str, Any]) -> bool:
    return node.get("type") == "object" and list(node.get("properties", {})) == ["fields", "rows"]


def split_literals(value: Any, node: dict[str, Any]) -> tuple[Any, list[Any]]:
    """Return ``value`` with every ``x-literal`` node nulled, and the values found there.

    The walk follows the output schema, Table Layout included, so it finds
    Literal Fields exactly as a Harness reading the schema would.
    """
    found: list[Any] = []

    def walk(value: Any, node: dict[str, Any]) -> Any:
        if node.get(LITERAL_KEYWORD):
            found.append(value)
            return None
        branches = node.get("anyOf", [node])
        if isinstance(value, dict):
            table = next((b for b in branches if _is_table(b)), None)
            if table is not None and list(value) == ["fields", "rows"]:
                positions = table["properties"]["rows"]["items"]["items"]["anyOf"]
                return {
                    "fields": value["fields"],
                    "rows": [[walk(c, p) for c, p in zip(row, positions)] for row in value["rows"]],
                }
            objects = [b for b in branches if b.get("type") == "object" and not _is_table(b)]
            record = next(b for b in objects if set(b["properties"]) == set(value))
            return {k: walk(v, record["properties"][k]) for k, v in value.items()}
        if isinstance(value, list):
            array = next(b for b in branches if b.get("type") == "array")
            return [walk(v, array["items"]) for v in value]
        return value

    return walk(value, node), found


def _from_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(zip(table["fields"], row)) for row in table["rows"]]


# --- The containment test --------------------------------------------------------------


@dataclass(frozen=True)
class Delivered:
    """One call as both deliveries returned it."""

    operation: str
    json: dict[str, Any]
    is_error: bool
    stdout: str
    stderr: str


def _call_both(harness: Harness, operation: str, arguments: dict[str, Any]) -> Delivered:
    code = harness.code_mode(operation, **arguments)
    cli = run_dataforge(["--progress", *to_argv(operation, arguments)])
    assert code.is_error == cli.is_error, (operation, code.json, cli.json)
    assert normalise_handles(canonical_json(code.json)) == normalise_handles(
        canonical_json(cli.json)
    )
    return Delivered(operation, code.json, code.is_error, cli.stdout, cli.stderr)


def test_every_operation_has_canary_calls() -> None:
    assert sorted(CALLS) == NAMES, "add the Operation's calls to CALLS"


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", NAMES)
def test_no_canary_escapes(
    name: str, policy: str, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LITERAL_POLICY_ENV, policy)
    canaries = write_canaries(tmp_path)
    calls = CALLS[name](_handles(harness, canaries))
    outcomes = [_call_both(harness, op, args) for op, args in calls]
    assert any(o.is_error for o in outcomes) and not all(o.is_error for o in outcomes)

    for out in outcomes:
        where = f"{out.operation} {'error' if out.is_error else 'result'} under {policy}"
        assert canaries.found_in(out.stderr) == [], f"{where}: stderr"
        if policy == "deny" or out.is_error:
            assert canaries.found_in(out.stdout) == [], where
            continue
        rest, _ = split_literals(out.json, OPERATIONS[out.operation].output_schema())
        assert canaries.found_in(canonical_json(rest)) == [], f"{where}: outside a Literal Field"


def test_allow_returns_the_canaries_in_literal_fields(
    harness: Harness, tmp_path: Path
) -> None:
    # The containment test proves nothing unless the canaries reach the
    # Literal Fields at all: under allow they must.
    canaries = write_canaries(tmp_path)
    handles = _handles(harness, canaries)
    for column in ("amount", "visits", "city", "when"):
        out = harness.call("value_summary", profile=handles.profile, column=column)
        _, literals = split_literals(out.json, OPERATIONS["value_summary"].output_schema())
        assert canaries.found_in(canonical_json(literals)), column
        assert out.json["warnings"]["rows"] == []


def test_deny_withholds_literal_fields_and_keeps_the_rest(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canaries = write_canaries(tmp_path)
    handles = _handles(harness, canaries)
    monkeypatch.setenv(LITERAL_POLICY_ENV, "deny")

    out = harness.call("value_summary", profile=handles.profile, column="city")
    _, literals = split_literals(out.json, OPERATIONS["value_summary"].output_schema())
    assert literals and all(v is None for v in literals)
    top = _from_table(out.json["top_values"])
    assert top and all(r["value"] is None and r["count"] > 0 and r["share"] > 0 for r in top)
    assert out.json["cardinality"] == 6

    withheld = [o for o in _from_table(out.json["omitted"]) if o["reason"] == "literal_policy"]
    assert {o["field"] for o in withheld} == {"top_values[].value", "rare_labels", "mode"}
    assert all(o["shown"] == 0 and o["operation"] is None for o in withheld)
    assert next(o for o in withheld if o["field"] == "top_values[].value")["total"] == len(top)
    paths = literal_field_paths(OPERATIONS["value_summary"].result_model)
    assert {o["field"] for o in withheld} <= set(paths)
    warnings = _from_table(out.json["warnings"])
    assert [w["code"] for w in warnings] == ["literal_policy"]

    numeric = harness.call("value_summary", profile=handles.profile, column="amount").json
    assert numeric["bounds"] == {"min": None, "max": None}
    assert all(r["value"] is None and r["percentile"] for r in _from_table(numeric["percentiles"]))
    assert [w["code"] for w in _from_table(numeric["warnings"])] == ["literal_policy"]

    target = harness.call("describe_target", profile=handles.profile, name="label").json
    classes = _from_table(target["classes"]["counts"])
    assert all(c["label"] is None and c["count"] > 0 for c in classes)


def test_deny_keeps_values_the_agent_supplied(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canaries = write_canaries(tmp_path)
    harness.declare(canaries.path)
    monkeypatch.setenv(LITERAL_POLICY_ENV, "deny")
    frame = harness.code_mode("load", path=str(canaries.path)).json["frame"]
    out = harness.call(
        "profile",
        frame=frame,
        targets=["label"],
        sentinels=[{"column": "amount", "values": [SENTINEL]}],
        semantic_types=[{"column": "ident", "semantic_type": "identifier"}],
    ).json
    declarations = out["declarations"]
    assert _from_table(declarations["sentinels"]) == [{"column": "amount", "values": [SENTINEL]}]
    assert _from_table(declarations["semantic_types"]) == [
        {"column": "ident", "semantic_type": "identifier"}
    ]
    assert declarations["targets"] == ["label"]
    assert _from_table(out["warnings"]) == [] or all(
        w["code"] != "literal_policy" for w in _from_table(out["warnings"])
    )


# --- The policy is a Workspace Setting, never an argument -----------------------------


@pytest.mark.parametrize("name", NAMES)
def test_no_operation_argument_is_a_workspace_setting(name: str) -> None:
    assert set(OPERATIONS[name].args_model.model_fields) & set(SETTING_NAMES) == set()


def test_the_policy_cannot_be_passed_as_an_argument(harness: Harness) -> None:
    # Code Mode only: on the CLI, `--literal-policy` is the setting's own
    # flag, the first source in the Workspace Settings order.
    out = harness.code_mode("list_handles", literal_policy="allow")
    assert out.code == "invalid_argument"


def test_the_policy_resolves_flag_then_env_then_config(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canaries = write_canaries(tmp_path)
    handles = _handles(harness, canaries)
    argv = to_argv("value_summary", {"profile": handles.profile, "column": "city"})

    def withheld(*flags: str) -> bool:
        out = run_dataforge([*flags, *argv])
        return any(o["reason"] == "literal_policy" for o in _from_table(out.json["omitted"]))

    harness.workspace.mkdir(parents=True, exist_ok=True)
    (harness.workspace / "config.toml").write_text('literal_policy = "deny"\n')
    assert withheld()
    monkeypatch.setenv(LITERAL_POLICY_ENV, "allow")
    assert not withheld()
    assert withheld("--literal-policy", "deny")
    monkeypatch.delenv(LITERAL_POLICY_ENV)
    assert not withheld("--literal-policy", "allow")


# --- No literal value ever enters a schema --------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_literal_field_is_marked_and_nullable(name: str) -> None:
    spec = OPERATIONS[name]
    marked: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get(LITERAL_KEYWORD):
                marked.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec.output_schema())
    assert bool(marked) == bool(literal_field_paths(spec.result_model))
    for node in marked:
        types = [b.get("type") for b in node.get("anyOf", [node])]
        assert "null" in types, f"{name}: a Literal Field must admit null for `deny`"
    assert LITERAL_KEYWORD not in json.dumps(spec.input_schema())


@pytest.mark.parametrize("name", NAMES)
def test_no_canary_enters_a_schema(name: str, tmp_path: Path) -> None:
    # Schemas are static per library version: building them after a dataset
    # was profiled must not pull any of its values in.
    canaries = write_canaries(tmp_path)
    spec = OPERATIONS[name]
    text = json.dumps([spec.input_schema(), spec.output_schema()])
    assert canaries.found_in(text) == []
