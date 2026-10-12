"""The Error Code registry covers every raise site and every warning site.

Every ``raise`` in the library must raise a Library Error built from a
registered code (ADR-0100). The scan reads the source's syntax tree, so a new
``raise ValueError(...)`` or an unregistered code fails here by file and line.

A raise is accepted when it is:

- a bare ``raise`` or ``raise exc`` re-raising a caught exception;
- ``DataForgeError.from_code("<code>", ...)`` with a literal, registered,
  non-warning code;
- a call to a function in the same module annotated to return
  ``DataForgeError`` (a builder such as ``_incompatible``), whose own
  ``from_code`` calls are checked by the same scan.

A warning site is a ``ResultWarning(code="<code>")``; its code must be
registered as a ``warning``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import dataforge_ml
from dataforge_ml._boundary._error_codes import CATEGORIES, REGISTRY
from dataforge_ml.records import LibraryError

SOURCE = Path(dataforge_ml.__file__).parent

#: Raise sites that may raise a builtin exception, by (file, function). Each
#: guards the error machinery itself or an Operation's definition at import
#: time, so it fires only on a library bug and never on an Agent's call.
EXEMPT: dict[tuple[str, str], str] = {
    ("records.py", "__init__"): "a code raised under the wrong category class",
    ("records.py", "from_code"): "a warning code raised as an error",
    ("records.py", "__getattr__"): "Python's attribute protocol needs AttributeError",
    ("_boundary/_error_codes.py", "entry_for"): "an unregistered code",
    ("_boundary/_operation.py", "_build_spec"): "an Operation defined outside the Strict Subset",
    ("_boundary/_operation.py", "decorate"): "an Operation registered twice",
    ("_boundary/_schema.py", "_inline"): "a recursive schema, found at definition time",
    ("_cli.py", "main"): "the console script's exit status, not an error",
}

#: ``warnings.warn`` sites left from before Result Warnings, per file. Each is
#: removed by the Operation ticket that moves its caveat into ``warnings``
#: (#593 profile, #600 fit_unit, #601 impute, #606 old surface). The count may
#: only go down; a new ``warnings.warn`` anywhere fails.
LEGACY_WARN_SITES: dict[str, int] = {
    "profiling/_correlation_profiler.py": 9,
    "profiling/_nonlinearity_profiler.py": 3,
    "imputation/_unit_fit.py": 1,
    "imputation/_fitted_imputer.py": 1,
    "_serialization.py": 1,
}


def _sources() -> list[tuple[str, ast.Module]]:
    return [
        (path.relative_to(SOURCE).as_posix(), ast.parse(path.read_text("utf-8")))
        for path in sorted(SOURCE.rglob("*.py"))
    ]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


def _is_from_code(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "from_code"


def _builders(tree: ast.Module) -> set[str]:
    # Functions annotated ``-> DataForgeError``: their result is a Library Error.
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and isinstance(node.returns, (ast.Name, ast.Constant))
        and getattr(node.returns, "id", getattr(node.returns, "value", None)) == "DataForgeError"
    }


def _raise_problem(node: ast.Raise, builders: set[str]) -> str | None:
    exc = node.exc
    if exc is None or isinstance(exc, ast.Name):
        return None
    if isinstance(exc, ast.Call):
        if _is_from_code(exc):
            return None  # its code is checked by test_every_coded_site_is_registered
        if isinstance(exc.func, ast.Name) and exc.func.id in builders:
            return None
    return f"raises {ast.unparse(exc)[:60]!r}, not a registered Library Error"


def test_every_raise_site_raises_a_library_error() -> None:
    problems = []
    for name, tree in _sources():
        parents = _parents(tree)
        builders = _builders(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            problem = _raise_problem(node, builders)
            if problem is None:
                continue
            if (name, _enclosing_function(node, parents)) in EXEMPT:
                continue
            problems.append(f"{name}:{node.lineno} {problem}")
    assert problems == []


def test_every_coded_site_is_registered() -> None:
    problems = []
    seen = 0
    for name, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_from_code(node):
                seen += 1
                code = node.args[0] if node.args else None
                if not (isinstance(code, ast.Constant) and isinstance(code.value, str)):
                    problems.append(f"{name}:{node.lineno} code is not a string literal")
                elif code.value not in REGISTRY:
                    problems.append(f"{name}:{node.lineno} {code.value!r} is not registered")
                elif REGISTRY[code.value].category == "warning":
                    problems.append(f"{name}:{node.lineno} {code.value!r} is a warning")
            elif isinstance(node.func, ast.Name) and node.func.id == "ResultWarning":
                seen += 1
                code = next((k.value for k in node.keywords if k.arg == "code"), None)
                if not (isinstance(code, ast.Constant) and isinstance(code.value, str)):
                    problems.append(f"{name}:{node.lineno} warning code is not a string literal")
                elif REGISTRY.get(code.value) is None or REGISTRY[code.value].category != "warning":
                    problems.append(f"{name}:{node.lineno} {code.value!r} is not a registered warning")
    assert seen, "the scan found the coded sites"
    assert problems == []


def _warn_sites(tree: ast.Module) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warn"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "warnings"
    )


def test_no_new_python_warning_sites() -> None:
    found = {name: _warn_sites(tree) for name, tree in _sources()}
    grown = {
        name: count
        for name, count in found.items()
        if count > LEGACY_WARN_SITES.get(name, 0)
    }
    assert grown == {}, "emit a coded ResultWarning instead of warnings.warn"


@pytest.mark.parametrize("name", sorted(LEGACY_WARN_SITES))
def test_legacy_warning_allowance_is_not_stale(name: str) -> None:
    # Lower the allowance when a ticket removes a site, so it cannot regrow.
    tree = ast.parse((SOURCE / name).read_text("utf-8"))
    assert _warn_sites(tree) == LEGACY_WARN_SITES[name]


def test_registry_entries_are_well_formed() -> None:
    optional = set(LibraryError.model_fields) - {"code", "message", "hint"}
    for code, entry in REGISTRY.items():
        assert entry.code == code
        assert code.isidentifier() and code == code.lower()
        assert entry.category in CATEGORIES
        assert entry.hint.strip()
        assert set(entry.fields) <= optional, code


def test_the_spec_named_codes_are_registered() -> None:
    named = {
        "invalid_argument": "InvalidInput",
        "invalid_dial_value": "InvalidInput",
        "unknown_column": "UnknownName",
        "unknown_handle": "UnknownName",
        "wrong_handle_kind": "InvalidInput",
        "input_not_declared": "InvalidInput",
        "incompatible_artifact": "IncompatibleArtifact",
        "unit_not_trainable": "Infeasible",
        "c2st_string_column": "Infeasible",
        "c2st_sample_floor": "Infeasible",
        "stratify_on_continuous_target": "Infeasible",
        "missing_fitted_columns": "InvalidInput",
        "unexpected": "Unexpected",
    }
    assert {code: REGISTRY[code].category for code in named} == named


def test_the_old_exception_classes_are_gone() -> None:
    for name in (
        "UnseenColumnError",
        "FittedColumnAbsentError",
        "OverrideCoercionError",
        "UnsupportedFormatError",
        "IncompatibleArtifactError",
        "UnitNotTrainableError",
        "C2STDtypeError",
        "C2STSampleFloorError",
    ):
        assert not hasattr(dataforge_ml, name)
        assert all(name not in ast.unparse(tree) for _, tree in _sources()), name
