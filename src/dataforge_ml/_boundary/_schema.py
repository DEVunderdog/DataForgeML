"""The Strict Subset: a post-pass over Pydantic schemas, and its checker (ADR-0101).

Pydantic emits ordinary JSON Schema. :func:`to_strict_subset` rewrites it into
the subset every vendor's strict mode accepts unchanged:

- ``$ref``/``$defs`` inlined (no recursion is allowed, so this terminates);
- every object closed (``additionalProperties: false``) with every property
  required, ``null`` meaning "the library's default";
- ``oneOf`` becomes ``anyOf``, ``const`` becomes a one-value ``enum``;
- constraint keywords are removed and stated in the description instead;
- ``title`` and ``default`` are dropped.

:func:`strict_subset_problems` is the independent checker the contract tests
run on every Operation schema.
"""

from __future__ import annotations

import copy
from typing import Any

MAX_DEPTH = 10
MAX_PROPERTIES = 5_000

_CONSTRAINTS: dict[str, str] = {
    "minimum": "at least {}",
    "maximum": "at most {}",
    "exclusiveMinimum": "greater than {}",
    "exclusiveMaximum": "less than {}",
    "multipleOf": "a multiple of {}",
    "minLength": "at least {} characters",
    "maxLength": "at most {} characters",
    "pattern": "matching the pattern `{}`",
    "format": "in {} format",
    "minItems": "at least {} items",
    "maxItems": "at most {} items",
    "uniqueItems": "with unique items",
}
_DROPPED = ("title", "default", "discriminator", "examples")
_FORBIDDEN = (
    "$ref",
    "$defs",
    "prefixItems",
    "oneOf",
    "allOf",
    "const",
    "patternProperties",
    *_DROPPED,
    *_CONSTRAINTS,
)


def _inline(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_inline(v, defs, seen) for v in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in seen:
            raise ValueError(f"recursive schema through {name!r} is outside the Strict Subset")
        target = _inline(copy.deepcopy(defs[name]), defs, (*seen, name))
        rest = {k: v for k, v in node.items() if k != "$ref"}
        return {**target, **_inline(rest, defs, seen)}
    return {k: _inline(v, defs, seen) for k, v in node.items() if k != "$defs"}


def _with_constraints(node: dict[str, Any], phrases: list[str]) -> None:
    if phrases:
        text = "Constraints: " + "; ".join(phrases) + "."
        node["description"] = (
            f"{node['description']} {text}" if node.get("description") else text
        )


def _rewrite(node: dict[str, Any]) -> list[str]:
    """Rewrite ``node`` in place; return constraint phrases still to be placed."""
    for key in _DROPPED:
        node.pop(key, None)
    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")
    if "allOf" in node and len(node["allOf"]) == 1:
        node.update(node.pop("allOf")[0])
    if "const" in node:
        node["enum"] = [node.pop("const")]

    phrases = [
        template.format(node.pop(key))
        for key, template in _CONSTRAINTS.items()
        if key in node and (key != "uniqueItems" or node[key])
    ]
    node.pop("uniqueItems", None)

    if "anyOf" in node:
        # A nullable union nests one anyOf in another; splice it flat.
        flat: list[Any] = []
        for branch in node["anyOf"]:
            key = "anyOf" if "anyOf" in branch else "oneOf"
            nested = key in branch and set(branch) <= {key, "discriminator"}
            flat += branch[key] if nested else [branch]
        node["anyOf"] = flat
    for branch in node.get("anyOf", []):
        phrases += _rewrite(branch)
    if "items" in node:
        phrases += _rewrite(node["items"])
    if node.get("type") == "object" or "properties" in node:
        node["type"] = "object"
        properties = node.setdefault("properties", {})
        for prop in properties.values():
            _with_constraints(prop, _rewrite(prop))
        node["additionalProperties"] = False
        node["required"] = list(properties)
    return phrases


def to_strict_subset(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON Schema into the Strict Subset.

    Parameters
    ----------
    schema : dict
        A schema from ``model_json_schema()``.

    Returns
    -------
    dict
        A new schema in the Strict Subset; the input is not modified.

    Raises
    ------
    ValueError
        If the schema is recursive.
    """
    defs = schema.get("$defs", {})
    out = _inline(copy.deepcopy(schema), defs, ())
    _with_constraints(out, _rewrite(out))
    return out


def strict_subset_problems(schema: dict[str, Any]) -> list[str]:
    """List every way ``schema`` breaks the Strict Subset.

    Parameters
    ----------
    schema : dict
        A JSON Schema.

    Returns
    -------
    list of str
        One line per problem, each starting with its JSON path. Empty when
        the schema is in the subset.
    """
    problems: list[str] = []
    count = 0

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal count
        if not isinstance(node, dict):
            problems.append(f"{path}: schema node is not an object")
            return
        for key in _FORBIDDEN:
            if key in node:
                problems.append(f"{path}: keyword `{key}` is outside the subset")
        if not any(k in node for k in ("type", "anyOf", "enum")):
            problems.append(f"{path}: typeless node")
        if node.get("type") == "object":
            if depth > MAX_DEPTH:
                problems.append(f"{path}: object depth {depth} exceeds {MAX_DEPTH}")
            properties = node.get("properties")
            if not isinstance(properties, dict):
                problems.append(f"{path}: object without `properties` (an open map)")
                properties = {}
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: `additionalProperties` is not false")
            if node.get("required") != list(properties):
                problems.append(f"{path}: not every property is required")
            count += len(properties)
            for name, prop in properties.items():
                walk(prop, f"{path}.{name}", depth + 1)
        if node.get("type") == "array":
            if "items" not in node:
                problems.append(f"{path}: array without `items`")
            else:
                walk(node["items"], f"{path}[]", depth)
        for i, branch in enumerate(node.get("anyOf", [])):
            walk(branch, f"{path}|{i}", depth)

    if schema.get("type") != "object":
        problems.append("$: root is not an object")
    walk(schema, "$", 1)
    if count > MAX_PROPERTIES:
        problems.append(f"$: {count} properties exceed {MAX_PROPERTIES}")
    return problems
