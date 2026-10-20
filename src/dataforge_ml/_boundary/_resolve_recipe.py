"""Internal code behind the ``resolve_recipe`` Operation (ADR-0089, ADR-0110).

It reads the routing Handle and the profile Handle the routing was made from,
resolves the recipe once, writes the recipe Handle and lists its units by
value, so ``derive_units`` is not an Operation.

The routing's own dials feed the recipe: the correlation thresholds and the
predictor cap that routing judged a column by are the ones its recipe resolves
with. Every other recipe dial takes its library default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._serialization import serialize
from ..records import DataForgeError
from ._detail import page
from ._error_codes import close_matches
from ._profile import _Issue
from ._route import numeric_dials
from ._workspace import InheritedDeclarations, read_profile, read_routing, write_handle

#: Most units the result lists by value; ``explain_routing`` pages through the columns behind the rest.
UNIT_CAP = 100
#: Most columns one unit lists by value.
UNIT_COLUMNS_CAP = 100


def _recipe_config(routing_dials: dict[str, Any]) -> Any:
    # A config holding the routing's own dials; every other field is the default.
    from ..config import PipelineConfig
    from ..imputation._config import ImputationConfig, NumericImputationConfig

    numeric = NumericImputationConfig(**numeric_dials(routing_dials))
    return PipelineConfig(imputation=ImputationConfig(numeric=numeric))


def _check_hyperparameters(entries: Sequence[Any], units: Sequence[Any]) -> list[_Issue]:
    """Check every ``hyperparameters`` entry against the routing's units."""
    from ..operations._hyperparameters import _RECORDS

    strategies = {u.unit_id: u.strategy for u in units}
    issues: list[_Issue] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        param, unit_id = f"hyperparameters[{i}]", entry.unit_id
        if unit_id not in strategies:
            issues.append(
                _Issue(
                    "unknown_unit",
                    f"{param}.unit_id",
                    f"the routing has no unit `{unit_id}`",
                    unknown=unit_id,
                )
            )
            continue
        if unit_id in seen:
            issues.append(
                _Issue("duplicate_key", f"{param}.unit_id", f"`{unit_id}` is listed twice")
            )
            continue
        seen.add(unit_id)
        strategy = strategies[unit_id]
        if strategy not in _RECORDS:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.unit_id",
                    f"`{unit_id}` is a `{strategy.value}` unit, which has no hyperparameters",
                )
            )
        elif entry.hyperparameters.kind != strategy.value:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.hyperparameters.kind",
                    f"`{unit_id}` is a `{strategy.value}` unit, so its record's `kind` "
                    f"is `{strategy.value}`, not `{entry.hyperparameters.kind}`",
                    unknown=strategy.value,
                )
            )
    return issues


def _issues_error(issues: list[_Issue], unit_ids: Sequence[str]) -> DataForgeError:
    # The first issue's code names the error; the rest ride along in ``issues``.
    first = issues[0]
    params = list(dict.fromkeys(i.param for i in issues))
    if len(issues) == 1:
        message = f"`{first.param}`: {first.problem}."
    else:
        message = (
            f"{len(issues)} problems in the arguments, first `{first.param}`: "
            f"{first.problem}. See `issues` for all of them."
        )
    fields: dict[str, Any] = {
        "param": first.param if len(params) == 1 else None,
        "params": params,
        "issues": [{"param": i.param, "problem": i.problem} for i in issues],
    }
    if first.code == "unknown_unit":
        unknown = [i.unknown for i in issues if i.code == "unknown_unit"]
        return DataForgeError.from_code(
            "unknown_unit", message, candidates=close_matches(unknown, unit_ids), **fields
        )
    if first.code == "duplicate_key":
        return DataForgeError.from_code("duplicate_key", message, **fields)
    # A kind mismatch names the one kind its unit takes.
    allowed = [first.unknown] if first.unknown is not None else None
    return DataForgeError.from_code("invalid_argument", message, allowed=allowed, **fields)


def _shown_hyperparameters(recipe: Any, unit: Any) -> dict[str, Any] | None:
    # The unit's merged dials as its strategy's record; None for a strategy with none.
    from ..operations._hyperparameters import _RECORDS, hyperparameters_record

    if unit.strategy not in _RECORDS:
        return None
    record = hyperparameters_record(unit.strategy, recipe.hyperparameters(unit.unit_id))
    return record.model_dump()


def resolve_routing_recipe(
    *, routing: str, hyperparameters: Sequence[Any] | None
) -> dict[str, Any]:
    """Resolve a routing Handle into a new recipe Handle and build the ``resolve_recipe`` result.

    Parameters
    ----------
    routing : str
        The routing Handle.
    hyperparameters : sequence or None
        The ``UnitHyperparameters`` records, as validated by the args model.

    Returns
    -------
    dict
        The fields of the ``resolve_recipe`` result.

    Raises
    ------
    DataForgeError
        ``unknown_unit``, ``duplicate_key`` or ``invalid_argument`` naming
        every problem ``hyperparameters`` entry, raised before any work; the
        routing Handle's own errors, and the profile Handle's when the
        routing's profile can no longer be read.
    """
    from ..imputation._recipe import resolve_recipe
    from ..imputation._units import derive_units

    stored, record = read_routing(routing, param="routing")
    units = derive_units(stored)
    issues = _check_hyperparameters(hyperparameters or [], units)
    if issues:
        raise _issues_error(issues, [u.unit_id for u in units])
    profile = record["profile"]
    result, _ = read_profile(profile, param="routing")
    recipe = resolve_recipe(stored, result, _recipe_config(record["dials"]["values"]))
    for entry in hyperparameters or []:
        recipe = recipe.with_hyperparameters(
            entry.unit_id, entry.hyperparameters.to_hyperparameters()
        )

    handle = write_handle(
        "recipe",
        size={"units": len(units)},
        payload_name="recipe.bin",
        write_payload=lambda tmp: tmp.write_bytes(serialize(recipe)),
        obj=recipe,
        declarations=InheritedDeclarations.from_dict(record["declarations"]),
        extra={"routing": routing, "profile": profile, "frame": record["frame"]},
    )

    shown, _, omitted = page(
        list(units), None, None, UNIT_CAP, field="units", operation="explain_routing"
    )
    rows = []
    for i, unit in enumerate(shown):
        columns, _, cut = page(
            list(unit.columns),
            None,
            None,
            UNIT_COLUMNS_CAP,
            field=f"units[{i}].columns",
            operation="explain_routing",
        )
        omitted += cut
        rows.append(
            {
                "unit_id": unit.unit_id,
                "strategy": unit.strategy.value,
                "columns": columns,
                "hyperparameters": _shown_hyperparameters(recipe, unit),
            }
        )
    return {
        "warnings": [],
        "omitted": omitted,
        "recipe": handle,
        "routing": routing,
        "profile": profile,
        "units": rows,
    }
