"""Internal code behind the ``resolve_recipe`` Operation (ADR-0089, ADR-0110).

It reads the routing Handle and the profile Handle the routing was made from,
resolves the recipe once, writes the recipe Handle and lists its units by
value, so ``derive_units`` is not an Operation.

The routing's own dials feed the recipe: the correlation thresholds and the
predictor cap that routing judged a column by are the ones its recipe resolves
with. Every other recipe dial takes its library default.
"""

from __future__ import annotations

from typing import Any

from .._serialization import serialize
from ._detail import page
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


def resolve_routing_recipe(*, routing: str) -> dict[str, Any]:
    """Resolve a routing Handle into a new recipe Handle and build the ``resolve_recipe`` result.

    Parameters
    ----------
    routing : str
        The routing Handle.

    Returns
    -------
    dict
        The fields of the ``resolve_recipe`` result.

    Raises
    ------
    DataForgeError
        The routing Handle's own errors, and the profile Handle's when the
        routing's profile can no longer be read.
    """
    from ..imputation._recipe import resolve_recipe
    from ..imputation._units import derive_units

    stored, record = read_routing(routing, param="routing")
    profile = record["profile"]
    result, _ = read_profile(profile, param="routing")
    recipe = resolve_recipe(stored, result, _recipe_config(record["dials"]["values"]))
    units = derive_units(recipe.routing)

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
            {"unit_id": unit.unit_id, "strategy": unit.strategy.value, "columns": columns}
        )
    return {
        "warnings": [],
        "omitted": omitted,
        "recipe": handle,
        "routing": routing,
        "profile": profile,
        "units": rows,
    }
