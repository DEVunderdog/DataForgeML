"""Internal code behind the ``compose_imputer`` Operation (ADR-0103, ADR-0110).

It checks that the unit Handles the Agent passes were fitted from the recipe
Handle and cover its units exactly, then writes an ``imputer`` Handle that is
only a manifest: the recipe and its unit Handles by unit id. Nothing aggregate
is serialized (ADR-0072); :func:`~._workspace.read_imputer` composes the
imputer again on use. The check reads unit records, never their payloads, so
composing again after refitting one unit writes one small record and nothing
else.
"""

from __future__ import annotations

from typing import Any

from ..records import DataForgeError
from ._detail import page
from ._resolve_recipe import UNIT_CAP
from ._workspace import (
    InheritedDeclarations,
    read_recipe,
    read_unit_record,
    write_handle,
)


def _issues_fields(params: list[str], problem: str) -> dict[str, Any]:
    return {
        "param": params[0] if len(params) == 1 else None,
        "params": params,
        "issues": [{"param": p, "problem": problem} for p in params],
    }


def compose_recipe_imputer(*, recipe: str, units: list[str]) -> dict[str, Any]:
    """Check unit Handles against a recipe Handle and write the imputer manifest.

    Parameters
    ----------
    recipe : str
        The recipe Handle.
    units : list of str
        The unit Handles, one per unit of the recipe, in any order.

    Returns
    -------
    dict
        The fields of the ``compose_imputer`` result.

    Raises
    ------
    DataForgeError
        ``unit_recipe_mismatch`` naming every unit fitted from another
        recipe; ``duplicate_key`` naming every unit Handle whose unit
        another one already covers; ``unit_coverage_mismatch`` with the
        ``columns`` of the recipe's units no Handle covers; the recipe and
        unit Handles' own errors.
    """
    from ..imputation._units import derive_units

    stored, record = read_recipe(recipe, param="recipe")
    derived = derive_units(stored.routing)
    unit_records = [
        read_unit_record(handle, param=f"units[{i}]") for i, handle in enumerate(units)
    ]

    foreign = [f"units[{i}]" for i, r in enumerate(unit_records) if r["recipe"] != recipe]
    if foreign:
        raise DataForgeError.from_code(
            "unit_recipe_mismatch",
            f"`{foreign[0]}`: {len(foreign)} unit(s) were fitted from another recipe, "
            f"not `{recipe}`.",
            **_issues_fields(foreign, "fitted from another recipe"),
        )

    by_id: dict[str, str] = {}
    repeated: list[str] = []
    for i, (handle, r) in enumerate(zip(units, unit_records, strict=True)):
        if r["unit_id"] in by_id:
            repeated.append(f"units[{i}]")
        else:
            by_id[r["unit_id"]] = handle
    if repeated:
        raise DataForgeError.from_code(
            "duplicate_key",
            f"`{repeated[0]}`: {len(repeated)} unit Handle(s) cover a unit an earlier "
            "one already covers.",
            **_issues_fields(repeated, "covers a unit an earlier entry already covers"),
        )

    missing = [u for u in derived if u.unit_id not in by_id]
    if missing:
        names = ", ".join(f"`{u.unit_id}`" for u in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise DataForgeError.from_code(
            "unit_coverage_mismatch",
            f"`units`: no unit Handle covers unit(s) {names}{more} of the recipe. "
            "Fit them with `fit_unit` and compose again.",
            columns=[c for u in missing for c in u.columns],
        )

    # Routing order, so the manifest reads like the recipe it was composed from.
    manifest = [{"unit_id": u.unit_id, "unit": by_id[u.unit_id]} for u in derived]
    handle = write_handle(
        "imputer",
        size={"units": len(manifest)},
        declarations=InheritedDeclarations.from_dict(record["declarations"]),
        extra={"recipe": recipe, "units": manifest},
    )

    strategies = {u.unit_id: u.strategy.value for u in derived}
    shown, _, omitted = page(manifest, None, None, UNIT_CAP, field="units", operation=None)
    return {
        "warnings": [],
        "omitted": omitted,
        "imputer": handle,
        "recipe": recipe,
        "units": [
            {"unit_id": e["unit_id"], "strategy": strategies[e["unit_id"]], "unit": e["unit"]}
            for e in shown
        ],
    }
