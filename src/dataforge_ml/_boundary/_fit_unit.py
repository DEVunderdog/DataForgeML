"""Internal code behind the ``fit_unit`` Operation (ADR-0071, ADR-0110).

It reads the recipe Handle and a frame (the one its profile was made from, or
a column-matched one the Agent passes, such as a split fold), trains one unit
on that frame, writes the fitted unit as a ``unit`` Handle, and returns
the unit's typed Fit Signals. There is no convenience chain: training a whole
recipe is one call per unit (ADR-0075).

``seed=null`` means the library's fixed default seed, so the same call fits
the same unit twice (ADR-0106).
"""

from __future__ import annotations

from typing import Any

from .._serialization import serialize
from ..records import DataForgeError
from ._detail import page
from ._error_codes import close_matches
from ._resolve_recipe import UNIT_COLUMNS_CAP, _shown_hyperparameters
from ._route import _capped_warning
from ._workspace import (
    InheritedDeclarations,
    read_frame,
    read_recipe,
    settings,
    write_handle,
)


def _unknown_unit(unit_id: str, unit_ids: list[str]) -> DataForgeError:
    return DataForgeError.from_code(
        "unknown_unit",
        f"`unit_id`: the recipe has no unit `{unit_id}`.",
        param="unit_id",
        candidates=close_matches([unit_id], unit_ids),
    )


def _missing_fitted_columns(missing: list[str]) -> DataForgeError:
    names = ", ".join(f"`{c}`" for c in missing)
    return DataForgeError.from_code(
        "missing_fitted_columns",
        f"`frame`: lacks column(s) {names} of the recipe's routing.",
        columns=missing,
    )


def fit_recipe_unit(
    *, recipe: str, unit_id: str, frame: str | None, seed: int | None
) -> dict[str, Any]:
    """Fit one unit of a recipe Handle on a frame and build the ``fit_unit`` result.

    Parameters
    ----------
    recipe : str
        The recipe Handle.
    unit_id : str
        The unit to fit, as ``resolve_recipe`` lists it.
    frame : str or None
        The frame Handle to fit on, such as a split fold; None means the
        frame the recipe's profile was made from.
    seed : int or None
        The seed of the stochastic strategies; None means the fixed default.

    Returns
    -------
    dict
        The fields of the ``fit_unit`` result.

    Raises
    ------
    DataForgeError
        ``unknown_unit`` with ``candidates`` when the recipe has no such
        unit, raised before the frame is read; ``missing_fitted_columns``
        when ``frame`` lacks a column of the recipe's routing;
        ``unit_not_trainable`` when the unit cannot train on the frame; the
        recipe and frame Handles' own errors.
    """
    from .._seed import DEFAULT_SEED
    from ..imputation._unit_fit import fit_unit
    from ..imputation._units import derive_units

    stored, record = read_recipe(recipe, param="recipe")
    units = {u.unit_id: u for u in derive_units(stored.routing)}
    if unit_id not in units:
        raise _unknown_unit(unit_id, list(units))
    unit = units[unit_id]
    if frame is None:
        frame = record["frame"]
        data, _ = read_frame(frame, param="recipe")
    else:
        data, _ = read_frame(frame, param="frame")
        missing = [c for c in stored.routing.column_routings if c not in data.columns]
        if missing:
            raise _missing_fitted_columns(missing)
    seed = DEFAULT_SEED if seed is None else seed
    fit = fit_unit(stored, unit, data, seed=seed, n_jobs_inner=settings().core_budget)

    handle = write_handle(
        "unit",
        size={"columns": len(unit.columns)},
        payload_name="unit.bin",
        write_payload=lambda tmp: tmp.write_bytes(serialize(fit.fitted)),
        obj=fit.fitted,
        declarations=InheritedDeclarations.from_dict(record["declarations"]),
        extra={
            "recipe": recipe,
            "frame": frame,
            "unit_id": unit_id,
            "strategy": unit.strategy.value,
        },
    )

    columns, _, omitted = page(
        list(unit.columns), None, None, UNIT_COLUMNS_CAP, field="columns",
        operation="explain_routing",
    )
    signals = fit.signals
    warnings = []
    for i, warning in enumerate(signals.warnings):
        if warning.columns is not None:
            warning, cut = _capped_warning(warning, warning.columns, i)
            omitted += cut
        warnings.append(warning)
    return {
        "warnings": warnings,
        "omitted": omitted,
        "unit": handle,
        "recipe": recipe,
        "frame": frame,
        "unit_id": unit_id,
        "strategy": unit.strategy.value,
        "columns": columns,
        "seed": seed,
        "signals": {
            "estimator": signals.estimator,
            "converged": signals.converged,
            "n_iter": signals.n_iter,
            "hyperparameters": _shown_hyperparameters(stored, unit),
        },
    }
