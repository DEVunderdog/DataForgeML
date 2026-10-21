"""The ``compose_imputer`` Operation: compose a recipe's fitted units into an imputer."""

from __future__ import annotations

from .._boundary._compose_imputer import compose_recipe_imputer
from .._boundary._operation import operation
from ..records import OutputRecord, Table, WireRecord
from .resolve_recipe import UnitStrategy


class ImputerUnit(WireRecord):
    """One unit of the imputer and the unit Handle that fills it."""

    unit_id: str
    """The unit, as ``resolve_recipe`` lists it, such as ``mice`` or ``median:age``."""
    strategy: UnitStrategy
    """The strategy the unit trained."""
    unit: str
    """The unit Handle composed for it, such as ``unit_3c4d5e6f``."""


class ComposeImputerResult(OutputRecord):
    """The imputer Handle and the unit Handle composed for each unit of the recipe."""

    imputer: str
    """Imputer Handle, such as ``imputer_7f3a9c2e``. It names the recipe and the unit Handles, and holds nothing else."""
    recipe: str
    """The recipe Handle read."""
    units: Table[ImputerUnit]
    """Every unit of the recipe, in routing order, with the unit Handle composed for it. Empty when no column fills values. At most 50; past 50, ``omitted`` counts the rest."""


@operation(effect="create", phase="impute", scales_with_data=False)
def compose_imputer(recipe: str, units: list[str]) -> ComposeImputerResult:
    """Compose a recipe's fitted units into an imputer Handle.

    Use when
    --------
    Call this after ``fit_unit`` has trained every unit ``resolve_recipe``
    listed, to get one imputer that fills the whole frame. Pass one unit
    Handle per unit, in any order. Every unit must have been fitted from
    this recipe, and every unit of the recipe needs one Handle; columns
    routed to ``passthrough``, ``dropped`` or ``indicator`` have no unit and
    need none. The imputer Handle names the recipe and its unit Handles and
    holds nothing else, so the call is quick and copies no fitted model. To
    change one unit, fit it again with ``fit_unit``, such as on another
    frame or seed, and compose again with the new unit Handle in place of
    the old one: that writes only a new imputer Handle, and the old imputer
    stays as it was. To change hyperparameters or estimates, resolve a new
    recipe and fit its units, since units of another recipe are refused.

    Parameters
    ----------
    recipe : str
        A recipe Handle, such as ``recipe_7f3a9c2e``.
    units : list of str
        Unit Handles, one per unit of the recipe, such as
        ``["unit_3c4d5e6f", "unit_9a8b7c6d"]``. Empty when the recipe has no
        unit.

    Returns
    -------
    ComposeImputerResult
        ``imputer`` and ``recipe`` Handles, and ``units``: each unit's
        ``unit_id`` and ``strategy`` with the ``unit`` Handle composed for it.

    Errors
    ------
    unknown_handle
        ``recipe`` or a ``units`` entry names no Handle of its kind in this
        Workspace. ``candidates`` lists Handles of that kind, newest first.
    wrong_handle_kind
        ``recipe`` or a ``units`` entry is a Handle of another kind.
    unit_recipe_mismatch
        Units were fitted from another recipe. ``params`` names them, such
        as ``units[2]``.
    duplicate_key
        Two unit Handles fill the same unit. ``params`` names every one after
        the first.
    unit_coverage_mismatch
        A unit of the recipe has no unit Handle. The message names the units;
        ``columns`` names the columns they fill.
    invalid_argument
        An argument has the wrong type.
    incompatible_artifact
        Another library format version wrote ``recipe`` or a unit.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    fit_unit : Make the unit Handles to compose.
    resolve_recipe : List the units a recipe needs.
    list_handles : Find unit and imputer Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge route --profile profile_1b2c3d4e --strategies '[{"column":"income","strategy":"median"}]'
    {...,"routing":"routing_5a6b7c8d",...}
    $ dataforge resolve-recipe --routing routing_5a6b7c8d
    {...,"recipe":"recipe_9e0f1a2b",...}
    $ dataforge fit-unit --recipe recipe_9e0f1a2b --unit-id median:income
    {...,"unit":"unit_3c4d5e6f",...}
    $ dataforge compose-imputer --recipe recipe_9e0f1a2b --units unit_3c4d5e6f
    {...,"imputer":"imputer_4d5e6f7a","recipe":"recipe_9e0f1a2b",...}
    """
    return ComposeImputerResult.model_validate(
        compose_recipe_imputer(recipe=recipe, units=units)
    )
