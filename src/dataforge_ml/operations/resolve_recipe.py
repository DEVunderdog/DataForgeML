"""The ``resolve_recipe`` Operation: resolve a routing into a fittable recipe."""

from __future__ import annotations

from typing import Literal

from .._boundary._operation import operation
from .._boundary._resolve_recipe import resolve_routing_recipe
from ..records import OutputRecord, Table, WireRecord
from ._hyperparameters import Hyperparameters

#: The strategies a unit trains: every routed strategy except ``dropped``,
#: ``passthrough`` and ``indicator``, which train nothing.
UnitStrategy = Literal[
    "mean",
    "median",
    "mode",
    "knn",
    "mice",
    "mnar",
    "constant",
    "cluster_conditional",
    "gmm_sampling",
]


class UnitHyperparameters(WireRecord):
    """Hyperparameters you set on one unit, over the ones the recipe resolves."""

    unit_id: str
    """The unit to tune, as ``resolve_recipe`` lists it: ``mice``, ``knn``, or ``<strategy>:<column>`` such as ``mnar:income``."""
    hyperparameters: Hyperparameters
    """The record of the unit's strategy, picked by ``kind``: ``mice``, ``knn``, ``mnar`` or ``cluster_conditional``. Only those strategies have hyperparameters. A null field keeps the resolved value."""


class RecipeUnit(WireRecord):
    """One unit of work ``fit_unit`` trains: a joint block or one column."""

    unit_id: str
    """The unit id ``fit_unit`` takes: ``mice`` or ``knn`` for a joint block, ``<strategy>:<column>`` for one column, such as ``median:age``."""
    strategy: UnitStrategy
    """The strategy the unit trains."""
    columns: list[str]
    """The columns the unit fills, in dataset order: every column of a joint block, or its one column. At most 100; past 100, ``omitted`` counts the rest and names ``explain_routing``."""
    hyperparameters: Hyperparameters | None
    """Every hyperparameter the unit trains with: the resolved value, or yours where you set one. Null for a strategy with none: every one but ``mice``, ``knn``, ``mnar`` and ``cluster_conditional``."""


class ResolveRecipeResult(OutputRecord):
    """The recipe Handle and every unit it trains."""

    recipe: str
    """Recipe Handle, such as ``recipe_7f3a9c2e``. It holds its own copy of the routing."""
    routing: str
    """The routing Handle read."""
    profile: str
    """The profile Handle the routing was made from."""
    units: Table[RecipeUnit]
    """Every unit, in routing order: a joint block where its first column sits in the dataset. Empty when no column fills values. At most 100; past 100, ``omitted`` counts the rest and names ``explain_routing``, whose columns give each one's id as ``<strategy>:<column>``."""


@operation(effect="create", phase="impute", scales_with_data=False)
def resolve_recipe(
    routing: str, hyperparameters: list[UnitHyperparameters] | None = None
) -> ResolveRecipeResult:
    """Resolve a routing into a recipe Handle that lists the units to fit.

    Use when
    --------
    Call this after ``route`` to get the work ``fit_unit`` does. The recipe
    fixes every unit's hyperparameters and every column's estimates from the
    profile the routing was made from, once, so each unit fits the same way
    every time. It reads the routing's own dials, so a dial you set on
    ``route`` holds here too. The result lists the units by value: the joint
    ``mice`` and ``knn`` blocks, and one unit per other column that fills
    values. Columns routed to ``passthrough`` or ``dropped``, and indicator
    columns, train nothing and have no unit. Each unit shows the
    hyperparameters it trains with. To tune a unit, pass ``hyperparameters``:
    the fields you set replace the resolved ones, and the rest stay. To
    change a recipe, call ``resolve_recipe`` again with every override you
    want; that makes a new recipe Handle and leaves the old one as it was.

    Parameters
    ----------
    routing : str
        A routing Handle, such as ``routing_7f3a9c2e``.
    hyperparameters : list of UnitHyperparameters or None
        Hyperparameters you set, one entry per unit. Null keeps every
        resolved value.

    Returns
    -------
    ResolveRecipeResult
        ``recipe``, ``routing`` and ``profile`` Handles, and ``units``: each
        with its ``unit_id``, ``strategy``, ``columns`` and
        ``hyperparameters``.

    Errors
    ------
    unknown_handle
        ``routing`` names no routing Handle in this Workspace. ``candidates``
        lists routing Handles, newest first.
    wrong_handle_kind
        ``routing`` is a Handle of another kind.
    unknown_unit
        A ``hyperparameters`` entry names a unit the routing does not have.
        ``candidates`` lists close matches; ``issues`` lists every problem
        entry.
    duplicate_key
        ``hyperparameters`` names the same unit twice.
    invalid_argument
        An argument has the wrong type, a hyperparameter record holds a key
        or value it does not take, or its ``kind`` is not the unit's
        strategy.
    incompatible_artifact
        Another library format version wrote ``routing`` or its profile.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    route : Make the routing Handle to resolve.
    list_handles : Find recipe Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge route --profile profile_1b2c3d4e --strategies '[{"column":"income","strategy":"knn"}]'
    {...,"routing":"routing_5a6b7c8d",...}
    $ dataforge resolve-recipe --routing routing_5a6b7c8d
    {...,"recipe":"recipe_9e0f1a2b","routing":"routing_5a6b7c8d","profile":"profile_1b2c3d4e",...}
    $ dataforge resolve-recipe --routing routing_5a6b7c8d --hyperparameters '[{"unit_id":"knn","hyperparameters":{"kind":"knn","n_neighbors":10}}]'
    {...,"recipe":"recipe_3c4d5e6f","routing":"routing_5a6b7c8d","profile":"profile_1b2c3d4e",...}
    """
    return ResolveRecipeResult.model_validate(
        resolve_routing_recipe(routing=routing, hyperparameters=hyperparameters)
    )
