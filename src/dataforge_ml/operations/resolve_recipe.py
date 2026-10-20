"""The ``resolve_recipe`` Operation: resolve a routing into a fittable recipe."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._resolve_recipe import resolve_routing_recipe
from ..records import LITERAL, OutputRecord, Table, WireRecord
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


Finite = Annotated[float, Field(allow_inf_nan=False)]


class SnapBounds(WireRecord):
    """The range a column's predictions are rounded and clipped into."""

    low: Finite
    """The smallest value a prediction may take."""
    high: Finite
    """The largest value a prediction may take. At least ``low``."""


class ColumnEstimatesOverride(WireRecord):
    """Estimates you set on one column, over the ones the recipe resolves from the profile."""

    column: str
    """The column, exactly as in the data. Its strategy must read estimates: ``cluster_conditional`` or ``gmm_sampling``, or ``knn`` or ``mice`` on a ``bounded_discrete`` column."""
    center1: Finite | None = None
    """First of the two mode centres a ``cluster_conditional`` or ``gmm_sampling`` column splits on. Null keeps the resolved value; when the column has none, set both centres."""
    center2: Finite | None = None
    """Second mode centre. Null keeps the resolved value."""
    feature_cols: list[str] | None = None
    """Numeric columns a ``cluster_conditional`` column measures its cluster centroids over, replacing the resolved list. Null keeps the resolved list."""
    domain_snap_bounds: SnapBounds | None = None
    """The range a ``bounded_discrete`` column on ``knn``, ``mice`` or ``gmm_sampling`` snaps its predictions into. Null keeps the resolved range, the column's observed one."""


class ShownSnapBounds(WireRecord):
    """The range a column's predictions are rounded and clipped into."""

    low: Annotated[float | None, LITERAL]
    """The smallest value a prediction may take. A Literal Field."""
    high: Annotated[float | None, LITERAL]
    """The largest value a prediction may take. A Literal Field."""


class RecipeEstimates(WireRecord):
    """The estimates one column trains with: the resolved ones, or yours where you set one."""

    column: str
    """The column."""
    center1: float | None
    """First mode centre; null when the column has none, such as a forced bimodal column whose profile found one mode. Such a column fits only once you set both centres."""
    center2: float | None
    """Second mode centre; null when the column has none."""
    feature_cols: list[str] | None
    """Columns a ``cluster_conditional`` column measures centroids over, in dataset order; null for any other strategy. At most 10; past 10, ``omitted`` counts the rest."""
    domain_snap_bounds: ShownSnapBounds | None
    """The range predictions snap into; null unless the column is ``bounded_discrete`` on ``knn``, ``mice`` or ``gmm_sampling``."""


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
    """Every unit, in routing order: a joint block where its first column sits in the dataset. Empty when no column fills values. At most 50; past 50, ``omitted`` counts the rest and names ``explain_routing``, whose columns give each one's id as ``<strategy>:<column>``."""
    estimates: Table[RecipeEstimates]
    """Every column that reads estimates, in routing order. Empty when none does. At most 15; past 15, ``omitted`` counts the rest."""


@operation(effect="create", phase="impute", scales_with_data=False)
def resolve_recipe(
    routing: str,
    hyperparameters: list[UnitHyperparameters] | None = None,
    estimates: list[ColumnEstimatesOverride] | None = None,
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
    the fields you set replace the resolved ones, and the rest stay.
    Estimates are facts the recipe reads off the profile for some columns:
    the two mode centres and centroid features of a bimodal column, and the
    range a ``bounded_discrete`` column's predictions snap into. The result
    shows them; pass ``estimates`` to replace one, field by field, as with
    ``hyperparameters``. A forced bimodal column the profile found one mode
    in has no centres, and fits only once you set both. ``feature_cols``
    replaces the whole list. To change a recipe, call ``resolve_recipe``
    again with every override you want; that makes a new recipe Handle and
    leaves the old one as it was.

    Parameters
    ----------
    routing : str
        A routing Handle, such as ``routing_7f3a9c2e``.
    hyperparameters : list of UnitHyperparameters or None
        Hyperparameters you set, one entry per unit. Null keeps every
        resolved value.
    estimates : list of ColumnEstimatesOverride or None
        Estimates you set, one entry per column. Null keeps every resolved
        estimate.

    Returns
    -------
    ResolveRecipeResult
        ``recipe``, ``routing`` and ``profile`` Handles; ``units``, each
        with its ``unit_id``, ``strategy``, ``columns`` and
        ``hyperparameters``; and ``estimates``, one per column that reads
        them.

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
    unknown_column
        An ``estimates`` entry names a column the routing does not have, as
        its ``column`` or in ``feature_cols``. ``candidates`` lists close
        matches.
    duplicate_key
        ``hyperparameters`` names the same unit twice, or ``estimates`` the
        same column.
    estimates_not_applicable
        An ``estimates`` entry names a column whose strategy reads no
        estimates, or sets a field its strategy does not read.
    invalid_argument
        An argument has the wrong type, a hyperparameter record holds a key
        or value it does not take, or its ``kind`` is not the unit's
        strategy; a ``feature_cols`` entry is the column itself or not
        numeric; or ``domain_snap_bounds`` has ``low`` above ``high``.
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
    $ dataforge route --profile profile_1b2c3d4e --strategies '[{"column":"income","strategy":"gmm_sampling"}]'
    {...,"routing":"routing_2e3f4a5b",...}
    $ dataforge resolve-recipe --routing routing_2e3f4a5b --estimates '[{"column":"income","center1":48000,"center2":66000}]'
    {...,"recipe":"recipe_7a8b9c0d","routing":"routing_2e3f4a5b","profile":"profile_1b2c3d4e",...}
    """
    return ResolveRecipeResult.model_validate(
        resolve_routing_recipe(
            routing=routing, hyperparameters=hyperparameters, estimates=estimates
        )
    )
