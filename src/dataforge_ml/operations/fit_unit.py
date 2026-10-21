"""The ``fit_unit`` Operation: train one unit of a recipe."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .._boundary._fit_unit import fit_recipe_unit
from .._boundary._operation import operation
from ..records import OutputRecord, WireRecord
from ._hyperparameters import Hyperparameters
from .resolve_recipe import UnitStrategy


class UnitFitSignals(WireRecord):
    """What one unit's fit did: the estimator that ran, whether it converged, and the dials it trained with."""

    estimator: str | None
    """Name of the estimator that ran, such as ``BayesianRidge``, ``KNNImputer`` or ``GaussianMixture``; null for a strategy that fits none, such as ``median``."""
    converged: bool | None
    """Whether an iterative model stopped before its iteration cap; null for a strategy with no iterations. When false, ``warnings`` says what to change."""
    n_iter: int | None
    """Iterations the model ran; null for a strategy with no iterations."""
    hyperparameters: Hyperparameters | None
    """Every hyperparameter the unit trained with: the recipe's resolved value, or yours where you set one on ``resolve_recipe``. Null for a strategy with none: every one but ``mice``, ``knn``, ``mnar`` and ``cluster_conditional``."""


class FitUnitResult(OutputRecord):
    """The fitted unit Handle and the signals of its fit."""

    unit: str
    """Fitted unit Handle, such as ``unit_7f3a9c2e``."""
    recipe: str
    """The recipe Handle read."""
    frame: str
    """The frame Handle the unit trained on: yours, or the frame the recipe's profile was made from."""
    unit_id: str
    """The unit trained, as ``resolve_recipe`` lists it."""
    strategy: UnitStrategy
    """The strategy the unit trained."""
    columns: list[str]
    """The columns the unit fills, in dataset order. At most 100; past 100, ``omitted`` counts the rest and names ``explain_routing``."""
    seed: int
    """The seed used: yours, or the fixed default when you passed null."""
    signals: UnitFitSignals
    """What the fit did. Caveats on it, such as a model that hit its iteration cap, are in ``warnings``."""


@operation(effect="create", phase="impute", scales_with_data=True)
def fit_unit(
    recipe: str,
    unit_id: str,
    frame: str | None = None,
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None,
) -> FitUnitResult:
    """Train one unit of a recipe on a frame and return a unit Handle.

    Use when
    --------
    Call this after ``resolve_recipe``, once per unit it lists, to train the
    imputer piece by piece. There is no call that trains them all: you fit
    each unit yourself, in any order, and can run them side by side. The
    unit trains on the frame the recipe's profile was made from, or on
    ``frame`` when you pass one, such as a ``split`` fold, with the
    hyperparameters and estimates the recipe holds. The result is a unit
    Handle and the fit's signals: the estimator that ran, whether an
    iterative model converged, how many iterations it ran, and the dials it
    trained with. Read ``warnings`` first: a model that hit its iteration
    cap fits, but comes back with a coded warning saying which dial to turn.
    To change how a unit trains, call ``resolve_recipe`` again with new
    hyperparameters or estimates and fit that recipe's unit. ``seed`` fixes
    the random draws, such as those ``gmm_sampling`` fills with; null uses a
    fixed default, so the same call gives the same unit twice.

    Parameters
    ----------
    recipe : str
        A recipe Handle, such as ``recipe_7f3a9c2e``.
    unit_id : str
        The unit to train, as ``resolve_recipe`` lists it: ``mice`` or ``knn``
        for a joint block, ``<strategy>:<column>`` for one column, such as
        ``median:age``.
    frame : str or None
        A frame Handle to train on, such as a ``split`` fold. It must have
        every column of the recipe's routing. Null uses the frame the
        recipe's profile was made from.
    seed : int or None
        Seed of the random draws, from 0 to 4294967295. Null uses the fixed
        default.

    Returns
    -------
    FitUnitResult
        ``unit``, ``recipe`` and ``frame`` Handles; the ``unit_id``,
        ``strategy`` and ``columns`` trained; the ``seed`` used; and
        ``signals``: ``estimator``, ``converged``, ``n_iter`` and
        ``hyperparameters``.

    Errors
    ------
    unknown_handle
        ``recipe`` or ``frame`` names no Handle of its kind in this
        Workspace. ``candidates`` lists Handles of that kind, newest first.
    wrong_handle_kind
        ``recipe`` or ``frame`` is a Handle of another kind.
    unknown_unit
        The recipe has no unit ``unit_id``. ``candidates`` lists close
        matches.
    missing_fitted_columns
        ``frame`` lacks columns of the recipe's routing. ``columns`` names
        them.
    unit_not_trainable
        The unit cannot train on the frame, such as a bimodal column with no
        mode centres or a column with too few values. ``columns`` names its
        columns; the message gives the reason.
    invalid_argument
        An argument has the wrong type, or ``seed`` is out of range.
    incompatible_artifact
        Another library format version wrote ``recipe`` or the frame.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    resolve_recipe : Make the recipe Handle and list its unit ids.
    split : Make folds to train on.
    list_handles : Find unit Handles made earlier.

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
    {...,"unit":"unit_3c4d5e6f","recipe":"recipe_9e0f1a2b","frame":"frame_7f3a9c2e","unit_id":"median:income","strategy":"median",...}
    """
    return FitUnitResult.model_validate(
        fit_recipe_unit(recipe=recipe, unit_id=unit_id, frame=frame, seed=seed)
    )
