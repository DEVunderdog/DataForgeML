"""The ``impute`` Operation: fill a frame with an imputer into a model-ready frame."""

from __future__ import annotations

from typing import Annotated

from .._boundary._impute import impute_frame
from .._boundary._operation import operation
from ..records import LITERAL, ColumnKeyedResult, DialPointer, Table, WireRecord
from .describe_target import CellValue
from .route import StrategyName


class ImputedColumn(WireRecord):
    """What ``impute`` did to one column."""

    column: str
    """Column name. A ``<column>_missing`` row is the indicator an ``mnar`` column adds."""
    strategy: StrategyName
    """The column's strategy. A ``dropped`` column is not in the new frame, unless it is a Drop Candidate: see ``drop_candidates``."""
    missing: int
    """Cells missing in the source frame, sentinels included; 0 for an indicator."""
    filled: int
    """Cells the strategy filled. ``missing - filled`` cells are still missing: those of an ``mnar`` or ``passthrough`` column, and any a model left. 0 for a ``dropped`` column."""
    fill_value: Annotated[CellValue, LITERAL]
    """The one value a ``mean``, ``median``, ``mode``, ``constant`` or ``mnar`` column fills with. An ``mnar`` column keeps its nulls: its fill is on record here, not applied. Null for every other strategy. A Literal Field."""


class DropCandidates(WireRecord):
    """The columns routing marked ``dropped`` because profiling flagged them ``missingness.drop_candidate``."""

    columns: list[str]
    """The Drop Candidates, in dataset order. They stay in the new frame with their missing values. At most 100; past 100, ``omitted`` counts the rest."""
    dial: DialPointer
    """The profiling threshold that flagged them. Raise it to flag fewer."""


class ImputeResult(ColumnKeyedResult):
    """The model-ready frame Handle and what was done to each column. ``result.get(name)`` finds one column."""

    frame: str
    """The new frame Handle, such as ``frame_5b6c7d8e``. It lacks hard-excluded and ``dropped`` columns other than Drop Candidates, has the indicator columns as Soft Exclusions later phases skip, and inherits the source frame's declarations."""
    imputer: str
    """The imputer Handle read."""
    source_frame: str
    """The frame Handle filled. It stays as it was, excluded columns included."""
    rows: int
    """Rows of the new frame: as many as the source frame."""
    columns: Table[ImputedColumn]
    """Every column the imputer was fitted on: those that fill or drop first, then ``passthrough``, each group in dataset order. At most 100; past 100, ``omitted`` counts the rest."""
    drop_candidates: DropCandidates | None
    """Columns the library would drop, kept in the new frame: every drop is the Agent's own. To drop them, pass them to ``route`` as ``exclude_columns`` and build the imputer again from there. Null when there are none."""


@operation(effect="create", phase="impute", scales_with_data=True)
def impute(imputer: str, frame: str) -> ImputeResult:
    """Fill a frame with an imputer into a new frame Handle ready to model on.

    Use when
    --------
    Call this after ``compose_imputer``, once per frame to fill: the frame
    the units were fitted on, a ``split`` fold or the test frame. The
    imputer applies what it learned and learns nothing from ``frame``, so
    filling a test frame leaks nothing. The new frame leaves out
    hard-excluded columns and the ``dropped`` columns you declared, and has
    every indicator column; the source frame keeps them. A Drop Candidate,
    a column routing marked ``dropped`` for its missing share, stays in the
    new frame with its missing values: ``drop_candidates`` lists them and
    the dial that flagged them, and dropping one is your call. An ``mnar``
    column keeps its nulls beside its ``<column>_missing`` indicator: its
    fill is on record in ``columns`` and is not applied. ``passthrough``
    columns keep their missing values too. ``frame`` must have every column
    the imputer fills or leaves as it is, and no column it was not fitted
    on, other than excluded ones. The new frame inherits the source frame's
    declarations and the exclusions the routing added, and declares each
    indicator column a Soft Exclusion: later phases keep it but skip it.

    Parameters
    ----------
    imputer : str
        An imputer Handle, such as ``imputer_7f3a9c2e``.
    frame : str
        A frame Handle to fill, such as ``frame_7f3a9c2e``.

    Returns
    -------
    ImputeResult
        ``frame`` (the new Handle), ``imputer`` and ``source_frame``
        Handles, ``rows``, and ``columns``: each column's ``strategy``,
        ``missing`` and ``filled`` cell counts and ``fill_value``, and
        ``drop_candidates`` with their Dial Pointer.

    Errors
    ------
    unknown_handle
        ``imputer`` or ``frame`` names no Handle of its kind in this
        Workspace. ``candidates`` lists Handles of that kind, newest first.
    wrong_handle_kind
        ``imputer`` or ``frame`` is a Handle of another kind.
    unseen_column
        ``frame`` has columns the imputer was not fitted on and no
        declaration excludes. ``columns`` names them.
    missing_fitted_columns
        ``frame`` lacks columns the imputer fills or leaves as it is.
        ``columns`` names them.
    invalid_argument
        An argument has the wrong type.
    incompatible_artifact
        Another library format version wrote ``imputer``, a Handle it names,
        or ``frame``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    compose_imputer : Make the imputer Handle.
    split : Make folds and a test frame to fill.
    export_frame : Write the new frame as a CSV file.

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
    {...,"imputer":"imputer_4d5e6f7a",...}
    $ dataforge impute --imputer imputer_4d5e6f7a --frame frame_7f3a9c2e
    {...,"frame":"frame_8b9c0d1e","imputer":"imputer_4d5e6f7a","source_frame":"frame_7f3a9c2e","rows":6,...}
    """
    return ImputeResult.model_validate(impute_frame(imputer=imputer, frame=frame))
