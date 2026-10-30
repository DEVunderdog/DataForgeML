"""The ``c2st`` Operation: test whether two frames come from the same distribution."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._c2st import run_c2st
from .._boundary._operation import operation
from ..records import OutputRecord, WireRecord

C2STSchemeName = Literal["split", "cross_validation"]


class C2STThresholds(WireRecord):
    """Every ``c2st`` dial. Null keeps every default."""

    min_rows: Annotated[int, Field(ge=2)] | None = None
    """The size floor: the smaller frame needs at least this many rows, or the call fails with ``c2st_sample_floor``. Below 30 the test flags identical frames as different more often than it should. Null means 30."""
    min_samples_leaf: Annotated[int, Field(ge=1)] | None = None
    """Fewest rows in a leaf of the classifier. Larger is smoother and harder to fool by memorising rows. Null means 5."""


class C2STRun(OutputRecord):
    """The pooled result of one Classifier Two-Sample Test over two frames."""

    reference: str
    """The reference frame Handle read."""
    candidate: str
    """The candidate frame Handle read."""
    reference_rows: int
    """Rows of ``reference``."""
    candidate_rows: int
    """Rows of ``candidate``. Each repeat draws the smaller frame's row count from each."""
    columns: int
    """Columns tested: those of ``reference`` that neither frame declares excluded or soft-excluded."""
    skipped: int
    """Columns of ``reference`` left out because a frame declares them excluded or soft-excluded."""
    scheme: C2STSchemeName
    """The scheme used."""
    repeats: int
    """The repeats run."""
    seed: int
    """The seed used: yours, or the fixed default when you passed null."""
    mean_repeat_z: float
    """The headline: the mean of ``repeat_z``. Near 0 means the classifier could not tell the frames apart; about 2 or more means it could, so the frames differ."""
    repeat_z: list[float]
    """One ``z`` per repeat, in repeat order: ``(accuracy - 0.5) * 2 * sqrt(n_test)``."""
    repeat_z_spread: float
    """Standard deviation of ``repeat_z``; 0 for one repeat. Large next to ``mean_repeat_z`` means the mean rests on lucky draws: raise ``repeats``."""
    train_accuracy: float
    """Mean training accuracy over the repeats. Near 1 beside a ``test_accuracy`` near 0.5 means the classifier memorised rows, which hides a difference rather than inventing one."""
    test_accuracy: float
    """Mean held-out accuracy over the repeats; 0.5 is chance."""
    n_test: int
    """Held-out rows scored in each repeat, half from each frame."""
    degenerate: bool
    """True when every repeat's classifier predicted one frame for every row, so ``z`` says nothing. Compare larger frames, or lower ``thresholds.min_samples_leaf``."""
    unseen_category: bool
    """True when a categorical or boolean value is in ``candidate`` and never in ``reference``."""


@operation(effect="read", phase="evaluate", scales_with_data=True)
def c2st(
    reference: str,
    candidate: str,
    repeats: Annotated[int, Field(ge=1, le=100)] | None = None,
    scheme: C2STSchemeName | None = None,
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None,
    thresholds: C2STThresholds | None = None,
) -> C2STRun:
    """Test whether two frames come from the same distribution with a Classifier Two-Sample Test.

    Use when
    --------
    Call this to check drift between two frames, such as a ``split``
    fold's ``train`` frame and the ``test`` frame. A classifier learns to
    tell ``reference`` rows from ``candidate`` rows; if it does no better
    than chance on held-out rows, the frames are indistinguishable. Each
    repeat subsamples the larger frame to the smaller one's size. Columns
    either frame declares excluded or soft-excluded are left out; every
    other column must be in both frames. String columns cannot be tested:
    exclude them on ``profile`` before ``split``. No Handle is written.

    Parameters
    ----------
    reference : str
        A frame Handle, such as ``frame_7f3a9c2e``: the baseline, such as
        the training rows.
    candidate : str
        A frame Handle to compare against ``reference``, such as the test
        rows.
    repeats : int or None
        Balanced subsampling repeats, from 1 to 100. More repeats steady
        ``mean_repeat_z``. Null means 5.
    scheme : str or None
        ``split`` fits once per repeat on a 70/30 carve; ``cross_validation``
        fits five times per repeat and scores every drawn row, at five times
        the cost. Null means ``split``.
    seed : int or None
        Seed of the subsampling, the carve and the classifier. Null means a
        fixed default, so the same call gives the same result.
    thresholds : C2STThresholds or None
        The size floor and the classifier's leaf size, each with its
        default and range. Null keeps every default.

    Returns
    -------
    C2STRun
        The Handles and their rows, the ``columns`` tested and ``skipped``,
        the ``scheme``, ``repeats`` and ``seed`` used, the headline
        ``mean_repeat_z`` with each ``repeat_z``, the accuracies, and the
        ``degenerate`` and ``unseen_category`` flags.

    Errors
    ------
    unknown_handle
        ``reference`` or ``candidate`` names no frame Handle in this
        Workspace. ``candidates`` lists frame Handles, newest first.
    wrong_handle_kind
        ``reference`` or ``candidate`` is a Handle of another kind.
    frames_not_aligned
        A column to test is in only one frame. ``columns`` names them.
    c2st_string_column
        A column to test is a string column. ``columns`` names them.
    c2st_sample_floor
        The smaller frame has fewer rows than ``thresholds.min_rows``.
    invalid_argument
        An argument has the wrong type or is out of its range, or every
        column is excluded.
    incompatible_artifact
        Another library format version wrote ``reference`` or ``candidate``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    split : Make the train, validation and test frames to compare.
    profile : Exclude string columns, which the frames ``split`` makes inherit.
    impute : Fill a frame before comparing it.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e --exclude-columns city
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge split --profile profile_1b2c3d4e --scheme '{"kind":"random","test_size":0.5}'
    {...,"profile":"profile_1b2c3d4e","frame":"frame_7f3a9c2e",...,"test":"frame_2c3d4e5f",...,"rows":[[0,0,"frame_3d4e5f6a",...]]},...}
    $ dataforge c2st --reference frame_3d4e5f6a --candidate frame_2c3d4e5f --thresholds '{"min_rows":3}'
    {...,"reference":"frame_3d4e5f6a","candidate":"frame_2c3d4e5f",...,"columns":2,"skipped":1,...}
    """
    return C2STRun.model_validate(
        run_c2st(
            reference=reference,
            candidate=candidate,
            repeats=repeats,
            scheme=scheme,
            seed=seed,
            thresholds=thresholds,
        )
    )
