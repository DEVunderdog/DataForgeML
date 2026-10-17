"""The ``split`` Operation: split a profiled frame into a test frame and folds."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._split import split_profile
from ..records import DialPointer, OutputRecord, Table, WireRecord

SchemeName = Literal[
    "random",
    "time",
    "group",
    "kfold",
    "group_kfold",
    "repeated_kfold",
    "holdout_cv",
    "profile_stratified",
    "profile_stratified_kfold",
]

TestSize = Annotated[float, Field(gt=0, lt=1)]
FoldCount = Annotated[int, Field(ge=2, le=100)]


# --- The scheme -------------------------------------------------------------------


class RandomScheme(WireRecord):
    """One shuffled train/test split."""

    kind: Literal["random"]
    """Always ``random``."""
    test_size: TestSize | None = None
    """Share of rows held out as ``test``. Null means 0.2."""
    stratify: bool | None = None
    """True keeps each target class's share the same in train and test. Null stratifies when the profile declares a target."""
    target: str | None = None
    """The declared target to stratify on. Null takes the profile's target; name one when it declares several."""


class TimeScheme(WireRecord):
    """One chronological split: the latest rows are ``test``, so no future row trains."""

    kind: Literal["time"]
    """Always ``time``."""
    time_column: str
    """The datetime or numeric column that orders the rows. Its sentinels count as missing, and rows with no time go to train."""
    test_size: TestSize | None = None
    """Share of rows, the latest ones, held out as ``test``. Null means 0.2 unless ``cutoff`` is set; set at most one of the two."""
    cutoff: datetime | float | None = None
    """Rows at or after this time are ``test``. A datetime for a datetime column (an ISO string on the CLI), a number for a numeric one. Null splits by ``test_size``."""


class GroupScheme(WireRecord):
    """One train/test split that keeps every group whole on one side."""

    kind: Literal["group"]
    """Always ``group``."""
    group_column: str
    """The column naming each row's group, such as a customer id. Rows missing it form one group."""
    test_size: TestSize | None = None
    """Share of groups held out as ``test``. Null means 0.2."""


class KFoldScheme(WireRecord):
    """``k`` shuffled folds: each row is in exactly one fold's validation frame."""

    kind: Literal["kfold"]
    """Always ``kfold``."""
    k: FoldCount | None = None
    """Number of folds. Null means 5."""
    stratify: bool | None = None
    """True keeps each target class's share the same in every fold. Null stratifies when the profile declares a target."""
    target: str | None = None
    """The declared target to stratify on. Null takes the profile's target; name one when it declares several."""


class GroupKFoldScheme(WireRecord):
    """``k`` folds that keep every group whole on one side of each fold."""

    kind: Literal["group_kfold"]
    """Always ``group_kfold``."""
    group_column: str
    """The column naming each row's group. Rows missing it form one group."""
    k: FoldCount | None = None
    """Number of folds, at most the number of groups. Null means 5."""
    stratify: bool | None = None
    """True balances target classes across folds as far as whole groups allow. Null stratifies when the profile declares a target."""
    target: str | None = None
    """The declared target to stratify on. Null takes the profile's target; name one when it declares several."""


class RepeatedKFoldScheme(WireRecord):
    """k-fold run ``repeats`` times, each with a new shuffle: ``k * repeats`` folds."""

    kind: Literal["repeated_kfold"]
    """Always ``repeated_kfold``."""
    k: FoldCount | None = None
    """Number of folds per repeat. Null means 5."""
    repeats: Annotated[int, Field(ge=1, le=50)] | None = None
    """Number of repeats. ``k * repeats`` may be at most 100. Null means 10."""
    stratify: bool | None = None
    """True keeps each target class's share the same in every fold. Null stratifies when the profile declares a target."""
    target: str | None = None
    """The declared target to stratify on. Null takes the profile's target; name one when it declares several."""


class HoldoutCVScheme(WireRecord):
    """A held-out ``test`` frame, then ``k`` folds over the remaining rows."""

    kind: Literal["holdout_cv"]
    """Always ``holdout_cv``."""
    test_size: TestSize | None = None
    """Share of rows held out as ``test``. Null means 0.2."""
    k: FoldCount | None = None
    """Number of folds over the rest. Null means 5."""
    stratify: bool | None = None
    """True stratifies the hold-out and the folds on the target. Null stratifies when the profile declares a target."""
    target: str | None = None
    """The declared target to stratify on. Null takes the profile's target; name one when it declares several."""


class ProfileStratifiedScheme(WireRecord):
    """One train/test split balanced on every signal the profile flags: target, missingness, extremes, rare labels."""

    kind: Literal["profile_stratified"]
    """Always ``profile_stratified``."""
    test_size: TestSize | None = None
    """Share of rows held out as ``test``. It also sets the viability floor, ``ceil(2 / min(test_size, 1 - test_size))`` rows. Null means 0.2."""
    target: str | None = None
    """The declared target to balance with the other signals. Null takes the profile's target, if any; name one when it declares several."""


class ProfileStratifiedKFoldScheme(WireRecord):
    """``k`` folds balanced on every signal the profile flags."""

    kind: Literal["profile_stratified_kfold"]
    """Always ``profile_stratified_kfold``."""
    k: FoldCount | None = None
    """Number of folds. It is also the viability floor, in rows. Null means 5."""
    target: str | None = None
    """The declared target to balance with the other signals. Null takes the profile's target, if any; name one when it declares several."""


Scheme = Annotated[
    RandomScheme | TimeScheme | GroupScheme | KFoldScheme | GroupKFoldScheme | RepeatedKFoldScheme | HoldoutCVScheme | ProfileStratifiedScheme | ProfileStratifiedKFoldScheme,
    Field(discriminator="kind"),
]


# --- Dials ------------------------------------------------------------------------


class StratificationDials(WireRecord):
    """Dials the profile-stratified schemes read. Null keeps every default in this group."""

    max_stratification_signals: Annotated[int, Field(ge=1)] | None = None
    """Most signals balanced at once. The cap is the smaller of this and ``n_rows / rows_per_signal``. Null means 50."""
    rows_per_signal: Annotated[int, Field(ge=1)] | None = None
    """Rows the data must have per balanced signal. Null means 10."""
    boolean_minority_threshold: Annotated[float, Field(gt=0, le=0.5)] | None = None
    """A boolean column whose rarer value is below this share becomes a signal. Null means 0.05."""
    redundancy_correlation_threshold: Annotated[float, Field(gt=0, le=1)] | None = None
    """Absolute correlation at or above which two signals count as one, and the less important is dropped. Null means 0.95."""


class SplitThresholds(WireRecord):
    """Every split dial, grouped. Null keeps every default."""

    stratification: StratificationDials | None = None
    """The signal gates of the profile-stratified schemes."""


# --- Result -----------------------------------------------------------------------


class Fold(WireRecord):
    """One fold's frames. A single train/test split has one fold with no validation frame."""

    index: int
    """Fold number, from 0, within its repeat."""
    repeat: int
    """Repeat number, from 0. Always 0 except for ``repeated_kfold``."""
    train: str
    """Frame Handle of the training rows."""
    validation: str | None
    """Frame Handle of the validation rows; null for a single train/test split, whose held-out rows are ``test``."""
    train_rows: int
    """Rows in ``train``."""
    validation_rows: int | None
    """Rows in ``validation``; null when it is null."""


class StratificationSummary(WireRecord):
    """How many signals each gate of a profile-stratified scheme kept. Counts only."""

    floor: int
    """The viability floor: fewest rows on each side a signal or label needs."""
    candidates: int
    """Signals the profile gave."""
    viable: int
    """Signals at or above the floor on both sides."""
    non_redundant: int
    """Viable signals left after near-duplicates were dropped."""
    cap: int
    """Most signals balanced at once."""
    kept: int
    """Signals balanced. 0 means the rows were split at random."""
    rows_to_train: int
    """Rows sent to train because their target class or rare value is below the floor."""
    labels_to_train: int
    """Target classes and rare values below the floor."""


class SplitFallback(WireRecord):
    """A step the split took because a threshold ruled out the planned one."""

    fallback: Literal["unstratified", "signals_capped", "rows_to_train"]
    """``unstratified``: no signal could be balanced, so rows were split at random. ``signals_capped``: some signals were not balanced. ``rows_to_train``: rows of labels below the floor all went to train."""
    count: int
    """Rows split at random, signals left out, or rows sent to train, by ``fallback``."""
    dials: Table[DialPointer]
    """The dials or ``scheme`` arguments that decided it, with their resolved values. Empty when the data had nothing to balance."""


class SplitPartitions(OutputRecord):
    """The test frame and the folds, each a frame Handle carrying the profile's declarations."""

    profile: str
    """The profile Handle that was split."""
    frame: str
    """The frame Handle the profile was made from, whose rows were split."""
    scheme: SchemeName
    """The scheme used."""
    seed: int
    """The seed used: yours, or the fixed default when you passed null."""
    stratified_on: str | None
    """The target the split balanced; null when it balanced none."""
    test: str | None
    """Frame Handle of the held-out rows; null for the k-fold schemes, which hold none out."""
    test_rows: int | None
    """Rows in ``test``; null when it is null."""
    folds: Table[Fold]
    """Every fold, in repeat then index order. At most 100."""
    stratification: StratificationSummary | None
    """Signal counts for the profile-stratified schemes; null for the others."""
    fallbacks: Table[SplitFallback]
    """Every fallback taken, each with the dials behind it. Empty when none."""


@operation(effect="create", phase="split", scales_with_data=True)
def split(
    profile: str,
    scheme: Scheme | None = None,
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None,
    thresholds: SplitThresholds | None = None,
) -> SplitPartitions:
    """Split the frame behind a profile into a test frame and folds, each a new frame Handle.

    Use when
    --------
    Call this after ``profile`` to make train, validation and test frames.
    Every scheme returns one shape: an optional ``test`` frame and a list of
    ``folds``, each with a ``train`` and an optional ``validation`` frame, so
    a train/test split is one fold with no validation and holdout-CV is a
    ``test`` plus ``k`` folds. Each new frame keeps the profile's
    declarations (targets, types, sentinels, exclusions), so a later
    ``profile`` of a fold may add to them but not contradict them. The
    profile-stratified schemes balance every signal the profile flags, not
    only the target; ``fallbacks`` reports each step a threshold forced, with
    the dial to turn. The source frame is never changed.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``. Its frame is split.
    scheme : RandomScheme or TimeScheme or GroupScheme or KFoldScheme or GroupKFoldScheme or RepeatedKFoldScheme or HoldoutCVScheme or ProfileStratifiedScheme or ProfileStratifiedKFoldScheme or None
        How to split, chosen by ``kind``; null means ``profile_stratified``
        with its defaults.
    seed : int or None
        Seed of every shuffle. Null means a fixed default, so the same call
        gives the same split.
    thresholds : SplitThresholds or None
        Dials of the profile-stratified schemes, each with its default and
        range. Null keeps every default.

    Returns
    -------
    SplitPartitions
        ``test`` and ``folds`` as frame Handles with their row counts, the
        ``seed`` used, ``stratified_on``, ``stratification`` (signal counts)
        and ``fallbacks``. ``warnings`` repeats each fallback
        (``split_unstratified``, ``stratification_signals_capped``,
        ``rare_rows_to_train``) and flags time-less rows sent to train
        (``null_time_rows_to_train``).

    Errors
    ------
    unknown_handle
        ``profile`` names no profile Handle in this Workspace. ``candidates``
        lists profile Handles, newest first.
    wrong_handle_kind
        ``profile`` is a Handle of another kind.
    unknown_column
        ``time_column``, ``group_column`` or ``target`` names no column of the
        frame. ``candidates`` lists close matches.
    declaration_conflict
        ``time_column`` or ``group_column`` is excluded.
    stratify_without_target
        ``stratify`` is true but the profile declares no target.
    stratify_on_continuous_target
        The target has too few rows per value to stratify on, as a
        continuous target does. The profile-stratified schemes bin it instead.
    invalid_argument
        An argument has the wrong type or is out of its range; ``target`` is
        not a declared target, or is unset when several are declared; ``k``
        is above the rows or groups; ``k * repeats`` is above 100; both
        ``test_size`` and ``cutoff`` are set; or ``cutoff`` does not match
        ``time_column``.
    incompatible_artifact
        Another library format version wrote ``profile``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    profile : Make the profile Handle to split, or profile a fold.
    export_frame : Write a fold's frame to CSV.
    list_handles : Find frame Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge split --profile profile_1b2c3d4e --scheme '{"kind":"kfold","k":3}'
    {...,"profile":"profile_1b2c3d4e","frame":"frame_7f3a9c2e","scheme":"kfold","seed":42,"stratified_on":null,"test":null,...}
    """
    return SplitPartitions.model_validate(
        split_profile(profile=profile, scheme=scheme, seed=seed, thresholds=thresholds)
    )
