"""The ``evaluate_imputation`` Operation: score an imputed frame with the imputation C2ST."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._evaluate_imputation import evaluate_imputation as run_evaluation
from .._boundary._operation import operation
from ..records import ColumnKeyedResult, DialPointer, Table, WireRecord

OutcomeName = Literal[
    "tested",
    "no_filled_cells",
    "no_observed_cells",
    "unfilled",
    "type_not_testable",
    "below_sample_floor",
    "uninformative",
]
VerdictName = Literal["flagged", "not_flagged", "no_verdict"]
AnnotationName = Literal["low_power", "imbalanced_piles", "degenerate", "unseen_category"]


class EvaluateImputationThresholds(WireRecord):
    """Every ``evaluate_imputation`` dial. Null keeps every default."""

    min_filled_cells: Annotated[int, Field(ge=2)] | None = None
    """The size floor: a column whose smaller pile, observed or filled rows, is under it is ``below_sample_floor`` and not tested. Null means 30."""
    min_samples_leaf: Annotated[int, Field(ge=1)] | None = None
    """Fewest rows in a leaf of the classifier. Larger is smoother and harder to fool by memorising rows; too large and it predicts one pile for every row, which is ``uninformative``. Null means 5."""
    low_power_below: Annotated[int, Field(ge=0)] | None = None
    """A tested column whose smaller pile is under this many rows is annotated ``low_power``: a pass there means not enough data to tell. Null means 250."""
    imbalance_warn_ratio: Annotated[float, Field(ge=0, le=1)] | None = None
    """A tested column whose smaller pile over its larger one is under this ratio is annotated ``imbalanced_piles``. Null means 0.5."""
    fdr_alpha: Annotated[float, Field(gt=0, le=1)] | None = None
    """The false discovery rate across the tested columns: a column whose Benjamini-Hochberg adjusted p-value is at or below it is ``flagged``. Raise it to flag more. Null means 0.05."""


class OutcomeCount(WireRecord):
    """How many columns had one outcome and verdict."""

    outcome: OutcomeName
    """The outcome."""
    verdict: VerdictName
    """The verdict."""
    count: int
    """Number of columns with both."""


class EvaluatedColumn(WireRecord):
    """One column's C2ST outcome and score."""

    column: str
    """Column name."""
    outcome: OutcomeName
    """What happened: ``tested``, or why the test could not run. ``no_filled_cells``: nothing was missing. ``no_observed_cells``: nothing was observed. ``unfilled``: the imputed frame is still missing where the source was, as for an ``mnar``, ``passthrough`` or Drop Candidate column. ``type_not_testable``: a text or identifier column. ``below_sample_floor``: too few rows in the smaller pile. ``uninformative``: the classifier predicted one pile for every row."""
    verdict: VerdictName
    """``flagged`` when the filled rows are told apart from the observed ones beyond the false discovery rate: go look at this column. ``not_flagged`` when tested and not flagged. ``no_verdict`` when not tested."""
    mean_frame_z: float | None
    """The score: how well a classifier tells the column's filled rows from its observed rows, as a ``z``. Near 0 means it could not; about 2 or more means it could. Null unless ``tested``."""
    p_adjusted: float | None
    """The Benjamini-Hochberg adjusted one-sided p-value of ``mean_frame_z`` across the tested columns. Null unless ``tested``."""
    annotations: list[AnnotationName]
    """Facts qualifying a tested column's score, which they never suppress: ``low_power``, ``imbalanced_piles``, ``degenerate`` (some repeats predicted one pile) and ``unseen_category`` (a filled row holds a category no observed row does). Empty when not tested."""
    n_observed: int
    """Rows where the column was observed in the source frame."""
    n_filled: int
    """Rows where the column was missing in the source frame, sentinels included: the rows the imputation filled."""


class OutcomeDials(WireRecord):
    """The dials that decide one outcome, verdict or annotation. Turn one to change which columns get it."""

    outcome: str
    """The outcome, verdict or annotation, as in ``columns``."""
    dials: Table[DialPointer]
    """Every dial its rule reads, with its resolved value."""


class EvaluateImputationResult(ColumnKeyedResult):
    """The C2ST report Handle, the outcome census and each column's outcome. ``result.get(name)`` finds one column."""

    c2st: str
    """The new C2ST report Handle, such as ``c2st_7f3a9c2e``. It holds every column's per-frame results and the provenance."""
    profile: str
    """The profile Handle read."""
    imputed: str
    """The imputed frame Handle read."""
    source_frame: str
    """The frame Handle ``imputed`` was filled from: the observed values it is scored against."""
    rows: int
    """Rows of ``imputed``."""
    seed: int
    """The seed used: yours, or the fixed default when you passed null."""
    census: Table[OutcomeCount]
    """Columns per outcome and verdict, over every evaluated column. Read it first."""
    columns: Table[EvaluatedColumn]
    """Every evaluated column: ``flagged`` ones first, then dataset order. Excluded and ``dropped`` columns are not evaluated. At most 100; past 100, ``omitted`` counts the rest and names ``describe_c2st_columns``."""
    outcome_dials: Table[OutcomeDials]
    """One entry per threshold-decided outcome, verdict or annotation any evaluated column has, with the dials that decide it."""


@operation(effect="create", phase="evaluate", scales_with_data=True)
def evaluate_imputation(
    profile: str,
    imputed: str,
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None,
    thresholds: EvaluateImputationThresholds | None = None,
) -> EvaluateImputationResult:
    """Score how well an imputed frame keeps its data's distribution with a Classifier Two-Sample Test.

    Use when
    --------
    Call this after ``impute`` to check the fill. For each column, a
    classifier learns to tell the rows where the column was filled from the
    rows where it was observed, reading whole rows of ``imputed``; if it does
    no better than chance, the fill is indistinguishable from real data.
    ``imputed`` is scored against the frame it was filled from, so pass the
    profile the imputer was routed on, even for a filled fold. Columns either
    Handle excludes and ``dropped`` columns are not evaluated. Read
    ``census`` first, then the ``flagged`` columns. A column filled with one
    value (``mean``, ``median``, ``mode`` or ``constant``) is expected to be
    flagged: judge a model-based column against those. Under missing at
    random, the observed and filled rows are meant to differ, so a flag cannot
    tell a bad fill from strongly MAR missingness: the ``z`` is true, but it
    may not be the imputation's fault. The result writes a ``c2st_`` Handle
    holding every column's per-frame results and the provenance.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``: the one the imputer
        was routed on. Its sentinels and semantic types decide which cells
        were filled and which columns can be tested.
    imputed : str
        A frame Handle ``impute`` made, such as ``frame_5b6c7d8e``.
    seed : int or None
        Seed of the subsampling, the carve and the classifier. Null means a
        fixed default, so the same call gives the same result.
    thresholds : EvaluateImputationThresholds or None
        The size floor, the classifier's leaf size, the annotation bands and
        the false discovery rate, each with its default and range. Null
        keeps every default.

    Returns
    -------
    EvaluateImputationResult
        ``c2st`` (the new Handle), the ``profile``, ``imputed`` and
        ``source_frame`` Handles, ``rows``, ``seed``, ``census`` (columns
        per outcome and verdict), ``columns`` (each one's outcome, verdict,
        score and annotations, at most 100) and ``outcome_dials``.

    Errors
    ------
    unknown_handle
        ``profile`` or ``imputed`` names no Handle of its kind in this
        Workspace. ``candidates`` lists Handles of that kind, newest first.
    wrong_handle_kind
        ``profile`` or ``imputed`` is a Handle of another kind.
    invalid_argument
        An argument has the wrong type or is out of its range, or
        ``imputed`` was not made by ``impute``.
    incompatible_artifact
        Another library format version wrote ``profile``, ``imputed`` or the
        frame it was filled from.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    impute : Make the imputed frame to score.
    route : Route a flagged column to another strategy.
    c2st : Compare two frames, such as a fold's train and test frames.

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
    {...,"frame":"frame_8b9c0d1e",...}
    $ dataforge evaluate-imputation --profile profile_1b2c3d4e --imputed frame_8b9c0d1e
    {...,"c2st":"c2st_6a7b8c9d","profile":"profile_1b2c3d4e","imputed":"frame_8b9c0d1e",...}
    """
    return EvaluateImputationResult.model_validate(
        run_evaluation(profile=profile, imputed=imputed, seed=seed, thresholds=thresholds)
    )
