"""The ``describe_c2st_columns`` Operation: per-frame C2ST results off a stored report."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .._boundary._describe_c2st_columns import DESCRIBE_C2ST_CAP
from .._boundary._describe_c2st_columns import describe_c2st_columns as describe_report
from .._boundary._operation import operation
from ..records import ColumnKeyedResult, Table, WireRecord
from .c2st import C2STSchemeName
from .evaluate_imputation import AnnotationName, OutcomeName, VerdictName


class C2STFrameResult(WireRecord):
    """The test on one imputed frame, pooled over its repeats."""

    frame: int
    """The frame's position, from 0. ``evaluate_imputation`` scores one frame, so this is 0."""
    mean_repeat_z: float
    """The mean of ``repeat_z``: this frame's score."""
    repeat_z: list[float]
    """One ``z`` per balanced subsampling repeat, in repeat order: ``(accuracy - 0.5) * 2 * sqrt(n_test)``."""
    repeat_z_spread: float
    """Standard deviation of ``repeat_z``; 0 for one repeat. Large next to ``mean_repeat_z`` means the mean rests on lucky draws."""
    train_accuracy: float
    """Mean training accuracy over the repeats. Near 1 beside a ``test_accuracy`` near 0.5 means the classifier memorised rows."""
    test_accuracy: float
    """Mean held-out accuracy over the repeats; 0.5 is chance."""
    n_test: int
    """Held-out rows scored in each repeat, half observed and half filled."""
    degenerate: bool
    """True when every repeat's classifier predicted one pile for every row."""
    unseen_category: bool
    """True when a filled row holds a category no observed row does."""


class C2STColumnScore(WireRecord):
    """Every number of a tested column."""

    mean_frame_z: float
    """The score: the mean of each frame's ``mean_repeat_z``. Near 0 means the filled rows could not be told from the observed ones; about 2 or more means they could."""
    frame_z_spread: float
    """Standard deviation of the frames' ``mean_repeat_z``; 0 for one frame."""
    p_value: float
    """The one-sided p-value of ``mean_frame_z``, before correction."""
    p_adjusted: float | None
    """The Benjamini-Hochberg adjusted p-value across the report's tested columns. At or below ``thresholds.fdr_alpha`` of ``evaluate_imputation`` is ``flagged``."""
    frames: Table[C2STFrameResult]
    """One result per imputed frame, in order."""


class C2STColumnDetail(WireRecord):
    """One evaluated column's C2ST result in full."""

    column: str
    """Column name."""
    outcome: OutcomeName
    """What happened: ``tested``, or why the test could not run, as in ``evaluate_imputation``."""
    verdict: VerdictName
    """``flagged``, ``not_flagged``, or ``no_verdict`` when not tested."""
    annotations: list[AnnotationName]
    """Facts qualifying a tested column's score, as in ``evaluate_imputation``. Empty when not tested."""
    n_observed: int
    """Rows where the column was observed in the source frame."""
    n_filled: int
    """Rows where the column was missing in the source frame: the rows the imputation filled."""
    score: C2STColumnScore | None
    """The numbers and per-frame results; null unless ``tested``."""


class ClassifierParam(WireRecord):
    """One parameter of the classifier that ran."""

    name: str
    """Parameter name."""
    value: str
    """Its value, as text."""


class C2STRunConfig(WireRecord):
    """Every setting the run used, defaults included."""

    repeats: int
    """Balanced subsampling repeats per frame."""
    scheme: C2STSchemeName
    """``split``: one fit per repeat on a 70/30 carve; ``cross_validation``: five."""
    min_samples_leaf: int
    """Fewest rows in a leaf of the classifier."""
    min_filled_cells: int
    """The size floor under which a column is ``below_sample_floor``."""
    low_power_below: int
    """The pile size under which a tested column is ``low_power``."""
    imbalance_warn_ratio: float
    """The pile ratio under which a tested column is ``imbalanced_piles``."""
    fdr_alpha: float
    """The false discovery rate deciding ``flagged``."""
    random_state: int | None
    """The seed used."""


class C2STRunProvenance(WireRecord):
    """What ran, so a score can be argued with later."""

    classifier_class: str
    """The classifier's fully qualified class name."""
    classifier_params: Table[ClassifierParam]
    """Every classifier parameter, defaults included, by name."""
    sklearn_version: str
    """The scikit-learn version that produced the numbers."""
    config: C2STRunConfig
    """The settings the run used."""


class DescribeC2STColumnsResult(ColumnKeyedResult):
    """One page of a C2ST report's columns in full, and the provenance. ``result.get(name)`` finds one column."""

    report: str
    """The c2st Handle read."""
    total: int
    """How many columns match ``names``."""
    offset: int
    """How many matching columns this page skipped."""
    columns: Table[C2STColumnDetail]
    """Matching columns: ``flagged`` ones first, then dataset order. At most 20; when more remain, ``omitted`` counts them and names ``describe_c2st_columns``."""
    provenance: C2STRunProvenance
    """The classifier, its parameters, the scikit-learn version and the settings of the run."""


@operation(effect="read", phase="evaluate", scales_with_data=False)
def describe_c2st_columns(
    report: str,
    names: list[str] | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=DESCRIBE_C2ST_CAP)] | None = None,
) -> DescribeC2STColumnsResult:
    """Read columns' full C2ST results off a c2st Handle, flagged columns first.

    Use when
    --------
    Call this after ``evaluate_imputation`` to dig into a column's score:
    each frame's ``z`` per repeat, its accuracies, and the provenance (the
    classifier, its parameters and the settings the run used). It reads the
    stored report only, so it is cheap and never recomputes. Filter with
    ``names``, then page with ``offset`` and ``limit``. The same arguments
    always return the same page.

    Parameters
    ----------
    report : str
        A c2st Handle ``evaluate_imputation`` made, such as ``c2st_6a7b8c9d``.
    names : list of str or None
        Only these columns. Null means every evaluated column.
    offset : int or None
        How many matching columns to skip, for the next page. Null means 0.
    limit : int or None
        Most columns to return, from 1 to 20. Null means 20.

    Returns
    -------
    DescribeC2STColumnsResult
        ``total`` matching columns and ``columns``: this page, flagged
        first, then in dataset order, each with its ``score`` and
        per-frame results when tested. ``provenance`` says what ran. When
        more remain, ``omitted`` counts them; call again with ``offset``
        raised by the page size.

    Errors
    ------
    unknown_handle
        ``report`` names no c2st Handle in this Workspace. ``candidates``
        lists c2st Handles, newest first.
    wrong_handle_kind
        ``report`` is a Handle of another kind.
    unknown_column
        A name in ``names`` was not evaluated: it is not a column, or it was
        excluded or ``dropped``. ``candidates`` lists close matches;
        ``issues`` lists every unknown name.
    invalid_argument
        An argument has the wrong type or is out of its range.
    incompatible_artifact
        Another library format version wrote ``report``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    evaluate_imputation : Make the c2st Handle and its outcome census.
    route : Route a flagged column to another strategy.

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
    {...,"c2st":"c2st_6a7b8c9d",...}
    $ dataforge describe-c2st-columns --report c2st_6a7b8c9d --names income
    {"warnings":...,"omitted":...,"report":"c2st_6a7b8c9d","total":1,"offset":0,...}
    """
    return DescribeC2STColumnsResult.model_validate(
        describe_report(report=report, names=names, offset=offset, limit=limit)
    )
