"""The ``describe_target`` Operation: one declared target's problem type and statistics."""

from __future__ import annotations

from typing import Annotated, Literal

from .._boundary._operation import operation
from .._boundary._profile_detail import describe_profile_target
from ..records import LITERAL, OutputRecord, Table, WireRecord
from .describe_columns import NumericDetail

#: A cell value: a label, a bound or a percentile, typed as the data holds it.
CellValue = bool | int | float | str | None


class ClassCount(WireRecord):
    """One class of a classification target."""

    label: Annotated[CellValue, LITERAL]
    """The class label, as the data holds it. A Literal Field."""
    count: int
    """Rows holding this class."""
    share: float
    """Share of all rows holding this class."""


class TargetClasses(WireRecord):
    """The classes of a classification target and how balanced they are."""

    class_count: int
    """Number of distinct classes."""
    dominant_class_ratio: float | None
    """Count of the most common class divided by the second most common."""
    normalized_entropy: float | None
    """Shannon entropy of the classes, from 0 (one class) to 1 (all equally common)."""
    counts: Table[ClassCount]
    """The most common classes, most common first. At most 20, and no more than the profile keeps; ``omitted`` counts the rest."""


class DescribeTargetResult(OutputRecord):
    """One declared target: its problem type and statistics."""

    profile: str
    """The profile Handle read."""
    column: str
    """The target column."""
    problem_type: Literal[
        "regression", "binary_classification", "multiclass_classification", "unknown"
    ]
    """The learning problem the target implies."""
    missing_count: int
    """Rows whose target is missing."""
    missing_ratio: float
    """Share of rows whose target is missing."""
    flags: list[
        Literal[
            "contains_missing",
            "high_imbalance",
            "severe_imbalance",
            "highly_skewed",
            "is_identifier",
        ]
    ]
    """Problems with the target: ``contains_missing``, ``high_imbalance`` (dominant class ratio above 5), ``severe_imbalance`` (above 20), ``highly_skewed`` or ``is_identifier``."""
    distribution: NumericDetail | None
    """Statistics of a numeric target, as ``describe_columns`` gives them; null for a classification target."""
    classes: TargetClasses | None
    """Classes of a classification target; null for a regression target."""


@operation(effect="read", phase="profile", scales_with_data=False)
def describe_target(profile: str, name: str) -> DescribeTargetResult:
    """Read one declared target's problem type and statistics off a profile Handle.

    Use when
    --------
    Call this after ``profile`` with ``targets`` set, to see what kind of
    learning problem a target poses and what to watch for: missing target
    values, class imbalance or a skewed numeric target. A classification
    target lists its most common classes with their labels; a regression
    target gives its distribution. Class labels are cell values, so they
    are Literal Fields. To rank features against the target, call
    ``top_associations`` with ``column`` set to it. It reads the stored
    profile only, so it is cheap and never recomputes.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    name : str
        A target declared when the profile was made.

    Returns
    -------
    DescribeTargetResult
        The target's ``problem_type``, missing values and ``flags``, plus
        ``distribution`` for a numeric target or ``classes`` for a
        classification target. When the target has more classes than are
        listed, ``omitted`` counts them.

    Errors
    ------
    unknown_handle
        ``profile`` names no profile Handle in this Workspace.
        ``candidates`` lists profile Handles, newest first.
    wrong_handle_kind
        ``profile`` is a Handle of another kind.
    unknown_column
        ``name`` is not a target declared on the profile. ``candidates``
        lists the declared targets closest to it.
    invalid_argument
        An argument has the wrong type.
    incompatible_artifact
        Another library format version wrote ``profile``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    profile : Declare targets and make the profile Handle.
    top_associations : Rank features against the target.
    value_summary : Read every class label, or a target's bounds and percentiles.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e --targets income
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge describe-target --profile profile_1b2c3d4e --name income
    {...,"column":"income","problem_type":"regression","missing_count":1,...}
    """
    return DescribeTargetResult.model_validate(
        describe_profile_target(profile=profile, name=name)
    )
