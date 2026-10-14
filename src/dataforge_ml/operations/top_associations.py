"""The ``top_associations`` Operation: the strongest column pairs off a stored profile."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._profile_detail import ASSOCIATIONS_CAP, top_profile_associations
from ..records import DialPointer, OutputRecord, Table, WireRecord

MetricName = Literal["pearson", "cramers_v", "eta_squared"]


class Association(WireRecord):
    """How strongly two columns go together, by one metric."""

    column_a: str
    """One column. When you filter by ``column``, always that column."""
    column_b: str
    """The other column. For ``eta_squared``, the categorical one when ``column`` is not set."""
    metric: MetricName
    """``pearson`` for two numeric columns, ``cramers_v`` for two categorical columns, ``eta_squared`` for a numeric and a categorical one."""
    value: float
    """The measure. Pearson runs from -1 to 1; Cramér's V and η² from 0 to 1. Strength is its absolute value."""
    near_redundant: bool
    """True when the strength is above the metric's dial in ``near_redundant_dials``: one column nearly repeats the other."""


class MetricDial(WireRecord):
    """The dial that decides ``near_redundant`` for one metric."""

    metric: MetricName
    """The metric."""
    dial: DialPointer
    """The ``profile`` dial and the value it had."""


class TopAssociationsResult(OutputRecord):
    """One page of column pairs, strongest first."""

    profile: str
    """The profile Handle read."""
    total: int
    """How many pairs match the filters."""
    offset: int
    """How many matching pairs this page skipped."""
    pairs: Table[Association]
    """Matching pairs by strength, strongest first; ties by metric (``pearson``, ``cramers_v``, ``eta_squared``), then dataset order. At most 100; when more remain, ``omitted`` counts them and names ``top_associations``."""
    near_redundant_dials: Table[MetricDial]
    """For each metric, the dial behind ``near_redundant``."""


@operation(effect="read", phase="profile", scales_with_data=False)
def top_associations(
    profile: str,
    column: str | None = None,
    metric: MetricName | None = None,
    min_strength: Annotated[float, Field(ge=0, le=1)] | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=ASSOCIATIONS_CAP)] | None = None,
) -> TopAssociationsResult:
    """List the most strongly associated column pairs off a profile Handle, strongest first.

    Use when
    --------
    Call this after ``profile`` to find columns that go together:
    near-duplicates to drop, predictors for imputation, or the features most
    related to a target. Pearson (two numeric columns), Cramér's V (two
    categorical columns) and η² (a numeric and a categorical column) come in
    one list sorted by strength, the absolute value of the measure. Set
    ``column`` to a target to rank features against it. Pairs whose measure
    could not be computed are left out. It reads the stored profile only, so
    it is cheap and never recomputes. The same arguments always return the
    same page.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    column : str or None
        Only pairs that include this column, which is then always
        ``column_a``. Null means every pair.
    metric : str or None
        Only this metric: ``pearson``, ``cramers_v`` or ``eta_squared``.
        Null means all three.
    min_strength : float or None
        Only pairs at least this strong, from 0 to 1. Null means no floor.
    offset : int or None
        How many matching pairs to skip, for the next page. Null means 0.
    limit : int or None
        Most pairs to return, from 1 to 100. Null means 100.

    Returns
    -------
    TopAssociationsResult
        ``total`` matching pairs and ``pairs``: this page, strongest first.
        ``near_redundant_dials`` names the dial behind each metric's
        ``near_redundant``. When more remain, ``omitted`` counts them; call
        again with ``offset`` raised by the page size.

    Errors
    ------
    unknown_handle
        ``profile`` names no profile Handle in this Workspace.
        ``candidates`` lists profile Handles, newest first.
    wrong_handle_kind
        ``profile`` is a Handle of another kind.
    unknown_column
        ``column`` is not a column of the profile. ``candidates`` lists
        close matches.
    invalid_argument
        An argument has the wrong type or is out of its range.
    incompatible_artifact
        Another library format version wrote ``profile``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    profile : Make the profile Handle and its Dataset Digest.
    describe_target : Read a target's problem type and statistics.
    missingness : Read which columns go missing together.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e --targets income
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge top-associations --profile profile_1b2c3d4e --column income
    {...,"profile":"profile_1b2c3d4e","total":1,"offset":0,"pairs":{"fields":["column_a","column_b","metric","value","near_redundant"],"rows":[["income","city","eta_squared",...,true]]},...}
    """
    return TopAssociationsResult.model_validate(
        top_profile_associations(
            profile=profile,
            column=column,
            metric=metric,
            min_strength=min_strength,
            offset=offset,
            limit=limit,
        )
    )
