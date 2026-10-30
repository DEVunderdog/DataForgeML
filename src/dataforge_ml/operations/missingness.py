"""The ``missingness`` Operation: how missing values spread across rows and columns."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._profile_detail import MISSING_PAIRS_CAP, profile_missingness
from ..records import DialPointer, OutputRecord, Table, WireRecord


class RowMissingness(WireRecord):
    """How many values each row is missing, across the profiled columns."""

    complete_share: float
    """Share of rows missing nothing."""
    one_to_two_share: float
    """Share of rows missing one or two values."""
    three_to_five_share: float
    """Share of rows missing three to five values."""
    over_five_share: float
    """Share of rows missing more than five values."""
    p90_missing_per_row: int
    """Values missing per row at the 90th percentile of rows."""
    drop_candidate_rows: int
    """Rows missing at least ``drop_candidate_dial``'s share of the profiled columns."""
    drop_candidate_share: float
    """Share of rows that are drop candidates."""
    drop_candidate_dial: DialPointer
    """The ``profile`` dial that decides which rows are drop candidates."""


class CoMissingPair(WireRecord):
    """Two columns whose missing values fall on the same rows."""

    column_a: str
    """One column. When you filter by ``column``, always that column."""
    column_b: str
    """The other column."""
    correlation: float
    """Correlation of the two columns' missing masks, from -1 to 1. Near 1 means they go missing on the same rows."""
    mar_suspect: bool
    """True when ``correlation`` is above ``mar_suspect_dial``: both columns are flagged ``missingness.mar_suspect``."""


class MissingnessResult(OutputRecord):
    """The row-missingness distribution and one page of co-missing pairs."""

    profile: str
    """The profile Handle read."""
    sparsity: float
    """Share of all cells that are missing."""
    columns_with_missing: int
    """Profiled columns missing at least one value."""
    rows: RowMissingness
    """How many values each row is missing."""
    total: int
    """How many pairs match the filter. Only columns missing at least one value form pairs."""
    offset: int
    """How many matching pairs this page skipped."""
    pairs: Table[CoMissingPair]
    """Matching pairs, most co-missing (highest ``correlation``) first, ties in dataset order. At most 100; when more remain, ``omitted`` counts them and names ``missingness``."""
    mar_suspect_dial: DialPointer
    """The ``profile`` dial behind ``mar_suspect``."""


@operation(effect="read", phase="profile", scales_with_data=False)
def missingness(
    profile: str,
    column: str | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=MISSING_PAIRS_CAP)] | None = None,
) -> MissingnessResult:
    """Read how missing values spread across rows, and which columns go missing together.

    Use when
    --------
    Call this after ``profile`` when columns have missing values, to decide
    whether to drop rows and whether values are missing at random. It gives
    how many values each row is missing, and the column pairs whose missing
    values fall on the same rows, most co-missing first. Columns that go
    missing together suggest the values are missing for a reason, which
    matters for imputation. It reads the stored profile only, so it is cheap
    and never recomputes. The same arguments always return the same page.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    column : str or None
        Only pairs that include this column, which is then always
        ``column_a``. Null means every pair.
    offset : int or None
        How many matching pairs to skip, for the next page. Null means 0.
    limit : int or None
        Most pairs to return, from 1 to 100. Null means 100.

    Returns
    -------
    MissingnessResult
        ``sparsity``, ``columns_with_missing``, the ``rows`` distribution,
        and ``pairs``: this page of ``total`` co-missing pairs, most
        co-missing first. ``mar_suspect_dial`` and ``rows.drop_candidate_dial``
        name the dials behind the two flags. When more pairs remain,
        ``omitted`` counts them; call again with ``offset`` raised by the
        page size.

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
    describe_columns : Read each column's missing-value counts.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge missingness --profile profile_1b2c3d4e
    {"warnings":...,"omitted":...,"profile":"profile_1b2c3d4e",...,"columns_with_missing":2,...}
    """
    return MissingnessResult.model_validate(
        profile_missingness(profile=profile, column=column, offset=offset, limit=limit)
    )
