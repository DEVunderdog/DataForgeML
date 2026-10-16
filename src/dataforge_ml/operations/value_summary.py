"""The ``value_summary`` Operation: one column's actual values, so sentinels can be spotted."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._profile_detail import VALUES_CAP, summarise_values
from ..records import LITERAL, OutputRecord, Table, WireRecord
from .describe_target import CellValue
from .profile import SemanticTypeName


class ValueCount(WireRecord):
    """One value and how often it occurs."""

    value: Annotated[CellValue, LITERAL]
    """The value, as the data holds it. A Literal Field."""
    count: int
    """Rows holding this value."""
    share: float
    """Share of all rows holding this value."""


class Bounds(WireRecord):
    """The smallest and largest value of a column."""

    min: Annotated[CellValue, LITERAL]
    """The smallest value; for a datetime column, the first date as ISO text. A Literal Field."""
    max: Annotated[CellValue, LITERAL]
    """The largest value; for a datetime column, the last date as ISO text. A Literal Field."""


class Percentile(WireRecord):
    """One percentile of a numeric column."""

    percentile: int
    """Which percentile: 1, 5, 25, 50, 75, 95 or 99."""
    value: Annotated[CellValue, LITERAL]
    """The value at that percentile. A Literal Field."""


class ValueSummaryResult(OutputRecord):
    """One column's top values, rare labels, bounds and percentiles."""

    profile: str
    """The profile Handle read."""
    column: str
    """The column summarised."""
    semantic_type: SemanticTypeName | None
    """The column's semantic type; null when it was skipped."""
    cardinality: int | None
    """Number of distinct non-missing values; null when skipped."""
    top_values: Table[ValueCount]
    """The most common values, most common first. The profile keeps up to 10 for a categorical, boolean or whole-number column and none for a continuous one; ``omitted`` counts the rest."""
    rare_labels: Annotated[list[CellValue] | None, LITERAL]
    """Labels of a categorical column rarer than ``thresholds.categorical.stratification_rare_threshold_pct`` of ``profile``, most common first. A Literal Field."""
    bounds: Bounds | None
    """The smallest and largest value of a numeric or datetime column; else null."""
    median: Annotated[CellValue, LITERAL]
    """The median of a numeric column; else null. A Literal Field."""
    mode: Annotated[CellValue, LITERAL]
    """The most common value; null for a datetime or text column. A Literal Field."""
    percentiles: Table[Percentile]
    """Percentiles of a numeric column, lowest first; empty for any other type."""


@operation(effect="read", phase="profile", scales_with_data=False)
def value_summary(
    profile: str,
    column: str,
    k: Annotated[int, Field(ge=1, le=VALUES_CAP)] | None = None,
) -> ValueSummaryResult:
    """Read one column's top values, rare labels, bounds and percentiles off a profile Handle.

    Use when
    --------
    Call this to see a column's actual values: to spot sentinels such as
    ``-999`` or ``"?"`` that stand for missing values, to learn the labels a
    categorical column uses, or to check a column's range. These are cell
    values, so each one is in a Literal Field; every other detail Operation
    holds none. To read a spotted sentinel as missing, call ``profile``
    again with it in ``sentinels``. It reads the stored profile only, so it
    is cheap and never recomputes.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    column : str
        A column of the profile.
    k : int or None
        Most entries in each list, from 1 to 20. Null means 20.

    Returns
    -------
    ValueSummaryResult
        ``top_values`` with counts, ``rare_labels``, ``bounds``, ``median``,
        ``mode`` and ``percentiles``, each filled where the column's type
        has it. When a list was cut, ``omitted`` counts the rest and names
        ``value_summary`` (raise ``k``) or ``query_frame`` (the profile keeps
        no more).

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
    profile : Declare sentinels and make the profile Handle.
    describe_columns : Read a column's statistics without cell values.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge value-summary --profile profile_1b2c3d4e --column city --k 2
    {...,"column":"city","semantic_type":"categorical","cardinality":3,"top_values":{"fields":["value","count","share"],"rows":[["north",2,...],["south",2,...]]},...}
    """
    return ValueSummaryResult.model_validate(
        summarise_values(profile=profile, column=column, k=k)
    )
