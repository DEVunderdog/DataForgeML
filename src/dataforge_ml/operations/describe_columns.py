"""The ``describe_columns`` Operation: full column profiles off a stored profile."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._profile_detail import DESCRIBE_CAP, describe_profile_columns
from ..records import ColumnKeyedResult, Table, WireRecord
from .profile import FlagDials, NumericKindName, SemanticTypeName

SkewSeverityName = Literal["normal", "moderate", "high", "severe"]
MissingSeverityName = Literal["minor", "moderate", "high", "severe"]
TypeFlagName = Literal[
    "numeric_coerced",
    "datetime_coerced",
    "boolean_candidate",
    "encoded_category",
    "identifier_column",
    "sequential_index",
    "float_sequential_index",
    "free_text_candidate",
    "user_override",
    "numeric_kind_override",
]


class ColumnMissingness(WireRecord):
    """How much of one column is missing."""

    null_count: int
    """Rows missing, sentinels and blank strings included."""
    null_ratio: float | None
    """Share of rows missing, sentinels and blank strings included."""
    standard_null_count: int
    """Rows that were null in the file itself, before sentinels were read as missing."""
    severity: MissingSeverityName | None
    """Missing-value severity, graded by the ``thresholds.missingness.severity_*`` dials of ``profile``; null when nothing is missing."""
    co_missing_with: list[str]
    """Columns whose missing rows go with this column's, above ``thresholds.missingness.mar_correlation_threshold``. At most 10; ``missingness`` lists every pair."""


class Tails(WireRecord):
    """Which tail of a numeric column holds more of its spread."""

    tag: Literal["symmetric", "right_heavy", "left_heavy"]
    """``right_heavy``, ``left_heavy`` or ``symmetric``, graded by the ``thresholds.numeric.tail_asymmetry_*`` dials of ``profile``."""
    right_share: float | None
    """Share of the spread beyond the median that lies in the right tail."""


class Bimodality(WireRecord):
    """The bimodality test readings of a numeric column."""

    dip_statistic: float | None
    """Hartigan's dip statistic."""
    dip_p_value: float | None
    """The dip test's p-value. Low means more than one mode."""
    separation: float | None
    """Separation between the two modes (Ashman's D)."""
    minority_weight: float | None
    """Share of rows in the smaller mode."""


class Nonlinearity(WireRecord):
    """How a numeric column relates to the other numeric columns. Routing reads these."""

    tag: Literal["linear", "monotonic_nonlinear", "complex_nonlinear", "unpredictable"]
    """The relationship, graded by the ``thresholds.nonlinearity`` dials of ``profile``."""
    spearman_pearson_discrepancy: float | None
    """Largest Spearman-minus-Pearson gap over the other numeric columns."""
    mean_mutual_information: float | None
    """Mean excess mutual information with the other numeric columns, in nats."""
    max_mutual_information: float | None
    """Largest mutual information with another numeric column, in nats."""
    r2_linear: float | None
    """R² of a linear model predicting this column from the others."""
    r2_rf: float | None
    """R² of a random forest predicting this column from the others."""
    r2_gap: float | None
    """``r2_rf`` minus ``r2_linear``."""
    heteroscedasticity_p_value: float | None
    """Breusch-Pagan p-value of the linear model's residuals. Low means uneven spread."""


class NumericDetail(WireRecord):
    """Statistics of a numeric column. No cell values: bounds and percentiles are in ``value_summary``."""

    kind: Literal["numeric"]
    """Always ``numeric``."""
    mean: float | None
    """Mean of the non-missing values."""
    std: float | None
    """Standard deviation."""
    skew: float | None
    """Skewness."""
    kurtosis: float | None
    """Excess kurtosis."""
    skew_severity: SkewSeverityName | None
    """Skew graded by the ``thresholds.numeric.skew_*`` dials of ``profile``."""
    kurtosis_tag: Literal["platykurtic", "mesokurtic", "leptokurtic"] | None
    """Kurtosis graded by the ``thresholds.numeric.kurt_*`` dials of ``profile``."""
    mean_median_ratio: float | None
    """Mean divided by median."""
    mode_share: float | None
    """Share of rows holding the most common value."""
    outlier_density: float | None
    """Share of values that are outliers, by ``thresholds.numeric.outlier_sigma_threshold``."""
    tails: Tails | None
    """Tail asymmetry; null when not measured."""
    bimodal: Bimodality | None
    """Bimodality test readings; null when the column was not tested further."""
    nonlinearity: Nonlinearity | None
    """Nonlinearity probe readings; null when the column was not probed."""


class CategoricalDetail(WireRecord):
    """Statistics of a categorical column. No labels: they are in ``value_summary``."""

    kind: Literal["categorical"]
    """Always ``categorical``."""
    unique_ratio: float | None
    """Distinct values per row."""
    mode_share: float | None
    """Share of rows holding the most common label."""
    rare_category_count: int
    """Labels rarer than ``thresholds.categorical.rare_threshold_pct`` of ``profile``."""
    rare_row_share: float | None
    """Share of rows holding one of those rare labels."""
    stratification_rare_label_count: int
    """Labels rarer than ``thresholds.categorical.stratification_rare_threshold_pct``, which a stratified split protects."""
    dominant_class_ratio: float | None
    """Count of the most common label divided by the second most common."""
    normalized_entropy: float | None
    """Shannon entropy of the labels, from 0 (one label) to 1 (all equally common)."""
    normalized_gini: float | None
    """Gini impurity of the labels, from 0 (one label) to 1 (all equally common)."""


class DatetimeDetail(WireRecord):
    """Statistics of a datetime column. The first and last dates are in ``value_summary``."""

    kind: Literal["datetime"]
    """Always ``datetime``."""
    range_days: float | None
    """Days from the first date to the last."""
    granularity: (
        Literal[
            "yearly",
            "monthly",
            "weekly",
            "daily",
            "hourly",
            "minutely",
            "secondly",
            "irregular",
        ]
        | None
    )
    """The spacing the dates follow."""
    median_gap_seconds: float | None
    """Median gap between consecutive dates, in seconds."""
    gap_cv: float | None
    """Coefficient of variation of the gaps between dates."""
    future_date_count: int
    """Dates after the time of profiling."""


class BooleanDetail(WireRecord):
    """Statistics of a boolean column."""

    kind: Literal["boolean"]
    """Always ``boolean``."""
    true_count: int
    """Rows holding true."""
    false_count: int
    """Rows holding false."""
    true_share: float | None
    """Share of non-missing rows holding true."""


class TextDetail(WireRecord):
    """Statistics of a free-text column."""

    kind: Literal["text"]
    """Always ``text``."""
    mean_tokens: float | None
    """Mean word count."""
    median_tokens: float | None
    """Median word count."""
    vocabulary_size: int
    """Distinct words."""
    min_chars: int
    """Shortest value, in characters."""
    max_chars: int
    """Longest value, in characters."""
    mean_chars: float | None
    """Mean length, in characters."""
    median_chars: float | None
    """Median length, in characters."""
    empty_share: float | None
    """Share of values that are empty."""
    whitespace_share: float | None
    """Share of values that are only whitespace."""


ColumnStats = Annotated[
    NumericDetail | CategoricalDetail | DatetimeDetail | BooleanDetail | TextDetail,
    Field(discriminator="kind"),
]


class ColumnDetail(WireRecord):
    """One column's full profile. No cell values."""

    column: str
    """Column name, exactly as in the data."""
    role: Literal["feature", "target", "skipped"]
    """``target`` for a declared target, ``skipped`` for a column in ``skip_columns`` (no statistics), else ``feature``."""
    semantic_type: SemanticTypeName | None
    """Detected or declared semantic type; null when skipped."""
    numeric_kind: NumericKindName | None
    """``continuous`` or ``bounded_discrete`` for a numeric column; else null."""
    dtype: str | None
    """The column's storage type as loaded, such as ``Int64`` or ``String``."""
    parsed_dtype: str | None
    """The storage type profiling read it as, after any coercion; null when skipped."""
    type_flags: list[TypeFlagName]
    """How the semantic type was decided, such as ``encoded_category`` or ``user_override``."""
    flags: list[str]
    """Flags raised on the column, each named ``<group>.<flag>``. ``flag_dials`` names the dials behind each."""
    cardinality: int | None
    """Number of distinct non-missing values; null when skipped."""
    missingness: ColumnMissingness | None
    """How much is missing; null when skipped."""
    stats: ColumnStats | None
    """Statistics for the column's semantic type, told apart by ``kind``; null when skipped, for an identifier, or when profiling failed."""


class DescribeColumnsResult(ColumnKeyedResult):
    """One page of full column profiles. ``result.get(name)`` finds one column."""

    profile: str
    """The profile Handle read."""
    total: int
    """How many columns match the filters."""
    offset: int
    """How many matching columns this page skipped."""
    columns: Table[ColumnDetail]
    """Matching columns: flagged ones first, then dataset order. At most 15; when more remain, ``omitted`` counts them and names ``describe_columns``."""
    flag_dials: Table[FlagDials]
    """One entry per flag raised on a column of this page, in ``columns`` order of first appearance."""


@operation(effect="read", phase="profile", scales_with_data=False)
def describe_columns(
    profile: str,
    names: list[str] | None = None,
    flagged_only: bool | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=DESCRIBE_CAP)] | None = None,
) -> DescribeColumnsResult:
    """Read full column profiles off a profile Handle, flagged columns first.

    Use when
    --------
    Call this after ``profile`` to read more about columns than the Dataset
    Digest shows: every statistic for the column's semantic type, the
    missing-value counts, the flags and how the type was decided. It reads
    the stored profile only, so it is cheap and never recomputes. It holds no
    cell values; for top values, bounds and percentiles call
    ``value_summary``. Filter with ``names`` or ``flagged_only``, then page
    with ``offset`` and ``limit``. The same arguments always return the same
    page.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    names : list of str or None
        Only these columns. Null means every profiled and skipped column.
    flagged_only : bool or None
        True keeps only columns that raised a flag. Null means false.
    offset : int or None
        How many matching columns to skip, for the next page. Null means 0.
    limit : int or None
        Most columns to return, from 1 to 15. Null means 15.

    Returns
    -------
    DescribeColumnsResult
        ``total`` matching columns and ``columns``: this page, flagged
        first, then in dataset order. Each column's ``stats`` is told apart
        by ``kind``. ``flag_dials`` names the dials behind each flag on the
        page. When more remain, ``omitted`` counts them; call again with
        ``offset`` raised by the page size.

    Errors
    ------
    unknown_handle
        ``profile`` names no profile Handle in this Workspace.
        ``candidates`` lists profile Handles, newest first.
    wrong_handle_kind
        ``profile`` is a Handle of another kind.
    unknown_column
        A name in ``names`` is not a column of the profile. ``candidates``
        lists close matches; ``issues`` lists every unknown name.
    invalid_argument
        An argument has the wrong type or is out of its range.
    incompatible_artifact
        Another library format version wrote ``profile``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    profile : Make the profile Handle and its Dataset Digest.
    value_summary : Read one column's top values, bounds and percentiles.
    missingness : Read every co-missing pair.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge describe-columns --profile profile_1b2c3d4e --names age,city
    {"warnings":...,"omitted":...,"profile":"profile_1b2c3d4e","total":2,"offset":0,...}
    """
    return DescribeColumnsResult.model_validate(
        describe_profile_columns(
            profile=profile,
            names=names,
            flagged_only=flagged_only,
            offset=offset,
            limit=limit,
        )
    )
