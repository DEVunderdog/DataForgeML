"""The ``profile`` Operation: profile a frame once and return its Dataset Digest."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .._boundary._operation import operation
from .._boundary._profile import profile_frame
from ..records import ColumnKeyedResult, DialPointer, Table, WireRecord

SemanticTypeName = Literal["numeric", "categorical", "datetime", "boolean", "text", "identifier"]
NumericKindName = Literal["continuous", "bounded_discrete"]
EpochUnitName = Literal["s", "ms", "us", "ns", "d"]

Ratio = Annotated[float, Field(ge=0, le=1)]
NonNegative = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=1)]


# --- Declarations -----------------------------------------------------------------


class SemanticTypeDeclaration(WireRecord):
    """A column's semantic type, set by you instead of detected."""

    column: str
    """Column name, exactly as in the data."""
    semantic_type: SemanticTypeName
    """The semantic type: ``numeric``, ``categorical``, ``datetime``, ``boolean``, ``text`` or ``identifier``."""


class NumericKindDeclaration(WireRecord):
    """A numeric column's kind, set by you instead of detected."""

    column: str
    """Column name, exactly as in the data. Its semantic type must be ``numeric``."""
    numeric_kind: NumericKindName
    """``continuous``, or ``bounded_discrete`` for a whole-number count or score with a small range."""


class SentinelDeclaration(WireRecord):
    """Values in one column that mean "missing", such as ``-999`` or ``"?"``."""

    column: str
    """Column name, exactly as in the data."""
    values: Annotated[list[float], Field(min_length=1)] | Annotated[list[str], Field(min_length=1)]
    """The values to read as missing: all numbers for a numeric column, or all strings for a text column. For a text column they replace the default markers (``NA``, ``NULL``, ``?``...) and match case-insensitively."""


class DatetimeFormatDeclaration(WireRecord):
    """How to read one datetime column. Give exactly one of ``format`` and ``epoch_unit``."""

    column: str
    """Column name, exactly as in the data."""
    format: str | None = None
    """A strftime format for a text column, such as ``%Y-%m-%d`` or ``%Y`` for bare years; or null."""
    epoch_unit: EpochUnitName | None = None
    """The unit of a numeric epoch column: ``s``, ``ms``, ``us``, ``ns`` or ``d``; or null."""

    @model_validator(mode="after")
    def _exactly_one(self) -> DatetimeFormatDeclaration:
        if (self.format is None) == (self.epoch_unit is None):
            raise ValueError("give exactly one of `format` and `epoch_unit`")
        if self.format is not None and not self.format:
            raise ValueError("`format` must not be empty")
        return self


# --- Dials ------------------------------------------------------------------------


class MissingnessDials(WireRecord):
    """Dials for missing values. Null keeps every default in this group."""

    severity_minor: Ratio | None = None
    """Null ratio below which a column's severity is ``minor``. Must not exceed ``severity_moderate``. Null means 0.01."""
    severity_moderate: Ratio | None = None
    """Null ratio below which severity is ``moderate``. Must not exceed ``severity_high``. Null means 0.05."""
    severity_high: Ratio | None = None
    """Null ratio below which severity is ``high``; at or above it, ``severe``. Null means 0.2."""
    mar_correlation_threshold: Ratio | None = None
    """Correlation between two columns' missing masks above which both are flagged ``missingness.mar_suspect``. Null means 0.6."""
    col_drop_threshold: Ratio | None = None
    """Null ratio above which a column is flagged ``missingness.drop_candidate``. Null means 0.5."""
    row_drop_threshold: Ratio | None = None
    """Share of profiled columns a row must be missing to count as a drop-candidate row. Null means 0.5."""


class TypeDetectionDials(WireRecord):
    """Dials for detecting each column's semantic type. Null keeps every default in this group."""

    numeric_coerce_threshold: Ratio | None = None
    """Share of a text column's values that must parse as numbers for it to be read as numeric. Null means 0.95."""
    datetime_coerce_threshold: Ratio | None = None
    """Share of a text column's values that must parse as datetimes for it to be read as datetime. Null means 0.8."""
    encoded_category_max_unique: Count | None = None
    """Most distinct values an integer column may have to be read as an encoded category. Null means 15."""
    encoded_category_max_ratio: Ratio | None = None
    """Most distinct values per non-null row an integer column may have to be read as an encoded category. Null means 0.05."""
    identifier_unique_ratio: Ratio | None = None
    """Distinct values per row above which a column is read as an identifier. Null means 0.99."""
    identifier_max_median_length: Count | None = None
    """Longest median length, in characters, a text column may have to be read as an identifier. Null means 40."""
    free_text_avg_words: Count | None = None
    """Median word count above which a text column is read as free text. Null means 3."""
    free_text_median_chars: Count | None = None
    """Median length, in characters, above which a multi-word text column is read as free text. Null means 20."""
    free_text_p90_chars: Count | None = None
    """90th-percentile length, in characters, above which a high-cardinality text column is read as free text. Null means 35."""
    free_text_min_unique_ratio: Ratio | None = None
    """Distinct values per row a text column needs, with ``free_text_p90_chars``, to be read as free text. Null means 0.4."""
    free_text_high_unique_with_spaces: Ratio | None = None
    """Distinct values per row above which a multi-word text column is read as free text whatever its length. Null means 0.7."""


class NumericDials(WireRecord):
    """Dials for numeric columns. Null keeps every default in this group."""

    skew_normal: NonNegative | None = None
    """Largest absolute skew still graded ``normal``. Must not exceed ``skew_moderate``. Null means 0.5."""
    skew_moderate: NonNegative | None = None
    """Largest absolute skew graded ``moderate``. Must not exceed ``skew_high``. Null means 1.0."""
    skew_high: NonNegative | None = None
    """Largest absolute skew graded ``high``; above it, ``severe``. Null means 2.0."""
    kurt_platykurtic_upper: float | None = None
    """Excess kurtosis below which a column is platykurtic. Must be below ``kurt_leptokurtic_lower``. Null means -1.0."""
    kurt_leptokurtic_lower: float | None = None
    """Excess kurtosis above which a column is leptokurtic. Null means 3.0."""
    near_constant_threshold: Ratio | None = None
    """Share of rows holding the most common value above which a column is flagged ``numeric.near_constant``. Null means 0.9."""
    scale_orders_of_magnitude: Count | None = None
    """Orders of magnitude a column's values must span to be flagged ``numeric.scale_anomaly``. Null means 3."""
    bimodal_dip_p_value_threshold: Ratio | None = None
    """Dip-test p-value below which a column is tested further for ``numeric.bimodal``. Null means 0.05."""
    bimodal_min_separation_threshold: NonNegative | None = None
    """Least separation (Ashman's D) between two modes for ``numeric.bimodal``. Null means 2.0."""
    bimodal_min_component_weight: Ratio | None = None
    """Least share of rows in the smaller mode for ``numeric.bimodal``. Null means 0.05."""
    tail_asymmetry_right_share_threshold: Ratio | None = None
    """Right-tail share above which a column's tails are ``right_heavy``. Must exceed ``tail_asymmetry_left_share_threshold``. Null means 0.6666666666666666."""
    tail_asymmetry_left_share_threshold: Ratio | None = None
    """Right-tail share below which a column's tails are ``left_heavy``. Null means 0.3333333333333333."""
    outlier_sigma_threshold: NonNegative | None = None
    """Standard deviations from the mean beyond which a value counts as an outlier. Null means 3.0."""
    high_outlier_density_threshold: Ratio | None = None
    """Share of outlier values above which a column is flagged ``numeric.high_outlier_density``. Null means 0.05."""


class CategoricalDials(WireRecord):
    """Dials for categorical columns. Null keeps every default in this group."""

    rare_threshold_pct: Ratio | None = None
    """Share of rows below which a category counts as rare. Null means 0.01."""
    stratification_rare_threshold_pct: Ratio | None = None
    """Share of rows below which a category counts as a rare label the stratified split protects. Null means 0.05."""
    mixed_type_min_minor_pct: Ratio | None = None
    """Least share (lower confidence bound) of the minority value type, numbers or text, for ``categorical.mixed_type``. Null means 0.05."""
    near_constant_threshold: Ratio | None = None
    """Share of rows holding the most common value above which a column is flagged ``categorical.near_constant``. Null means 0.9."""


class CorrelationDials(WireRecord):
    """Dials for associations between columns. Null keeps every default in this group."""

    near_redundant_pearson_threshold: Ratio | None = None
    """Absolute Pearson or Spearman correlation above which two numeric columns are near-redundant. Null means 0.95."""
    near_redundant_cramer_v_threshold: Ratio | None = None
    """Cramér's V above which two categorical columns are near-redundant. Null means 0.8."""
    near_redundant_eta_squared_threshold: Ratio | None = None
    """η² above which a numeric and a categorical column are near-redundant. Null means 0.5."""


class DatetimeDials(WireRecord):
    """Dials for datetime columns. Null keeps every default in this group."""

    mnar_null_ratio_threshold: Ratio | None = None
    """Null ratio, unparseable values included, above which a column is flagged ``datetime.mnar_suspected``. Null means 0.05."""
    high_gap_cv_threshold: NonNegative | None = None
    """Coefficient of variation of the gaps between dates above which a column is flagged ``datetime.high_gap_variance``. Null means 1.0."""
    recent_window_fraction: Annotated[float, Field(gt=0, le=1)] | None = None
    """Latest share of the date range checked for ``datetime.recent_date_missing``. Null means 0.1."""


class NonlinearityDials(WireRecord):
    """Dials for the nonlinearity probes on numeric columns. Null keeps every default in this group."""

    spearman_pearson_discrepancy_threshold: Annotated[float, Field(ge=0, le=2)] | None = None
    """Largest Spearman-minus-Pearson gap over predictors above which a column reads as monotonic nonlinear. Null means 0.1."""
    mutual_information_threshold: NonNegative | None = None
    """Mean excess mutual information, in nats, above which a column reads as complex nonlinear. Null means 0.05."""
    r2_gap_threshold: NonNegative | None = None
    """Random-forest R² minus linear R² above which a column reads as complex nonlinear. Null means 0.1."""
    heteroscedasticity_p_value_threshold: Ratio | None = None
    """Breusch-Pagan p-value below which a column's residuals count as heteroscedastic. Null means 0.05."""
    r2_rf_unpredictable_threshold: Annotated[float, Field(le=1)] | None = None
    """Random-forest R² below which a column reads as unpredictable. Null means 0.05."""
    min_rows: Annotated[int, Field(ge=2)] | None = None
    """Fewest complete rows, across a column and its predictors, to probe it. Null means 20."""
    bootstrap_sample_size: Annotated[int, Field(ge=2)] | None = None
    """Most rows sampled for the R² probes. Null means 500."""


class ProfileThresholds(WireRecord):
    """Every profiling dial, grouped. Null keeps every default."""

    missingness: MissingnessDials | None = None
    """Missing-value severity and the missingness flags."""
    type_detection: TypeDetectionDials | None = None
    """How each column's semantic type is detected."""
    numeric: NumericDials | None = None
    """Skew, kurtosis, tails, outliers and the numeric flags."""
    categorical: CategoricalDials | None = None
    """Rare labels and the categorical flags."""
    correlation: CorrelationDials | None = None
    """Near-redundancy between columns."""
    datetime: DatetimeDials | None = None
    """The datetime flags."""
    nonlinearity: NonlinearityDials | None = None
    """The nonlinearity probes routing reads."""


# --- Result -----------------------------------------------------------------------


class DigestColumn(WireRecord):
    """One column at low resolution. No literal values."""

    column: str
    """Column name, exactly as in the data."""
    role: Literal["feature", "target", "skipped"]
    """``target`` for a declared target, ``skipped`` for a column in ``skip_columns`` (no statistics), else ``feature``."""
    semantic_type: SemanticTypeName | None
    """Detected or declared semantic type; null when skipped."""
    numeric_kind: NumericKindName | None
    """``continuous`` or ``bounded_discrete`` for a numeric column; else null."""
    null_ratio: float | None
    """Share of rows missing, sentinels and blank strings included; null when skipped."""
    severity: Literal["minor", "moderate", "high", "severe"] | None
    """Missing-value severity, graded by the ``thresholds.missingness.severity_*`` dials; null when nothing is missing."""
    cardinality: int | None
    """Number of distinct non-missing values; null when skipped."""
    skew: float | None
    """Skewness of a numeric column; else null."""
    outlier_density: float | None
    """Share of a numeric column's values that are outliers; else null."""
    flags: list[str]
    """Flags raised on the column, each named ``<group>.<flag>``, such as ``numeric.bimodal``. ``flag_dials`` names the dials behind each."""


class TargetSummary(WireRecord):
    """One declared target."""

    column: str
    """Target column name."""
    problem_type: Literal["regression", "binary_classification", "multiclass_classification", "unknown"]
    """The learning problem the target implies."""
    missing_ratio: float
    """Share of rows whose target is missing."""


class FlagDials(WireRecord):
    """The dials that decide one flag. Turn one to change which columns raise the flag."""

    flag: str
    """The flag, as in ``columns[].flags``."""
    dials: Table[DialPointer]
    """Every dial the flag's rule reads, with its resolved value. Empty when no dial decides it, as for ``missingness.fully_null``."""


class ProfileDeclarations(WireRecord):
    """What profiling read from you, including what the frame inherited. Later Operations read these from the profile."""

    semantic_types: Table[SemanticTypeDeclaration]
    """Declared semantic types."""
    numeric_kinds: Table[NumericKindDeclaration]
    """Declared numeric kinds."""
    sentinels: Table[SentinelDeclaration]
    """Declared sentinel values."""
    datetime_formats: Table[DatetimeFormatDeclaration]
    """Declared datetime formats and epoch units."""
    targets: list[str]
    """Declared targets."""
    exclude_columns: list[str]
    """Columns excluded from this and every later phase. They are not profiled."""
    skip_columns: list[str]
    """Columns skipped by profiling only."""


class ProfileResult(ColumnKeyedResult):
    """The profile Handle and the Dataset Digest. ``result.get(name)`` finds one column."""

    profile: str
    """Profile Handle, such as ``profile_7f3a9c2e``. It carries its frame and these declarations."""
    frame: str
    """The frame Handle that was profiled."""
    n_rows: int
    """Number of rows."""
    n_columns: int
    """Number of columns in the frame, excluded ones included."""
    duplicate_rows: int
    """Number of rows that repeat an earlier row exactly."""
    duplicate_ratio: float
    """Share of rows that repeat an earlier row."""
    sparsity: float
    """Share of all cells that are missing."""
    targets: Table[TargetSummary]
    """Declared targets and their problem types."""
    declarations: ProfileDeclarations
    """Your declarations as recorded on the profile."""
    columns: Table[DigestColumn]
    """Profiled and skipped columns: flagged ones first, then dataset order. At most 100; past 100, ``omitted`` counts the rest and names ``describe_columns``."""
    flag_dials: Table[FlagDials]
    """One entry per flag raised on any column, in ``columns`` order of first appearance."""


@operation(effect="create", phase="profile", scales_with_data=True)
def profile(
    frame: str,
    semantic_types: list[SemanticTypeDeclaration] | None = None,
    numeric_kinds: list[NumericKindDeclaration] | None = None,
    sentinels: list[SentinelDeclaration] | None = None,
    datetime_formats: list[DatetimeFormatDeclaration] | None = None,
    targets: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    skip_columns: list[str] | None = None,
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None,
    thresholds: ProfileThresholds | None = None,
) -> ProfileResult:
    """Profile a frame once into a profile Handle and return its Dataset Digest.

    Use when
    --------
    Call this once per frame, right after ``load``. It computes every column
    statistic, association and nonlinearity probe once and stores them in a
    profile Handle, which carries its frame. Every later step reads the
    profile, never the raw data. The Dataset Digest is the dataset at low
    resolution: row and column counts, duplicates, sparsity, targets, your
    declarations, and one row per column with flagged columns first. It holds
    no cell values. Declarations made here, and those the frame inherited,
    are recorded on the profile and passed down to every frame made from it.
    To change a declaration or a dial, call ``profile`` again; that makes a
    new profile Handle and leaves the old one as it was.

    Parameters
    ----------
    frame : str
        A frame Handle, such as ``frame_7f3a9c2e``.
    semantic_types : list of SemanticTypeDeclaration or None
        Semantic types you set instead of detection, one entry per column.
        Null means detect every column.
    numeric_kinds : list of NumericKindDeclaration or None
        Numeric kinds you set instead of detection, one entry per numeric
        column. Null means detect them.
    sentinels : list of SentinelDeclaration or None
        Values that mean "missing" in a column, one entry per column. Null
        means none beyond blank strings and the default text markers.
    datetime_formats : list of DatetimeFormatDeclaration or None
        How to read datetime columns, one entry per column. Null means infer
        each format.
    targets : list of str or None
        Target columns. Null means none.
    exclude_columns : list of str or None
        Columns to leave out of this and every later phase, added to what the
        frame already excludes. Null means none.
    skip_columns : list of str or None
        Columns profiling skips but later phases still see. Null means none.
    seed : int or None
        Seed for every random step: mixture fits, mutual information and the
        nonlinearity probes. Null means a fixed default, so the same call
        gives the same result.
    thresholds : ProfileThresholds or None
        Dials, grouped by what they judge, each with its default and range.
        Null keeps every default.

    Returns
    -------
    ProfileResult
        ``profile`` and ``frame`` Handles, the dataset card (``n_rows``,
        ``n_columns``, ``duplicate_rows``, ``duplicate_ratio``,
        ``sparsity``, ``targets``, ``declarations``), ``columns`` (at most
        100, flagged first) and ``flag_dials``: the dials behind each flag
        raised. ``warnings`` lists the computations that fell back.

    Errors
    ------
    unknown_handle
        ``frame`` names no frame Handle in this Workspace. ``candidates``
        lists frame Handles, newest first.
    wrong_handle_kind
        ``frame`` is a Handle of another kind.
    unknown_column
        A declaration names a column the frame lacks. ``candidates`` lists
        close matches; ``issues`` lists every problem argument.
    duplicate_key
        A declaration list names the same column twice.
    declaration_conflict
        Two declarations contradict: a numeric kind on a column declared
        non-numeric, a target that is excluded or skipped, or a declaration
        that differs from one the frame inherited.
    invalid_dial_value
        Two dials are out of order, such as ``severity_minor`` above
        ``severity_moderate``.
    invalid_argument
        An argument has the wrong type or is out of its range.
    override_coercion_failed
        No value of a column converts to its declared type.
    incompatible_artifact
        Another library format version wrote ``frame``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    load : Make the frame Handle to profile.
    list_handles : Find profile Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e --targets income
    {...,"profile":"profile_1b2c3d4e","frame":"frame_7f3a9c2e","n_rows":6,"n_columns":3,...}
    """
    return ProfileResult.model_validate(
        profile_frame(
            frame=frame,
            semantic_types=semantic_types,
            numeric_kinds=numeric_kinds,
            sentinels=sentinels,
            datetime_formats=datetime_formats,
            targets=targets,
            exclude_columns=exclude_columns,
            skip_columns=skip_columns,
            seed=seed,
            thresholds=thresholds,
        )
    )
