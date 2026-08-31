"""
Result dataclasses for categorical column profiling.

These complement TabularProfileResult and are populated by
CategoricalProfiler, which is opt-in via ProfileConfig.categorical_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Sub-config
# ---------------------------------------------------------------------------


@dataclass
class CategoricalProfileConfig:
    """
    Threshold configuration for the categorical column sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``CategoricalProfileConfig()`` produces identical behaviour to
    the pre-config implementation.

    Note: ``near_constant_threshold`` is independent of the equivalent field
    in ``NumericProfileConfig`` — they share a default but are separately
    tunable.

    Parameters
    ----------
    rare_threshold_pct : float
        Fraction of total rows below which a category is counted as rare for
        diagnostic purposes (``RareCategoryStats.rare_category_count``).
    stratification_rare_threshold_pct : float
        Fraction of total rows below which a category is added to
        ``RareCategoryStats.rare_label_values``, used by the stratified
        splitter to protect minority classes.
    mixed_type_min_minor_pct : float
        Minimum Wilson-interval lower bound for the minority type fraction
        required to set ``CategoricalFlag.MixedType``.
    near_constant_threshold : float
        Mode frequency above which a column receives
        ``CategoricalFlag.NearConstant``. Expressed as a fraction of total
        rows (e.g. 0.90 = 90%).
    """

    rare_threshold_pct: float = 0.01
    stratification_rare_threshold_pct: float = 0.05
    mixed_type_min_minor_pct: float = 0.05
    near_constant_threshold: float = 0.90

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "rare_threshold_pct": self.rare_threshold_pct,
            "stratification_rare_threshold_pct": self.stratification_rare_threshold_pct,
            "mixed_type_min_minor_pct": self.mixed_type_min_minor_pct,
            "near_constant_threshold": self.near_constant_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CategoricalProfileConfig:
        """
        Construct a ``CategoricalProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        CategoricalProfileConfig
            Reconstructed config instance.
        """
        return cls(
            rare_threshold_pct=float(data.get("rare_threshold_pct", 0.01)),
            stratification_rare_threshold_pct=float(
                data.get("stratification_rare_threshold_pct", 0.05)
            ),
            mixed_type_min_minor_pct=float(data.get("mixed_type_min_minor_pct", 0.05)),
            near_constant_threshold=float(data.get("near_constant_threshold", 0.90)),
        )

# ---------------------------------------------------------------------------
# Categorical stats dataclasses (canonical home — config.py re-exports these)
# ---------------------------------------------------------------------------


class CategoricalFlag(StrEnum):
    MixedType = "mixed_type"
    NearConstant = "near_constant"


@dataclass
class TopValueEntry:
    """
    A single entry in the top-value frequency table for a categorical column.

    Attributes
    ----------
    value : object
        The category value.
    count : int
        Absolute occurrence count in the column.
    percentage : float
        Fraction of total non-null rows this value represents, in [0, 1].
    """

    value: object
    count: int
    percentage: float

    def to_dict(self) -> dict:
        """
        Serialise this entry to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``value``, ``count``, ``percentage``.
        """
        return {"value": self.value, "count": self.count, "percentage": self.percentage}


@dataclass
class RareCategoryStats:
    """
    Summary of rare categories detected in a categorical column.

    A category is considered rare when its row count falls below
    ``CategoricalProfileConfig.rare_threshold_pct``.  ``rare_label_values``
    uses the stricter ``stratification_rare_threshold_pct`` threshold and
    feeds the stratified splitter to protect minority classes.

    Attributes
    ----------
    threshold_pct : float
        The diagnostic rare threshold that produced this summary.
    rare_category_count : int
        Number of distinct category values below ``threshold_pct``.
    total_rare_rows : int
        Total row count across all rare categories.
    rare_row_percentage : float
        Fraction of total rows occupied by rare categories, in [0, 1].
    rare_label_values : list
        Category values that fall below ``rare_label_threshold_pct``,
        used by the stratified splitter.
    rare_label_threshold_pct : float
        The stratification threshold applied to populate ``rare_label_values``.
    """

    threshold_pct: float
    rare_category_count: int = 0
    total_rare_rows: int = 0
    rare_row_percentage: float = 0.0
    rare_label_values: list = field(default_factory=list)
    rare_label_threshold_pct: float = 0.05

    def to_dict(self) -> dict:
        """
        Serialise this summary to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "threshold_pct": self.threshold_pct,
            "rare_category_count": self.rare_category_count,
            "total_rare_rows": self.total_rare_rows,
            "rare_row_percentage": self.rare_row_percentage,
            "rare_label_values": self.rare_label_values,
            "rare_label_threshold_pct": self.rare_label_threshold_pct,
        }


@dataclass
class ImbalanceMetrics:
    """
    Class-balance statistics for a categorical column.

    Attributes
    ----------
    dominant_class_ratio : float, optional
        Ratio of the most-frequent to second-most-frequent class frequency.
        None when cardinality < 2.
    normalized_shannon_entropy : float, optional
        Shannon entropy of the class distribution scaled to [0, 1] by dividing
        by log2(cardinality). None when cardinality < 2.
    normalized_gini : float, optional
        Gini impurity of the class distribution scaled to [0, 1] by dividing
        by (1 - 1/cardinality). None when cardinality < 2.
    """

    dominant_class_ratio: float | None = None
    normalized_shannon_entropy: float | None = None
    normalized_gini: float | None = None

    def to_dict(self) -> dict:
        """
        Serialise these metrics to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``dominant_class_ratio``, ``normalized_shannon_entropy``, ``normalized_gini``.
        """
        return {
            "dominant_class_ratio": self.dominant_class_ratio,
            "normalized_shannon_entropy": self.normalized_shannon_entropy,
            "normalized_gini": self.normalized_gini,
        }


@dataclass
class CategoricalStats:
    """
    Complete categorical profile for a single column.

    Aggregates cardinality, mode information, rare-category summary,
    imbalance metrics, and diagnostic flags computed by the categorical
    sub-processor.

    Attributes
    ----------
    cardinality : int
        Number of distinct non-null values.
    unique_ratio : float
        Fraction of rows with a unique value (cardinality / row_count).
    mode_frequency : float
        Fraction of rows occupied by the most common value, in [0, 1].
    top_values : list[TopValueEntry]
        Top-k most frequent categories sorted by descending count.
    rare_categories : RareCategoryStats
        Summary of categories occurring below the rare threshold.
    imbalance : ImbalanceMetrics
        Class-balance statistics.
    flags : list[CategoricalFlag]
        Diagnostic flags raised for this column (e.g. ``MixedType``,
        ``NearConstant``).
    """

    cardinality: int = 0
    unique_ratio: float = 0.0
    mode_frequency: float = 0.0
    top_values: list[TopValueEntry] = field(default_factory=list)
    rare_categories: RareCategoryStats = field(
        default_factory=lambda: RareCategoryStats(threshold_pct=0.01),
    )
    imbalance: ImbalanceMetrics = field(default_factory=ImbalanceMetrics)
    flags: list[CategoricalFlag] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialise this profile to a plain dictionary.

        Returns
        -------
        dict
            All fields serialised recursively; nested objects call their own
            ``to_dict`` methods.
        """
        return {
            "cardinality": self.cardinality,
            "unique_ratio": self.unique_ratio,
            "mode_frequency": self.mode_frequency,
            "top_values": [v.to_dict() for v in self.top_values],
            "rare_categories": self.rare_categories.to_dict(),
            "imbalance": self.imbalance.to_dict(),
            "flags": [str(f) for f in self.flags],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CategoricalStats":
        """
        Reconstruct this profile from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        CategoricalStats
            Reconstructed profile instance.
        """
        return cls(
            cardinality=data.get("cardinality", 0),
            unique_ratio=data.get("unique_ratio", 0.0),
            mode_frequency=data.get("mode_frequency", 0.0),
            top_values=[TopValueEntry(**v) for v in data.get("top_values", [])],
            rare_categories=RareCategoryStats(
                **data.get("rare_categories", {"threshold_pct": 0.01})
            ),
            imbalance=ImbalanceMetrics(**data.get("imbalance", {})),
            flags=[CategoricalFlag(f) for f in data.get("flags", [])],
        )


CategoricalColumnProfile = CategoricalStats


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Markdown rendering helpers (Rendering Contract, ADR-0086)
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if value is None:
        return "not computed"
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_seq(values: list) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


def _categorical_stats_lines(stats: "CategoricalStats") -> list[str]:
    """Build the ``####``-rooted body for one column's categorical statistics."""
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| cardinality | {stats.cardinality:,} |",
        f"| unique_ratio | {stats.unique_ratio:.2%} |",
        f"| mode_frequency | {stats.mode_frequency:.2%} |",
        f"| flags | {_fmt_seq(stats.flags)} |",
        "",
        "#### Top Values\n",
    ]
    if stats.top_values:
        lines.append("| Value | Count | Percentage |")
        lines.append("|---|---|---|")
        for entry in stats.top_values:
            lines.append(
                f"| {_fmt(entry.value)} | {entry.count:,} | {entry.percentage:.2%} |"
            )
    else:
        lines.append("none")
    lines.append("")

    lines.append("#### Rare Categories\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    rare = stats.rare_categories
    lines.append(f"| threshold_pct | {_fmt(rare.threshold_pct)} |")
    lines.append(f"| rare_category_count | {rare.rare_category_count:,} |")
    lines.append(f"| total_rare_rows | {rare.total_rare_rows:,} |")
    lines.append(f"| rare_row_percentage | {rare.rare_row_percentage:.2%} |")
    lines.append(f"| rare_label_threshold_pct | {_fmt(rare.rare_label_threshold_pct)} |")
    lines.append(f"| rare_label_values | {_fmt_seq(rare.rare_label_values)} |")
    lines.append("")

    lines.append("#### Imbalance\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key, value in stats.imbalance.to_dict().items():
        lines.append(f"| {key} | {_fmt(value)} |")

    return lines


@dataclass
class CategoricalProfileResult:
    """
    Categorical profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, CategoricalColumnProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, CategoricalStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the categorical profile as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels and places each column's detail beneath
        them. Every field of every :class:`CategoricalStats` is covered; absent
        values render a stated absence rather than a bare ``None``.

        Returns
        -------
        str
            Markdown document with a summary table followed by one detail
            section per profiled column.
        """
        analysed = ", ".join(self.analysed_columns) if self.analysed_columns else "none"
        lines = ["# Categorical Profile\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| analysed_columns | {analysed} |")
        lines.append("")

        lines.append(
            "| Column | Cardinality | Unique ratio | Mode frequency "
            "| Rare categories | Flags |"
        )
        lines.append("|---|---|---|---|---|---|")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(
                    f"| `{name}` | {stats.cardinality:,} "
                    f"| {stats.unique_ratio:.2%} | {stats.mode_frequency:.2%} "
                    f"| {stats.rare_categories.rare_category_count:,} "
                    f"| {_fmt_seq(stats.flags)} |"
                )
        else:
            lines.append("| none | | | | | |")
        lines.append("")

        lines.append("## Column Details\n")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(f"### `{name}`\n")
                lines.extend(_categorical_stats_lines(stats))
                lines.append("")
        else:
            lines.append("none")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Categorical Profile document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
