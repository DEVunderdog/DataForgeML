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
    value: object
    count: int
    percentage: float

    def to_dict(self) -> dict:
        return {"value": self.value, "count": self.count, "percentage": self.percentage}


@dataclass
class RareCategoryStats:
    threshold_pct: float
    rare_category_count: int = 0
    total_rare_rows: int = 0
    rare_row_percentage: float = 0.0
    rare_label_values: list = field(default_factory=list)
    rare_label_threshold_pct: float = 0.05

    def to_dict(self) -> dict:
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
    class_ratio: float = 0.0
    shannon_entropy: float = 0.0
    gini_impurity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "class_ratio": self.class_ratio,
            "shannon_entropy": self.shannon_entropy,
            "gini_impurity": self.gini_impurity,
        }


@dataclass
class CategoricalStats:
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
        return {
            "cardinality": self.cardinality,
            "unique_ratio": self.unique_ratio,
            "mode_frequency": self.mode_frequency,
            "top_values": [v.to_dict() for v in self.top_values],
            "rare_categories": self.rare_categories.to_dict(),
            "imbalance": self.imbalance.to_dict(),
            "flags": [str(f) for f in self.flags],
        }


CategoricalColumnProfile = CategoricalStats


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


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

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Categorical Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)
