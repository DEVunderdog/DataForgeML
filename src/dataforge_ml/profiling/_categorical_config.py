"""
Result dataclasses for categorical column profiling.

These complement TabularProfileResult and are populated by
CategoricalProfiler, which is opt-in via ProfileConfig.categorical_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Categorical stats dataclasses (canonical home — config.py re-exports these)
# ---------------------------------------------------------------------------


class CategoricalFlag(StrEnum):
    MixedType = "mixed_type"
    FreeText = "free_text"
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

    def to_dict(self) -> dict:
        return {
            "threshold_pct": self.threshold_pct,
            "rare_category_count": self.rare_category_count,
            "total_rare_rows": self.total_rare_rows,
            "rare_row_percentage": self.rare_row_percentage,
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
