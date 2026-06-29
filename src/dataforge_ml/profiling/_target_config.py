"""
Configuration and result dataclasses for Target Variable profiling.

Determines the nature of the predictive task (Regression vs Classification)
and flags critical issues like missing labels or severe imbalances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from ._categorical_config import CategoricalColumnProfile
from ._numeric_config import ColumnNumericProfile

class TargetProblemType(StrEnum):
    Regression = "regression"
    BinaryClassification = "binary_classification"
    MulticlassClassification = "multiclass_classification"
    Unknown = "unknown"

class TargetFlag(StrEnum):
    ContainsMissing = "contains_missing"      # Target has >0 missing values; must drop or reframe
    HighImbalance = "high_imbalance"          # Class ratio > 5 (requires handling in Phase 5)
    SevereImbalance = "severe_imbalance"      # Class ratio > 20 (accuracy metric is meaningless)
    HighlySkewed = "highly_skewed"            # Numeric target is severely skewed (consider log transform)
    IsIdentifier = "is_identifier"            # Target looks like an ID column (useless for modeling)

@dataclass
class TargetProfileResult:
    """Profile specific to the designated target variable.

    Captures the inferred ``TargetProblemType``, missingness, and any
    ``TargetFlag`` signals (imbalance, skew, identifier-like behaviour).
    Delegates detailed statistics to ``ColumnNumericProfile`` for
    ``Regression`` targets and to ``CategoricalColumnProfile`` for
    classification targets.
    """

    column: str
    problem_type: TargetProblemType

    # Missingness (Critical for targets)
    missing_count: int = 0
    missing_ratio: float = 0.0

    # Underlying profile data depending on the problem type
    numeric_profile: Optional[ColumnNumericProfile] = None
    categorical_profile: Optional[CategoricalColumnProfile] = None

    flags: list[TargetFlag] = field(default_factory=list)

    def has_flag(self, flag: TargetFlag) -> bool:
        """Check whether a specific ``TargetFlag`` is set on the target column.

        Parameters
        ----------
        flag : TargetFlag
            The flag to test.

        Returns
        -------
        bool
            ``True`` if ``flag`` is present in :attr:`flags`, ``False``
            otherwise.
        """
        return flag in self.flags

    def to_dict(self) -> dict:
        """Serialise the target profile to a plain dictionary.

        Returns
        -------
        dict
            All fields keyed by field name.  ``problem_type`` is serialised
            as its string value; ``numeric_profile`` and
            ``categorical_profile`` are expanded via their own ``to_dict()``
            methods when present; ``flags`` are serialised as their string
            values.
        """
        return {
            "column": self.column,
            "problem_type": str(self.problem_type),
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "numeric_profile": self.numeric_profile.to_dict() if self.numeric_profile else None,
            "categorical_profile": self.categorical_profile.to_dict() if self.categorical_profile else None,
            "flags": [str(f) for f in self.flags],
        }

    def __str__(self) -> str:
        lines = [
            "=== Target Variable Profile ===",
            f"  Column        : {self.column}",
            f"  Problem Type  : {self.problem_type}",
            f"  Missingness   : {self.missing_count:,} rows ({self.missing_ratio:.2%})",
        ]
        
        if self.has_flag(TargetFlag.ContainsMissing):
            lines.append("    [!] WARNING: Target contains missing values. Imputation is not recommended.")

        if self.categorical_profile and self.problem_type in (TargetProblemType.BinaryClassification, TargetProblemType.MulticlassClassification):
            im = self.categorical_profile.imbalance
            lines.append(f"  Classes       : {self.categorical_profile.cardinality:,}")
            lines.append(f"  Class Ratio   : {im.class_ratio:.2f}")
            lines.append(f"  Gini Impurity : {im.gini_impurity:.4f}")
            
        if self.numeric_profile and self.problem_type == TargetProblemType.Regression:
            lines.append(f"  Mean / Median : {self.numeric_profile.mean:.4f} / {self.numeric_profile.median:.4f}")
            lines.append(f"  Skewness      : {self.numeric_profile.skewness:.4f} [{self.numeric_profile.skew_severity}]")

        if self.flags:
            lines.append(f"  Flags         : {', '.join(self.flags)}")
            
        return "\n".join(lines)