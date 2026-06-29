"""
Result dataclass for boolean column profiling.

Populated by BooleanProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BooleanStats:
    """Value distribution statistics for a single Boolean column.

    Counts and ratios are computed over the non-missing rows of the column.
    ``true_ratio`` and ``false_ratio`` sum to ``1.0``.  ``mode`` is ``None``
    only when the column contains no non-missing values.
    """

    true_count: int = 0
    false_count: int = 0
    true_ratio: float = 0.0
    false_ratio: float = 0.0
    mode: Optional[bool] = None

    def to_dict(self) -> dict:
        """Serialise the boolean statistics to a plain dictionary.

        Returns
        -------
        dict
            All fields keyed by field name.
        """
        return {
            "true_count": self.true_count,
            "false_count": self.false_count,
            "true_ratio": self.true_ratio,
            "false_ratio": self.false_ratio,
            "mode": self.mode,
        }


@dataclass
class BooleanProfileResult:
    """
    Boolean profile for all eligible columns.

    Attributes
    ----------
    columns : dict[str, BooleanStats]
        Per-column boolean profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection
        and eligibility check).
    """

    columns: dict[str, BooleanStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)
