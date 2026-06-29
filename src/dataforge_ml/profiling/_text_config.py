"""
Result dataclass for free-text column profiling.

Populated by TextProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextStats:
    """Lexical statistics for a single free-text column.

    Token counts are whitespace-split approximations.  Character length
    statistics cover all non-null values.  ``empty_ratio`` and
    ``whitespace_ratio`` are computed over the total row count including
    missing values.
    """

    avg_token_count: float = 0.0
    median_token_count: float = 0.0
    vocabulary_size: int = 0
    char_length_min: int = 0
    char_length_max: int = 0
    char_length_mean: float = 0.0
    char_length_median: float = 0.0
    empty_ratio: float = 0.0
    whitespace_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Serialise the text statistics to a plain dictionary.

        Returns
        -------
        dict
            All fields keyed by field name.
        """
        return {
            "avg_token_count": self.avg_token_count,
            "median_token_count": self.median_token_count,
            "vocabulary_size": self.vocabulary_size,
            "char_length_min": self.char_length_min,
            "char_length_max": self.char_length_max,
            "char_length_mean": self.char_length_mean,
            "char_length_median": self.char_length_median,
            "empty_ratio": self.empty_ratio,
            "whitespace_ratio": self.whitespace_ratio,
        }


@dataclass
class TextProfileResult:
    """
    Text profile for all eligible columns.

    Attributes
    ----------
    columns : dict[str, TextStats]
        Per-column text profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection
        and eligibility check).
    """

    columns: dict[str, TextStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)
