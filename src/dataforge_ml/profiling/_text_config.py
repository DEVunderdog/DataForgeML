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

    @classmethod
    def from_dict(cls, data: dict) -> "TextStats":
        """Reconstruct the text statistics from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        TextStats
            Reconstructed instance.
        """
        return cls(
            avg_token_count=data.get("avg_token_count", 0.0),
            median_token_count=data.get("median_token_count", 0.0),
            vocabulary_size=data.get("vocabulary_size", 0),
            char_length_min=data.get("char_length_min", 0),
            char_length_max=data.get("char_length_max", 0),
            char_length_mean=data.get("char_length_mean", 0.0),
            char_length_median=data.get("char_length_median", 0.0),
            empty_ratio=data.get("empty_ratio", 0.0),
            whitespace_ratio=data.get("whitespace_ratio", 0.0),
        )


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

    def to_markdown(self) -> str:
        """Render the text profile as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels. Every field of the result is covered;
        empty collections render a stated absence rather than a bare ``None``.

        Returns
        -------
        str
            Markdown document with the analysed column scope, a token/vocabulary
            table and a character-length table, one row per profiled column.
        """
        lines = ["# Text Profile\n"]

        lines.append("## Summary\n")
        analysed = ", ".join(self.analysed_columns) if self.analysed_columns else "none"
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| analysed_columns | {analysed} |")
        lines.append("")

        lines.append("## Tokens and Vocabulary\n")
        lines.append(
            "| Column | Avg tokens | Median tokens | Vocabulary size "
            "| Empty % | Whitespace % |"
        )
        lines.append("|---|---|---|---|---|---|")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(
                    f"| `{name}` | {stats.avg_token_count:.2f} "
                    f"| {stats.median_token_count:.2f} | {stats.vocabulary_size:,} "
                    f"| {stats.empty_ratio:.2%} | {stats.whitespace_ratio:.2%} |"
                )
        else:
            lines.append("| none | | | | | |")
        lines.append("")

        lines.append("## Character Length\n")
        lines.append("| Column | Min | Max | Mean | Median |")
        lines.append("|---|---|---|---|---|")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(
                    f"| `{name}` | {stats.char_length_min:,} "
                    f"| {stats.char_length_max:,} | {stats.char_length_mean:.2f} "
                    f"| {stats.char_length_median:.2f} |"
                )
        else:
            lines.append("| none | | | | |")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Text Profile document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
