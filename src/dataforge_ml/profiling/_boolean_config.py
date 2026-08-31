"""
Result dataclass for boolean column profiling.

Populated by BooleanProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class BooleanFlag(StrEnum):
    FormatMismatch = "format_mismatch"


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
    flags: list[BooleanFlag] = field(default_factory=list)

    def has_flag(self, flag: BooleanFlag) -> bool:
        """Check whether a specific ``BooleanFlag`` is set on this column.

        Parameters
        ----------
        flag : BooleanFlag
            The flag to test.

        Returns
        -------
        bool
            ``True`` if ``flag`` is present in :attr:`flags`, ``False``
            otherwise.
        """
        return flag in self.flags

    def to_dict(self) -> dict:
        """Serialise the boolean statistics to a plain dictionary.

        Returns
        -------
        dict
            All fields keyed by field name.  ``flags`` are serialised as their
            string values.
        """
        return {
            "true_count": self.true_count,
            "false_count": self.false_count,
            "true_ratio": self.true_ratio,
            "false_ratio": self.false_ratio,
            "mode": self.mode,
            "flags": [str(f) for f in self.flags],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BooleanStats":
        """Reconstruct the boolean statistics from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        BooleanStats
            Reconstructed instance.
        """
        return cls(
            true_count=data.get("true_count", 0),
            false_count=data.get("false_count", 0),
            true_ratio=data.get("true_ratio", 0.0),
            false_ratio=data.get("false_ratio", 0.0),
            mode=data.get("mode"),
            flags=[BooleanFlag(f) for f in data.get("flags", [])],
        )


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

    def to_markdown(self) -> str:
        """Render the boolean profile as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels. Every field of the result is covered;
        empty collections render a stated absence rather than a bare ``None``.

        Returns
        -------
        str
            Markdown document with the analysed column scope and one row of
            value-distribution statistics per profiled column.
        """
        lines = ["# Boolean Profile\n"]

        lines.append("## Summary\n")
        analysed = ", ".join(self.analysed_columns) if self.analysed_columns else "none"
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| analysed_columns | {analysed} |")
        lines.append("")

        lines.append("## Columns\n")
        lines.append(
            "| Column | True count | True % | False count | False % | Mode | Flags |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        if self.columns:
            for name, stats in self.columns.items():
                mode = "not computed" if stats.mode is None else str(stats.mode)
                flags = (
                    ", ".join(str(f) for f in stats.flags) if stats.flags else "none"
                )
                lines.append(
                    f"| `{name}` | {stats.true_count:,} | {stats.true_ratio:.2%} "
                    f"| {stats.false_count:,} | {stats.false_ratio:.2%} "
                    f"| {mode} | {flags} |"
                )
        else:
            lines.append("| none | | | | | | |")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Boolean Profile document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
