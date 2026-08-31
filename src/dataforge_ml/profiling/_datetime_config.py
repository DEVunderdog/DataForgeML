"""
Result dataclasses and sub-config for datetime column profiling.

Populated by DatetimeProfiler, which is opt-in via
ProfileConfig.datetime_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Optional


class InferredGranularity(StrEnum):
    Yearly = "yearly"
    Monthly = "monthly"
    Weekly = "weekly"
    Daily = "daily"
    Hourly = "hourly"
    Minutely = "minutely"
    Secondly = "secondly"
    Irregular = "irregular"


class DatetimeFlag(StrEnum):
    FutureDates = "future_dates"
    HighGapVariance = "high_gap_variance"
    MnarSuspected = "mnar_suspected"
    RecentDateMissing = "recent_date_missing"
    FormatMismatch = "format_mismatch"


@dataclass
class DatetimeStats:
    """Statistical summary of a single Datetime column.

    Produced by ``DatetimeProfiler`` for each opted-in column.  Stores
    range, gap regularity, and inferred granularity.
    """

    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    date_range_days: Optional[float] = None
    future_date_count: int = 0
    inferred_granularity: Optional[InferredGranularity] = None
    median_gap_seconds: Optional[float] = None
    gap_cv: Optional[float] = None
    flags: list[DatetimeFlag] = field(default_factory=list)

    def has_flag(self, flag: DatetimeFlag) -> bool:
        """Check whether a specific ``DatetimeFlag`` is set on this column.

        Parameters
        ----------
        flag : DatetimeFlag
            The flag to test.

        Returns
        -------
        bool
            ``True`` if ``flag`` is present in :attr:`flags`, ``False``
            otherwise.
        """
        return flag in self.flags

    def to_dict(self) -> dict:
        """Serialise the datetime statistics to a plain dictionary.

        Returns
        -------
        dict
            All fields keyed by field name.  ``inferred_granularity`` is
            serialised as its string value; ``flags`` are serialised as
            their string values.
        """
        return {
            "min_date": self.min_date.isoformat() if self.min_date else None,
            "max_date": self.max_date.isoformat() if self.max_date else None,
            "date_range_days": self.date_range_days,
            "future_date_count": self.future_date_count,
            "inferred_granularity": str(self.inferred_granularity) if self.inferred_granularity else None,
            "median_gap_seconds": self.median_gap_seconds,
            "gap_cv": self.gap_cv,
            "flags": [str(f) for f in self.flags],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatetimeStats":
        """Reconstruct the datetime statistics from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        DatetimeStats
            Reconstructed instance.
        """
        raw_granularity = data.get("inferred_granularity")
        raw_min_date = data.get("min_date")
        raw_max_date = data.get("max_date")
        return cls(
            min_date=datetime.fromisoformat(raw_min_date) if raw_min_date else None,
            max_date=datetime.fromisoformat(raw_max_date) if raw_max_date else None,
            date_range_days=data.get("date_range_days"),
            future_date_count=data.get("future_date_count", 0),
            inferred_granularity=(
                InferredGranularity(raw_granularity) if raw_granularity else None
            ),
            median_gap_seconds=data.get("median_gap_seconds"),
            gap_cv=data.get("gap_cv"),
            flags=[DatetimeFlag(f) for f in data.get("flags", [])],
        )


# ---------------------------------------------------------------------------
# Markdown rendering helpers (Rendering Contract, ADR-0086)
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if value is None:
        return "not computed"
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_seq(values: list) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


@dataclass
class DatetimeProfileResult:
    """
    Datetime profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, ColumnDatetimeProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, DatetimeStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the datetime profile as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels. Every field of every
        :class:`DatetimeStats` is covered; absent values render a stated
        absence rather than a bare ``None``.

        Returns
        -------
        str
            Markdown document with a summary table followed by one detail
            section per profiled column.
        """
        analysed = ", ".join(self.analysed_columns) if self.analysed_columns else "none"
        lines = ["# Datetime Profile\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| analysed_columns | {analysed} |")
        lines.append("")

        lines.append("| Column | Min date | Max date | Granularity | Flags |")
        lines.append("|---|---|---|---|---|")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(
                    f"| `{name}` | {_fmt(stats.min_date)} | {_fmt(stats.max_date)} "
                    f"| {_fmt(stats.inferred_granularity)} "
                    f"| {_fmt_seq(stats.flags)} |"
                )
        else:
            lines.append("| none | | | | |")
        lines.append("")

        lines.append("## Column Details\n")
        if self.columns:
            for name, stats in self.columns.items():
                lines.append(f"### `{name}`\n")
                lines.append("| Field | Value |")
                lines.append("|---|---|")
                lines.append(f"| min_date | {_fmt(stats.min_date)} |")
                lines.append(f"| max_date | {_fmt(stats.max_date)} |")
                lines.append(f"| date_range_days | {_fmt(stats.date_range_days)} |")
                lines.append(f"| future_date_count | {stats.future_date_count:,} |")
                lines.append(
                    f"| inferred_granularity | {_fmt(stats.inferred_granularity)} |"
                )
                lines.append(
                    f"| median_gap_seconds | {_fmt(stats.median_gap_seconds)} |"
                )
                lines.append(f"| gap_cv | {_fmt(stats.gap_cv)} |")
                lines.append(f"| flags | {_fmt_seq(stats.flags)} |")
                lines.append("")
        else:
            lines.append("none")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Datetime Profile document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


# ---------------------------------------------------------------------------
# Sub-config
# ---------------------------------------------------------------------------


@dataclass
class DatetimeProfileConfig:
    """
    Threshold configuration for the datetime sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``DatetimeProfileConfig()`` produces identical behaviour to
    the pre-config implementation.

    Parameters
    ----------
    mnar_null_ratio_threshold : float
        Null ratio above which a datetime column receives the
        ``DatetimeFlag.MnarSuspected`` flag.  A column whose parse-level null
        ratio (including values that failed datetime coercion) exceeds this
        value is considered potentially Missing Not At Random.
    high_gap_cv_threshold : float
        Coefficient of variation (std / mean) of consecutive time gaps above
        which a column receives the ``DatetimeFlag.HighGapVariance`` flag,
        indicating irregular temporal spacing.
    recent_window_fraction : float
        Fraction of the total date range considered the "recent" window when
        checking for ``DatetimeFlag.RecentDateMissing``.  A value of 0.10
        means the last 10 % of the observed date range is examined for data
        sparsity.
    """

    mnar_null_ratio_threshold: float = 0.05
    high_gap_cv_threshold: float = 1.0
    recent_window_fraction: float = 0.10

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "mnar_null_ratio_threshold": self.mnar_null_ratio_threshold,
            "high_gap_cv_threshold": self.high_gap_cv_threshold,
            "recent_window_fraction": self.recent_window_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DatetimeProfileConfig:
        """
        Construct a ``DatetimeProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        DatetimeProfileConfig
            Reconstructed config instance.
        """
        return cls(
            mnar_null_ratio_threshold=float(
                data.get("mnar_null_ratio_threshold", 0.05)
            ),
            high_gap_cv_threshold=float(data.get("high_gap_cv_threshold", 1.0)),
            recent_window_fraction=float(data.get("recent_window_fraction", 0.10)),
        )
