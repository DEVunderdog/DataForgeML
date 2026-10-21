"""Internal code behind the five profile detail Operations (ADR-0104).

Each detail Operation reads only the stored profile Handle: the profile
object, plus the column order, resolved dials and declarations ``profile``
wrote on the Handle record. None of them reads the frame or recomputes.

Every detail list follows the same rules: filters first, then
``offset``/``limit`` with ``limit`` at most the cap, a stated sort rule, and
an ``omitted`` entry naming the Operation that returns the rest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..records import DataForgeError
from ._detail import page as _page
from ._detail import unknown_columns
from ._profile import FLAG_DIALS, _Dials, _flags
from ._workspace import read_profile

#: Most column profiles one ``describe_columns`` page holds (ADR-0104).
DESCRIBE_CAP = 15
#: Most co-missing columns listed on one column profile; ``missingness`` has the rest.
CO_MISSING_CAP = 10
#: Most pairs one ``top_associations`` page holds (ADR-0104).
ASSOCIATIONS_CAP = 100
#: Most co-missing pairs one ``missingness`` page holds (ADR-0104).
MISSING_PAIRS_CAP = 100
#: Most entries of each kind ``value_summary`` returns (ADR-0104).
VALUES_CAP = 20
#: Most classes ``describe_target`` lists; ``value_summary`` has the rest.
CLASSES_CAP = 20
#: The three association metrics, in tie-break order.
METRICS: tuple[str, ...] = ("pearson", "cramers_v", "eta_squared")
#: The dial that decides ``near_redundant`` for each metric.
_NEAR_REDUNDANT_DIAL = {
    "pearson": "correlation.near_redundant_pearson_threshold",
    "cramers_v": "correlation.near_redundant_cramer_v_threshold",
    "eta_squared": "correlation.near_redundant_eta_squared_threshold",
}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class _Stored:
    """A profile Handle as the detail Operations read it."""

    handle: str
    profile: Any
    columns: list[str]
    dials: _Dials
    targets: list[str]
    skipped: frozenset[str]
    excluded: frozenset[str]
    cardinality: dict[str, int]

    def role(self, column: str) -> str:
        if column in self.targets:
            return "target"
        return "skipped" if column in self.skipped else "feature"

    def position(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.columns)}


def _read(handle: str) -> _Stored:
    profile, record = read_profile(handle, param="profile")
    declarations = record["declarations"]
    return _Stored(
        handle=handle,
        profile=profile,
        columns=list(record["columns"]),
        dials=_Dials.from_record(record),
        targets=list(declarations["targets"]),
        skipped=frozenset(record["skip_columns"]),
        excluded=frozenset(declarations["exclusions"]),
        cardinality=dict(record["cardinality"]),
    )


def _unknown_columns(
    stored: _Stored,
    unknown: Sequence[tuple[str, str]],
    *,
    pool: Sequence[str] | None = None,
    what: str = "column in the profile",
) -> DataForgeError:
    """One ``unknown_column`` error naming every ``(param, name)`` at fault."""
    return unknown_columns(
        unknown,
        pool=list(stored.columns if pool is None else pool),
        excluded=stored.excluded,
        holder="profile",
        what=what,
    )


def _check_column(stored: _Stored, column: str | None, param: str) -> None:
    if column is not None and column not in stored.columns:
        raise _unknown_columns(stored, [(param, column)])


def _flag_dials(stored: _Stored, flags: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"flag": f, "dials": [stored.dials.pointer(p) for p in FLAG_DIALS.get(f, ())]}
        for f in dict.fromkeys(flags)
    ]


# --- describe_columns -------------------------------------------------------------


def _numeric_stats(stats: Any) -> dict[str, Any]:
    bimodal = stats.bimodal_stats
    has_nonlinearity = stats.nonlinearity_tag is not None
    return {
        "kind": "numeric",
        "mean": _finite(stats.mean),
        "std": _finite(stats.std),
        "skew": _finite(stats.skewness),
        "kurtosis": _finite(stats.kurtosis),
        "skew_severity": None
        if stats.skewness_severity is None
        else stats.skewness_severity.value,
        "kurtosis_tag": None
        if stats.kurtosis_tag is None
        else stats.kurtosis_tag.value,
        "mean_median_ratio": _finite(stats.mean_median_ratio),
        "mode_share": _finite(stats.mode_frequency),
        "outlier_density": _finite(stats.outlier_density),
        "tails": (
            None
            if stats.tail_asymmetry_tag is None
            else {
                "tag": stats.tail_asymmetry_tag.value,
                "right_share": _finite(stats.tail_asymmetry_share),
            }
        ),
        "bimodal": (
            None
            if bimodal is None
            else {
                "dip_statistic": _finite(bimodal.dip_statistic),
                "dip_p_value": _finite(bimodal.dip_p_value),
                "separation": _finite(bimodal.cluster_separation),
                "minority_weight": _finite(bimodal.minority_weight),
            }
        ),
        "nonlinearity": (
            None
            if not has_nonlinearity
            else {
                "tag": stats.nonlinearity_tag.value,
                "spearman_pearson_discrepancy": _finite(
                    stats.spearman_pearson_discrepancy
                ),
                "mean_mutual_information": _finite(stats.mean_mutual_information),
                "max_mutual_information": _finite(stats.max_mutual_information),
                "r2_linear": _finite(stats.r2_linear),
                "r2_rf": _finite(stats.r2_rf),
                "r2_gap": _finite(stats.r2_gap),
                "heteroscedasticity_p_value": _finite(stats.heteroscedasticity_p_value),
            }
        ),
    }


def _categorical_stats(stats: Any) -> dict[str, Any]:
    rare = stats.rare_categories
    imbalance = stats.imbalance
    return {
        "kind": "categorical",
        "unique_ratio": _finite(stats.unique_ratio),
        "mode_share": _finite(stats.mode_frequency),
        "rare_category_count": rare.rare_category_count,
        "rare_row_share": _finite(rare.rare_row_percentage),
        "stratification_rare_label_count": len(rare.rare_label_values),
        "dominant_class_ratio": _finite(imbalance.dominant_class_ratio),
        "normalized_entropy": _finite(imbalance.normalized_shannon_entropy),
        "normalized_gini": _finite(imbalance.normalized_gini),
    }


def _stats(column_profile: Any) -> dict[str, Any] | None:
    from ..profiling._boolean_config import BooleanStats
    from ..profiling._categorical_config import CategoricalStats
    from ..profiling._datetime_config import DatetimeStats
    from ..profiling._numeric_config import NumericStats
    from ..profiling._text_config import TextStats

    stats = column_profile.stats
    if isinstance(stats, NumericStats):
        return _numeric_stats(stats)
    if isinstance(stats, CategoricalStats):
        return _categorical_stats(stats)
    if isinstance(stats, DatetimeStats):
        return {
            "kind": "datetime",
            "range_days": _finite(stats.date_range_days),
            "granularity": (
                None
                if stats.inferred_granularity is None
                else stats.inferred_granularity.value
            ),
            "median_gap_seconds": _finite(stats.median_gap_seconds),
            "gap_cv": _finite(stats.gap_cv),
            "future_date_count": stats.future_date_count,
        }
    if isinstance(stats, BooleanStats):
        return {
            "kind": "boolean",
            "true_count": stats.true_count,
            "false_count": stats.false_count,
            "true_share": _finite(stats.true_ratio),
        }
    if isinstance(stats, TextStats):
        return {
            "kind": "text",
            "mean_tokens": _finite(stats.avg_token_count),
            "median_tokens": _finite(stats.median_token_count),
            "vocabulary_size": stats.vocabulary_size,
            "min_chars": stats.char_length_min,
            "max_chars": stats.char_length_max,
            "mean_chars": _finite(stats.char_length_mean),
            "median_chars": _finite(stats.char_length_median),
            "empty_share": _finite(stats.empty_ratio),
            "whitespace_share": _finite(stats.whitespace_ratio),
        }
    return None


def _column_detail(
    stored: _Stored, name: str, index: int, omitted: list[dict[str, Any]]
) -> dict[str, Any]:
    column_profile = stored.profile.columns[name]
    skipped = name in stored.skipped
    missingness = None if skipped else column_profile.missingness
    semantic_type = None if skipped else column_profile.semantic_type
    numeric_kind = column_profile.numeric_kind
    co_missing: list[str] = []
    if missingness is not None:
        co_missing = list(missingness.correlated_with)
        if len(co_missing) > CO_MISSING_CAP:
            omitted.append(
                {
                    "field": f"columns[{index}].missingness.co_missing_with",
                    "shown": CO_MISSING_CAP,
                    "total": len(co_missing),
                    "operation": "missingness",
                }
            )
    return {
        "column": name,
        "role": stored.role(name),
        "semantic_type": None if semantic_type is None else semantic_type.value,
        "numeric_kind": (
            numeric_kind.value
            if numeric_kind is not None
            and semantic_type is not None
            and semantic_type.value == "numeric"
            else None
        ),
        "dtype": column_profile.original_dtype or None,
        "parsed_dtype": None if skipped else column_profile.inferred_dtype or None,
        "type_flags": [] if skipped else [f.value for f in column_profile.type_flags],
        "flags": [] if skipped else _flags(column_profile),
        "cardinality": stored.cardinality.get(name),
        "missingness": (
            None
            if missingness is None
            else {
                "null_count": missingness.effective_null_count,
                "null_ratio": _finite(missingness.effective_null_ratio),
                "standard_null_count": missingness.standard_null_count,
                "severity": None
                if missingness.severity is None
                else missingness.severity.value,
                "co_missing_with": co_missing[:CO_MISSING_CAP],
            }
        ),
        "stats": None if skipped else _stats(column_profile),
    }


def describe_profile_columns(
    *,
    profile: str,
    names: Sequence[str] | None,
    flagged_only: bool | None,
    offset: int | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return one page of full column profiles, flagged first, without literal values.

    Parameters
    ----------
    profile : str
        The profile Handle.
    names : sequence of str or None
        Only these columns; None means every column.
    flagged_only : bool or None
        True keeps only columns that raised a flag.
    offset, limit : int or None
        The page; None means 0 and :data:`DESCRIBE_CAP`.

    Returns
    -------
    dict
        The fields of the ``describe_columns`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every unknown name, plus the profile
        Handle's own errors.
    """
    stored = _read(profile)
    names = list(dict.fromkeys(names or []))
    unknown = [
        (f"names[{i}]", n) for i, n in enumerate(names) if n not in stored.columns
    ]
    if unknown:
        raise _unknown_columns(stored, unknown)

    chosen = [c for c in stored.columns if not names or c in names]
    flags = {
        c: [] if c in stored.skipped else _flags(stored.profile.columns[c])
        for c in chosen
    }
    if flagged_only:
        chosen = [c for c in chosen if flags[c]]
    chosen.sort(key=lambda c: not flags[c])  # stable: flagged first, then dataset order
    page, offset, omitted = _page(
        chosen,
        offset,
        limit,
        DESCRIBE_CAP,
        field="columns",
        operation="describe_columns",
    )
    columns = [_column_detail(stored, c, i, omitted) for i, c in enumerate(page)]
    return {
        "warnings": [],
        "omitted": omitted,
        "profile": profile,
        "total": len(chosen),
        "offset": offset,
        "columns": columns,
        "flag_dials": _flag_dials(stored, [f for c in page for f in flags[c]]),
    }


# --- top_associations -------------------------------------------------------------


def _associations(stored: _Stored) -> list[dict[str, Any]]:
    """Every measured pair, as ``{column_a, column_b, metric, value, near_redundant}``."""
    found: list[dict[str, Any]] = []
    correlation = stored.profile.dataset.feature_correlation
    if correlation is None:
        return found
    for pair in correlation.pairwise:
        found.append(
            {
                "column_a": pair.col_a,
                "column_b": pair.col_b,
                "metric": "pearson",
                "value": _finite(pair.pearson_r),
                "near_redundant": pair.near_redundant,
            }
        )
    for pair in correlation.cramer_v_pairs:
        found.append(
            {
                "column_a": pair.col_a,
                "column_b": pair.col_b,
                "metric": "cramers_v",
                "value": _finite(pair.cramer_v),
                "near_redundant": pair.near_redundant,
            }
        )
    for pair in correlation.eta_squared_pairs:
        found.append(
            {
                "column_a": pair.numeric_col,
                "column_b": pair.categorical_col,
                "metric": "eta_squared",
                "value": _finite(pair.eta_squared),
                "near_redundant": pair.near_redundant,
            }
        )
    return [p for p in found if p["value"] is not None]


def top_profile_associations(
    *,
    profile: str,
    column: str | None,
    metric: str | None,
    min_strength: float | None,
    offset: int | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return one page of associations, strongest first.

    Parameters
    ----------
    profile : str
        The profile Handle.
    column : str or None
        Only pairs with this column, written as ``column_a``.
    metric : str or None
        Only this metric.
    min_strength : float or None
        Only pairs at least this strong.
    offset, limit : int or None
        The page; None means 0 and :data:`ASSOCIATIONS_CAP`.

    Returns
    -------
    dict
        The fields of the ``top_associations`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` for an unknown ``column``, plus the profile
        Handle's own errors.
    """
    stored = _read(profile)
    _check_column(stored, column, "column")
    position = stored.position()
    pairs = _associations(stored)
    if column is not None:
        pairs = [
            p
            if p["column_a"] == column
            else {**p, "column_a": column, "column_b": p["column_a"]}
            for p in pairs
            if column in (p["column_a"], p["column_b"])
        ]
    if metric is not None:
        pairs = [p for p in pairs if p["metric"] == metric]
    if min_strength is not None:
        pairs = [p for p in pairs if abs(p["value"]) >= min_strength]
    pairs.sort(
        key=lambda p: (
            -abs(p["value"]),
            METRICS.index(p["metric"]),
            position.get(p["column_a"], len(position)),
            position.get(p["column_b"], len(position)),
        )
    )
    page, offset, omitted = _page(
        pairs,
        offset,
        limit,
        ASSOCIATIONS_CAP,
        field="pairs",
        operation="top_associations",
    )
    return {
        "warnings": [],
        "omitted": omitted,
        "profile": profile,
        "total": len(pairs),
        "offset": offset,
        "pairs": page,
        "near_redundant_dials": [
            {"metric": m, "dial": stored.dials.pointer(_NEAR_REDUNDANT_DIAL[m])}
            for m in METRICS
        ],
    }


# --- missingness ------------------------------------------------------------------


def _co_missing_pairs(stored: _Stored) -> list[dict[str, Any]]:
    matrix = stored.profile.dataset.missingness_matrix or {}
    threshold = stored.dials.values["missingness.mar_correlation_threshold"]
    columns = [c for c in stored.columns if c in matrix]
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = _finite(matrix[a].get(b))
            if value is None:
                continue
            pairs.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "correlation": value,
                    "mar_suspect": value > threshold,
                }
            )
    return pairs


def profile_missingness(
    *,
    profile: str,
    column: str | None,
    offset: int | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return the row-missingness distribution and one page of co-missing pairs.

    Parameters
    ----------
    profile : str
        The profile Handle.
    column : str or None
        Only pairs with this column, written as ``column_a``.
    offset, limit : int or None
        The page; None means 0 and :data:`MISSING_PAIRS_CAP`.

    Returns
    -------
    dict
        The fields of the ``missingness`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` for an unknown ``column``, plus the profile
        Handle's own errors.
    """
    stored = _read(profile)
    _check_column(stored, column, "column")
    position = stored.position()
    pairs = _co_missing_pairs(stored)
    if column is not None:
        pairs = [
            p
            if p["column_a"] == column
            else {**p, "column_a": column, "column_b": p["column_a"]}
            for p in pairs
            if column in (p["column_a"], p["column_b"])
        ]
    pairs.sort(
        key=lambda p: (
            -p["correlation"],
            position[p["column_a"]],
            position[p["column_b"]],
        )
    )
    page, offset, omitted = _page(
        pairs, offset, limit, MISSING_PAIRS_CAP, field="pairs", operation="missingness"
    )
    rows = stored.profile.dataset.row_distribution
    dataset = stored.profile.dataset
    columns_missing = sum(
        1
        for c in stored.columns
        if c not in stored.skipped
        and (m := stored.profile.columns[c].missingness) is not None
        and m.effective_null_count > 0
    )
    return {
        "warnings": [],
        "omitted": omitted,
        "profile": profile,
        "sparsity": _finite(dataset.overall_sparsity) or 0.0,
        "columns_with_missing": columns_missing,
        "rows": {
            "complete_share": _finite(rows.complete_row_fraction) or 0.0,
            "one_to_two_share": _finite(rows.pct_one_to_two) or 0.0,
            "three_to_five_share": _finite(rows.pct_three_to_five) or 0.0,
            "over_five_share": _finite(rows.pct_over_five) or 0.0,
            "p90_missing_per_row": rows.row_missingness_p90,
            "drop_candidate_rows": rows.drop_candidate_row_count,
            "drop_candidate_share": _finite(rows.pct_over_half_missing) or 0.0,
            "drop_candidate_dial": stored.dials.pointer(
                "missingness.row_drop_threshold"
            ),
        },
        "total": len(pairs),
        "offset": offset,
        "pairs": page,
        "mar_suspect_dial": stored.dials.pointer(
            "missingness.mar_correlation_threshold"
        ),
    }


# --- describe_target --------------------------------------------------------------


def describe_profile_target(*, profile: str, name: str) -> dict[str, Any]:
    """Return one declared target's problem type and statistics.

    Parameters
    ----------
    profile : str
        The profile Handle.
    name : str
        A declared target.

    Returns
    -------
    dict
        The fields of the ``describe_target`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` when ``name`` is not a declared target, plus the
        profile Handle's own errors.
    """
    stored = _read(profile)
    target = stored.profile.targets.get(name)
    if target is None:
        if not stored.targets:
            raise DataForgeError.from_code(
                "unknown_column",
                f"`name`: the profile has no targets, so `{name}` is not one. "
                f"Declare targets with `profile --targets`.",
                param="name",
                columns=[name],
            )
        raise _unknown_columns(
            stored, [("name", name)], pool=stored.targets, what="declared target"
        )

    omitted: list[dict[str, Any]] = []
    classes = None
    distribution = None
    if target.categorical_profile is not None:
        categorical = target.categorical_profile
        entries = list(categorical.top_values)
        cardinality = categorical.cardinality
        if cardinality > min(len(entries), CLASSES_CAP):
            omitted.append(
                {
                    "field": "classes.counts",
                    "shown": min(len(entries), CLASSES_CAP),
                    "total": cardinality,
                    "operation": "value_summary",
                }
            )
        classes = {
            "class_count": cardinality,
            "dominant_class_ratio": _finite(categorical.imbalance.dominant_class_ratio),
            "normalized_entropy": _finite(
                categorical.imbalance.normalized_shannon_entropy
            ),
            "counts": [
                {
                    "label": _scalar(e.value),
                    "count": e.count,
                    "share": _finite(e.percentage) or 0.0,
                }
                for e in entries[:CLASSES_CAP]
            ],
        }
    if target.numeric_profile is not None:
        distribution = _numeric_stats(target.numeric_profile)
    return {
        "warnings": [],
        "omitted": omitted,
        "profile": profile,
        "column": name,
        "problem_type": target.problem_type.value,
        "missing_count": target.missing_count,
        "missing_ratio": _finite(target.missing_ratio) or 0.0,
        "flags": [f.value for f in target.flags],
        "distribution": distribution,
        "classes": classes,
    }


# --- value_summary ----------------------------------------------------------------


def _scalar(value: Any) -> bool | int | float | str | None:
    """A cell value as a JSON scalar: numbers stay numbers, datetimes become ISO text."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _cut(
    values: list[Any],
    k: int,
    field: str,
    total: int,
    omitted: list[dict[str, Any]],
) -> list[Any]:
    # A larger k shows more only while the profile holds more than k; past
    # what the profile stores, no Operation returns the rest yet.
    shown = values[:k]
    if total > len(shown):
        omitted.append(
            {
                "field": field,
                "shown": len(shown),
                "total": total,
                "operation": "value_summary" if len(values) > k else "query_frame",
            }
        )
    return shown


def summarise_values(*, profile: str, column: str, k: int | None) -> dict[str, Any]:
    """Return a column's top values, rare labels, bounds and percentiles.

    Parameters
    ----------
    profile : str
        The profile Handle.
    column : str
        A profiled column.
    k : int or None
        Most entries of each list; None means :data:`VALUES_CAP`.

    Returns
    -------
    dict
        The fields of the ``value_summary`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` for an unknown ``column``, plus the profile
        Handle's own errors.
    """
    from ..profiling._boolean_config import BooleanStats
    from ..profiling._categorical_config import CategoricalStats
    from ..profiling._datetime_config import DatetimeStats
    from ..profiling._numeric_config import NumericStats

    stored = _read(profile)
    _check_column(stored, column, "column")
    k = VALUES_CAP if k is None else k
    column_profile = stored.profile.columns[column]
    stats = None if column in stored.skipped else column_profile.stats
    omitted: list[dict[str, Any]] = []
    out: dict[str, Any] = {
        "warnings": [],
        "omitted": omitted,
        "profile": profile,
        "column": column,
        "semantic_type": (
            None
            if column in stored.skipped or column_profile.semantic_type is None
            else column_profile.semantic_type.value
        ),
        "cardinality": stored.cardinality.get(column),
        "top_values": [],
        "rare_labels": [],
        "bounds": None,
        "median": None,
        "mode": None,
        "percentiles": [],
    }
    cardinality = stored.cardinality.get(column) or 0

    def counts(
        entries: Sequence[Any], value: Callable[[Any], Any]
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "value": value(e.value),
                "count": e.count,
                "share": _finite(e.percentage) or 0.0,
            }
            for e in entries
        ]
        if not rows:  # a continuous column keeps no top values
            return []
        return _cut(rows, k, "top_values", max(cardinality, len(rows)), omitted)

    if isinstance(stats, NumericStats):
        p = stats.percentiles
        out["top_values"] = counts(stats.top_values, _scalar)
        if stats.min is not None or stats.max is not None:
            out["bounds"] = {"min": _scalar(stats.min), "max": _scalar(stats.max)}
        out["median"] = _scalar(stats.median)
        out["mode"] = _scalar(stats.mode)
        out["percentiles"] = [
            {"percentile": q, "value": _scalar(getattr(p, f"p{q}"))}
            for q in (1, 5, 25, 50, 75, 95, 99)
            if getattr(p, f"p{q}") is not None
        ][:k]
    elif isinstance(stats, CategoricalStats):
        out["top_values"] = counts(stats.top_values, _scalar)
        rare = [_scalar(v) for v in stats.rare_categories.rare_label_values]
        out["rare_labels"] = _cut(rare, k, "rare_labels", len(rare), omitted)
        out["mode"] = _scalar(stats.top_values[0].value) if stats.top_values else None
    elif isinstance(stats, DatetimeStats):
        if stats.min_date is not None or stats.max_date is not None:
            out["bounds"] = {
                "min": _scalar(stats.min_date),
                "max": _scalar(stats.max_date),
            }
    elif isinstance(stats, BooleanStats):
        n = stats.true_count + stats.false_count
        rows = [
            {"value": v, "count": c, "share": _finite(c / n) if n else 0.0}
            for v, c in sorted(
                ((True, stats.true_count), (False, stats.false_count)),
                key=lambda e: -e[1],
            )
            if c
        ]
        out["top_values"] = rows[:k]
        out["mode"] = stats.mode
    return out
