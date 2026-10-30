"""Internal code behind the ``evaluate_imputation`` Operation (ADR-0087, ADR-0110).

It reads a profile Handle and a frame Handle ``impute`` made, scores the
imputed frame against the frame it was filled from with the imputation C2ST,
writes the whole report as a ``c2st_`` Handle and returns a bounded summary:
the outcome census, each column's outcome and score, flagged columns first,
and the dials that decided each threshold outcome (ADR-0104, ADR-0106).

The original frame is the imputed frame's own ``source_frame``, so the two
are row-aligned whatever frame the profile was made from: a fold filled by an
imputer routed on the whole frame is scored with that frame's profile.
Columns the imputed frame lacks (``dropped`` ones) and every column a
declaration excludes are left out of the census; the adapter resolves the
rest. Outcomes are per-column data only, never copied into warnings.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from threadpoolctl import threadpool_limits

from ..records import DataForgeError
from ._detail import page
from ._operation import progress_observer
from ._workspace import read_frame, read_profile, read_record, settings, write_handle

#: The Operation dial paths are relative to.
OPERATION = "evaluate_imputation"
#: Most per-column records one result lists (ADR-0102).
COLUMNS_CAP = 100

#: Each threshold-decided outcome, verdict or annotation, as in ``columns``,
#: to the dials that decide it, as paths under ``thresholds``.
OUTCOME_DIALS: dict[str, tuple[str, ...]] = {
    "below_sample_floor": ("min_filled_cells",),
    "uninformative": ("min_samples_leaf",),
    "flagged": ("fdr_alpha",),
    "not_flagged": ("fdr_alpha",),
    "low_power": ("low_power_below",),
    "imbalanced_piles": ("imbalance_warn_ratio",),
    "degenerate": ("min_samples_leaf",),
}


def dial_defaults() -> dict[str, Any]:
    """Return every ``evaluate_imputation`` dial's default, keyed by its name under ``thresholds``.

    Returns
    -------
    dict
        ``min_filled_cells``, ``min_samples_leaf``, ``low_power_below``,
        ``imbalance_warn_ratio`` and ``fdr_alpha`` to their library defaults.
    """
    from ..evaluation._config import C2STConfig

    config = C2STConfig()
    return {
        "min_filled_cells": config.min_filled_cells,
        "min_samples_leaf": config.min_samples_leaf,
        "low_power_below": config.low_power_below,
        "imbalance_warn_ratio": config.imbalance_warn_ratio,
        "fdr_alpha": config.fdr_alpha,
    }


def _pointer(dial: str, values: dict[str, Any], given: set[str]) -> dict[str, Any]:
    return {
        "operation": OPERATION,
        "dial": f"thresholds.{dial}",
        "value": values[dial],
        "source": "argument" if dial in given else "default",
    }


def _facts(record: Any) -> dict[str, Any]:
    # A column's outcome and pile sizes, as both the stored report and the
    # result row hold them.
    return {
        "column": record.column,
        "outcome": str(record.outcome),
        "verdict": str(record.verdict),
        "annotations": [str(a) for a in record.annotations],
        "n_observed": record.n_observed,
        "n_filled": record.n_filled,
    }


def _stored(report: Any) -> dict[str, Any]:
    # The whole report as JSON: every column with its per-frame results, and
    # the provenance. ``describe_c2st_columns`` reads it back.
    def column(record: Any) -> dict[str, Any]:
        score = record.score
        return {
            **_facts(record),
            "score": None
            if score is None
            else {
                "mean_frame_z": score.mean_frame_z,
                "frame_z_spread": score.frame_z_spread,
                "p_value": score.p_value,
                "p_adjusted": score.p_adjusted,
                "frames": [
                    {**asdict(frame), "repeat_z": list(frame.repeat_z)}
                    for frame in score.frames
                ],
            },
        }

    provenance = report.provenance
    return {
        "columns": [column(record) for record in report.columns.values()],
        "provenance": {
            "classifier_class": provenance.classifier_class,
            "classifier_params": dict(provenance.classifier_params),
            "sklearn_version": provenance.sklearn_version,
            "config": provenance.config.to_dict(),
        },
    }


def _row(record: Any) -> dict[str, Any]:
    score = record.score
    return {
        **_facts(record),
        "mean_frame_z": None if score is None else score.mean_frame_z,
        "p_adjusted": None if score is None else score.p_adjusted,
    }


def _census(report: Any) -> list[dict[str, Any]]:
    # Columns per (outcome, verdict) pair, in the vocabularies' own order.
    from ..evaluation.imputation._records import C2STOutcome, C2STVerdict

    counts: dict[tuple[str, str], int] = {}
    for record in report.columns.values():
        key = (str(record.outcome), str(record.verdict))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"outcome": str(outcome), "verdict": str(verdict), "count": counts[key]}
        for outcome in C2STOutcome
        for verdict in C2STVerdict
        if (key := (str(outcome), str(verdict))) in counts
    ]


def evaluate_imputation(
    *, profile: str, imputed: str, seed: int | None, thresholds: Any
) -> dict[str, Any]:
    """Score an imputed frame Handle with the imputation C2ST and build the result.

    Parameters
    ----------
    profile : str
        The profile Handle the imputation was routed on.
    imputed : str
        A frame Handle ``impute`` made.
    seed : int or None
        The seed; None means the fixed default.
    thresholds : EvaluateImputationThresholds or None
        The dials record.

    Returns
    -------
    dict
        The fields of the ``evaluate_imputation`` result.

    Raises
    ------
    DataForgeError
        ``invalid_argument`` when ``imputed`` was not made by ``impute``; plus
        each Handle's own errors.
    """
    from .._seed import DEFAULT_SEED
    from ..evaluation._config import C2STConfig
    from ..evaluation.imputation._adapter import imputation_score_c2st
    from ..evaluation.imputation._records import C2STVerdict

    stored_profile, profile_record = read_profile(profile, param="profile")
    imputed_frame, imputed_declared = read_frame(imputed, param="imputed")
    # ``split`` stamps ``source_frame`` on its frames too; only ``impute``
    # stamps the imputer.
    imputed_record = read_record(imputed, "frame", param="imputed")
    source = imputed_record.get("source_frame")
    if "imputer" not in imputed_record:
        raise DataForgeError.from_code(
            "invalid_argument",
            "`imputed` must be a frame Handle `impute` made, so the frame it was "
            "filled from is known; this one was not.",
            param="imputed",
        )
    original, _ = read_frame(source, param="imputed")

    excluded = {
        *imputed_declared.exclusions,
        *imputed_declared.soft_exclusions,
        *profile_record["declarations"]["exclusions"],
        *(c for c in original.columns if c not in imputed_frame.columns),
    }

    values = dial_defaults()
    given: set[str] = set()
    if thresholds is not None:
        for name, value in thresholds:
            if value is not None:
                values[name] = value
                given.add(name)
    seed = DEFAULT_SEED if seed is None else seed
    # The C2ST classifier's OpenMP threads, capped as in ``c2st`` (ADR-0081).
    with threadpool_limits(limits=settings().core_budget, user_api="openmp"):
        report = imputation_score_c2st(
            original,
            [imputed_frame],
            stored_profile,
            exclude_columns=sorted(excluded),
            config=C2STConfig(seed=seed, **values),
            observer=progress_observer(),
        )

    stored = _stored(report)
    handle = write_handle(
        "c2st",
        size={"columns": len(report)},
        payload_name="report.json",
        write_payload=lambda tmp: tmp.write_text(json.dumps(stored), "utf-8"),
        obj=report,
        extra={
            "profile": profile,
            "imputed": imputed,
            "source_frame": source,
            "seed": seed,
            "dials": {"values": values, "set_by_caller": sorted(given)},
        },
    )

    records = sorted(
        report.columns.values(), key=lambda r: r.verdict is not C2STVerdict.Flagged
    )  # stable: flagged first, then dataset order
    rows, _, omitted = page(
        [_row(r) for r in records],
        None,
        None,
        COLUMNS_CAP,
        field="columns",
        operation="describe_c2st_columns",
    )
    present = {
        name
        for record in report.columns.values()
        for name in (str(record.outcome), str(record.verdict), *map(str, record.annotations))
    }
    return {
        "warnings": [],
        "omitted": omitted,
        "c2st": handle,
        "profile": profile,
        "imputed": imputed,
        "source_frame": source,
        "rows": imputed_frame.height,
        "seed": seed,
        "census": _census(report),
        "columns": rows,
        "outcome_dials": [
            {"outcome": name, "dials": [_pointer(d, values, given) for d in dials]}
            for name, dials in OUTCOME_DIALS.items()
            if name in present
        ],
    }
