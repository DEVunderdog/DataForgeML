"""Internal code behind the ``describe_c2st_columns`` Operation (ADR-0087, ADR-0104).

It reads only the stored report of a ``c2st_`` Handle, which
``evaluate_imputation`` wrote whole: every evaluated column with its
per-frame results, and the provenance. Nothing is recomputed. Columns follow
``evaluate_imputation``'s order, flagged first, then dataset order, and are
paged like every detail list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._detail import page, unknown_columns
from ._workspace import read_c2st

#: Most columns one ``describe_c2st_columns`` page holds (ADR-0102).
DESCRIBE_C2ST_CAP = 20


def _score(score: dict[str, Any] | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "mean_frame_z": score["mean_frame_z"],
        "frame_z_spread": score["frame_z_spread"],
        "p_value": score["p_value"],
        "p_adjusted": score["p_adjusted"],
        "frames": [
            {
                "frame": index,
                "mean_repeat_z": frame["mean_repeat_z"],
                "repeat_z": list(frame["repeat_z"]),
                "repeat_z_spread": frame["repeat_z_spread"],
                "train_accuracy": frame["train_accuracy"],
                "test_accuracy": frame["test_accuracy"],
                "n_test": frame["n_test"],
                "degenerate": frame["degenerate"],
                "unseen_category": frame["unseen_category"],
            }
            for index, frame in enumerate(score["frames"])
        ],
    }


def _column(column: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": column["column"],
        "outcome": column["outcome"],
        "verdict": column["verdict"],
        "annotations": list(column["annotations"]),
        "n_observed": column["n_observed"],
        "n_filled": column["n_filled"],
        "score": _score(column["score"]),
    }


def _provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    params = provenance["classifier_params"]
    return {
        "classifier_class": provenance["classifier_class"],
        "classifier_params": [
            {"name": name, "value": params[name]} for name in sorted(params)
        ],
        "sklearn_version": provenance["sklearn_version"],
        "config": dict(provenance["config"]),
    }


def describe_c2st_columns(
    *,
    report: str,
    names: Sequence[str] | None,
    offset: int | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return one page of a C2ST report's columns, with per-frame results and provenance.

    Parameters
    ----------
    report : str
        The c2st Handle.
    names : sequence of str or None
        Only these columns; None means every evaluated column.
    offset, limit : int or None
        The page; None means 0 and :data:`DESCRIBE_C2ST_CAP`.

    Returns
    -------
    dict
        The fields of the ``describe_c2st_columns`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every name the report did not evaluate,
        plus the c2st Handle's own errors.
    """
    stored, _ = read_c2st(report, param="report")
    columns = stored["columns"]
    evaluated = [c["column"] for c in columns]
    names = list(dict.fromkeys(names or []))
    unknown = [(f"names[{i}]", n) for i, n in enumerate(names) if n not in evaluated]
    if unknown:
        raise unknown_columns(
            unknown,
            pool=evaluated,
            excluded=(),
            holder="C2ST report",
            what="evaluated column in the C2ST report",
        )

    chosen = [c for c in columns if not names or c["column"] in names]
    chosen.sort(key=lambda c: c["verdict"] != "flagged")  # stable: flagged first
    cut, offset, omitted = page(
        chosen,
        offset,
        limit,
        DESCRIBE_C2ST_CAP,
        field="columns",
        operation="describe_c2st_columns",
    )
    return {
        "warnings": [],
        "omitted": omitted,
        "report": report,
        "total": len(chosen),
        "offset": offset,
        "columns": [_column(c) for c in cut],
        "provenance": _provenance(stored["provenance"]),
    }
