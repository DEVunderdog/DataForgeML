"""Internal code behind the ``c2st`` Operation (ADR-0087, ADR-0110).

It reads two frame Handles, leaves out every column either frame declares
excluded or soft-excluded, checks the rest before any work, and runs the
generic Classifier Two-Sample Test on them. The result is returned by value:
no Handle is written.

Both frames must then have the same columns; their order may differ, since
the candidate is read in the reference's order. Dtype is the whole type
contract, as in the generic test: a ``pl.String`` column raises
``c2st_string_column``, and the Agent leaves it out by declaring it excluded
on ``profile``.
"""

from __future__ import annotations

from typing import Any

from threadpoolctl import threadpool_limits

from ..records import DataForgeError
from ._operation import progress_observer
from ._workspace import read_frame, settings

#: The stage name the repeat progress is stamped with.
OPERATION = "c2st"


def dial_defaults() -> dict[str, Any]:
    """Return every ``c2st`` dial's default, keyed by its name under ``thresholds``.

    Returns
    -------
    dict
        ``min_rows`` and ``min_samples_leaf`` to their library defaults.
    """
    from ..evaluation._c2st import DEFAULT_MIN_SAMPLES_LEAF, MIN_SAMPLE_FLOOR

    return {"min_rows": MIN_SAMPLE_FLOOR, "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF}


def _tested_columns(
    reference: Any, candidate: Any, skipped: set[str]
) -> tuple[list[str], list[str]]:
    # The reference's columns, in its order, that no declaration skips, and
    # every such column only one frame has.
    columns = [c for c in reference.columns if c not in skipped]
    others = {c for c in candidate.columns if c not in skipped}
    return columns, sorted(set(columns) ^ others)


def run_c2st(
    *,
    reference: str,
    candidate: str,
    repeats: int | None,
    scheme: str | None,
    seed: int | None,
    thresholds: Any,
) -> dict[str, Any]:
    """Run C2ST over two frame Handles and build the ``c2st`` result.

    Parameters
    ----------
    reference : str
        The reference frame Handle.
    candidate : str
        The candidate frame Handle.
    repeats : int or None
        Balanced subsampling repeats; None means the library default.
    scheme : str or None
        ``split`` or ``cross_validation``; None means ``split``.
    seed : int or None
        The seed; None means the fixed default.
    thresholds : C2STThresholds or None
        The dials record.

    Returns
    -------
    dict
        The fields of the ``c2st`` result.

    Raises
    ------
    DataForgeError
        ``frames_not_aligned`` when the frames' tested columns differ;
        ``invalid_argument`` when no column is left to test;
        ``c2st_string_column`` and ``c2st_sample_floor`` from the test; plus
        each frame Handle's own errors.
    """
    from .._seed import DEFAULT_SEED
    from ..evaluation._c2st import C2STScheme, _run_c2st
    from ..evaluation._config import C2STConfig
    from ..observability import Emitter

    reference_frame, reference_declared = read_frame(reference, param="reference")
    candidate_frame, candidate_declared = read_frame(candidate, param="candidate")
    skipped = {
        *reference_declared.exclusions,
        *reference_declared.soft_exclusions,
        *candidate_declared.exclusions,
        *candidate_declared.soft_exclusions,
    }
    columns, differing = _tested_columns(reference_frame, candidate_frame, skipped)
    if differing:
        raise DataForgeError.from_code(
            "frames_not_aligned",
            f"`reference` and `candidate` must have the same columns to test; "
            f"{len(differing)} column(s) are in only one of them. Exclude them on "
            f"`profile`, or compare frames from the same source.",
            columns=differing,
        )
    if not columns:
        raise DataForgeError.from_code(
            "invalid_argument",
            "`reference`: every column is excluded, so there is nothing to test.",
            param="reference",
        )
    n_skipped = reference_frame.width - len(columns)
    reference_frame = reference_frame.select(columns)
    candidate_frame = candidate_frame.select(columns)

    dials = dial_defaults()
    if thresholds is not None:
        dials.update({k: v for k, v in thresholds if v is not None})
    repeats = C2STConfig().repeats if repeats is None else repeats
    resolved_scheme = C2STScheme.Split if scheme is None else C2STScheme(scheme)
    seed = DEFAULT_SEED if seed is None else seed
    emitter = Emitter("evaluate", OPERATION, progress_observer(), total=repeats)
    # The classifier has no ``n_jobs``; it takes an OpenMP thread per core
    # unless capped, so concurrent calls oversubscribe the cores (ADR-0081).
    with threadpool_limits(limits=settings().core_budget, user_api="openmp"):
        result = _run_c2st(
            reference_frame,
            candidate_frame,
            repeats=repeats,
            scheme=resolved_scheme,
            seed=seed,
            min_rows=dials["min_rows"],
            min_samples_leaf=dials["min_samples_leaf"],
            on_repeat=lambda index, total: emitter.item(None, f"repeat {index} of {total}"),
            floor_param="thresholds.min_rows",
        )
    return {
        "warnings": [],
        "omitted": [],
        "reference": reference,
        "candidate": candidate,
        "reference_rows": reference_frame.height,
        "candidate_rows": candidate_frame.height,
        "columns": len(columns),
        "skipped": n_skipped,
        "scheme": str(resolved_scheme),
        "repeats": repeats,
        "seed": seed,
        "mean_repeat_z": result.mean_repeat_z,
        "repeat_z": list(result.repeat_z),
        "repeat_z_spread": result.repeat_z_spread,
        "train_accuracy": result.train_accuracy,
        "test_accuracy": result.test_accuracy,
        "n_test": result.n_test,
        "degenerate": result.degenerate,
        "unseen_category": result.unseen_category,
    }
