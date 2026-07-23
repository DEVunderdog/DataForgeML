"""Structured, ephemeral observability for one imputation unit's fit (ADR-0074).

A fit reports what happened through a :class:`FitSignals` value the caller reads
straight off the :class:`~dataforge_ml.imputation.UnitFitResult` it gets back —
not an event stream and not opaque strings. The record is **ephemeral**: it
describes one fit's runtime, never the fitted state, so it is never part of any
persisted artifact (ADR-0072).

Genuine warnings are **dual-channelled** (ADR-0074): every entry in
:attr:`FitSignals.warnings` is also raised through the standard :mod:`warnings`
machinery under :class:`ImputationFitWarning`, so a caller who never inspects the
record still sees them on stderr and can suppress, escalate, or route them with
the ordinary ``filterwarnings`` toolkit.

This module carries no dependency on the training surface; both the fitters and
the surface import from it, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._config import ImputationStrategy

__all__ = [
    "FitSignals",
    "ImputationFitWarning",
]


class ImputationFitWarning(UserWarning):
    """Category for warnings raised while fitting an imputation unit.

    Every entry recorded in :attr:`FitSignals.warnings` is also emitted through
    the standard :mod:`warnings` channel under this category (ADR-0074), so a
    caller who never inspects the returned record is not silently deprived of
    them. Suppress, escalate, or route them with the ordinary toolkit, e.g.
    ``warnings.filterwarnings("ignore", category=ImputationFitWarning)``.

    The two conditions that raise it are a model that hit its iteration cap
    without converging, and a strategy forced past its routing threshold (KNN
    above the row/feature caps, or Regression below the row floor) — the latter
    is warned, never blocked (ADR-0071).
    """


@dataclass(frozen=True)
class FitSignals:
    """Structured record of one imputation unit's fit (ADR-0074).

    The observability value :func:`~dataforge_ml.imputation.fit_unit` returns on
    every :class:`~dataforge_ml.imputation.UnitFitResult`, letting the caller
    answer "did this fit converge? which estimator ran? how long did it take?"
    without parsing strings. A typed core carries the facts any consumer branches
    on; ``notes`` is a free-form bag for strategy-specific minutiae that earn no
    typed field, kept separate from ``warnings`` so the cheap question "did
    anything go wrong?" is answered by reading ``warnings`` alone.

    The record is ephemeral — it describes the fit's runtime, not the fitted
    state, and is never serialized (ADR-0072). An untrainable unit raises
    :class:`~dataforge_ml.imputation.UnitNotTrainableError` and produces no
    record: observability describes fits that happened (ADR-0074).

    Attributes
    ----------
    unit_id : str
        The plan id of the trained unit.
    strategy : ImputationStrategy
        The strategy that was executed.
    estimator : str or None
        Human-readable name of the estimator that ran (e.g.
        ``"RandomForestRegressor"``, ``"KNNImputer"``); ``None`` for strategies
        that fit no estimator, such as the scalar fills.
    converged : bool or None
        Whether an iterative model stopped before its iteration cap; ``None`` for
        strategies with no iterative-convergence notion.
    n_iter : int or None
        The iteration count an iterative model settled on; ``None`` when there is
        none.
    duration_s : float
        Wall-clock seconds the fit took, measured with ``perf_counter``. Spans
        the whole :func:`~dataforge_ml.imputation.fit_unit` call, normalisation
        included.
    warnings : tuple[str, ...]
        Actionable fit-time warnings — a hit iteration cap, or a strategy forced
        past its routing threshold. Each is also raised through the standard
        :mod:`warnings` channel under :class:`ImputationFitWarning` (ADR-0074).
    notes : tuple[str, ...]
        Informational fit-time detail (resolved dials, scaling, the routing
        branch taken) that does not earn a typed field.
    """

    unit_id: str
    strategy: ImputationStrategy
    estimator: Optional[str] = None
    converged: Optional[bool] = None
    n_iter: Optional[int] = None
    duration_s: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
