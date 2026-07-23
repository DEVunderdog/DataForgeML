"""Stateless training surface for the user-orchestrated imputation layer.

The execution half of the Decision/Execution split, redrawn around a single
primitive (ADR-0071, ADR-0075): the user keeps the pure
:class:`ImputationDecision` produced by
:func:`~dataforge_ml.imputation.decide`, trains each planned unit with their
own :func:`fit_unit` loop — batch scheduling is user-owned (ADR-0075) — and
composes the fitted units into a whole-frame imputer with
:meth:`FittedImputer.compose`. There is no executor, no state machine, and no
store: a fit either produces a fitted unit or raises
:class:`UnitNotTrainableError` (single-track failure, ADR-0071).

The frame is normalised off the plan's declared sentinel maps before the unit
trains (ADR-0068), so a raw frame may be handed straight in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Optional

import polars as pl

from ..config import PipelineConfig, SemanticType
from ..utils._null_normalization import _resolve_effective_nulls
from ._config import ImputationStrategy, ImputationUnit
from ._fit_signals import FitSignals, ImputationFitWarning
from ._fitters import UnitFitContext, _dispatch_unit_fit

if TYPE_CHECKING:
    from ._config import ImputationDecision
    from ._fitted_imputer import FittedUnit

__all__ = [
    "FitSignals",
    "ImputationFitWarning",
    "UnitFitResult",
    "UnitNotTrainableError",
    "fit_unit",
]


class UnitNotTrainableError(RuntimeError):
    """Raised when a planned unit cannot be trained.

    Single-track failure (ADR-0071): a unit that cannot train always raises,
    carrying a structured payload — the unit id, its strategy, its columns, and
    the fitter's reason — rather than silently degrading to a scalar. The
    fitter's own diagnosis is chained as the cause.

    Attributes
    ----------
    unit_id : str
        The id of the unit that could not train.
    strategy : ImputationStrategy
        The strategy the unit was planned to execute.
    columns : tuple[str, ...]
        The columns the unit owns.
    reason : str
        The fitter's human-readable explanation of why training failed.
    """

    def __init__(
        self,
        unit_id: str,
        strategy: ImputationStrategy,
        columns: tuple[str, ...],
        reason: str,
    ) -> None:
        self.unit_id = unit_id
        self.strategy = strategy
        self.columns = tuple(columns)
        self.reason = reason
        names = ", ".join(f"'{c}'" for c in self.columns)
        super().__init__(
            f"Unit '{unit_id}' ({strategy}) could not train column(s) {names}: "
            f"{reason}"
        )


@dataclass(frozen=True)
class UnitFitResult:
    """A trained unit bundled with a structured record of the fit.

    What :func:`fit_unit` returns: the fitted unit that can transform its own
    columns, alongside the plan-carried facts that identify it (``unit_id``,
    ``strategy``, ``columns``) and the structured
    :class:`~dataforge_ml.imputation.FitSignals` the fit produced (ADR-0074).

    Attributes
    ----------
    unit_id : str
        The plan id of the trained unit.
    strategy : ImputationStrategy
        The strategy that was executed.
    columns : tuple[str, ...]
        The columns the unit owns.
    fitted : FittedUnit
        The trained unit; transforms its own columns with no completeness
        requirement.
    signals : FitSignals
        The structured, ephemeral record of the fit — estimator, convergence,
        duration, notes, and any warnings. Read it to answer "did this fit
        converge? which estimator ran? how long did it take?" without parsing
        strings.
    """

    unit_id: str
    strategy: ImputationStrategy
    columns: tuple[str, ...]
    fitted: "FittedUnit"
    signals: FitSignals


def _numeric_config(decision: "ImputationDecision"):
    """Rebuild the numeric imputation config from the plan's config snapshot.

    The plan carries the serialised :class:`PipelineConfig` it was decided
    under, so a training function needs no config argument — everything a
    fitter honours is reachable from the decision alone.
    """
    return PipelineConfig.from_dict(decision.config_snapshot).imputation.numeric


def _fit_context(
    decision: "ImputationDecision", random_seed: Optional[int]
) -> UnitFitContext:
    """Build the context the shared fitters read the plan through."""
    return UnitFitContext(
        column_decisions=decision.column_decisions,
        config=_numeric_config(decision),
        feature_columns=tuple(
            col
            for col, d in decision.column_decisions.items()
            if d.semantic_type == SemanticType.Numeric
        ),
        random_seed=random_seed,
    )


def _unit(decision: "ImputationDecision", unit_id: str) -> ImputationUnit:
    """Return the plan's unit for ``unit_id``, or raise ``KeyError``."""
    for unit in decision.units:
        if unit.unit_id == unit_id:
            return unit
    known = ", ".join(f"'{u.unit_id}'" for u in decision.units)
    raise KeyError(
        f"Plan carries no unit '{unit_id}'. Known units: {known or '(none)'}."
    )


def _emit_warnings(signals: FitSignals) -> None:
    """Dual-channel every recorded warning through the standard mechanism.

    Each entry already sits in ``signals.warnings``; re-raising it under
    :class:`ImputationFitWarning` (ADR-0074) gives a caller who never inspects
    the record stderr visibility and the ordinary ``filterwarnings`` toolkit.
    """
    for message in signals.warnings:
        warnings.warn(message, category=ImputationFitWarning, stacklevel=3)


def fit_unit(
    decision: "ImputationDecision",
    unit_id: str,
    df: pl.DataFrame,
    random_seed: Optional[int] = None,
    n_jobs_inner: int = -1,
) -> UnitFitResult:
    """Train exactly one planned unit and bundle it into a :class:`UnitFitResult`.

    The one training primitive of the user-orchestrated flow: batch scheduling
    is user-owned (ADR-0075), so training a whole plan is a caller-written loop
    of ``fit_unit`` calls. The frame is normalised off the plan's declared
    sentinel maps (ADR-0068) before the fit, so a raw frame may be handed
    straight in.

    **Concurrency — pin the inner layer when you parallelise the loop.** The
    caller owns the outer/inner split (ADR-0069): parallelism lives in exactly
    one layer, and ``n_jobs_inner`` is how you say which. A sequential loop —
    one ``fit_unit`` at a time — takes the default ``n_jobs_inner=-1``, which
    opens each fit's inner sklearn parallelism to every core. A self-parallelised
    drive — units fitted side by side in your own thread pool — must pass
    ``n_jobs_inner=1`` on every call, otherwise each concurrent fit also fans out
    to every core and the two layers oversubscribe the machine. The value never
    moves the result (ADR-0069); it only changes how the cores are spent.

    Parameters
    ----------
    decision : ImputationDecision
        The immutable plan. Supplies the unit recipe, the per-column decisions a
        fitter honours, the config snapshot, and the sentinel maps used to
        normalise ``df``.
    unit_id : str
        The id of the unit to train.
    df : pl.DataFrame
        Training data, raw or already normalised.
    random_seed : int, optional
        Seed for the stochastic strategies (GMM sampling).
    n_jobs_inner : int, default -1
        Inner estimator ``n_jobs`` (ADR-0056). Leave at ``-1`` for a sequential
        drive; pin to ``1`` per call when your own loop already fits units
        concurrently. Never affects results.

    Returns
    -------
    UnitFitResult
        The trained unit plus its structured fit record. The record's
        ``duration_s`` spans this whole call, normalisation included.

    Raises
    ------
    KeyError
        If ``unit_id`` names no unit in the plan.
    UnitNotTrainableError
        If the unit cannot train, carrying its structured payload — the sole
        failure track (ADR-0071).
    """
    start = perf_counter()
    unit = _unit(decision, unit_id)
    train_df = _resolve_effective_nulls(
        df,
        numeric_sentinels=decision.numeric_sentinels,
        string_sentinels=decision.string_sentinels,
    )
    ctx = _fit_context(decision, random_seed)
    outcome = _dispatch_unit_fit(unit, train_df, ctx, n_jobs_inner)
    if outcome.fitted is None or outcome.signals is None:
        raise UnitNotTrainableError(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            columns=unit.columns,
            reason=outcome.fallback_reason or str(unit.strategy),
        )
    signals = replace(outcome.signals, duration_s=perf_counter() - start)
    _emit_warnings(signals)
    return UnitFitResult(
        unit_id=unit.unit_id,
        strategy=unit.strategy,
        columns=unit.columns,
        fitted=outcome.fitted,
        signals=signals,
    )
