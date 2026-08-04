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

import joblib
import polars as pl

from ..config import PipelineConfig, SemanticType
from ..utils._null_normalization import _resolve_effective_nulls
from ._config import ImputationStrategy, ImputationUnit, ModelChoice
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
    "core_budget",
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
        custom_estimators=decision.custom_estimators,
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

    **Concurrency — budget the inner layer when you parallelise the loop.** The
    caller owns the outer/inner split (ADR-0069): parallelism lives in exactly
    one layer, and ``n_jobs_inner`` is how you say which. A sequential loop —
    one ``fit_unit`` at a time — takes the default ``n_jobs_inner=-1``, which
    opens each fit's inner sklearn parallelism to every core. A self-parallelised
    drive — units fitted side by side in your own thread pool — calls
    :func:`core_budget` once before the loop and passes each unit's value here;
    a drive that parallelises and passes nothing leaves every concurrent fit
    fanning out to every core, so the two layers oversubscribe the machine. The
    value never moves the result (ADR-0069); it only changes how the cores are
    spent.

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
        drive; when your own loop already fits units concurrently, pass this
        unit's value from :func:`core_budget` (ADR-0081). Never affects results.

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


def _block_model_choice(
    decision: "ImputationDecision", columns: tuple[str, ...]
) -> Optional[ModelChoice]:
    """Return the estimator family the plan stamped on a block of columns.

    A joint block trains one estimator, so every column carries the same choice
    and the first one that has it answers for all — the same rule the MICE
    fitter reads the block through. ``None`` means the block routed to no
    estimator family and cannot train at all.
    """
    for col in columns:
        col_decision = decision.column_decisions.get(col)
        if col_decision is not None and col_decision.model_choice is not None:
            return col_decision.model_choice
    return None


def core_budget(
    decision: "ImputationDecision",
    max_workers: Optional[int],
    total_cores: Optional[int] = None,
) -> dict[str, int]:
    """Compute the whole-plan inner-parallelism budget: ``unit_id -> n_jobs_inner``.

    The library owns this **arithmetic** and nothing else (ADR-0081): the pool,
    the loop, the submission order and the failure policy stay user-owned
    (ADR-0075 is unamended). Call it once immediately before your own
    :func:`fit_unit` loop and pass each unit's value as that call's
    ``n_jobs_inner``.

    The arithmetic is heavy-aware rather than degree-proportional. A plan holds
    at most two joint blocks, and at most one unit — the ``"mice"`` block, and
    only when its ``model_choice`` is
    :attr:`~dataforge_ml.ModelChoice.RandomForestRegressor` — can absorb inner
    parallelism at all, so the MICE block receives
    ``max(1, total_cores - 1_if_knn_present)`` and every other unit receives
    ``1``. Dividing the cores evenly across the outer degree instead would hand
    the MICE block a ``1`` and leave the machine idle.

    A free function and not a property on
    :class:`~dataforge_ml.ImputationDecision`: the plan is derived purely from
    ``(profile, shape, config)``, and hanging a machine fact on it would make the
    same serialized plan answer differently on a different box (ADR-0072's
    precedent). It answers for the whole plan at once because the
    reserved-for-KNN term is a fact about the plan, not about any one unit.

    Parameters
    ----------
    decision : ImputationDecision
        The plan to price. Read-only — ``units``, and ``column_decisions`` for
        the MICE block's ``model_choice``.
    max_workers : int or None
        The outer degree of the drive the budget is for. ``1`` is a sequential
        one-unit-at-a-time drive, which is outer-degree-one and gets the wide
        ``-1`` for MICE (ADR-0069). Anything greater is a parallel pool, and
        ``None`` — a :class:`~concurrent.futures.ThreadPoolExecutor` of unknown
        degree — takes that same parallel branch rather than being read as
        sequential or rejected.
    total_cores : int, optional
        Overrides core detection, for a caller subdividing a box across
        processes. Detection is ``joblib.cpu_count()``, which respects cgroup
        quotas and is the same detector sklearn uses for its own ``n_jobs=-1``,
        so the budget and sklearn's actual fan-out agree on how big the box is.

    Returns
    -------
    dict[str, int]
        A mapping whose keys are exactly the plan's unit ids and whose values are
        the ``n_jobs_inner`` each unit should be fitted with.

    Notes
    -----
    **All ones is a legitimate answer.** ``BayesianRidge`` and
    ``GradientBoostingRegressor`` have no ``n_jobs`` to spend (ADR-0069), so a
    MICE block routed to either is priced at ``1``, with no error and no warning
    — ``1`` is the truthful number, and a warning would fire on a correct plan.
    On default config this covers every ``ComplexNonlinear`` frame at or above
    ``gradient_boost_min_rows``, which routes to
    ``GradientBoostingRegressor``: there is no speedup to be had there. The
    condition is branch-specific, not a frame-size ceiling —
    ``MonotonicNonlinear`` takes ``RandomForestRegressor`` at any row count and
    stays able to spend a budget.

    **A user-supplied estimator is priced at ``1`` in every branch**, the
    sequential ``max_workers=1`` one included (ADR-0083). The library never sets
    a foreign estimator's parameters, so ``1`` is the truthful count of cores
    *the library* spends; ``total_cores`` remains the hatch for reserving the
    rest. There is no ``n_jobs`` probe: it would fire on correct plans while
    missing ``nthread``, ``num_threads``, and every internal pool that is not
    exposed as a parameter.

    The budget is therefore structurally blind to parallelism configured inside
    a foreign object, and the documented pattern for one is to **fit that unit
    outside your pool**, where it has the machine to itself, driving the
    remaining units through the pool on this budget. ``total_cores`` is the
    reservation hatch: pass it reduced by whatever the self-parallelising fit
    will take, so the two never contend for the same cores.

    **The budget is a snapshot.** It describes the plan as it was when the call
    returned. A plan edit in between — :meth:`ImputationDecision.with_model_choice`,
    :meth:`ImputationDecision.with_hyperparameters` — or a fresh
    :func:`decide` silently invalidates it, and nothing detects that. The
    mitigation is placement: compute the budget immediately before the loop that
    consumes it, not enforcement.
    """
    cores = joblib.cpu_count() if total_cores is None else total_cores
    budget = {unit.unit_id: 1 for unit in decision.units}

    mice_unit = next(
        (u for u in decision.units if u.strategy == ImputationStrategy.MICE), None
    )
    if mice_unit is None:
        return budget

    model_choice = _block_model_choice(decision, mice_unit.columns)
    if model_choice == ModelChoice.Custom:
        # A user-supplied estimator is never configured by the library, so 1 is
        # the truthful count of cores *the library* spends on it — in every
        # branch, the sequential one included (ADR-0083). No introspection and
        # no new knob: total_cores is already the reservation hatch.
        return budget

    if max_workers == 1:
        # Outer degree one: the inner layer gets the whole machine (ADR-0069).
        # Unconditional, so a sequential driver reads the same -1 here that
        # fit_unit defaults to — the function stays purely additive.
        budget[mice_unit.unit_id] = -1
        return budget

    if model_choice != ModelChoice.RandomForestRegressor:
        # BayesianRidge and GradientBoostingRegressor have no n_jobs to spend
        # (ADR-0069), and a block that resolved no choice cannot train at all;
        # 1 is the truthful number and no warning is owed on a correct plan.
        return budget

    knn_present = any(u.strategy == ImputationStrategy.KNN for u in decision.units)
    budget[mice_unit.unit_id] = max(1, cores - (1 if knn_present else 0))
    return budget
