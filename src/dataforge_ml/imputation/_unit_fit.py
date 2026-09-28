"""Stateless training surface for the user-orchestrated imputation layer.

The execution half of the layered imputation door (ADR-0071, ADR-0075,
ADR-0084): the user keeps the :class:`~dataforge_ml.imputation.ImputationRecipe`
produced by :func:`~dataforge_ml.imputation.resolve_recipe`, trains each unit
from :func:`~dataforge_ml.imputation.derive_units` with their own
:func:`fit_unit` loop — batch scheduling is user-owned (ADR-0075) — and
composes the fitted units into a whole-frame imputer with
:meth:`FittedImputer.compose`. There is no executor, no state machine, and no
store: a fit either produces a fitted unit or raises
:class:`UnitNotTrainableError` (single-track failure, ADR-0071).

The frame is normalised off the recipe's declared sentinel maps before the
unit trains (ADR-0068), so a raw frame may be handed straight in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

import polars as pl

from ..utils._dtype_floor import _apply_dtype_floor
from ..utils._null_normalization import _resolve_effective_nulls
from ._config import ImputationStrategy, ImputationUnit, _md_cell
from ._fit_signals import FitSignals, ImputationFitWarning
from ._fitters import UnitFitContext, _dispatch_unit_fit
from ._units import derive_units

if TYPE_CHECKING:
    from ._fitted_imputer import FittedUnit
    from ._recipe import ImputationRecipe

__all__ = [
    "FitSignals",
    "ImputationFitWarning",
    "UnitFitResult",
    "UnitNotTrainableError",
    "fit_unit",
]


class UnitNotTrainableError(RuntimeError):
    """Raised when a routed unit cannot be trained.

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
            f"Unit '{unit_id}' ({strategy}) could not train column(s) {names}: {reason}"
        )


@dataclass(frozen=True)
class UnitFitResult:
    """A trained unit bundled with a structured record of the fit.

    What :func:`fit_unit` returns: the fitted unit that can transform its own
    columns, alongside the recipe-carried facts that identify it (``unit_id``,
    ``strategy``, ``columns``) and the structured
    :class:`~dataforge_ml.imputation.FitSignals` the fit produced (ADR-0074).

    Attributes
    ----------
    unit_id : str
        The id of the trained unit.
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
    fitted: FittedUnit
    signals: FitSignals

    def to_markdown(self) -> str:
        """Render the fit result as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086): it owns the
        ``#`` and ``##`` heading levels and delegates to the
        :class:`~dataforge_ml.imputation.FitSignals` fragment beneath them. The
        fitted unit itself is opaque state, so it is reported by type name only.

        Returns
        -------
        str
            Markdown document naming the unit, its strategy, its columns and the
            fitted unit's type, followed by the fit record.
        """
        fitted = type(self.fitted).__name__ if self.fitted is not None else "none"
        lines = [
            f"# Unit Fit — `{self.unit_id}`\n",
            "## Summary\n",
            "| Field | Value |",
            "|---|---|",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| columns | {_md_cell(self.columns)} |",
            f"| fitted | {_md_cell(fitted)} |",
            "",
            "## Fit Record\n",
            self.signals.to_markdown()
            if self.signals is not None
            else "not recorded",
        ]
        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Unit Fit document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


def _fit_context(
    recipe: ImputationRecipe, random_seed: int | None
) -> UnitFitContext:
    """Build the context the shared fitters read the recipe through."""
    from ._recipe import _active_numeric_columns

    return UnitFitContext(
        column_routings=recipe.routing.column_routings,
        column_estimates=recipe.column_estimates,
        unit_hyperparameters={
            unit.unit_id: recipe.hyperparameters(unit.unit_id)
            for unit in derive_units(recipe.routing)
        },
        feature_columns=tuple(_active_numeric_columns(recipe.routing)),
        mice_estimator=recipe.routing.mice_estimator,
        random_seed=random_seed,
        mice_model_choice=recipe.routing.mice_model_choice,
    )


def _resolve(recipe: ImputationRecipe, unit: ImputationUnit) -> ImputationUnit:
    """Match ``unit`` against the recipe's routing by id, or raise.

    An unknown id raises ``KeyError``. A known id whose ``strategy`` or
    ``columns`` differ from the recipe's own unit of that id can only have
    come from a different routing — training it would silently train a
    mismatched recipe, so this raises ``ValueError`` naming both instead.
    """
    if isinstance(unit, str):
        # The pre-4.x signature took the id. Say so, rather than letting the
        # attribute lookup below fail with a bare AttributeError.
        raise TypeError(
            f"fit_unit() takes an ImputationUnit, not the id {unit!r}. Get the "
            f"unit from derive_units(recipe.routing) and pass that."
        )
    recipe_units = {u.unit_id: u for u in derive_units(recipe.routing)}
    recipe_unit = recipe_units.get(unit.unit_id)
    if recipe_unit is None:
        known = ", ".join(f"'{u}'" for u in recipe_units)
        raise KeyError(
            f"Recipe's routing carries no unit '{unit.unit_id}'. Known units: "
            f"{known or '(none)'}."
        )
    if recipe_unit.strategy != unit.strategy or recipe_unit.columns != unit.columns:
        raise ValueError(
            f"Unit '{unit.unit_id}' does not match the recipe's routing: "
            f"passed {unit!r}, recipe's routing carries {recipe_unit!r}. This "
            f"unit was derived from a different routing — re-derive it with "
            f"derive_units(recipe.routing) before fitting."
        )
    return recipe_unit


def _emit_warnings(signals: FitSignals) -> None:
    """Dual-channel every recorded warning through the standard mechanism.

    Each entry already sits in ``signals.warnings``; re-raising it under
    :class:`ImputationFitWarning` (ADR-0074) gives a caller who never inspects
    the record stderr visibility and the ordinary ``filterwarnings`` toolkit.
    """
    for message in signals.warnings:
        warnings.warn(message, category=ImputationFitWarning, stacklevel=3)


def fit_unit(
    recipe: ImputationRecipe,
    unit: ImputationUnit,
    df: pl.DataFrame,
    random_seed: int | None = None,
    n_jobs_inner: int = -1,
) -> UnitFitResult:
    """Train exactly one unit and bundle it into a :class:`UnitFitResult`.

    The one training primitive of the user-orchestrated flow: batch scheduling
    is user-owned (ADR-0075), so training a whole routing is a caller-written
    loop of ``fit_unit`` calls. The frame is normalised off the recipe's
    declared sentinel maps (ADR-0068) before the fit, so a raw frame may be
    handed straight in.

    **Concurrency — the caller owns the outer/inner split** (ADR-0069):
    parallelism lives in exactly one layer, and ``n_jobs_inner`` is how you
    say which. A sequential loop — one ``fit_unit`` at a time — takes the
    default ``n_jobs_inner=-1``. A self-parallelised drive picks its own
    per-unit value so the two layers do not oversubscribe the machine. The
    value never moves the result (ADR-0069); it only changes how the cores
    are spent.

    Parameters
    ----------
    recipe : ImputationRecipe
        The immutable recipe. Supplies the routing every unit is matched
        against, the per-unit hyperparameters, the profile-derived estimates,
        and the sentinel maps used to normalise ``df``.
    unit : ImputationUnit
        The unit to train, taken from
        :func:`~dataforge_ml.imputation.derive_units`. Matched against the
        recipe's own routing by ``unit_id``; a mismatch on ``strategy`` or
        ``columns`` raises rather than silently training the recipe's version.
    df : pl.DataFrame
        Training data, raw or already normalised. A working copy is taken at
        entry: effective nulls are resolved and the **Dtype Floor** enforced
        against the routing's semantic types. The caller's frame is never
        mutated.
    random_seed : int, optional
        Seed for the stochastic strategies (GMM sampling).
    n_jobs_inner : int, default -1
        Inner estimator ``n_jobs`` (ADR-0056), unused by every fitter reachable
        today. Accepted for signature stability with the (not-yet-implemented)
        model-based fitters.

    Returns
    -------
    UnitFitResult
        The trained unit plus its structured fit record. The record's
        ``duration_s`` spans this whole call, normalisation included.

    Raises
    ------
    KeyError
        If ``unit`` names no unit in the recipe's routing.
    TypeError
        If ``unit`` is a unit id string rather than an
        :class:`~dataforge_ml.ImputationUnit` — the pre-4.x signature.
    ValueError
        If ``unit``'s ``strategy`` or ``columns`` differ from the recipe's own
        unit of the same id — it was derived from a different routing.
    UnitNotTrainableError
        If the unit cannot train, carrying its structured payload — the sole
        failure track (ADR-0071).
    """
    start = perf_counter()
    unit = _resolve(recipe, unit)
    # Phase entry: effective nulls first, then the Dtype Floor off the
    # routing's semantic types. The floor casts away the string namespace the
    # sentinel rules need, so the order is fixed.
    train_df = _apply_dtype_floor(
        _resolve_effective_nulls(
            df,
            numeric_sentinels=recipe.numeric_sentinels,
            string_sentinels=recipe.string_sentinels,
        ),
        {
            name: r.semantic_type
            for name, r in recipe.routing.column_routings.items()
        },
    )
    ctx = _fit_context(recipe, random_seed)
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

