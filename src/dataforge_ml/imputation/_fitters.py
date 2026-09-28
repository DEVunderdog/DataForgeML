"""
Unit-shaped fitters — the one training implementation the surface calls.

Every imputation strategy that learns something from the training frame
trains here, in a function keyed on ``(ImputationUnit, train_df,
UnitFitContext)``. The stateless training surface (``_unit_fit``) dispatches
into these functions through :func:`_dispatch_unit_fit`.

Everything a fitter needs beyond the frame is on :class:`UnitFitContext` —
built by the training surface from an :class:`~dataforge_ml.imputation.ImputationRecipe`
and the unit being trained. Nothing here reads the Phase 1 profile directly: a
fitter is handed a recipe-derived context and a frame, never the profile the
recipe was resolved from.

Like ``_fitted_units``, this module must not import the training surface; the
dependency edge runs surface → fitters, never back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from ._config import ColumnRouting, ImputationStrategy, ModelChoice
from ._fit_signals import FitSignals
from ._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)
from ._recipe import ColumnEstimates
from ._regression_estimator_factory import (
    RegressionEstimatorFactory,
    _CoreInvariantRandomForest,
)
from ._utils import _compute_mean, _compute_median, _compute_mode, _df_to_numpy

if TYPE_CHECKING:
    from ._config import ImputationUnit
    from ._fitted_imputer import FittedUnit


__all__ = [
    "UnitFitContext",
    "UnitFitOutcome",
    "_dispatch_unit_fit",
    "fit_cluster_unit",
    "fit_gmm_unit",
    "fit_knn_unit",
    "fit_mice_unit",
    "fit_scalar_unit",
]


_SCALAR_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.Mean,
        ImputationStrategy.Median,
        ImputationStrategy.Mode,
        ImputationStrategy.Constant,
        ImputationStrategy.MNAR,
    }
)


@dataclass(frozen=True)
class UnitFitContext:
    """Everything a unit-shaped fitter needs beyond the unit and the frame.

    Built by the training surface (:func:`~dataforge_ml.imputation.fit_unit`)
    from an :class:`~dataforge_ml.imputation.ImputationRecipe`.

    Parameters
    ----------
    column_routings : Mapping[str, ColumnRouting]
        The owning routing's per-column entries. Read for ``constant_fill``
        and ``grouping_variable`` — the two config declarations that complete
        a strategy.
    column_estimates : Mapping[str, ColumnEstimates]
        The owning recipe's per-column profile-derived estimates. Read for
        the bimodal centres, ``feature_cols`` and ``domain_snap_bounds``.
    unit_hyperparameters : Mapping[str, dict[str, Any]]
        Per-unit merged hyperparameters (``recipe.hyperparameters(unit_id)``),
        keyed by unit id.
    feature_columns : tuple[str, ...], optional
        The active numeric column population MICE and KNN both widen their
        predictor set into (ADR-0079, ADR-0093) — every active
        ``SemanticType.Numeric`` column, not just a block's own membership.
    mice_estimator : Any, optional
        The owning routing's user-supplied MICE estimator
        (:attr:`~dataforge_ml.imputation.ImputationRouting.mice_estimator`),
        held by identity and never cloned. Read only when the block's
        ``model_choice`` is :attr:`~dataforge_ml.ModelChoice.Custom`.
    random_seed : int, optional
        Seed for the stochastic strategies (GMM sampling).
    mice_model_choice : ModelChoice, optional
        The owning routing's
        :attr:`~dataforge_ml.imputation.ImputationRouting.mice_model_choice` —
        the estimator family the MICE block trains with. ``None`` when no
        column routes to MICE.
    """

    column_routings: Mapping[str, ColumnRouting]
    column_estimates: Mapping[str, ColumnEstimates] = field(default_factory=dict)
    unit_hyperparameters: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    mice_estimator: Any = None
    random_seed: int | None = None
    mice_model_choice: ModelChoice | None = None


@dataclass(frozen=True)
class UnitFitOutcome:
    """What one unit-shaped fitter learned, plus why it learned nothing.

    A fitter never decides what to do about a unit it cannot train — it reports
    ``fitted=None`` with a reason and leaves the policy (raise) to the surface
    that called it (ADR-0071).

    Parameters
    ----------
    fitted : FittedUnit or None
        The trained unit, or ``None`` when this unit cannot train.
    signals : FitSignals or None, optional
        The structured, ephemeral record of the fit (ADR-0074) — estimator,
        convergence, notes, and any warnings. Its ``duration_s`` is left unset
        here and stamped by the training surface, which owns the clock. ``None``
        when ``fitted`` is ``None``: a failure produces no record.
    fallback_reason : str or None, optional
        Why ``fitted`` is ``None``; always set when it is.
    """

    fitted: FittedUnit | None = None
    signals: FitSignals | None = None
    fallback_reason: str | None = None


def _hyperparameters(unit: ImputationUnit, ctx: UnitFitContext) -> dict[str, Any]:
    """Read a unit's recipe-resolved hyperparameters as a plain dict."""
    return dict(ctx.unit_hyperparameters.get(unit.unit_id, {}))


def _estimator_name(estimator: Any) -> str:
    """Human-readable name of a built estimator for the fit signals.

    A signal names the estimator family the user chose, so the RandomForest
    branch reports the ``ModelChoice`` it was routed to rather than the
    core-invariance wrapper it happens to be built as (ADR-0069).
    """
    if isinstance(estimator, Pipeline):
        return "Pipeline(StandardScaler+BayesianRidge)"
    if isinstance(estimator, _CoreInvariantRandomForest):
        return "RandomForestRegressor"
    return type(estimator).__name__


def _domain_snap_bounds(
    ctx: UnitFitContext, columns: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Collect the recipe's domain-snap bounds for the columns that carry them."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        estimates = ctx.column_estimates.get(col)
        if estimates is not None and estimates.domain_snap_bounds is not None:
            bounds[col] = estimates.domain_snap_bounds
    return bounds


_CENTRAL_TENDENCY: dict[str, Any] = {
    "mean": _compute_mean,
    "median": _compute_median,
    "mode": _compute_mode,
}


def _fill_scalar_predictors(
    train_df: pl.DataFrame, ctx: UnitFitContext, extra_cols: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Pre-fill a joint block's scalar-owned predictors to match the serve frame.

    At serve time every model-based unit reads the same pre-model snapshot,
    which already carries the scalar fills (``FittedImputer.transform``
    applies Mean / Median / Mode before any model unit runs). A joint block's
    fit frame must see the same filled values for those predictors rather
    than their raw nulls, or the fit trains against a distribution it will
    never serve against (#418). The fill is recomputed here from ``train_df``
    under each predictor's own decided strategy — never read off a sibling
    ``FittedScalar`` unit's learned state — so ``fit_unit`` calls stay
    independent of one another (ADR-0074). A predictor routed to a
    model-based strategy (KNN / MICE) is left untouched: it arrives raw at
    serve time too, so raw is already the matching frame.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context; each predictor's decided strategy is read off
        ``ctx.column_routings``.
    extra_cols : list[str]
        Predictor columns outside the block's own membership.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        ``train_df`` (or a copy with the scalar-owned predictors' nulls
        filled) paired with the names of the columns that were filled.
    """
    filled_cols = []
    fill_exprs = []
    for c in extra_cols:
        routing = ctx.column_routings.get(c)
        if routing is None:
            continue
        tendency = _CENTRAL_TENDENCY.get(str(routing.strategy))
        if tendency is None:
            continue
        fill_exprs.append(pl.col(c).fill_null(tendency(train_df, c)))
        filled_cols.append(c)
    if not fill_exprs:
        return train_df, filled_cols
    return train_df.with_columns(fill_exprs), filled_cols


def _dispatch_unit_fit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
    n_jobs_inner: int = 1,
) -> UnitFitOutcome:
    """Train one unit, dispatching on its strategy.

    The single door the training surface drives: given a routed unit and the
    training frame, learn that unit's fitted state. Every non-model-based
    strategy that trains something is reachable from here.

    Parameters
    ----------
    unit : ImputationUnit
        The unit to train.
    train_df : pl.DataFrame
        Training split; every learned value comes from here.
    ctx : UnitFitContext
        Recipe-derived context the fitter honours.
    n_jobs_inner : int, default 1
        Unused by every fitter in this module; accepted for a uniform dispatch
        signature with the (not-yet-implemented) model-based fitters.

    Returns
    -------
    UnitFitOutcome
        The trained unit, or a reason it could not train.

    Raises
    ------
    ValueError
        If ``unit.strategy`` is one no fitter in this module handles — every
        structural strategy (Dropped / Passthrough / Indicator), which trains
        nothing and must not be projected into a unit.
    """
    strategy = unit.strategy
    if strategy in _SCALAR_STRATEGIES:
        return fit_scalar_unit(unit, train_df, ctx)
    if strategy == ImputationStrategy.MICE:
        return fit_mice_unit(unit, train_df, ctx, n_jobs_inner=n_jobs_inner)
    if strategy == ImputationStrategy.KNN:
        return fit_knn_unit(unit, train_df, ctx)
    if strategy == ImputationStrategy.GMMSampling:
        return fit_gmm_unit(unit, train_df, ctx)
    if strategy == ImputationStrategy.ClusterConditional:
        return fit_cluster_unit(unit, train_df, ctx)
    raise ValueError(
        f"Unit '{unit.unit_id}' carries strategy '{strategy}', which no fitter "
        f"in this module handles. Structural strategies (Dropped / Passthrough "
        f"/ Indicator) train nothing and must not be projected into a unit."
    )


def fit_scalar_unit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Learn the scalar fill value for a Mean / Median / Mode / Constant / MNAR unit.

    ``Constant`` reads the user's declared value and never touches ``train_df``.
    ``MNAR`` computes the central tendency the recipe chose for it, rounded to a
    whole number when the column is integer-typed. The value is exposed, not
    applied: :meth:`FittedImputer.transform` leaves an MNAR column's nulls in
    place, and the unit's own ``transform`` is how a user opts into the fill
    (ADR-0098). The signals say so, since nulls in the output are otherwise
    the only clue.

    Parameters
    ----------
    unit : ImputationUnit
        The single-column unit to train.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context.

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_units.FittedScalar`, or no fit
        when a ``Constant`` column has no declared value.
    """
    col = unit.columns[0]

    def _signals(notes: tuple[str, ...] = ()) -> FitSignals:
        return FitSignals(unit_id=unit.unit_id, strategy=unit.strategy, notes=notes)

    if unit.strategy == ImputationStrategy.Constant:
        routing = ctx.column_routings.get(col)
        declared = routing.constant_fill if routing is not None else None
        if declared is None:
            return UnitFitOutcome(
                fallback_reason=(
                    f"constant: column '{col}' is routed to Constant but carries "
                    f"no constant_fill on the routing"
                )
            )
        return UnitFitOutcome(
            fitted=FittedScalar(target_col=col, fill_value=declared),
            signals=_signals(notes=("fill: declared constant value",)),
        )

    if unit.strategy == ImputationStrategy.MNAR:
        hyp = _hyperparameters(unit, ctx)
        tendency = hyp["central_tendency"]
        fill_value = {
            "mean": _compute_mean,
            "median": _compute_median,
            "mode": _compute_mode,
        }[tendency](train_df, col)
        if train_df[col].dtype.is_integer():
            fill_value = float(round(fill_value))
        return UnitFitOutcome(
            fitted=FittedScalar(target_col=col, fill_value=fill_value),
            signals=_signals(
                notes=(
                    f"central_tendency: {tendency}",
                    "fill computed, not applied: FittedImputer.transform leaves ",
                    "this MNAR column's nulls in place; read ",
                    "ColumnImputationRecord.fill_value or call this unit's ",
                    "transform to apply it",
                )
            ),
        )

    fill_value = {
        ImputationStrategy.Mean: _compute_mean,
        ImputationStrategy.Median: _compute_median,
        ImputationStrategy.Mode: _compute_mode,
    }[unit.strategy](train_df, col)
    return UnitFitOutcome(
        fitted=FittedScalar(target_col=col, fill_value=fill_value),
        signals=_signals(notes=(f"central_tendency: {unit.strategy}",)),
    )


def fit_mice_unit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
    n_jobs_inner: int = 1,
) -> UnitFitOutcome:
    """Fit the joint MICE block as one ``IterativeImputer`` over the full active-numeric matrix.

    The block trains a single estimator, built from ``ctx.mice_model_choice``
    — the choice :func:`~dataforge_ml.imputation.route` resolved off the
    Estimator Ladder (ADR-0094), or that ``with_model_choice`` set explicitly.
    :attr:`~dataforge_ml.ModelChoice.Custom` is the one choice nothing is
    built for: the user's own estimator is taken off ``ctx.mice_estimator``
    and used as-is (ADR-0090). It is not cloned, its ``n_jobs`` is not set,
    and a pre-fitted one is accepted without a raise or a warning —
    ``IterativeImputer`` clones it per column and refits from scratch, so its
    prior state is inert. A ``Custom`` block whose slot is empty — every
    routing reloaded from bytes — reports a reason rather than falling back
    to a library estimator.

    The predictor set is widened past the block's own membership to every
    column in ``ctx.feature_columns`` — every active
    ``SemanticType.Numeric`` column (ADR-0079), the block's own columns
    included. The block still writes back only its own columns:
    :class:`~dataforge_ml.imputation._fitted_imputer.FittedMICE` carries the
    ``all_cols`` / ``columns`` split that generalizes the former per-column
    regression unit's single-target write-back restriction block-wide.

    A widened predictor routed to a scalar strategy (Mean / Median / Mode) is
    filled with its own decided central tendency before this block fits
    (:func:`_fill_scalar_predictors`), so the frame trained on matches the
    scalar-filled pre-model snapshot this block will later serve against —
    uniformly, regardless of ``model_choice`` (#418).

    Parameters
    ----------
    unit : ImputationUnit
        The ``"mice"`` unit. ``max_iter`` / ``tol`` / ``initial_strategy`` /
        ``n_nearest_features`` are read from its hyperparameters.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context. ``feature_columns`` supplies the widened
        predictor set; ``mice_model_choice`` and ``mice_estimator`` the
        estimator to build.
    n_jobs_inner : int, default 1
        Inner estimator ``n_jobs`` (ADR-0056).

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_imputer.FittedMICE` plus the
        estimator / initial-strategy / predictor / convergence signals, or no
        fit when the routing carries no ``mice_model_choice`` at all (a bare
        ``author()`` declaration with no ``with_model_choice`` edit), or is
        planned ``Custom`` with no estimator carried.
    """
    from ._fitted_imputer import FittedMICE

    cols = tuple(unit.columns)
    model_choice = ctx.mice_model_choice
    if model_choice is None:
        return UnitFitOutcome(
            fallback_reason=(
                f"mice: unit '{unit.unit_id}' carries no mice_model_choice. "
                f"route() resolves one off the Estimator Ladder whenever the "
                f"block is non-empty; a hand-authored routing needs "
                f"routing.with_model_choice(choice) before this block can train."
            )
        )

    hyp = _hyperparameters(unit, ctx)
    max_iter = hyp["max_iter"]
    tol = hyp["tol"]
    initial_strategy = hyp["initial_strategy"]
    n_nearest_features = hyp["n_nearest_features"]

    # Widen the predictor set past the block's own membership: every active
    # numeric column is a candidate predictor (ADR-0079), mirroring the
    # feat_cols the former per-column regression fitter read. The block
    # still owns and writes back only its own columns (cols), not all_cols.
    # A predictor whose frame dtype is not numeric cannot enter the joint
    # matrix. Only the frame can answer that: a routing's semantic types are
    # route-time claims, and the manual door stamps Numeric on every column
    # it routes (ADR-0083), so a Passthrough string column would otherwise
    # be widened into.
    extra_cols = [
        c
        for c in ctx.feature_columns
        if c not in cols and c in train_df.columns and train_df.schema[c].is_numeric()
    ]
    all_cols = list(cols) + extra_cols

    # Close the scalar-half train/serve skew (#418): the widened predictors
    # must be filled exactly as the serve-time pre-model snapshot fills them.
    fit_df, scalar_filled_cols = _fill_scalar_predictors(train_df, ctx, extra_cols)

    if model_choice == ModelChoice.Custom:
        # The label says "look elsewhere": the instance rides on the
        # routing's mice_estimator slot and is used as-is, never cloned and
        # never reconfigured (ADR-0090). An empty slot means this routing
        # was reloaded from bytes, which never carries the estimator — that
        # cannot train, and saying so is the whole point of the label being
        # a value rather than None.
        estimator = ctx.mice_estimator
        if estimator is None:
            return UnitFitOutcome(
                fallback_reason=(
                    f"mice: the block is routed with ModelChoice.Custom but no "
                    f"estimator is carried for unit '{unit.unit_id}'. A "
                    f"user-supplied estimator is never serialized, so a "
                    f"reloaded routing must be re-supplied with "
                    f"routing.with_model_choice(estimator)."
                )
            )
    else:
        estimator = RegressionEstimatorFactory.build_from_choice(
            model_choice, n_jobs=n_jobs_inner
        )
    model = IterativeImputer(
        estimator=estimator,
        random_state=0,
        max_iter=max_iter,
        tol=tol,
        initial_strategy=initial_strategy,
        n_nearest_features=n_nearest_features,
    )
    model.fit(_df_to_numpy(fit_df, all_cols))

    if n_nearest_features is None:
        n_nearest_note = "n_nearest_features: all predictors used"
    else:
        n_nearest_note = f"n_nearest_features: {n_nearest_features}"

    if initial_strategy == "median":
        initial_strategy_note = "initial_strategy: median (skewed column detected)"
    else:
        initial_strategy_note = "initial_strategy: mean (all columns normal-skew)"

    converged = bool(model.n_iter_ < max_iter)
    warnings_: tuple[str, ...] = ()
    if not converged:
        warnings_ = (
            f"MICE hit its iteration cap without converging: max_iter={max_iter} ",
            "reached; consider increasing base_max_iter",
        )

    return UnitFitOutcome(
        fitted=FittedMICE(
            model=model,
            columns=list(cols),
            all_cols=all_cols,
            domain_snap_bounds=_domain_snap_bounds(ctx, cols),
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            estimator=_estimator_name(estimator),
            converged=converged,
            n_iter=int(model.n_iter_),
            warnings=warnings_,
            notes=(
                initial_strategy_note,
                n_nearest_note,
                f"predictors: block owns {len(cols)} columns, fit widened to ",
                f"{len(all_cols)} active numeric columns",
                f"scalar_predictor_fill: {len(scalar_filled_cols)} widened ",
                "predictor(s) filled to their own decided central tendency ",
                "before fit (train/serve parity, #418)",
            ),
        ),
    )


def fit_knn_unit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Fit the joint KNN block as one ``KNNImputer`` over the full active-numeric matrix.

    The distance space is widened past the block's own membership to every
    column in ``ctx.feature_columns`` (ADR-0093) — a block-only distance
    made a one-column KNN block a silent training-mean fill, since
    ``KNNImputer`` falls back to the column mean when a receiver shares no
    observed coordinate with any donor. The block still writes back only its
    own columns; :class:`~dataforge_ml.imputation._fitted_imputer._FittedKNN`
    carries the ``all_cols`` / ``columns`` split, matching
    :class:`~dataforge_ml.imputation._fitted_imputer.FittedMICE`. A widened
    predictor routed to a scalar strategy is filled to its own decided
    central tendency before the fit, exactly as the MICE block does
    (:func:`_fill_scalar_predictors`), so the training matrix's donor pool
    matches the frame it will serve on.

    The scaling params are learned here and stay learned fitted-state inside
    the unit — the recipe carries nothing about them (ADR-0062).

    Parameters
    ----------
    unit : ImputationUnit
        The ``"knn"`` unit. ``n_neighbors`` and ``weights`` are read from its
        hyperparameters.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context. ``feature_columns`` supplies the widened
        distance space.

    Returns
    -------
    UnitFitOutcome
        A fitted KNN unit plus the params and scaling signals.
    """
    from ._fitted_imputer import _FittedKNN

    cols = tuple(unit.columns)
    hyp = _hyperparameters(unit, ctx)
    n_neighbors = hyp["n_neighbors"]
    weights = hyp["weights"]

    extra_cols = [
        c
        for c in ctx.feature_columns
        if c not in cols and c in train_df.columns and train_df.schema[c].is_numeric()
    ]
    all_cols = list(cols) + extra_cols

    fit_df, scalar_filled_cols = _fill_scalar_predictors(train_df, ctx, extra_cols)

    arr = _df_to_numpy(fit_df, all_cols)

    # NaN-safe StandardScaler: missing cells stay NaN for KNNImputer to fill.
    col_means = np.nanmean(arr, axis=0)
    col_stds = np.nanstd(arr, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    arr_scaled = (arr - col_means) / col_stds

    model = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    model.fit(arr_scaled)

    return UnitFitOutcome(
        fitted=_FittedKNN(
            model=model,
            col_means=col_means,
            col_stds=col_stds,
            columns=list(cols),
            all_cols=all_cols,
            domain_snap_bounds=_domain_snap_bounds(ctx, cols),
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            estimator="KNNImputer",
            notes=(
                f"knn_params: n_neighbors={n_neighbors}, weights={weights} | ",
                f"distance space widened to {len(all_cols)} active numeric ",
                f"columns (block owns {len(cols)})",
                "knn_scaling: applied StandardScaler (nanmean/nanstd) across ",
                f"{len(all_cols)} feature columns",
                f"scalar_predictor_fill: {len(scalar_filled_cols)} widened ",
                "predictor(s) filled to their own decided central tendency ",
                "before fit (train/serve parity)",
            ),
        ),
    )


def fit_gmm_unit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Fit a two-component ``GaussianMixture`` for a bimodal column.

    The recipe's resolve-time bimodal centres seed the mixture, so the fit
    refines the modes Phase 1 already found rather than searching for them
    again.

    Parameters
    ----------
    unit : ImputationUnit
        The ``"gmm_sampling:{column}"`` unit. Its ``center1`` and ``center2``
        are read off the recipe's column estimates.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context.

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_units.FittedGMMSampling`, or
        no fit when the recipe carries no bimodal centres for the column, or the
        column has fewer than two observed values.
    """
    col = unit.columns[0]
    estimates = ctx.column_estimates.get(col)
    center1 = estimates.center1 if estimates is not None else None
    center2 = estimates.center2 if estimates is not None else None
    if center1 is None or center2 is None:
        return UnitFitOutcome(
            fallback_reason=(
                f"gmm_sampling: column '{col}' carries no bimodal centres on "
                f"the recipe. Set them with "
                f"ImputationRecipe.with_estimates('{col}', center1=..., center2=...)."
            )
        )

    series = train_df[col].drop_nulls()
    if len(series) < 2:
        return UnitFitOutcome(
            fallback_reason=(
                f"gmm_sampling: column '{col}' has fewer than two observed values"
            )
        )

    gmm = GaussianMixture(
        n_components=2,
        means_init=np.array([[center1], [center2]]),
        random_state=ctx.random_seed,
    )
    gmm.fit(series.to_numpy().reshape(-1, 1))

    return UnitFitOutcome(
        fitted=FittedGMMSampling(
            center1=gmm.means_[0][0],
            center2=gmm.means_[1][0],
            std1=np.sqrt(gmm.covariances_[0][0][0]),
            std2=np.sqrt(gmm.covariances_[1][0][0]),
            weight1=gmm.weights_[0],
            weight2=gmm.weights_[1],
            target_col=col,
            domain_snap_bounds=(
                estimates.domain_snap_bounds if estimates is not None else None
            ),
            random_seed=ctx.random_seed,
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            estimator="GaussianMixture",
            notes=("components: 2 (bimodal centres seeded from the recipe)",),
        ),
    )


def fit_cluster_unit(
    unit: ImputationUnit,
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Learn per-cluster fill values for a bimodal column.

    Two branches, chosen by whether the user declared a grouping variable for
    the column: group-wise central tendency over that variable, or
    centre-assignment against the recipe's two bimodal centres with a feature
    centroid per cluster so inference can assign an unseen row to one of them.

    Parameters
    ----------
    unit : ImputationUnit
        The ``"cluster_conditional:{column}"`` unit. ``central_tendency`` is
        read from the recipe's merged hyperparameters; ``center1`` /
        ``center2`` / ``feature_cols`` from the recipe's column estimates;
        ``grouping_variable`` off the routing.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Recipe-derived context.

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_units.FittedClusterConditional`,
        or no fit when the recipe carries no bimodal centres for the column, or
        the column has no observed values.
    """
    col = unit.columns[0]
    estimates = ctx.column_estimates.get(col)
    center1 = estimates.center1 if estimates is not None else None
    center2 = estimates.center2 if estimates is not None else None
    if center1 is None or center2 is None:
        return UnitFitOutcome(
            fallback_reason=(
                f"cluster_conditional: column '{col}' carries no bimodal "
                f"centres on the recipe. Set them with "
                f"ImputationRecipe.with_estimates('{col}', center1=..., center2=...)."
            )
        )

    use_mean = _hyperparameters(unit, ctx)["central_tendency"] == "mean"
    snap = estimates.domain_snap_bounds if estimates is not None else None
    routing = ctx.column_routings.get(col)
    grouping_var = routing.grouping_variable if routing is not None else None

    if grouping_var and grouping_var in train_df.columns:
        df_valid = train_df.select([col, grouping_var]).drop_nulls()
        if len(df_valid) == 0:
            return UnitFitOutcome(
                fallback_reason=(
                    f"cluster_conditional: column '{col}' has no rows with both "
                    f"a value and a '{grouping_var}' group"
                )
            )
        agg = pl.col(col).mean() if use_mean else pl.col(col).median()
        aggs = df_valid.group_by(grouping_var).agg(agg)
        return UnitFitOutcome(
            fitted=FittedClusterConditional(
                grouping_variable=grouping_var,
                group_fills=dict(
                    zip(aggs[grouping_var].to_list(), aggs[col].to_list())
                ),
                fill_1=None,
                fill_2=None,
                feature_centroid_1=None,
                feature_centroid_2=None,
                feature_cols=None,
                center1=center1,
                center2=center2,
                target_col=col,
                domain_snap_bounds=snap,
            ),
            signals=FitSignals(
                unit_id=unit.unit_id,
                strategy=unit.strategy,
                notes=(
                    f"branch: group-wise central tendency over '{grouping_var}'",
                    f"central_tendency: {'mean' if use_mean else 'median'}",
                ),
            ),
        )

    feat_cols = [c for c in (estimates.feature_cols or ()) if c in train_df.columns]
    df_valid = train_df.select([col] + feat_cols).drop_nulls(subset=[col])
    if len(df_valid) == 0:
        return UnitFitOutcome(
            fallback_reason=(
                f"cluster_conditional: column '{col}' has no observed values"
            )
        )

    vals = df_valid[col].to_numpy()
    mask1 = np.abs(vals - center1) <= np.abs(vals - center2)
    mask2 = ~mask1

    central = np.mean if use_mean else np.median
    fill1 = float(central(vals[mask1])) if mask1.any() else center1
    fill2 = float(central(vals[mask2])) if mask2.any() else center2

    centroid1 = None
    centroid2 = None
    if feat_cols:
        feat_arr = df_valid.select(feat_cols).to_numpy()
        # A feature that is entirely null within a cluster has no centroid
        # coordinate; zero keeps the distance defined rather than NaN-poisoning
        # every comparison against it.
        if mask1.any():
            centroid1 = np.nan_to_num(np.nanmean(feat_arr[mask1], axis=0))
        if mask2.any():
            centroid2 = np.nan_to_num(np.nanmean(feat_arr[mask2], axis=0))

    return UnitFitOutcome(
        fitted=FittedClusterConditional(
            grouping_variable=None,
            group_fills=None,
            fill_1=fill1,
            fill_2=fill2,
            feature_centroid_1=centroid1,
            feature_centroid_2=centroid2,
            feature_cols=list(feat_cols) if feat_cols else None,
            center1=center1,
            center2=center2,
            target_col=col,
            domain_snap_bounds=snap,
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            notes=(
                "branch: centre-assignment against the recipe's bimodal centres",
                f"central_tendency: {'mean' if use_mean else 'median'}",
            ),
        ),
    )
