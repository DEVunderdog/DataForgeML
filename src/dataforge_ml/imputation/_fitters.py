"""
Unit-shaped fitters — the one training implementation the surface calls.

Every imputation strategy that learns something from the training frame trains
here, in a function keyed on ``(ImputationUnit, train_df, hyperparameters,
n_jobs_inner)``. The stateless training surface (``_unit_fit``) dispatches into
these functions through :func:`_dispatch_unit_fit`.

Everything a fitter needs beyond the frame is either on the unit (the
decision-carried hyperparameters, ADR-0062), on the owning plan's
``ColumnImputationDecision`` (``model_choice`` and the facts about the data —
``domain_snap_bounds``, the bimodal centres, ``feature_cols``,
``grouping_variable``, ``constant_fill``), or on
:class:`UnitFitContext`. Nothing here reads the Phase 1 profile: a fitter is
handed a plan and a frame, never the profile the plan was derived from.

Like ``_fitted_units``, this module must not import the training surface; the
dependency edge runs surface → fitters, never back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from ._config import (
    ColumnImputationDecision,
    ImputationStrategy,
    ModelChoice,
    NumericImputationConfig,
)
from ._fit_signals import FitSignals
from ._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)
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

    Both engines can build this: the legacy path from the plan it already calls
    :func:`~dataforge_ml.imputation._decision_assembler.decide` to obtain, the
    executor from the plan it was constructed with.

    Parameters
    ----------
    column_decisions : Mapping[str, ColumnImputationDecision]
        The owning plan's per-column decisions. Read for ``model_choice`` and
        the facts about the data (``domain_snap_bounds``, the bimodal centres,
        ``feature_cols``, ``grouping_variable``, ``constant_fill``) — the
        decide-time facts a fitter honours rather than re-derives. No fitter
        reads ``config`` for any of them (ADR-0083).
    config : NumericImputationConfig
        Numeric imputation configuration.
    feature_columns : tuple[str, ...], optional
        The numeric column population a model-based unit may predict from. The
        joint MICE block's predictors are these columns minus its own owned
        columns (ADR-0079).
    random_seed : int, optional
        Seed for the stochastic strategies (GMM sampling).
    custom_estimators : Mapping[str, Any], optional
        The owning plan's unit-keyed user-supplied estimators (ADR-0083). Only
        the MICE fitter reads it, and only for a unit whose ``model_choice`` is
        :attr:`~dataforge_ml.ModelChoice.Custom`. The instance is the caller's
        own object, passed through by identity and never cloned here.
    """

    column_decisions: Mapping[str, ColumnImputationDecision]
    config: NumericImputationConfig
    feature_columns: tuple[str, ...] = ()
    random_seed: Optional[int] = None
    custom_estimators: Mapping[str, Any] = field(default_factory=dict)


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

    fitted: "Optional[FittedUnit]" = None
    signals: Optional[FitSignals] = None
    fallback_reason: Optional[str] = None


def _oversize_warning(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> Optional[str]:
    """Detect a strategy forced past its routing threshold (ADR-0074).

    Reconstructed purely from the unit's shape against the config thresholds — the
    plan stores no flag for it: KNN over ``knn_max_rows`` / ``knn_max_features``.
    Returns the warning text, or ``None`` when the shape sits within its
    routing envelope. It warns; it never blocks (ADR-0071).
    """
    n_rows = train_df.height
    cfg = ctx.config
    if unit.strategy == ImputationStrategy.KNN:
        n_features = len(unit.columns)
        if n_rows > cfg.knn_max_rows or n_features > cfg.knn_max_features:
            return (
                f"KNN forced past its routing threshold: {n_rows} rows "
                f"(cap {cfg.knn_max_rows}), {n_features} features "
                f"(cap {cfg.knn_max_features}); routing would have preferred "
                f"MICE at this shape"
            )
    return None


def _hyperparameters(unit: "ImputationUnit") -> dict[str, Any]:
    """Read a unit's decision-carried hyperparameters as a plain dict."""
    return dict(unit.hyperparameters) if unit.hyperparameters else {}


def _block_model_choice(
    ctx: UnitFitContext, columns: tuple[str, ...]
) -> Optional[ModelChoice]:
    """Return the estimator family the plan stamped on a block of columns.

    The joint blocks train one estimator for the whole block, so every column in
    the block carries the same choice; the first one that has it answers for all.
    ``None`` means the block routed to no estimator family (the ``Unpredictable``
    branch of ``RegressionEstimatorFactory``), which is the caller's cue to
    degrade.
    """
    for col in columns:
        decision = ctx.column_decisions.get(col)
        if decision is not None and decision.model_choice is not None:
            return decision.model_choice
    return None


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
    """Collect the plan's domain-snap bounds for the columns that carry them."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        decision = ctx.column_decisions.get(col)
        if decision is not None and decision.domain_snap_bounds is not None:
            bounds[col] = decision.domain_snap_bounds
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
    model-based strategy (KNN / MICE) is left untouched: it
    arrives raw at serve time too, so raw is already the matching frame.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Plan decisions; each predictor's decided strategy is read from here.
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
        decision = ctx.column_decisions.get(c)
        if decision is None:
            continue
        tendency = _CENTRAL_TENDENCY.get(str(decision.strategy))
        if tendency is None:
            continue
        fill_exprs.append(pl.col(c).fill_null(tendency(train_df, c)))
        filled_cols.append(c)
    if not fill_exprs:
        return train_df, filled_cols
    return train_df.with_columns(fill_exprs), filled_cols


def _dispatch_unit_fit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
    n_jobs_inner: int = 1,
) -> UnitFitOutcome:
    """Train one unit, dispatching on its strategy.

    The single door the training surface drives: given a plan's unit and the
    training frame, learn that unit's fitted state. Every strategy that trains
    something is reachable from here — a strategy with no fitter is a plan the
    engine should never have produced, and says so rather than silently
    returning an empty model.

    Parameters
    ----------
    unit : ImputationUnit
        The unit to train. Its ``hyperparameters`` are the resolved dials.
    train_df : pl.DataFrame
        Training split; every learned value comes from here.
    ctx : UnitFitContext
        Plan decisions and configuration the fitter honours.
    n_jobs_inner : int, default 1
        Inner estimator ``n_jobs`` (ADR-0056); ``1`` when nested under outer
        parallelism, ``-1`` for a unit running alone. Never affects results.

    Returns
    -------
    UnitFitOutcome
        The trained unit, or a reason it could not train.

    Raises
    ------
    ValueError
        If ``unit.strategy`` is one no fitter handles.
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
        f"handles. Structural strategies (Dropped / Passthrough / Indicator) "
        f"train nothing and must not be projected into a unit."
    )


def fit_scalar_unit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Learn the scalar fill value for a Mean / Median / Mode / Constant / MNAR unit.

    ``Constant`` reads the user's declared value and never touches ``train_df``.
    ``MNAR`` computes the central tendency the plan chose for it, rounded to a
    whole number when the column is integer-typed.

    Parameters
    ----------
    unit : ImputationUnit
        The single-column unit to train.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Plan decisions and configuration.

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
        decision = ctx.column_decisions.get(col)
        declared = decision.constant_fill if decision is not None else None
        if declared is None:
            return UnitFitOutcome(
                fallback_reason=(
                    f"constant: column '{col}' is routed to Constant but carries "
                    f"no constant_fill on the plan"
                )
            )
        return UnitFitOutcome(
            fitted=FittedScalar(target_col=col, fill_value=declared),
            signals=_signals(notes=("fill: declared constant value",)),
        )

    if unit.strategy == ImputationStrategy.MNAR:
        hyp = _hyperparameters(unit)
        tendency = hyp["central_tendency"]
        fill_value = _CENTRAL_TENDENCY[tendency](train_df, col)
        if train_df[col].dtype.is_integer():
            fill_value = float(round(fill_value))
        return UnitFitOutcome(
            fitted=FittedScalar(target_col=col, fill_value=fill_value),
            signals=_signals(notes=(f"central_tendency: {tendency}",)),
        )

    fill_value = _CENTRAL_TENDENCY[str(unit.strategy)](train_df, col)
    return UnitFitOutcome(
        fitted=FittedScalar(target_col=col, fill_value=fill_value),
        signals=_signals(notes=(f"central_tendency: {unit.strategy}",)),
    )


def fit_mice_unit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
    n_jobs_inner: int = 1,
) -> UnitFitOutcome:
    """Fit the joint MICE block as one ``IterativeImputer`` over the full active-numeric matrix.

    The block trains a single estimator, built from the ``model_choice`` the plan
    stamped on the block. A block whose columns are all ``Unpredictable`` carries
    no model choice and cannot train — it reports a reason instead.

    :attr:`~dataforge_ml.ModelChoice.Custom` is the one choice nothing is built
    for: the user's own estimator is taken off ``ctx.custom_estimators`` and used
    as-is (ADR-0083). It is not cloned, its ``n_jobs`` is not set, and a
    pre-fitted one is accepted without a raise or a warning — ``IterativeImputer``
    clones it per column and refits from scratch, so its prior state is inert. A
    ``Custom`` block whose slot is empty — every plan reloaded from bytes —
    reports a reason rather than falling back to a library estimator.

    The predictor set is widened past the block's own membership to every
    column in ``ctx.feature_columns`` — every active ``SemanticType.Numeric``
    column in the plan (ADR-0079), the same full feature set the former
    per-column regression fitter already read. The block still writes back only
    its own columns; :class:`~dataforge_ml.imputation._fitted_imputer.FittedMICE`
    carries the ``all_cols`` / ``columns`` split that generalizes the former
    per-column regression unit's single-target write-back restriction block-wide.

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
        Plan decisions and configuration. ``feature_columns`` supplies the
        widened predictor set.
    n_jobs_inner : int, default 1
        Inner estimator ``n_jobs`` (ADR-0056).

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_imputer.FittedMICE` plus the
        estimator / initial-strategy / predictor / convergence signals, or no fit
        when the block has no estimator family.
    """
    from ._fitted_imputer import FittedMICE

    cols = tuple(unit.columns)
    model_choice = _block_model_choice(ctx, cols)
    if model_choice is None:
        return UnitFitOutcome(
            fallback_reason="mice: all MICE columns Unpredictable; regression unsuitable"
        )

    hyp = _hyperparameters(unit)
    max_iter = hyp["max_iter"]
    tol = hyp["tol"]
    initial_strategy = hyp["initial_strategy"]
    n_nearest_features = hyp["n_nearest_features"]

    # Widen the predictor set past the block's own membership: every active
    # numeric column is a candidate predictor (ADR-0079), mirroring the feat_cols
    # the former per-column regression fitter read. The block still owns and
    # writes back only its own columns (cols), not all_cols.
    # A predictor whose frame dtype is not numeric cannot enter the joint matrix.
    # Only the frame can answer that: the plan's semantic types are decide-time
    # claims, and the manual door stamps ``Numeric`` on every column it plans
    # (ADR-0083), so a Passthrough string column would otherwise be widened into.
    extra_cols = [
        c
        for c in ctx.feature_columns
        if c not in cols
        and c in train_df.columns
        and train_df.schema[c].is_numeric()
    ]
    all_cols = list(cols) + extra_cols

    # Close the scalar-half train/serve skew (#418): the widened predictors
    # must be filled exactly as the serve-time pre-model snapshot fills them.
    fit_df, scalar_filled_cols = _fill_scalar_predictors(train_df, ctx, extra_cols)

    if model_choice == ModelChoice.Custom:
        # The label says "look elsewhere": the instance rides on the plan's
        # unit-keyed map and is used as-is, never cloned and never reconfigured
        # (ADR-0083). An empty slot means this plan was reloaded from bytes,
        # which never carry the estimator — that cannot train, and saying so is
        # the whole point of the label being a value rather than None.
        estimator = ctx.custom_estimators.get(unit.unit_id)
        if estimator is None:
            return UnitFitOutcome(
                fallback_reason=(
                    f"mice: the block is planned with ModelChoice.Custom but no "
                    f"estimator is carried for unit '{unit.unit_id}'. A "
                    f"user-supplied estimator is never serialized, so a reloaded "
                    f"plan must be re-authored with "
                    f"estimators={{'{unit.unit_id}': estimator}}."
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
        n_nearest_note = (
            f"n_nearest_features: all predictors used (block size "
            f"{len(cols)} <= {ctx.config.mice_n_nearest_features_min_cols})"
        )
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
            f"MICE hit its iteration cap without converging: max_iter={max_iter} "
            f"reached; consider increasing base_max_iter",
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
                f"predictors: block owns {len(cols)} columns, fit widened to "
                f"{len(all_cols)} active numeric columns",
                f"scalar_predictor_fill: {len(scalar_filled_cols)} widened "
                f"predictor(s) filled to their own decided central tendency "
                f"before fit (train/serve parity, #418)",
            ),
        ),
    )


def fit_knn_unit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Fit the joint KNN block as one ``KNNImputer`` over a standardized matrix.

    The scaling params are learned here and stay learned fitted-state inside the
    unit — the plan carries nothing about them (ADR-0062).

    Parameters
    ----------
    unit : ImputationUnit
        The ``"knn"`` unit. ``n_neighbors`` and ``weights`` are read from its
        hyperparameters.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Plan decisions and configuration.

    Returns
    -------
    UnitFitOutcome
        A fitted KNN unit plus the params and scaling signals.
    """
    from ._fitted_imputer import _FittedKNN

    cols = tuple(unit.columns)
    hyp = _hyperparameters(unit)
    n_neighbors = hyp["n_neighbors"]
    weights = hyp["weights"]

    arr = _df_to_numpy(train_df, list(cols))

    # NaN-safe StandardScaler: missing cells stay NaN for KNNImputer to fill.
    col_means = np.nanmean(arr, axis=0)
    col_stds = np.nanstd(arr, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    arr_scaled = (arr - col_means) / col_stds

    model = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    model.fit(arr_scaled)

    warnings_: tuple[str, ...] = ()
    oversize = _oversize_warning(unit, train_df, ctx)
    if oversize is not None:
        warnings_ = (oversize,)

    return UnitFitOutcome(
        fitted=_FittedKNN(
            model=model,
            col_means=col_means,
            col_stds=col_stds,
            columns=list(cols),
            domain_snap_bounds=_domain_snap_bounds(ctx, cols),
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            estimator="KNNImputer",
            warnings=warnings_,
            notes=(
                f"knn_params: n_neighbors={n_neighbors}, weights={weights} | "
                f"n_features={len(cols)}",
                f"knn_scaling: applied StandardScaler (nanmean/nanstd) "
                f"across {len(cols)} feature columns",
            ),
        ),
    )


def fit_gmm_unit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Fit a two-component ``GaussianMixture`` for a bimodal column.

    The plan's decide-time bimodal centres seed the mixture, so the fit refines
    the modes Phase 1 already found rather than searching for them again.

    Parameters
    ----------
    unit : ImputationUnit
        The ``"gmm_sampling:{column}"`` unit. Its ``center1`` and ``center2``
        are read off the column's decision.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Plan decisions and configuration.

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_units.FittedGMMSampling`, or
        no fit when the plan carries no bimodal centres or the column has fewer
        than two observed values.
    """
    col = unit.columns[0]
    decision = ctx.column_decisions.get(col)
    center1 = decision.center1 if decision is not None else None
    center2 = decision.center2 if decision is not None else None
    if center1 is None or center2 is None:
        return UnitFitOutcome(
            fallback_reason=(
                f"gmm_sampling: column '{col}' carries no bimodal centres on the plan"
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
                decision.domain_snap_bounds if decision is not None else None
            ),
            random_seed=ctx.random_seed,
        ),
        signals=FitSignals(
            unit_id=unit.unit_id,
            strategy=unit.strategy,
            estimator="GaussianMixture",
            notes=("components: 2 (bimodal centres seeded from the plan)",),
        ),
    )


def fit_cluster_unit(
    unit: "ImputationUnit",
    train_df: pl.DataFrame,
    ctx: UnitFitContext,
) -> UnitFitOutcome:
    """Learn per-cluster fill values for a bimodal column.

    Two branches, chosen by whether the user declared a grouping variable for the
    column: group-wise central tendency over that variable, or centre-assignment
    against the plan's two bimodal centres with a feature centroid per cluster so
    inference can assign an unseen row to one of them.

    Parameters
    ----------
    unit : ImputationUnit
        The ``"cluster_conditional:{column}"`` unit. ``central_tendency`` is
        read from its hyperparameters; ``center1`` / ``center2`` /
        ``feature_cols`` / ``grouping_variable`` off the column's decision.
    train_df : pl.DataFrame
        Training split.
    ctx : UnitFitContext
        Plan decisions and configuration.

    Returns
    -------
    UnitFitOutcome
        A :class:`~dataforge_ml.imputation._fitted_units.FittedClusterConditional`,
        or no fit when the plan carries no bimodal centres or the column has no
        observed values.
    """
    col = unit.columns[0]
    decision = ctx.column_decisions.get(col)
    center1 = decision.center1 if decision is not None else None
    center2 = decision.center2 if decision is not None else None
    if center1 is None or center2 is None:
        return UnitFitOutcome(
            fallback_reason=(
                f"cluster_conditional: column '{col}' carries no bimodal centres "
                f"on the plan"
            )
        )

    use_mean = _hyperparameters(unit)["central_tendency"] == "mean"
    snap = decision.domain_snap_bounds
    grouping_var = decision.grouping_variable

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

    feat_cols = [c for c in (decision.feature_cols or ()) if c in train_df.columns]
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
                "branch: centre-assignment against the plan's bimodal centres",
                f"central_tendency: {'mean' if use_mean else 'median'}",
            ),
        ),
    )
