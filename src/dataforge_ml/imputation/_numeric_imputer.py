"""
NumericImputer — Phase 2 sub-processor for SemanticType.Numeric columns.

Applies the Numeric Imputation Decision Priority (see issue #5) during fit()
and returns a _NumericFitBundle.  All fill values and models are computed
exclusively from train_df; the profile is used only for strategy routing.
Strategy routing is delegated to _StrategyRouter (pure, DataFrame-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline

from ..config import SemanticType
from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag
from ..profiling._numeric_config import (
    NonlinearityTag,
    NumericStats,
    SkewSeverity,
)
from ._config import (
    ColumnImputationRecord,
    ImputationFitDiagnostic,
    ImputationStrategy,
    NumericImputationConfig,
)
from ._regression_estimator_factory import RegressionEstimatorFactory
from ._strategy_router import _StrategyRouter
from ._utils import _df_to_numpy

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile, StructuralProfileResult


@dataclass
class FittedRegression:
    """Fitted state produced by _fit_regression for a single target column.

    Parameters
    ----------
    model : Any
        Fitted ``IterativeImputer`` that handles missing feature values
        internally.
    target_idx : int
        Index of the target column in the joint array. Always ``0`` by
        construction; stored explicitly so inference never relies on a
        positional convention.
    all_cols : list[str]
        Full column list ``[col] + feat_cols`` in joint-array order.
    signals : list[str]
        Human-readable entries recording the estimator choice and any
        convergence warning. Appended to ``ColumnImputationRecord.signals``
        by the caller after a successful fit.
    max_iter_used : int
        The effective ``max_iter`` value passed to ``IterativeImputer``.
        Used by the diagnostic to determine whether the model converged
        (``n_iter_ < max_iter_used``) or was stopped by the iteration cap.
    """

    model: Any
    target_idx: int
    all_cols: list[str]
    signals: list[str] = field(default_factory=list)
    max_iter_used: int = 0


@dataclass
class _NumericFitBundle:
    """Return bundle from NumericImputer.fit()."""

    records: list[ColumnImputationRecord]
    models: dict[str, Any] = field(default_factory=dict)
    model_cols: dict[str, list[str]] = field(default_factory=dict)


class NumericImputer:
    """Stateless sub-processor for numeric column imputation."""

    def fit(
        self,
        train_df: pl.DataFrame,
        columns: list[str],
        profile: StructuralProfileResult,
        config: NumericImputationConfig,
        mnar_columns: set[str],
    ) -> _NumericFitBundle:
        n_rows = len(train_df)
        n_features = len(columns)

        _validate_model_based_size_guards(config, n_rows, n_features)

        # Detect multi-MAR: ≥2 non-dropped, non-MNAR columns with MARSuspect flag
        mar_candidates: set[str] = set()
        for col in columns:
            cp = profile.columns.get(col)
            if cp is None or cp.missingness is None or col in mnar_columns:
                continue
            if cp.missingness.has_flag(MissingnessFlag.DropCandidate):
                continue
            if cp.missingness.has_flag(MissingnessFlag.MARSuspect):
                mar_candidates.add(col)
        multi_mar = len(mar_candidates) >= 2

        feature_correlation = profile.dataset.feature_correlation

        router = _StrategyRouter()

        # First pass: determine strategy + compute scalar fill values
        records: list[ColumnImputationRecord] = []
        for col in columns:
            cp = profile.columns.get(col)
            if cp is None:
                continue

            strategy, signals = router.route(
                col=col,
                cp=cp,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                mnar_columns=mnar_columns,
                feature_correlation=feature_correlation,
                per_column_strategy=config.per_column_strategy or {},
                per_column_constant_fill=config.per_column_constant_fill or {},
            )

            fill_value = _resolve_fill_value(
                train_df, col, cp, strategy, config.per_column_constant_fill or {}
            )
            indicator_added = strategy == ImputationStrategy.MNAR
            domain_snap_bounds = _resolve_domain_snap_bounds(cp, strategy)

            records.append(
                ColumnImputationRecord(
                    column=col,
                    semantic_type=SemanticType.Numeric,
                    strategy=strategy,
                    fill_value=fill_value,
                    indicator_added=indicator_added,
                    signals=signals,
                    domain_snap_bounds=domain_snap_bounds,
                )
            )

        # Second pass: fit model-based strategies
        mice_cols = [r.column for r in records if r.strategy == ImputationStrategy.MICE]
        knn_cols = [r.column for r in records if r.strategy == ImputationStrategy.KNN]
        reg_cols = [
            r.column for r in records if r.strategy == ImputationStrategy.Regression
        ]

        models: dict[str, Any] = {}
        model_cols: dict[str, list[str]] = {}

        if mice_cols:
            arr = _df_to_numpy(train_df, mice_cols)

            # Collect NonlinearityTag and NumericStats for each MICE column; default to Linear when absent
            mice_tags: list[NonlinearityTag] = []
            mice_stats: list[Optional[NumericStats]] = []
            for col in mice_cols:
                cp = profile.columns.get(col)
                stats = (
                    cp.stats
                    if cp is not None and isinstance(cp.stats, NumericStats)
                    else None
                )
                tag = (
                    stats.nonlinearity_tag
                    if stats is not None and stats.nonlinearity_tag is not None
                    else NonlinearityTag.Linear
                )
                mice_tags.append(tag)
                mice_stats.append(stats)

            winning_tag = _mice_winning_tag(mice_tags)

            # All-Unpredictable → skip MICE; fall back each column to Median individually
            if winning_tag == NonlinearityTag.Unpredictable:
                mice_col_set = set(mice_cols)
                for i, rec in enumerate(records):
                    if rec.column in mice_col_set:
                        records[i] = _fallback_to_median(
                            train_df,
                            rec.column,
                            rec,
                            "mice: all MICE columns Unpredictable; regression unsuitable",
                        )
            else:
                estimator = RegressionEstimatorFactory.build(
                    winning_tag, n_rows, config
                )
                if isinstance(estimator, Pipeline):
                    estimator_name = "Pipeline(StandardScaler+BayesianRidge)"
                else:
                    estimator_name = type(estimator).__name__

                max_iter = _compute_mice_max_iter(
                    winning_tag, arr, mice_stats, profile, mice_cols, config
                )
                if config.mice_max_iter is not None:
                    max_iter = config.mice_max_iter
                tol = _compute_mice_tol(winning_tag, mice_stats)
                initial_strategy = _mice_initial_strategy(mice_stats)
                n_nearest_features, n_nearest_signal = _compute_mice_n_nearest_features(
                    arr, profile, mice_cols, config
                )

                mice_model = IterativeImputer(
                    estimator=estimator,
                    random_state=0,
                    max_iter=max_iter,
                    tol=tol,
                    initial_strategy=initial_strategy,
                    n_nearest_features=n_nearest_features,
                )
                mice_model.fit(arr)
                models["mice"] = mice_model
                model_cols["mice"] = list(mice_cols)

                estimator_signal = (
                    f"mice_estimator: {estimator_name} (tag={winning_tag})"
                )
                if initial_strategy == "median":
                    initial_strategy_signal = (
                        "mice_initial_strategy: median (skewed column detected)"
                    )
                else:
                    initial_strategy_signal = (
                        "mice_initial_strategy: mean (all columns normal-skew)"
                    )
                mice_col_set = set(mice_cols)
                for rec in records:
                    if rec.column in mice_col_set:
                        rec.signals.append(estimator_signal)
                        rec.signals.append(initial_strategy_signal)
                        rec.signals.append(n_nearest_signal)

                if mice_model.n_iter_ == max_iter:
                    convergence_signal = (
                        f"mice_convergence_warning: max_iter={max_iter} reached; "
                        f"consider increasing base_max_iter"
                    )
                else:
                    convergence_signal = (
                        f"mice_converged: {mice_model.n_iter_} iterations "
                        f"(max_iter={max_iter})"
                    )
                for rec in records:
                    if rec.column in mice_col_set:
                        rec.signals.append(convergence_signal)

                mice_diagnostics = _compute_mice_diagnostics(
                    arr=arr,
                    mice_cols=mice_cols,
                    mice_model=mice_model,
                    config=config,
                    max_iter=max_iter,
                    estimator=estimator,
                    tol=tol,
                    initial_strategy=initial_strategy,
                    n_nearest_features=n_nearest_features,
                )
                for rec in records:
                    if rec.column in mice_col_set:
                        rec.diagnostic = mice_diagnostics.get(rec.column)

        if knn_cols:
            from ._fitted_imputer import _FittedKNN

            arr = _df_to_numpy(train_df, knn_cols)
            n_knn_features = len(knn_cols)
            n_rows = len(train_df)

            # --- Signal computation ---
            total_cells = arr.size
            miss_frac = (
                float(np.isnan(arr).sum() / total_cells) if total_cells > 0 else 0.0
            )
            complete_rows = int((~np.isnan(arr).any(axis=1)).sum())
            complete_frac = complete_rows / n_rows if n_rows > 0 else 0.0

            # --- Adaptive n_neighbors formula ---
            base_k = max(config.knn_min_neighbors, int(np.sqrt(n_knn_features)))
            k_raw = base_k * (1.0 + miss_frac) * (1.0 / max(complete_frac, 0.1)) ** 0.5
            adaptive_raw = max(config.knn_min_neighbors, int(k_raw))
            n_neighbors = min(
                adaptive_raw,
                n_rows - 1,
                config.knn_max_neighbors,
            )
            n_neighbors = max(1, n_neighbors)
            if config.knn_n_neighbors is not None:
                k_capped: Optional[bool] = None
                n_neighbors = config.knn_n_neighbors
            else:
                k_capped = adaptive_raw > (n_rows - 1)

            # --- Reliability-based weights formula ---
            reliability_high = (
                miss_frac < config.knn_distance_weight_max_null_ratio
                and n_knn_features <= config.knn_distance_weight_max_features
            )
            weights = "distance" if reliability_high else "uniform"

            # --- NaN-safe StandardScaler ---
            col_means = np.nanmean(arr, axis=0)
            col_stds = np.nanstd(arr, axis=0)
            col_stds[col_stds == 0.0] = 1.0
            arr_scaled = (arr - col_means) / col_stds  # NaN cells remain NaN

            # --- Fit KNNImputer on scaled matrix ---
            knn_model = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            knn_model.fit(arr_scaled)

            # --- Store _FittedKNN ---
            models["knn"] = _FittedKNN(
                model=knn_model,
                col_means=col_means,
                col_stds=col_stds,
            )
            model_cols["knn"] = list(knn_cols)

            # --- Append signals to each KNN column's record ---
            knn_params_signal = (
                f"knn_params: n_neighbors={n_neighbors}, weights={weights} | "
                f"n_features={n_knn_features}, miss_frac={miss_frac:.2f}, "
                f"complete_frac={complete_frac:.2f}"
            )
            knn_scaling_signal = (
                f"knn_scaling: applied StandardScaler (nanmean/nanstd) "
                f"across {n_knn_features} feature columns"
            )
            for rec in records:
                if rec.column in knn_cols:
                    rec.signals.append(knn_params_signal)
                    rec.signals.append(knn_scaling_signal)

            knn_col_set = set(knn_cols)
            knn_diagnostics = _compute_knn_diagnostics(
                arr=arr,
                knn_cols=knn_cols,
                fitted_knn=models["knn"],
                config=config,
                n_neighbors_used=n_neighbors,
                weights=weights,
                k_capped=k_capped,
            )
            for rec in records:
                if rec.column in knn_col_set:
                    rec.diagnostic = knn_diagnostics.get(rec.column)

        if reg_cols:
            for col in reg_cols:
                feat_cols = [c for c in columns if c != col]
                cp = profile.columns.get(col)
                stats = (
                    cp.stats
                    if cp is not None and isinstance(cp.stats, NumericStats)
                    else None
                )
                tag = (
                    stats.nonlinearity_tag
                    if stats is not None and stats.nonlinearity_tag is not None
                    else NonlinearityTag.Linear
                )

                rec_idx = next(i for i, r in enumerate(records) if r.column == col)
                record = records[rec_idx]

                fitted = _fit_regression(
                    train_df, col, feat_cols, tag, n_rows, config, stats
                )

                if fitted is None:
                    if not feat_cols:
                        reason = "no feature columns available"
                    elif tag == NonlinearityTag.Unpredictable:
                        reason = "nonlinearity_tag=Unpredictable: regression unsuitable"
                    else:
                        reason = "insufficient target observations for regression"
                    if record.domain_snap_bounds is not None:
                        records[rec_idx] = _fallback_to_mode(
                            train_df, col, record, reason
                        )
                    else:
                        records[rec_idx] = _fallback_to_median(
                            train_df, col, record, reason
                        )
                    continue

                model_key = f"regression:{col}"
                if model_key in models:
                    raise RuntimeError(
                        f"Duplicate regression model key: column '{col}' appears more than "
                        f"once in the regression column list."
                    )

                models[model_key] = fitted
                model_cols[model_key] = fitted.all_cols

                for signal in fitted.signals:
                    record.signals.append(signal)

                record.diagnostic = _compute_regression_diagnostic(
                    train_df=train_df,
                    col=col,
                    feat_cols=feat_cols,
                    fitted_reg=fitted,
                    tag=tag,
                    config=config,
                    n_rows=n_rows,
                )

        return _NumericFitBundle(records=records, models=models, model_cols=model_cols)


# ---------------------------------------------------------------------------
# Fit-time size-guard validation for per_column_strategy model-based overrides
# ---------------------------------------------------------------------------


def _validate_model_based_size_guards(
    config: NumericImputationConfig,
    n_rows: int,
    n_features: int,
) -> None:
    for col, strategy in (config.per_column_strategy or {}).items():
        if strategy == ImputationStrategy.Regression:
            if n_rows < config.regression_min_rows:
                raise ValueError(
                    f"Column '{col}': per_column_strategy=Regression requires "
                    f"n_rows >= regression_min_rows={config.regression_min_rows:,} "
                    f"(got n_rows={n_rows:,}). "
                    f"Increase the dataset size or lower regression_min_rows in NumericImputationConfig."
                )
        elif strategy == ImputationStrategy.KNN:
            if n_rows > config.knn_max_rows:
                raise ValueError(
                    f"Column '{col}': per_column_strategy=KNN requires "
                    f"n_rows <= knn_max_rows={config.knn_max_rows:,} "
                    f"(got n_rows={n_rows:,}). "
                    f"Raise knn_max_rows in NumericImputationConfig."
                )
            if n_features > config.knn_max_features:
                raise ValueError(
                    f"Column '{col}': per_column_strategy=KNN requires "
                    f"n_features <= knn_max_features={config.knn_max_features} "
                    f"(got n_features={n_features}). "
                    f"Raise knn_max_features in NumericImputationConfig."
                )
        # MICE: same size guard as the Severe threshold path in _mcar_model_strategy — no guard applied


# ---------------------------------------------------------------------------
# Fill value and domain snap helpers (called by NumericImputer.fit after routing)
# ---------------------------------------------------------------------------


def _resolve_fill_value(
    train_df: pl.DataFrame,
    col: str,
    cp: "ColumnProfile",
    strategy: ImputationStrategy,
    per_column_constant_fill: "Optional[dict[str, float]]" = None,
) -> Optional[float]:
    """Compute the scalar fill value for a column after strategy routing.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split used to compute statistics.
    col : str
        Column name.
    cp : ColumnProfile
        Phase 1 profile; used to read skewness and numeric kind for MNAR and
        BoundedDiscrete snapping.
    strategy : ImputationStrategy
        Strategy returned by ``_StrategyRouter.route()``.
    per_column_constant_fill : dict[str, float], optional
        User-declared constant fill values.  When ``strategy`` is ``Constant``
        and ``col`` is present in this dict, the declared value is returned
        directly without touching ``train_df``.

    Returns
    -------
    float or None
        Computed fill value, or ``None`` for model-based and structural
        strategies that do not use a scalar fill.
    """
    if strategy == ImputationStrategy.Constant:
        if per_column_constant_fill is not None and col in per_column_constant_fill:
            return per_column_constant_fill[col]
        return None

    if strategy in (
        ImputationStrategy.Dropped,
        ImputationStrategy.Passthrough,
        ImputationStrategy.KNN,
        ImputationStrategy.Regression,
        ImputationStrategy.MICE,
    ):
        return None

    if strategy == ImputationStrategy.Mode:
        return _compute_mode(train_df, col)

    if strategy == ImputationStrategy.MNAR:
        if cp.numeric_kind == NumericKind.BoundedDiscrete:
            return _compute_mode(train_df, col)
        mnar_stats = cp.stats if isinstance(cp.stats, NumericStats) else None
        skew_sev = mnar_stats.skewness_severity if mnar_stats is not None else None
        if skew_sev == SkewSeverity.Normal:
            fill_value: float = _compute_mean(train_df, col)
        else:
            fill_value = _compute_median(train_df, col)
        if train_df[col].dtype.is_integer():
            fill_value = float(round(fill_value))
        return fill_value

    if strategy == ImputationStrategy.Mean:
        return _compute_mean(train_df, col)

    if strategy == ImputationStrategy.Median:
        return _compute_median(train_df, col)

    return None


def _resolve_domain_snap_bounds(
    cp: "ColumnProfile",
    strategy: ImputationStrategy,
) -> Optional[tuple[float, float]]:
    """Return domain-snap bounds for BoundedDiscrete model-based strategies.

    Parameters
    ----------
    cp : ColumnProfile
        Phase 1 profile; ``numeric_kind`` and ``stats.min`` / ``stats.max``
        are read to determine whether snapping applies.
    strategy : ImputationStrategy
        Strategy returned by ``_StrategyRouter.route()``.

    Returns
    -------
    tuple[float, float] or None
        ``(min, max)`` when the column is BoundedDiscrete and the strategy is
        model-based; ``None`` otherwise.
    """
    if cp.numeric_kind != NumericKind.BoundedDiscrete:
        return None
    if strategy not in (
        ImputationStrategy.KNN,
        ImputationStrategy.Regression,
        ImputationStrategy.MICE,
    ):
        return None
    stats = cp.stats if isinstance(cp.stats, NumericStats) else None
    if stats is not None and stats.min is not None and stats.max is not None:
        return (stats.min, stats.max)
    return None


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------


def _fit_regression(
    train_df: pl.DataFrame,
    col: str,
    feat_cols: list[str],
    tag: NonlinearityTag,
    n_rows: int,
    config: NumericImputationConfig,
    stats: Optional[NumericStats] = None,
) -> FittedRegression | None:
    """Fit a single-column IterativeImputer for regression-based imputation.

    Returns ``None`` when fitting cannot proceed: no feature columns, the
    ``Unpredictable`` tag routes the factory to return ``None``, or the
    target column has fewer than two non-null observations.  The caller is
    responsible for routing a ``None`` return to ``_fallback_to_median``.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split.
    col : str
        Target column to impute.
    feat_cols : list[str]
        Predictor columns.
    tag : NonlinearityTag
        Nonlinearity classification for ``col`` from Phase 1.
    n_rows : int
        Number of rows in ``train_df``.
    config : NumericImputationConfig
        Imputation configuration supplying estimator thresholds and
        ``base_max_iter``.
    stats : NumericStats, optional
        Phase 1 numeric statistics for ``col``.  Used to derive IQR-relative
        ``tol`` and the R² gap signal for ``max_iter`` computation.

    Returns
    -------
    FittedRegression or None
        Fitted regression bundle on success; ``None`` to signal fallback.
    """
    if not feat_cols:
        return None

    if len(train_df[col].drop_nulls()) < 2:
        return None

    estimator = RegressionEstimatorFactory.build(tag, n_rows, config)
    if estimator is None:
        return None

    all_cols = [col] + feat_cols
    assert all_cols[0] == col, (
        f"Expected target column '{col}' at index 0 of the joint array, "
        f"got '{all_cols[0]}'. Column order was modified before array construction."
    )

    arr = _df_to_numpy(train_df, all_cols)
    max_iter = _compute_max_iter(tag, feat_cols, arr, stats, config)
    _reg_override = (config.per_column_max_iter or {}).get(col)
    if _reg_override is not None:
        max_iter = int(_reg_override)
    tol = _compute_tol(tag, stats)

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        tol=tol,
        random_state=0,
    )
    imputer.fit(arr)

    if isinstance(estimator, Pipeline):
        estimator_name = "Pipeline(StandardScaler+BayesianRidge)"
    else:
        estimator_name = type(estimator).__name__

    signals: list[str] = [f"regression_estimator: {estimator_name} (tag={tag})"]
    if imputer.n_iter_ == max_iter:
        signals.append(
            f"convergence_warning: max_iter={max_iter} reached; "
            f"consider increasing via NumericImputationConfig"
        )

    return FittedRegression(
        model=imputer,
        target_idx=0,
        all_cols=all_cols,
        signals=signals,
        max_iter_used=max_iter,
    )


def _fallback_to_median(
    train_df: pl.DataFrame,
    col: str,
    record: ColumnImputationRecord,
    reason: str,
) -> ColumnImputationRecord:
    """Return a copy of ``record`` updated to Median strategy.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split used to compute the median fill value.
    col : str
        Column name; used to look up values in ``train_df``.
    record : ColumnImputationRecord
        Original record for the column.  Not mutated.
    reason : str
        Human-readable explanation appended to ``signals`` as
        ``"fallback_to_median: <reason>"``.

    Returns
    -------
    ColumnImputationRecord
        New record with ``strategy=Median``, a computed ``fill_value``, and
        ``reason`` appended to ``signals``.
    """
    from dataclasses import replace

    signals = list(record.signals) + [f"fallback_to_median: {reason}"]
    return replace(
        record,
        strategy=ImputationStrategy.Median,
        fill_value=_compute_median(train_df, col),
        signals=signals,
    )


def _fallback_to_mode(
    train_df: pl.DataFrame,
    col: str,
    record: ColumnImputationRecord,
    reason: str,
) -> ColumnImputationRecord:
    """Return a copy of ``record`` updated to Mode strategy.

    Used when ``_fit_regression`` returns ``None`` for a BoundedDiscrete column.
    ``domain_snap_bounds`` is cleared because the model-based bounds are no
    longer relevant when falling back to a scalar fill.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split used to compute the mode fill value.
    col : str
        Column name; used to look up values in ``train_df``.
    record : ColumnImputationRecord
        Original record for the column.  Not mutated.
    reason : str
        Human-readable explanation appended to ``signals`` as
        ``"fallback_to_mode: <reason>"``.

    Returns
    -------
    ColumnImputationRecord
        New record with ``strategy=Mode``, a computed ``fill_value``,
        ``domain_snap_bounds=None``, and ``reason`` appended to ``signals``.
    """
    from dataclasses import replace

    signals = list(record.signals) + [f"fallback_to_mode: {reason}"]
    return replace(
        record,
        strategy=ImputationStrategy.Mode,
        fill_value=_compute_mode(train_df, col),
        domain_snap_bounds=None,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Dynamic IterativeImputer parameter helpers
# ---------------------------------------------------------------------------

_MICE_TAG_PRECEDENCE: dict[NonlinearityTag, int] = {
    NonlinearityTag.Unpredictable: 0,
    NonlinearityTag.Linear: 1,
    NonlinearityTag.MonotonicNonlinear: 2,
    NonlinearityTag.ComplexNonlinear: 3,
}


def _mice_winning_tag(tags: list[NonlinearityTag]) -> NonlinearityTag:
    return max(tags, key=lambda t: _MICE_TAG_PRECEDENCE.get(t, 0))


def _compute_max_iter(
    tag: NonlinearityTag,
    feat_cols: list[str],
    arr: np.ndarray,
    stats: Optional[NumericStats],
    config: NumericImputationConfig,
) -> int:
    base = config.base_max_iter

    # Signal 1: ComplexNonlinear → more iterations required
    if tag == NonlinearityTag.ComplexNonlinear:
        base += 5

    # Signal 2: count of feature columns with missing values
    feat_arr = arr[:, 1:]
    n_missing_feat_cols = int(np.isnan(feat_arr).any(axis=0).sum())
    base += n_missing_feat_cols * 2

    # Signal 3: low R² gap indicates a near-linear relationship → fewer iterations
    if stats is not None and stats.r2_gap is not None and stats.r2_gap < 0.05:
        base = max(1, base - 3)

    # Signal 4: high pairwise correlation among missing-feature columns → more iterations
    if feat_arr.shape[1] >= 2:
        missing_feat_mask = np.isnan(feat_arr).any(axis=0)
        missing_feat_arr = feat_arr[:, missing_feat_mask]
        if missing_feat_arr.shape[1] >= 2:
            try:
                n_mf = missing_feat_arr.shape[1]
                corr_sum, n_pairs = 0.0, 0
                for i in range(n_mf):
                    for j in range(i + 1, n_mf):
                        valid = ~(
                            np.isnan(missing_feat_arr[:, i])
                            | np.isnan(missing_feat_arr[:, j])
                        )
                        if valid.sum() >= 2:
                            c = np.corrcoef(
                                missing_feat_arr[valid, i], missing_feat_arr[valid, j]
                            )[0, 1]
                            if np.isfinite(c):
                                corr_sum += abs(c)
                                n_pairs += 1
                if n_pairs > 0 and (corr_sum / n_pairs) >= 0.7:
                    base += 3
            except Exception:  # noqa: BLE001
                pass

    # Signal 5: low complete-row fraction → more iterations needed
    complete_fraction = (~np.isnan(arr).any(axis=1)).sum() / len(arr)
    if complete_fraction < 0.2:
        base += 5
    elif complete_fraction < 0.5:
        base += 3

    return max(1, base)


def _compute_tol(
    tag: NonlinearityTag,
    stats: Optional[NumericStats],
) -> float:
    if stats is not None and stats.iqr is not None and stats.iqr > 0:
        scaling_factor = 5e-5 if tag == NonlinearityTag.ComplexNonlinear else 1e-4
        return max(1e-7, stats.iqr * scaling_factor)
    return 1e-3


def _compute_mice_max_iter(
    winning_tag: NonlinearityTag,
    arr: np.ndarray,
    mice_stats: list[Optional[NumericStats]],
    profile: StructuralProfileResult,
    mice_cols: list[str],
    config: NumericImputationConfig,
) -> int:
    """Compute ``max_iter`` for the MICE ``IterativeImputer`` from seven data signals.

    Mirrors ``_compute_max_iter`` for the single-column Regression strategy but
    uses MICE-specific aggregation: minimum R² gap across the block (worst-case
    convergence speed), maximum pairwise inter-column Pearson ``|r|`` (strongest
    coupling driver), and full-matrix missingness fraction.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    arr : np.ndarray
        Numeric array of the full MICE column matrix, shape ``(n_rows, n_mice_cols)``.
        NaN values mark missing cells.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column in the same order as columns in
        ``arr``.  Entries may be ``None`` when stats were not computed.
    profile : StructuralProfileResult
        Full Phase 1 result.  Used to read pre-computed Pearson correlations from
        ``feature_correlation`` when available.
    mice_cols : list[str]
        Column names corresponding to columns of ``arr``.
    config : NumericImputationConfig
        Imputation configuration supplying ``base_max_iter``.

    Returns
    -------
    int
        Computed ``max_iter`` value, always at least ``1``.
    """
    base = config.base_max_iter

    # Signal 1: ComplexNonlinear → more iterations required
    if winning_tag == NonlinearityTag.ComplexNonlinear:
        base += 5

    # Signal 2: full MICE matrix missingness fraction
    total_cells = arr.size
    miss_frac = float(np.isnan(arr).sum() / total_cells) if total_cells > 0 else 0.0
    if miss_frac >= 0.4:
        base += 5
    elif miss_frac >= 0.2:
        base += 3
    elif miss_frac >= 0.1:
        base += 2

    # Signal 3: minimum r2_gap across MICE columns; small gap means near-linear → fewer iterations
    r2_gaps = [s.r2_gap for s in mice_stats if s is not None and s.r2_gap is not None]
    if r2_gaps and min(r2_gaps) < 0.05:
        base = max(1, base - 3)

    # Signal 4: maximum pairwise Pearson |r| among MICE columns; high correlation → more iterations
    max_pearson = _mice_max_inter_correlation(arr, profile, mice_cols)
    if max_pearson is not None and max_pearson >= 0.7:
        base += 3

    # Signal 5: low complete-row fraction → more iterations needed
    complete_fraction = float((~np.isnan(arr).any(axis=1)).sum()) / len(arr)
    if complete_fraction < 0.2:
        base += 5
    elif complete_fraction < 0.5:
        base += 3

    return max(1, base)


def _compute_mice_tol(
    winning_tag: NonlinearityTag,
    mice_stats: list[Optional[NumericStats]],
) -> float:
    """Compute the convergence tolerance for the MICE ``IterativeImputer``.

    Uses the minimum IQR across all MICE columns so that tolerance is
    calibrated to the narrowest-range column in the block.  Applies tighter
    scaling when the block contains complex non-linear structure.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column.  Entries may be ``None``
        when stats were not computed.

    Returns
    -------
    float
        Convergence tolerance, always at least ``1e-7``.  Falls back to
        ``1e-3`` when no IQR is available.
    """
    iqrs = [
        s.iqr for s in mice_stats if s is not None and s.iqr is not None and s.iqr > 0
    ]
    if iqrs:
        min_iqr = min(iqrs)
        scaling_factor = (
            5e-5 if winning_tag == NonlinearityTag.ComplexNonlinear else 1e-4
        )
        return max(1e-7, min_iqr * scaling_factor)
    return 1e-3


_MICE_SKEW_TRIGGERS_MEDIAN: frozenset[SkewSeverity] = frozenset(
    {
        SkewSeverity.Moderate,
        SkewSeverity.High,
        SkewSeverity.Severe,
    }
)


def _mice_initial_strategy(mice_stats: list[Optional[NumericStats]]) -> str:
    """Determine the ``initial_strategy`` for the MICE ``IterativeImputer``.

    Returns ``"median"`` when any MICE column has ``SkewSeverity >= Moderate``;
    otherwise returns ``"mean"``.

    Parameters
    ----------
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column.  Entries may be ``None``
        when stats were not computed.

    Returns
    -------
    str
        Either ``"median"`` or ``"mean"``.
    """
    for stats in mice_stats:
        if stats is not None and stats.skewness_severity in _MICE_SKEW_TRIGGERS_MEDIAN:
            return "median"
    return "mean"


def _mice_max_inter_correlation(
    arr: np.ndarray,
    profile: StructuralProfileResult,
    mice_cols: list[str],
) -> Optional[float]:
    feature_corr = profile.dataset.feature_correlation
    if feature_corr is not None and len(mice_cols) >= 2:
        values = []
        for i in range(len(mice_cols)):
            for j in range(i + 1, len(mice_cols)):
                r = feature_corr.get_pearson(mice_cols[i], mice_cols[j])
                if r is not None:
                    values.append(abs(r))
        if values:
            return max(values)

    if arr.shape[1] >= 2:
        try:
            n_cols = arr.shape[1]
            max_r = 0.0
            found = False
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    valid = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
                    if valid.sum() >= 2:
                        c = np.corrcoef(arr[valid, i], arr[valid, j])[0, 1]
                        if np.isfinite(c):
                            max_r = max(max_r, abs(c))
                            found = True
            if found:
                return max_r
        except Exception:  # noqa: BLE001
            pass
    return None


def _compute_mice_n_nearest_features(
    arr: np.ndarray,
    profile: "StructuralProfileResult",
    mice_cols: list[str],
    config: NumericImputationConfig,
) -> tuple[Optional[int], str]:
    """Compute ``n_nearest_features`` for the MICE ``IterativeImputer``.

    For blocks at or below ``mice_n_nearest_features_min_cols`` columns all
    predictors are used (``n_nearest_features=None``). For larger blocks, the
    number of informative predictors per column is counted using value-level
    Pearson correlations and the median count across columns is returned,
    capped at ``mice_max_nearest_features``.

    Correlations are read from ``CorrelationProfiler`` output stored in
    ``profile.dataset.feature_correlation`` when available; otherwise they are
    computed directly from ``arr``.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of the full MICE column matrix, shape
        ``(n_rows, n_mice_cols)``. NaN values mark missing cells.
    profile : StructuralProfileResult
        Full Phase 1 result. Used to read pre-computed Pearson correlations
        from ``feature_correlation`` when available.
    mice_cols : list[str]
        Column names corresponding to columns of ``arr``.
    config : NumericImputationConfig
        Imputation configuration supplying ``mice_n_nearest_features_min_cols``,
        ``mice_max_nearest_features``, and ``mice_correlation_threshold``.

    Returns
    -------
    tuple[int or None, str]
        Computed ``n_nearest_features`` value (``None`` for small blocks) and
        a human-readable signal string recording the decision.
    """
    n_cols = len(mice_cols)
    if n_cols <= config.mice_n_nearest_features_min_cols:
        return None, (
            f"mice_n_nearest_features: all predictors used "
            f"— block ({n_cols} cols) at or below min_cols threshold "
            f"({config.mice_n_nearest_features_min_cols})"
        )

    threshold = config.mice_correlation_threshold
    feature_corr = profile.dataset.feature_correlation
    counts: list[int] = []

    for i, col_i in enumerate(mice_cols):
        count = 0
        for j, col_j in enumerate(mice_cols):
            if i == j:
                continue
            r: Optional[float] = None
            if feature_corr is not None:
                r = feature_corr.get_pearson(col_i, col_j)
            if r is None:
                valid = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
                if valid.sum() >= 2:
                    try:
                        c = np.corrcoef(arr[valid, i], arr[valid, j])[0, 1]
                        if np.isfinite(c):
                            r = float(c)
                    except Exception:  # noqa: BLE001
                        pass
            if r is not None and abs(r) > threshold:
                count += 1
        counts.append(count)

    median_count = int(np.median(counts)) if counts else 0
    n_nearest = min(max(1, median_count), config.mice_max_nearest_features)

    return n_nearest, (
        f"mice_n_nearest_features: {n_nearest} "
        f"(median informative predictors={median_count}, "
        f"capped at mice_max_nearest_features={config.mice_max_nearest_features}, "
        f"threshold={threshold})"
    )


# ---------------------------------------------------------------------------
# Scalar statistics helpers
# ---------------------------------------------------------------------------


def _clean(series: pl.Series) -> pl.Series:
    return series.drop_nulls()


def _compute_mean(df: pl.DataFrame, col: str) -> float:
    val = _clean(df[col]).mean()
    return float(val) if val is not None else 0.0


def _compute_median(df: pl.DataFrame, col: str) -> float:
    val = _clean(df[col]).median()
    return float(val) if val is not None else 0.0


def _compute_mode(df: pl.DataFrame, col: str) -> float:
    clean = _clean(df[col])
    if len(clean) == 0:
        return 0.0
    modes = clean.mode().sort()
    return float(modes[0])


# ---------------------------------------------------------------------------
# Fit diagnostic helpers (ImputationFitDiagnostic computation)
# ---------------------------------------------------------------------------


def _compute_regression_diagnostic(
    train_df: pl.DataFrame,
    col: str,
    feat_cols: list[str],
    fitted_reg: FittedRegression,
    tag: NonlinearityTag,
    config: NumericImputationConfig,
    n_rows: int,
) -> ImputationFitDiagnostic:
    """Compute fit quality metrics for a single Regression-strategy column.

    Runs k-fold cross-validated R² on complete rows when enough are available
    (k = ``config.refit_r2_cv_folds``), then collects distribution statistics
    from the imputed values the fitted model produces for the originally null
    rows.  The final stored model is never re-trained during this function.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split used during ``fit()``.
    col : str
        Target column name.
    feat_cols : list[str]
        Predictor columns used by this regression model.
    fitted_reg : FittedRegression
        The final fitted regression bundle already stored in ``models``.
    tag : NonlinearityTag
        Nonlinearity tag for ``col``; used to select the same estimator class
        for the temporary diagnostic model.
    config : NumericImputationConfig
        Imputation configuration; ``refit_r2_min_complete_rows`` and
        ``refit_r2_cv_folds`` control the R² evaluation.
    n_rows : int
        Number of rows in ``train_df``.

    Returns
    -------
    ImputationFitDiagnostic
        Populated diagnostic instance.
    """

    all_cols = [col] + feat_cols
    arr = _df_to_numpy(train_df, all_cols)

    # Observed stats from non-null target values
    target_arr = arr[:, 0]
    obs_vals = target_arr[~np.isnan(target_arr)]
    observed_mean = float(np.mean(obs_vals)) if len(obs_vals) > 0 else 0.0
    observed_std = float(np.std(obs_vals)) if len(obs_vals) > 0 else 0.0

    # Complete rows for k-fold CV
    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())

    r2_train: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    if n_complete >= config.refit_r2_min_complete_rows:
        arr_complete = arr[np.where(complete_mask)[0]]
        n_folds = config.refit_r2_cv_folds

        rng = np.random.default_rng(0)
        perm = rng.permutation(n_complete)
        arr_shuffled = arr_complete[perm]

        fold_size = n_complete // n_folds
        fold_r2s: list[float] = []
        fold_rmses: list[float] = []
        fold_maes: list[float] = []

        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_folds - 1 else n_complete

            arr_val_sub = arr_shuffled[val_start:val_end]
            arr_train_sub = np.concatenate(
                [arr_shuffled[:val_start], arr_shuffled[val_end:]]
            )

            y_true = arr_val_sub[:, 0]
            if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
                continue

            estimator_sub = RegressionEstimatorFactory.build(
                tag, len(arr_train_sub), config
            )
            if estimator_sub is None:
                continue

            max_iter_sub = _compute_max_iter(tag, feat_cols, arr_train_sub, None, config)
            tol_sub = _compute_tol(tag, None)
            temp_imputer = IterativeImputer(
                estimator=estimator_sub,
                max_iter=max_iter_sub,
                tol=tol_sub,
                random_state=0,
            )
            temp_imputer.fit(arr_train_sub)

            arr_val_masked = arr_val_sub.copy()
            arr_val_masked[:, 0] = np.nan
            arr_val_filled = temp_imputer.transform(arr_val_masked)

            y_pred = arr_val_filled[:, 0]
            try:
                r2_fold, rmse_fold, mae_fold = _compute_fold_metrics(y_true, y_pred)
                fold_r2s.append(r2_fold)
                fold_rmses.append(rmse_fold)
                fold_maes.append(mae_fold)
            except Exception:  # noqa: BLE001
                pass

        if fold_r2s:
            r2_train = float(np.mean(fold_r2s))
            rmse = float(np.mean(fold_rmses))
            mae = float(np.mean(fold_maes))

    # Imputed values: apply final model to the full training array
    null_mask = np.isnan(target_arr)
    imputed_mean = 0.0
    imputed_std = 0.0
    if null_mask.any():
        arr_filled = fitted_reg.model.transform(arr)
        imputed_vals = arr_filled[null_mask, 0]
        imputed_mean = float(np.mean(imputed_vals))
        imputed_std = float(np.std(imputed_vals))

    variance_ratio = imputed_std / observed_std if observed_std > 0.0 else 0.0
    converged = fitted_reg.model.n_iter_ < fitted_reg.max_iter_used
    n_iter = int(fitted_reg.model.n_iter_)

    return ImputationFitDiagnostic(
        r2_train=r2_train,
        rmse=rmse,
        mae=mae,
        converged=converged,
        n_iter=n_iter,
        imputed_mean=imputed_mean,
        imputed_std=imputed_std,
        observed_mean=observed_mean,
        observed_std=observed_std,
        variance_ratio=variance_ratio,
    )


def _compute_knn_diagnostics(
    arr: np.ndarray,
    knn_cols: list[str],
    fitted_knn: Any,
    config: NumericImputationConfig,
    n_neighbors_used: int,
    weights: str,
    k_capped: Optional[bool],
) -> dict[str, ImputationFitDiagnostic]:
    """Compute per-column fit quality metrics for all KNN-strategy columns.

    Runs k-fold cross-validated R² using one shared throwaway KNN model per
    fold, then collects distribution statistics for each column's imputed
    values.  The final stored model is never re-trained during this function.

    Parameters
    ----------
    arr : np.ndarray
        Raw (unscaled) KNN matrix of shape ``(n_rows, n_knn_cols)``.  NaN
        marks missing cells.
    knn_cols : list[str]
        Column names in the same order as columns in ``arr``.
    fitted_knn : Any
        The ``_FittedKNN`` instance already stored in ``models["knn"]``.
        Provides ``col_means``, ``col_stds``, and the fitted ``KNNImputer``.
    config : NumericImputationConfig
        Imputation configuration; ``refit_r2_min_complete_rows`` and
        ``refit_r2_cv_folds`` control the R² evaluation.
    n_neighbors_used : int
        Actual ``n_neighbors`` used when fitting the final KNN model; stored
        on every returned ``ImputationFitDiagnostic`` and reused for the
        throwaway fold models.
    weights : str
        ``weights`` strategy used when fitting the final KNN model; reused
        for the throwaway models.
    k_capped : bool, optional
        Pre-computed ``k_capped`` flag (see ``ImputationFitDiagnostic``).
        ``None`` when the ``knn_n_neighbors`` override is active.

    Returns
    -------
    dict[str, ImputationFitDiagnostic]
        One entry per column in ``knn_cols``.  ``converged`` and ``n_iter``
        are always ``None`` (not applicable to KNN).
    """

    col_means: np.ndarray = fitted_knn.col_means
    col_stds: np.ndarray = fitted_knn.col_stds

    arr_scaled = (arr - col_means) / col_stds

    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())

    knn_r2: dict[str, Optional[float]] = {col: None for col in knn_cols}
    knn_rmse: dict[str, Optional[float]] = {col: None for col in knn_cols}
    knn_mae: dict[str, Optional[float]] = {col: None for col in knn_cols}

    if n_complete >= config.refit_r2_min_complete_rows:
        arr_complete_scaled = arr_scaled[np.where(complete_mask)[0]]
        n_folds = config.refit_r2_cv_folds

        rng = np.random.default_rng(0)
        perm = rng.permutation(n_complete)
        arr_shuffled = arr_complete_scaled[perm]

        fold_size = n_complete // n_folds
        col_fold_r2s: dict[str, list[float]] = {col: [] for col in knn_cols}
        col_fold_rmses: dict[str, list[float]] = {col: [] for col in knn_cols}
        col_fold_maes: dict[str, list[float]] = {col: [] for col in knn_cols}

        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_folds - 1 else n_complete

            arr_val_sub = arr_shuffled[val_start:val_end]
            arr_train_sub = np.concatenate(
                [arr_shuffled[:val_start], arr_shuffled[val_end:]]
            )

            temp_knn = KNNImputer(n_neighbors=n_neighbors_used, weights=weights)
            temp_knn.fit(arr_train_sub)

            for k, col_k in enumerate(knn_cols):
                y_true = arr_val_sub[:, k]
                if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
                    continue

                arr_val_masked = arr_val_sub.copy()
                arr_val_masked[:, k] = np.nan
                arr_val_filled = temp_knn.transform(arr_val_masked)

                y_pred = arr_val_filled[:, k]
                y_pred_inv = y_pred * col_stds[k] + col_means[k]
                y_true_inv = y_true * col_stds[k] + col_means[k]
                try:
                    r2_fold, rmse_fold, mae_fold = _compute_fold_metrics(y_true_inv, y_pred_inv)
                    col_fold_r2s[col_k].append(r2_fold)
                    col_fold_rmses[col_k].append(rmse_fold)
                    col_fold_maes[col_k].append(mae_fold)
                except Exception:  # noqa: BLE001
                    pass

        for col_k in knn_cols:
            if col_fold_r2s[col_k]:
                knn_r2[col_k] = float(np.mean(col_fold_r2s[col_k]))
                knn_rmse[col_k] = float(np.mean(col_fold_rmses[col_k]))
                knn_mae[col_k] = float(np.mean(col_fold_maes[col_k]))

    # Apply final model to full matrix to obtain imputed values for null rows
    arr_scaled_filled = fitted_knn.model.transform(arr_scaled)
    arr_filled = arr_scaled_filled * col_stds + col_means

    diagnostics: dict[str, ImputationFitDiagnostic] = {}
    for k, col_k in enumerate(knn_cols):
        col_arr = arr[:, k]

        obs_vals = col_arr[~np.isnan(col_arr)]
        observed_mean = float(np.mean(obs_vals)) if len(obs_vals) > 0 else 0.0
        observed_std = float(np.std(obs_vals)) if len(obs_vals) > 0 else 0.0

        null_mask = np.isnan(col_arr)
        imputed_mean = 0.0
        imputed_std = 0.0
        if null_mask.any():
            imputed_vals = arr_filled[null_mask, k]
            imputed_mean = float(np.mean(imputed_vals))
            imputed_std = float(np.std(imputed_vals))

        variance_ratio = imputed_std / observed_std if observed_std > 0.0 else 0.0

        diagnostics[col_k] = ImputationFitDiagnostic(
            r2_train=knn_r2[col_k],
            rmse=knn_rmse[col_k],
            mae=knn_mae[col_k],
            converged=None,
            n_iter=None,
            imputed_mean=imputed_mean,
            imputed_std=imputed_std,
            observed_mean=observed_mean,
            observed_std=observed_std,
            variance_ratio=variance_ratio,
            n_neighbors_used=n_neighbors_used,
            k_capped=k_capped,
        )

    return diagnostics


def _compute_mice_diagnostics(
    arr: np.ndarray,
    mice_cols: list[str],
    mice_model: Any,
    config: NumericImputationConfig,
    max_iter: int,
    estimator: Any,
    tol: float,
    initial_strategy: str,
    n_nearest_features: Optional[int],
) -> dict[str, ImputationFitDiagnostic]:

    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())

    mice_r2: dict[str, Optional[float]] = {col: None for col in mice_cols}
    mice_rmse: dict[str, Optional[float]] = {col: None for col in mice_cols}
    mice_mae: dict[str, Optional[float]] = {col: None for col in mice_cols}

    if n_complete >= config.refit_r2_min_complete_rows:
        arr_complete = arr[np.where(complete_mask)[0]]
        n_folds = config.refit_r2_cv_folds

        rng = np.random.default_rng(0)
        perm = rng.permutation(n_complete)
        arr_shuffled = arr_complete[perm]

        fold_size = n_complete // n_folds
        col_fold_r2s: dict[str, list[float]] = {col: [] for col in mice_cols}
        col_fold_rmses: dict[str, list[float]] = {col: [] for col in mice_cols}
        col_fold_maes: dict[str, list[float]] = {col: [] for col in mice_cols}

        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_folds - 1 else n_complete

            arr_val_sub = arr_shuffled[val_start:val_end]
            arr_train_sub = np.concatenate(
                [arr_shuffled[:val_start], arr_shuffled[val_end:]]
            )

            temp_mice = IterativeImputer(
                estimator=estimator,
                random_state=0,
                max_iter=max_iter,
                tol=tol,
                initial_strategy=initial_strategy,
                n_nearest_features=n_nearest_features,
            )
            temp_mice.fit(arr_train_sub)

            for k, col_k in enumerate(mice_cols):
                y_true = arr_val_sub[:, k]
                if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
                    continue

                arr_val_masked = arr_val_sub.copy()
                arr_val_masked[:, k] = np.nan
                arr_val_filled = temp_mice.transform(arr_val_masked)

                y_pred = arr_val_filled[:, k]
                try:
                    r2_fold, rmse_fold, mae_fold = _compute_fold_metrics(y_true, y_pred)
                    col_fold_r2s[col_k].append(r2_fold)
                    col_fold_rmses[col_k].append(rmse_fold)
                    col_fold_maes[col_k].append(mae_fold)
                except Exception:  # noqa: BLE001
                    pass

        for col_k in mice_cols:
            if col_fold_r2s[col_k]:
                mice_r2[col_k] = float(np.mean(col_fold_r2s[col_k]))
                mice_rmse[col_k] = float(np.mean(col_fold_rmses[col_k]))
                mice_mae[col_k] = float(np.mean(col_fold_maes[col_k]))

    arr_filled = mice_model.transform(arr)
    converged = mice_model.n_iter_ < max_iter
    n_iter = int(mice_model.n_iter_)

    diagnostics: dict[str, ImputationFitDiagnostic] = {}
    for k, col_k in enumerate(mice_cols):
        col_arr = arr[:, k]

        obs_vals = col_arr[~np.isnan(col_arr)]
        observed_mean = float(np.mean(obs_vals)) if len(obs_vals) > 0 else 0.0
        observed_std = float(np.std(obs_vals)) if len(obs_vals) > 0 else 0.0

        null_mask = np.isnan(col_arr)
        imputed_mean = 0.0
        imputed_std = 0.0
        if null_mask.any():
            imputed_vals = arr_filled[null_mask, k]
            imputed_mean = float(np.mean(imputed_vals))
            imputed_std = float(np.std(imputed_vals))

        variance_ratio = imputed_std / observed_std if observed_std > 0.0 else 0.0

        diagnostics[col_k] = ImputationFitDiagnostic(
            r2_train=mice_r2[col_k],
            rmse=mice_rmse[col_k],
            mae=mice_mae[col_k],
            converged=converged,
            n_iter=n_iter,
            imputed_mean=imputed_mean,
            imputed_std=imputed_std,
            observed_mean=observed_mean,
            observed_std=observed_std,
            variance_ratio=variance_ratio,
        )

    return diagnostics


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Compute R2, RMSE, and MAE for a single validation fold.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    tuple[float, float, float]
        (r2, rmse, mae). R2 defaults to 0.0 if computation fails.
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    if np.array_equal(y_true, y_pred):
        return 1.0, 0.0, 0.0

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    rmse = float(np.sqrt(max(0.0, mean_squared_error(y_true, y_pred))))
    mae = float(max(0.0, mean_absolute_error(y_true, y_pred)))

    return r2, rmse, mae
