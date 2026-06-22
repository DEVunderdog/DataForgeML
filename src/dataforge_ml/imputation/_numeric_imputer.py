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
from ._config import ColumnImputationRecord, ImputationStrategy, NumericImputationConfig
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
    """

    model: Any
    target_idx: int
    all_cols: list[str]
    signals: list[str] = field(default_factory=list)


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
            )

            fill_value = _resolve_fill_value(train_df, col, cp, strategy)
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
            n_neighbors = min(
                max(config.knn_min_neighbors, int(k_raw)),
                n_rows - 1,
                config.knn_max_neighbors,
            )
            n_neighbors = max(1, n_neighbors)

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

        return _NumericFitBundle(records=records, models=models, model_cols=model_cols)


# ---------------------------------------------------------------------------
# Fill value and domain snap helpers (called by NumericImputer.fit after routing)
# ---------------------------------------------------------------------------


def _resolve_fill_value(
    train_df: pl.DataFrame,
    col: str,
    cp: "ColumnProfile",
    strategy: ImputationStrategy,
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

    Returns
    -------
    float or None
        Computed fill value, or ``None`` for model-based and structural
        strategies that do not use a scalar fill.
    """
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
