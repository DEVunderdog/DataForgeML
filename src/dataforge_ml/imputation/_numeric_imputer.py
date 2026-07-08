"""
NumericImputer — Phase 2 sub-processor for SemanticType.Numeric columns.

Applies the Numeric Imputation Decision Priority (see issue #5) during fit()
and returns a _NumericFitBundle.  All fill values and models are computed
exclusively from train_df; the profile is used only for strategy routing.
Strategy routing is delegated to _StrategyRouter (pure, DataFrame-free).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

from ..config import SemanticType
from ..observability import Emitter
from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag
from ..profiling._numeric_config import (
    NonlinearityTag,
    NumericStats,
    SkewSeverity,
)
from ._config import (
    ColumnImputationRecord,
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
        Read by the Evaluation phase to determine whether the model converged
        (``n_iter_ < max_iter_used``) or was stopped by the iteration cap.
    """

    model: Any
    target_idx: int
    all_cols: list[str]
    signals: list[str] = field(default_factory=list)
    max_iter_used: int = 0


@dataclass
class FittedClusterConditional:
    """Fitted state for Cluster-Conditional Imputation (branches 1 and 3).

    Parameters
    ----------
    grouping_variable : str or None
        Name of the categorical column used for grouping (branch 1).
    group_fills : dict of Any to float or None
        Mapping from group values to fill values (branch 1).
    fill_1 : float or None
        Fill value for cluster 1 (branch 3).
    fill_2 : float or None
        Fill value for cluster 2 (branch 3).
    feature_centroid_1 : np.ndarray or None
        Feature centroid for cluster 1 (branch 3).
    feature_centroid_2 : np.ndarray or None
        Feature centroid for cluster 2 (branch 3).
    feature_cols : list of str or None
        Names of the feature columns used to compute distances (branch 3).
    center1 : float
        Mean of cluster 1 from the univariate bimodal fit.
    center2 : float
        Mean of cluster 2 from the univariate bimodal fit.
    """
    grouping_variable: Optional[str]
    group_fills: Optional[dict[Any, float]]
    fill_1: Optional[float]
    fill_2: Optional[float]
    feature_centroid_1: Optional[np.ndarray]
    feature_centroid_2: Optional[np.ndarray]
    feature_cols: Optional[list[str]]
    center1: float
    center2: float

    def to_dict(self) -> dict:
        return {
            "grouping_variable": self.grouping_variable,
            "group_fills": self.group_fills,
            "fill_1": self.fill_1,
            "fill_2": self.fill_2,
            "feature_centroid_1": self.feature_centroid_1.tolist() if self.feature_centroid_1 is not None else None,
            "feature_centroid_2": self.feature_centroid_2.tolist() if self.feature_centroid_2 is not None else None,
            "feature_cols": self.feature_cols,
            "center1": self.center1,
            "center2": self.center2,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FittedClusterConditional":
        return cls(
            grouping_variable=data.get("grouping_variable"),
            group_fills=data.get("group_fills"),
            fill_1=data.get("fill_1"),
            fill_2=data.get("fill_2"),
            feature_centroid_1=np.array(data["feature_centroid_1"]) if data.get("feature_centroid_1") is not None else None,
            feature_centroid_2=np.array(data["feature_centroid_2"]) if data.get("feature_centroid_2") is not None else None,
            feature_cols=data.get("feature_cols"),
            center1=float(data["center1"]) if data.get("center1") is not None else 0.0,
            center2=float(data["center2"]) if data.get("center2") is not None else 0.0,
        )



@dataclass
class FittedGMMSampling:
    """Fitted state for GMM Sampling Imputation (branch 4).

    Parameters
    ----------
    center1 : float
        Mean of the first GMM component.
    center2 : float
        Mean of the second GMM component.
    std1 : float
        Standard deviation of the first GMM component.
    std2 : float
        Standard deviation of the second GMM component.
    weight1 : float
        Mixing weight of the first GMM component.
    weight2 : float
        Mixing weight of the second GMM component.
    """
    center1: float
    center2: float
    std1: float
    std2: float
    weight1: float
    weight2: float

    def to_dict(self) -> dict:
        return {
            "center1": self.center1,
            "center2": self.center2,
            "std1": self.std1,
            "std2": self.std2,
            "weight1": self.weight1,
            "weight2": self.weight2,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FittedGMMSampling":
        return cls(
            center1=float(data["center1"]),
            center2=float(data["center2"]),
            std1=float(data["std1"]),
            std2=float(data["std2"]),
            weight1=float(data["weight1"]),
            weight2=float(data["weight2"]),
        )



@dataclass
class _NumericFitBundle:
    """Return bundle from NumericImputer.fit()."""

    records: list[ColumnImputationRecord]
    models: dict[str, Any] = field(default_factory=dict)
    model_cols: dict[str, list[str]] = field(default_factory=dict)


_MODEL_BASED_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.MICE,
        ImputationStrategy.KNN,
        ImputationStrategy.Regression,
        ImputationStrategy.ClusterConditional,
        ImputationStrategy.GMMSampling,
    }
)


def _resolve_fit_workers(max_workers: Optional[int], n_units: int) -> int:
    """Resolve the thread count for the concurrent numeric fit.

    Never exceeds the number of independent units of work; ``max_workers=None``
    auto-sizes to the available CPU count, and ``max_workers=1`` forces a
    sequential fit.
    """
    if n_units <= 1:
        return 1
    if max_workers is None:
        resolved = os.cpu_count() or 1
    else:
        resolved = max_workers
    return max(1, min(resolved, n_units))


class NumericImputer:
    """Stateless sub-processor for numeric column imputation."""

    def fit(
        self,
        train_df: pl.DataFrame,
        columns: list[str],
        profile: StructuralProfileResult,
        config: NumericImputationConfig,
        mnar_columns: set[str],
        random_seed: Optional[int] = None,
        emitter: Optional[Emitter] = None,
    ) -> _NumericFitBundle:
        """Route numeric columns and learn their fill values and models.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training split; all fill values and models are computed here.
        columns : list[str]
            Numeric columns to fit.
        profile : StructuralProfileResult
            Full-dataset Phase 1 profile, used for strategy routing.
        config : NumericImputationConfig
            Numeric imputation configuration.
        mnar_columns : set[str]
            Columns the user has declared MNAR.
        random_seed : int, optional
            Seed for the stochastic strategies (GMM sampling).
        emitter : Emitter, optional
            Progress Emitter threaded from ``ImputationOrchestrator`` so that
            per-column ``item`` heartbeats and finer ``substep`` heartbeats emit
            from inside the fit as work happens; ``None`` disables emission.

        Returns
        -------
        _NumericFitBundle
            Records, fitted models, and per-model column lists.  ``fit`` does
            only routing + model learning; fit-quality diagnostics are the
            opt-in Evaluation phase's job (ADR-0058).
        """
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

        # Scalar strategies (Mean/Median/Mode/Constant/MNAR) learn their fill
        # value in the routing pass above, so their ``item`` heartbeat fires now
        # — the moment that column is actually fitted.  Model-based columns emit
        # their heartbeat from inside their strategy block below (ADR-0055).
        if emitter is not None:
            for rec in records:
                if rec.strategy not in _MODEL_BASED_STRATEGIES:
                    emitter.item(rec.column)

        # Second pass: fit model-based strategies
        mice_cols = [r.column for r in records if r.strategy == ImputationStrategy.MICE]
        knn_cols = [r.column for r in records if r.strategy == ImputationStrategy.KNN]
        reg_cols = [
            r.column for r in records if r.strategy == ImputationStrategy.Regression
        ]
        cluster_cols = [
            r.column for r in records if r.strategy == ImputationStrategy.ClusterConditional
        ]
        gmm_cols = [
            r.column for r in records if r.strategy == ImputationStrategy.GMMSampling
        ]

        models: dict[str, Any] = {}
        model_cols: dict[str, list[str]] = {}

        # Under the independence rule (ADR-0056) the mutually-independent
        # strategy blocks (MICE, KNN, the Regression set, GMM, cluster) and the
        # independent columns within a per-column strategy carry no shared data
        # dependency, so each is assembled as its own unit of work and the units
        # fit concurrently on threads.  Routing (the first pass above) stays
        # sequential because fitting consumes its output, and the joint MICE/KNN
        # models stay internally serial (a solver's columns are coupled).  Fixed
        # seeds and fold splits make a concurrent fit byte-identical to a
        # sequential one; ``config.max_workers`` caps the parallelism and
        # ``n_jobs_inner`` (resolved below) pins nested estimators to one core.
        units: list[Callable[[], None]] = []

        def _run_mice() -> None:
            if emitter is not None:
                emitter.substep(
                    f"MICE block fitting over {len(mice_cols)} columns"
                )
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
                
                if stats is not None and stats.bimodal_stats is not None and tag == NonlinearityTag.Linear:
                    tag = NonlinearityTag.MonotonicNonlinear

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
                    winning_tag, n_rows, config, n_jobs=n_jobs_inner
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

            # The joint MICE fit is complete (or fell back to per-column
            # Median); the heartbeat for every column in the block fires now.
            if emitter is not None:
                for col in mice_cols:
                    emitter.item(col)

        if mice_cols:
            units.append(_run_mice)

        def _run_knn() -> None:
            from ._fitted_imputer import _FittedKNN

            if emitter is not None:
                emitter.substep(
                    f"KNN block fitting over {len(knn_cols)} columns"
                )

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
                n_neighbors = config.knn_n_neighbors

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

            # The joint KNN fit is complete; heartbeat every column in the block.
            if emitter is not None:
                for col in knn_cols:
                    emitter.item(col)

        if knn_cols:
            units.append(_run_knn)

        if reg_cols:
            n_reg = len(reg_cols)

            def _run_regression_col(col: str, reg_pos: int) -> None:
                if emitter is not None:
                    emitter.substep(
                        f"regression fitting: {col}",
                        column=col,
                        index=reg_pos,
                        total=n_reg,
                    )
                    emitter.item(col)
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
                if stats is not None and stats.bimodal_stats is not None and tag == NonlinearityTag.Linear:
                    tag = NonlinearityTag.MonotonicNonlinear

                fitted = _fit_regression(
                    train_df, col, feat_cols, tag, n_rows, config, stats,
                    n_jobs=n_jobs_inner,
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
                    return

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

            for reg_pos, col in enumerate(reg_cols, start=1):
                units.append(partial(_run_regression_col, col, reg_pos))

        if gmm_cols:
            n_gmm = len(gmm_cols)

            def _run_gmm_col(col: str, gmm_pos: int) -> None:
                if emitter is not None:
                    emitter.substep(
                        f"GMM sampling fitting: {col}",
                        column=col,
                        index=gmm_pos,
                        total=n_gmm,
                    )
                    emitter.item(col)
                series = train_df[col].drop_nulls()
                if len(series) < 2:
                    return
                cp = profile.columns.get(col)
                stats = cp.stats if cp is not None and isinstance(cp.stats, NumericStats) else None
                if stats and stats.bimodal_stats:
                    c1, c2 = stats.bimodal_stats.center1, stats.bimodal_stats.center2
                    
                    gmm = GaussianMixture(
                        n_components=2, 
                        means_init=np.array([[c1], [c2]]),
                        random_state=random_seed
                    )
                    gmm.fit(series.to_numpy().reshape(-1, 1))
                    models[f"gmm:{col}"] = FittedGMMSampling(
                        center1=gmm.means_[0][0],
                        center2=gmm.means_[1][0],
                        std1=np.sqrt(gmm.covariances_[0][0][0]),
                        std2=np.sqrt(gmm.covariances_[1][0][0]),
                        weight1=gmm.weights_[0],
                        weight2=gmm.weights_[1]
                    )

            for gmm_pos, col in enumerate(gmm_cols, start=1):
                units.append(partial(_run_gmm_col, col, gmm_pos))

        if cluster_cols:
            n_cluster = len(cluster_cols)

            def _run_cluster_col(col: str, cluster_pos: int) -> None:
                if emitter is not None:
                    emitter.substep(
                        f"cluster-conditional fitting: {col}",
                        column=col,
                        index=cluster_pos,
                        total=n_cluster,
                    )
                    emitter.item(col)
                cp = profile.columns.get(col)
                stats = cp.stats if cp is not None and isinstance(cp.stats, NumericStats) else None
                if not stats or not stats.bimodal_stats:
                    return
                
                c1, c2 = stats.bimodal_stats.center1, stats.bimodal_stats.center2
                
                grouping_var = config.bimodal_grouping_variables.get(col)
                if grouping_var and grouping_var in train_df.columns:
                    # Branch 1
                    df_valid = train_df.select([col, grouping_var]).drop_nulls()
                    if len(df_valid) == 0:
                        return
                    
                    if stats.skewness_severity == SkewSeverity.Normal:
                        aggs = df_valid.group_by(grouping_var).agg(pl.col(col).mean())
                    else:
                        aggs = df_valid.group_by(grouping_var).agg(pl.col(col).median())
                    
                    group_fills = dict(zip(aggs[grouping_var].to_list(), aggs[col].to_list()))
                    models[f"cluster:{col}"] = FittedClusterConditional(
                        grouping_variable=grouping_var,
                        group_fills=group_fills,
                        fill_1=None,
                        fill_2=None,
                        feature_centroid_1=None,
                        feature_centroid_2=None,
                        feature_cols=None,
                        center1=c1,
                        center2=c2
                    )
                else:
                    # Branch 3
                    # Find correlated features
                    feat_cols = []
                    if profile.dataset.feature_correlation is not None:
                        col_corrs = profile.dataset.feature_correlation.pearson_matrix.get(col, {})
                        feat_cols = [c for c, r in col_corrs.items() if c != col and abs(r) > config.bimodal_correlation_threshold and c in train_df.columns]
                    
                    df_valid = train_df.select([col] + feat_cols).drop_nulls(subset=[col])
                    if len(df_valid) == 0:
                        return
                        
                    vals = df_valid[col].to_numpy()
                    dist1 = np.abs(vals - c1)
                    dist2 = np.abs(vals - c2)
                    mask1 = dist1 <= dist2
                    mask2 = ~mask1
                    
                    if stats.skewness_severity == SkewSeverity.Normal:
                        fill1 = float(np.mean(vals[mask1])) if mask1.any() else c1
                        fill2 = float(np.mean(vals[mask2])) if mask2.any() else c2
                    else:
                        fill1 = float(np.median(vals[mask1])) if mask1.any() else c1
                        fill2 = float(np.median(vals[mask2])) if mask2.any() else c2
                    
                    centroid1 = None
                    centroid2 = None
                    
                    if feat_cols:
                        feat_arr = df_valid.select(feat_cols).to_numpy()
                        # handle nulls in feature cols by nanmean
                        if mask1.any():
                            centroid1 = np.nanmean(feat_arr[mask1], axis=0)
                        if mask2.any():
                            centroid2 = np.nanmean(feat_arr[mask2], axis=0)
                            
                        # If a feature is completely null in a cluster, replace nan with 0
                        if centroid1 is not None:
                            centroid1 = np.nan_to_num(centroid1)
                        if centroid2 is not None:
                            centroid2 = np.nan_to_num(centroid2)
                    
                    models[f"cluster:{col}"] = FittedClusterConditional(
                        grouping_variable=None,
                        group_fills=None,
                        fill_1=fill1,
                        fill_2=fill2,
                        feature_centroid_1=centroid1,
                        feature_centroid_2=centroid2,
                        feature_cols=feat_cols if feat_cols else None,
                        center1=c1,
                        center2=c2
                    )

            for cluster_pos, col in enumerate(cluster_cols, start=1):
                units.append(partial(_run_cluster_col, col, cluster_pos))

        # Resolve the worker count from how many independent units there are,
        # then pin nested estimators to a single core to avoid oversubscription;
        # a lone unit (or a forced-sequential ``max_workers=1`` fit) keeps full
        # inner parallelism.
        n_units = len(units)
        workers = _resolve_fit_workers(config.max_workers, n_units)
        n_jobs_inner = 1 if workers > 1 else -1

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(unit) for unit in units]
                for future in futures:
                    # Re-raise any worker exception (e.g. a hard size-guard
                    # failure) instead of silently swallowing it.
                    future.result()
        else:
            for unit in units:
                unit()

        return _NumericFitBundle(
            records=records,
            models=models,
            model_cols=model_cols,
        )


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
        ImputationStrategy.GMMSampling,
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
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Inner estimator ``n_jobs`` (ADR-0056); ``1`` when nested under the
        outer thread parallelism, ``-1`` for a column fitted alone.  Never
        affects results for a fixed ``random_state``.

    Returns
    -------
    FittedRegression or None
        Fitted regression bundle on success; ``None`` to signal fallback.
    """
    if not feat_cols:
        return None

    if len(train_df[col].drop_nulls()) < 2:
        return None

    estimator = RegressionEstimatorFactory.build(tag, n_rows, config, n_jobs=n_jobs)
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


