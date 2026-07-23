"""
EvaluationOrchestrator — opt-in, stateless fit-quality phase (ADR-0058).

Evaluation is split into two clearly-named calls so cost is legible from the
call site: ``inspect(fitted_imputer, train_df)`` is the cheap, retrain-free
sanity check ("do the imputed values look sensible?"), and
``score_accuracy(fitted_imputer, train_df, profile)`` is the expensive,
cross-validated held-out accuracy measurement — an irreducible refit.  Neither
is welded into the imputation fit path; both run deliberately, on the
user's clock.  The orchestrator is a phase orchestrator in the same shape as
every other — construct it with an optional ``observer=`` and call an entry
point — and it is stateless: the caller passes the ``FittedImputer`` and the
data back in rather than the imputer hoarding the frame, so that object stays
serializable.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import polars as pl

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from ..config import PipelineConfig
from ..observability import Emitter, Observer, _ObservabilityMixin
from ..profiling._numeric_config import (
    NumericStats,
    SkewSeverity,
)
from ._config import (
    _MODEL_BASED_STRATEGIES,
    AccuracyDiagnostic,
    AccuracyReport,
    ImputationStrategy,
    InspectionDiagnostic,
    InspectionReport,
)
from ._utils import _df_to_numpy
from ..utils._null_normalization import _resolve_effective_nulls

if TYPE_CHECKING:
    from ._fitted_imputer import FittedImputer
    from ..profiling._config import StructuralProfileResult


def _resolve_fit_workers(max_workers: Optional[int], n_units: int) -> int:
    """Resolve the thread count for the concurrent evaluation units.

    Never exceeds the number of independent units of work; ``max_workers=None``
    auto-sizes to the available CPU count, and ``max_workers=1`` forces a
    sequential run.
    """
    if n_units <= 1:
        return 1
    if max_workers is None:
        resolved = os.cpu_count() or 1
    else:
        resolved = max_workers
    return max(1, min(resolved, n_units))


def _units_by_id(fitted_imputer: "FittedImputer") -> dict:
    """Key a composed imputer's ordered unit list back by plan unit id.

    ``FittedImputer`` holds a bare ordered unit list since the compose collapse
    (ADR-0071); Evaluation reads the model-based units by the plan ids the
    routing produced, so this reconstructs those ids. A fitted unit no longer
    restates its strategy (ADR-0074), so the strategy is read from the unit's
    concrete type. The joint MICE and KNN blocks are the two ids that are not
    ``"{strategy}:{column}"``.
    """
    from ._fitted_imputer import FittedMICE, _FittedKNN

    models: dict = {}
    for unit in fitted_imputer.units:
        if isinstance(unit, FittedMICE):
            models["mice"] = unit
        elif isinstance(unit, _FittedKNN):
            models["knn"] = unit
        else:
            strategy = _unit_strategy(unit)
            cols = unit.target_columns
            if strategy is not None and cols:
                models[f"{strategy}:{cols[0]}"] = unit
    return models


def _unit_strategy(unit: Any) -> Optional[ImputationStrategy]:
    """Recover a fitted unit's strategy from its concrete type (ADR-0074).

    A fitted unit carries only its fitted state and no longer restates the
    strategy it executed, so Evaluation reads it back from the class. Returns
    ``None`` for a unit type with no strategy mapping.
    """
    from ._fitted_units import (
        FittedClusterConditional,
        FittedGMMSampling,
        FittedRegression,
    )

    if isinstance(unit, FittedRegression):
        return ImputationStrategy.Regression
    if isinstance(unit, FittedClusterConditional):
        return ImputationStrategy.ClusterConditional
    if isinstance(unit, FittedGMMSampling):
        return ImputationStrategy.GMMSampling
    return None


def _read_model_metadata(
    strategy: ImputationStrategy,
    column: str,
    models: dict,
    n_rows: int,
    knn_n_neighbors_override: int | None,
) -> tuple:
    """Read convergence/neighbour metadata straight off the fitted models.

    Returns ``(converged, n_iter, n_neighbors_used, k_capped)``.  Regression and
    MICE contribute ``converged``/``n_iter`` from their fitted
    ``IterativeImputer``; KNN contributes ``n_neighbors_used`` (the model's own
    ``n_neighbors``) and ``k_capped`` (the count was forced to ``n_rows − 1``);
    the bimodal strategies contribute none.  ``k_capped`` is ``None`` whenever a
    ``knn_n_neighbors`` override bypassed the adaptive formula.
    """
    converged = None
    n_iter = None
    n_neighbors_used = None
    k_capped = None

    if strategy == ImputationStrategy.Regression:
        fitted_reg = models.get(f"regression:{column}")
        if fitted_reg is not None:
            # ``max_iter_used`` was stripped from the fitted unit (ADR-0074);
            # the effective cap lives on the fitted IterativeImputer itself.
            converged = bool(fitted_reg.model.n_iter_ < fitted_reg.model.max_iter)
            n_iter = int(fitted_reg.model.n_iter_)
    elif strategy == ImputationStrategy.MICE:
        mice_model = models.get("mice")
        if mice_model is not None:
            converged = bool(mice_model.n_iter_ < mice_model.max_iter)
            n_iter = int(mice_model.n_iter_)
    elif strategy == ImputationStrategy.KNN:
        fitted_knn = models.get("knn")
        if fitted_knn is not None:
            n_neighbors_used = int(fitted_knn.model.n_neighbors)
            if knn_n_neighbors_override is None:
                k_capped = n_neighbors_used == max(1, n_rows - 1)

    return converged, n_iter, n_neighbors_used, k_capped


# ---------------------------------------------------------------------------
# Held-out accuracy: cross-validated fold scoring (relocated from
# ``_numeric_imputer._compute_*_diagnostics`` — the Evaluation module is now
# the home of the CV-fold logic, ADR-0058).  Each helper returns only the three
# held-out accuracy numbers ``(r2_cv, rmse, mae)``; distribution and model
# metadata belong to :meth:`EvaluationOrchestrator.inspect`.
# ---------------------------------------------------------------------------


def _fold_slices(n_complete: int, n_folds: int):
    """Yield ``(fold_idx, val_start, val_end)`` for the k-fold split.

    The final fold absorbs the remainder so every complete row lands in exactly
    one validation fold, matching the split used during fit-time diagnostics.
    """
    fold_size = n_complete // n_folds
    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_folds - 1 else n_complete
        yield fold_idx, val_start, val_end


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


def _score_regression_cv(
    train_df: pl.DataFrame,
    col: str,
    fitted_reg: Any,
    config: Any,
    emitter: Optional[Emitter],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Cross-validate a single Regression-strategy column's held-out accuracy.

    The fold stand-in freezes M's recipe — the ``estimator``, ``max_iter``, and
    ``tol`` read straight off the fitted ``IterativeImputer`` — and only refits
    it on the fold's training rows (ADR-0059).  Nothing is re-derived from the
    fold or from the profile, so the score is a faithful baseline for M.
    """
    feat_cols = list(fitted_reg.all_cols[1:])
    all_cols = [col] + feat_cols
    arr = _df_to_numpy(train_df, all_cols)
    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())
    if n_complete < config.refit_r2_min_complete_rows:
        return None, None, None

    # Reuse the exact IterativeImputer recipe ``fit()`` settled on, read straight
    # off the fitted Regression model rather than rebuilding it per fold.
    estimator = fitted_reg.model.estimator
    max_iter = fitted_reg.model.max_iter
    tol = fitted_reg.model.tol

    n_folds = config.refit_r2_cv_folds
    rng = np.random.default_rng(0)
    arr_shuffled = arr[np.where(complete_mask)[0]][rng.permutation(n_complete)]

    fold_r2s: list[float] = []
    fold_rmses: list[float] = []
    fold_maes: list[float] = []
    for fold_idx, val_start, val_end in _fold_slices(n_complete, n_folds):
        if emitter is not None:
            emitter.substep(
                f"{col} regression accuracy fold",
                column=col,
                index=fold_idx + 1,
                total=n_folds,
            )
        arr_val_sub = arr_shuffled[val_start:val_end]
        arr_train_sub = np.concatenate(
            [arr_shuffled[:val_start], arr_shuffled[val_end:]]
        )

        y_true = arr_val_sub[:, 0]
        if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
            continue

        temp_imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            tol=tol,
            random_state=0,
        )
        temp_imputer.fit(arr_train_sub)

        arr_val_masked = arr_val_sub.copy()
        arr_val_masked[:, 0] = np.nan
        y_pred = temp_imputer.transform(arr_val_masked)[:, 0]
        try:
            r2_f, rmse_f, mae_f = _compute_fold_metrics(y_true, y_pred)
            fold_r2s.append(r2_f)
            fold_rmses.append(rmse_f)
            fold_maes.append(mae_f)
        except Exception:  # noqa: BLE001
            pass

    if fold_r2s:
        return (
            float(np.mean(fold_r2s)),
            float(np.mean(fold_rmses)),
            float(np.mean(fold_maes)),
        )
    return None, None, None


def _score_knn_cv(
    train_df: pl.DataFrame,
    knn_cols: list[str],
    fitted_knn: Any,
    config: Any,
    emitter: Optional[Emitter],
) -> dict[str, tuple[Optional[float], Optional[float], Optional[float]]]:
    """Cross-validate held-out accuracy for every KNN-strategy column.

    Freezes M's recipe (``n_neighbors`` and ``weights``) but recomputes the
    feature scaling from each fold's training rows only — reusing M's full-data
    ``col_means``/``col_stds`` would leak the held-out fold into the scaling
    (ADR-0059).
    """
    result: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {
        col: (None, None, None) for col in knn_cols
    }
    arr = _df_to_numpy(train_df, knn_cols)
    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())
    if n_complete < config.refit_r2_min_complete_rows:
        return result

    n_neighbors = int(fitted_knn.model.n_neighbors)
    weights = fitted_knn.model.weights

    n_folds = config.refit_r2_cv_folds
    rng = np.random.default_rng(0)
    arr_shuffled = arr[np.where(complete_mask)[0]][rng.permutation(n_complete)]

    col_r2s: dict[str, list[float]] = {col: [] for col in knn_cols}
    col_rmses: dict[str, list[float]] = {col: [] for col in knn_cols}
    col_maes: dict[str, list[float]] = {col: [] for col in knn_cols}

    for fold_idx, val_start, val_end in _fold_slices(n_complete, n_folds):
        if emitter is not None:
            emitter.substep(
                "KNN accuracy fold", index=fold_idx + 1, total=n_folds
            )
        arr_val_raw = arr_shuffled[val_start:val_end]
        arr_train_raw = np.concatenate(
            [arr_shuffled[:val_start], arr_shuffled[val_end:]]
        )

        # Scaling is a learned parameter: recompute it on the fold's training
        # rows only, then apply the same transform to the validation rows.
        fold_means = np.nanmean(arr_train_raw, axis=0)
        fold_stds = np.nanstd(arr_train_raw, axis=0)
        fold_stds[fold_stds == 0.0] = 1.0
        arr_train_sub = (arr_train_raw - fold_means) / fold_stds
        arr_val_sub = (arr_val_raw - fold_means) / fold_stds

        temp_knn = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        temp_knn.fit(arr_train_sub)

        for k, col_k in enumerate(knn_cols):
            y_true = arr_val_sub[:, k]
            if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
                continue
            arr_val_masked = arr_val_sub.copy()
            arr_val_masked[:, k] = np.nan
            y_pred = temp_knn.transform(arr_val_masked)[:, k]
            y_pred_inv = y_pred * fold_stds[k] + fold_means[k]
            y_true_inv = y_true * fold_stds[k] + fold_means[k]
            try:
                r2_f, rmse_f, mae_f = _compute_fold_metrics(y_true_inv, y_pred_inv)
                col_r2s[col_k].append(r2_f)
                col_rmses[col_k].append(rmse_f)
                col_maes[col_k].append(mae_f)
            except Exception:  # noqa: BLE001
                pass

    for col_k in knn_cols:
        if col_r2s[col_k]:
            result[col_k] = (
                float(np.mean(col_r2s[col_k])),
                float(np.mean(col_rmses[col_k])),
                float(np.mean(col_maes[col_k])),
            )
    return result


def _score_mice_cv(
    train_df: pl.DataFrame,
    mice_cols: list[str],
    mice_model: Any,
    config: Any,
    emitter: Optional[Emitter],
) -> dict[str, tuple[Optional[float], Optional[float], Optional[float]]]:
    """Cross-validate held-out accuracy for every MICE-strategy column."""
    result: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {
        col: (None, None, None) for col in mice_cols
    }
    arr = _df_to_numpy(train_df, mice_cols)
    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())
    if n_complete < config.refit_r2_min_complete_rows:
        return result

    # Reuse the exact IterativeImputer parameters ``fit()`` settled on, read
    # straight off the fitted MICE model rather than recomputing them.
    estimator = mice_model.estimator
    max_iter = mice_model.max_iter
    tol = mice_model.tol
    initial_strategy = mice_model.initial_strategy
    n_nearest_features = mice_model.n_nearest_features

    n_folds = config.refit_r2_cv_folds
    rng = np.random.default_rng(0)
    arr_shuffled = arr[np.where(complete_mask)[0]][rng.permutation(n_complete)]

    col_r2s: dict[str, list[float]] = {col: [] for col in mice_cols}
    col_rmses: dict[str, list[float]] = {col: [] for col in mice_cols}
    col_maes: dict[str, list[float]] = {col: [] for col in mice_cols}

    for fold_idx, val_start, val_end in _fold_slices(n_complete, n_folds):
        if emitter is not None:
            emitter.substep(
                "MICE accuracy fold", index=fold_idx + 1, total=n_folds
            )
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
            y_pred = temp_mice.transform(arr_val_masked)[:, k]
            try:
                r2_f, rmse_f, mae_f = _compute_fold_metrics(y_true, y_pred)
                col_r2s[col_k].append(r2_f)
                col_rmses[col_k].append(rmse_f)
                col_maes[col_k].append(mae_f)
            except Exception:  # noqa: BLE001
                pass

    for col_k in mice_cols:
        if col_r2s[col_k]:
            result[col_k] = (
                float(np.mean(col_r2s[col_k])),
                float(np.mean(col_rmses[col_k])),
                float(np.mean(col_maes[col_k])),
            )
    return result


def _score_cluster_cv(
    train_df: pl.DataFrame,
    col: str,
    feat_cols: list[str],
    c1: float,
    c2: float,
    stats: Optional[NumericStats],
    config: Any,
    emitter: Optional[Emitter],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Cross-validate held-out accuracy for a feature-based Cluster-Conditional column."""
    all_cols = [col] + feat_cols
    arr = _df_to_numpy(train_df, all_cols)
    complete_mask = ~np.isnan(arr).any(axis=1)
    n_complete = int(complete_mask.sum())
    if n_complete < config.refit_r2_min_complete_rows:
        return None, None, None

    n_folds = config.refit_r2_cv_folds
    rng = np.random.default_rng(0)
    arr_shuffled = arr[np.where(complete_mask)[0]][rng.permutation(n_complete)]

    is_normal = stats is not None and stats.skewness_severity == SkewSeverity.Normal

    fold_r2s: list[float] = []
    fold_rmses: list[float] = []
    fold_maes: list[float] = []
    for fold_idx, val_start, val_end in _fold_slices(n_complete, n_folds):
        if emitter is not None:
            emitter.substep(
                f"{col} cluster-conditional accuracy fold",
                column=col,
                index=fold_idx + 1,
                total=n_folds,
            )
        arr_val_sub = arr_shuffled[val_start:val_end]
        arr_train_sub = np.concatenate(
            [arr_shuffled[:val_start], arr_shuffled[val_end:]]
        )

        y_true = arr_val_sub[:, 0]
        if len(y_true) < 2 or float(np.std(y_true)) == 0.0:
            continue

        y_train = arr_train_sub[:, 0]
        mask1_train = np.abs(y_train - c1) <= np.abs(y_train - c2)
        mask2_train = ~mask1_train
        if is_normal:
            fill1 = float(np.mean(y_train[mask1_train])) if mask1_train.any() else c1
            fill2 = float(np.mean(y_train[mask2_train])) if mask2_train.any() else c2
        else:
            fill1 = float(np.median(y_train[mask1_train])) if mask1_train.any() else c1
            fill2 = float(np.median(y_train[mask2_train])) if mask2_train.any() else c2

        feat_train = arr_train_sub[:, 1:]
        centroid1 = (
            np.nan_to_num(np.nanmean(feat_train[mask1_train], axis=0))
            if mask1_train.any()
            else None
        )
        centroid2 = (
            np.nan_to_num(np.nanmean(feat_train[mask2_train], axis=0))
            if mask2_train.any()
            else None
        )

        feat_val = np.nan_to_num(arr_val_sub[:, 1:])
        y_pred = np.zeros_like(y_true)
        for i in range(len(feat_val)):
            d1 = (
                np.linalg.norm(feat_val[i] - centroid1)
                if centroid1 is not None
                else float("inf")
            )
            d2 = (
                np.linalg.norm(feat_val[i] - centroid2)
                if centroid2 is not None
                else float("inf")
            )
            y_pred[i] = fill1 if d1 <= d2 else fill2

        try:
            r2_f, rmse_f, mae_f = _compute_fold_metrics(y_true, y_pred)
            fold_r2s.append(r2_f)
            fold_rmses.append(rmse_f)
            fold_maes.append(mae_f)
        except Exception:  # noqa: BLE001
            pass

    if fold_r2s:
        return (
            float(np.mean(fold_r2s)),
            float(np.mean(fold_rmses)),
            float(np.mean(fold_maes)),
        )
    return None, None, None


class EvaluationOrchestrator(_ObservabilityMixin):
    """
    Stateless fit-quality Evaluation orchestrator (ADR-0058).

    Mirrors the "each phase is an orchestrator you pass ``observer=`` to" shape.
    Exposes two entry points whose cost is legible from the call:
    :meth:`inspect` is the cheap, retrain-free check that reuses the models
    ``fit()`` already learned (returning an :class:`InspectionReport`), and
    :meth:`score_accuracy` is the expensive, cross-validated held-out accuracy
    measurement (returning an :class:`AccuracyReport`).  Both take the
    ``FittedImputer`` and the data back in, so nothing is retained on the
    orchestrator between calls and the imputer stays serializable.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  Defaults to ``PipelineConfig()`` when omitted.
    observer : Callable[[PipelineEvent], None], optional
        Live Progress Observer invoked with each :class:`PipelineEvent` emitted
        during :meth:`score_accuracy`.  ``None`` (the default) disables Progress;
        Trace still flows to the ``dataforge_ml`` logger regardless (ADR-0054).
    """

    _PHASE = "evaluation"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        observer: Observer | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._observer: Observer | None = observer

    def inspect(
        self,
        fitted_imputer: FittedImputer,
        train_df: pl.DataFrame,
    ) -> InspectionReport:
        """
        Cheaply check whether imputed values look sensible — no retraining.

        Reuses the models ``fit()`` already learned (handed in via
        ``fitted_imputer``) rather than re-routing or re-fitting anything.  It
        runs :meth:`FittedImputer.transform` to fill the holes, then compares
        the filled cells against the observed values to produce per-column
        distributional statistics, and reads convergence/neighbour metadata
        straight off the fitted models.  No ``profile`` is required and no
        cross-validated refit is paid, so the call stays instant (ADR-0058).

        Parameters
        ----------
        fitted_imputer : FittedImputer
            The imputer returned by :meth:`FittedImputer.compose`.  Its
            records and fitted models are reused as-is.
        train_df : pl.DataFrame
            Data to inspect against.  Passed back in explicitly because
            Evaluation is stateless — the ``FittedImputer`` does not retain it.

        Returns
        -------
        InspectionReport
            Per-column inspection diagnostics.  Holds one entry per model-based
            column (KNN, Regression, MICE, and the bimodal strategies); scalar,
            Passthrough, Dropped, Constant, and MNAR columns carry no entry.
        """
        # Fill the holes through the real application path so inspection
        # reflects exactly the imputation ``transform`` would apply — never a
        # re-implementation of model inference (ADR-0058).
        filled_df = fitted_imputer.transform(train_df).dataframe
        models_by_id = _units_by_id(fitted_imputer)

        # Locate the originally-null cells the same way ``transform`` does:
        # normalise effective nulls first, then read the null mask.
        resolved = _resolve_effective_nulls(
            train_df,
            numeric_sentinels=fitted_imputer.numeric_sentinels,
            string_sentinels=fitted_imputer.string_sentinels,
        )
        n_rows = resolved.height
        numeric_cfg = self._config.imputation.numeric

        columns: dict[str, InspectionDiagnostic] = {}
        for col, rec in fitted_imputer.records.items():
            if rec.decision.strategy not in _MODEL_BASED_STRATEGIES:
                continue
            if col not in resolved.columns or col not in filled_df.columns:
                continue

            observed = resolved[col].drop_nulls().to_numpy()
            observed_mean = float(np.mean(observed)) if observed.size else 0.0
            observed_std = float(np.std(observed)) if observed.size else 0.0

            null_mask = resolved[col].is_null().to_numpy()
            imputed_mean = 0.0
            imputed_std = 0.0
            if null_mask.any():
                imputed_vals = filled_df[col].to_numpy()[null_mask]
                imputed_mean = float(np.mean(imputed_vals))
                imputed_std = float(np.std(imputed_vals))

            variance_ratio = (
                imputed_std / observed_std if observed_std > 0.0 else 0.0
            )

            converged, n_iter, n_neighbors_used, k_capped = _read_model_metadata(
                strategy=rec.decision.strategy,
                column=col,
                models=models_by_id,
                n_rows=n_rows,
                knn_n_neighbors_override=numeric_cfg.knn_n_neighbors,
            )

            columns[col] = InspectionDiagnostic(
                imputed_mean=imputed_mean,
                imputed_std=imputed_std,
                observed_mean=observed_mean,
                observed_std=observed_std,
                variance_ratio=variance_ratio,
                converged=converged,
                n_iter=n_iter,
                n_neighbors_used=n_neighbors_used,
                k_capped=k_capped,
            )

        return InspectionReport(columns=columns)

    def score_accuracy(
        self,
        fitted_imputer: FittedImputer,
        train_df: pl.DataFrame,
        profile: StructuralProfileResult,
    ) -> AccuracyReport:
        """
        Measure honest held-out accuracy per model-based column — the refit path.

        Held-out accuracy (``r2_cv``, ``rmse``, ``mae``) is irreducibly a refit:
        the models ``fit()`` learned have already seen every cell, so scoring
        them in-sample is optimistically biased.  This method therefore
        cross-validates on folds of the complete rows.  It does **not** re-route:
        each column's strategy is read from ``fitted_imputer.records``.  Each
        fold stand-in freezes M's recipe (estimator, ``max_iter``, ``tol``,
        ``n_neighbors``, ``weights``) read off the fitted model and only
        re-learns M's parameters — coefficients, and KNN feature scaling — on
        the fold's training rows (ADR-0059), so the score is a faithful baseline
        for M.  The fold work parallelizes across independent columns and
        strategy blocks under the independence rule (ADR-0056) and emits
        ``substep`` heartbeats (ADR-0055), so a multi-minute accuracy run stays
        observable (ADR-0058).

        Parameters
        ----------
        fitted_imputer : FittedImputer
            The imputer returned by :meth:`FittedImputer.compose`.  Its
            records supply the per-column strategy and its models supply the
            fold-estimator parameters; neither is re-derived.
        train_df : pl.DataFrame
            Data to score against.  Passed back in explicitly because Evaluation
            is stateless — the ``FittedImputer`` does not retain it.
        profile : StructuralProfileResult
            Full-dataset profile from Phase 1.  Used only for the
            Cluster-Conditional fold fills (skewness selects mean vs. median),
            never to re-route and never for Regression/KNN scoring.

        Returns
        -------
        AccuracyReport
            Per-column held-out accuracy.  Holds one entry per model-based
            column (KNN, Regression, MICE, and the bimodal strategies); scalar,
            Passthrough, Dropped, Constant, and MNAR columns carry no entry.
            The three fields are ``None`` when fewer than
            ``refit_r2_min_complete_rows`` complete rows are available, or when
            the strategy has no held-out truth to score (GMM-Sampling and the
            grouping-variable Cluster-Conditional branch).
        """
        self._config.imputation.validate()

        resolved = _resolve_effective_nulls(
            train_df,
            numeric_sentinels=fitted_imputer.numeric_sentinels,
            string_sentinels=fitted_imputer.string_sentinels,
        )
        config = self._config.imputation.numeric
        models_by_id = _units_by_id(fitted_imputer)

        # Reuse the strategy decisions already recorded at fit time (ADR-0058);
        # routing is never recomputed here.
        def _cols_for(strategy: ImputationStrategy) -> list[str]:
            return [
                col
                for col, rec in fitted_imputer.records.items()
                if rec.decision.strategy == strategy and col in resolved.columns
            ]

        mice_cols = _cols_for(ImputationStrategy.MICE)
        knn_cols = _cols_for(ImputationStrategy.KNN)
        reg_cols = _cols_for(ImputationStrategy.Regression)
        cluster_cols = _cols_for(ImputationStrategy.ClusterConditional)
        gmm_cols = _cols_for(ImputationStrategy.GMMSampling)

        n_model_cols = (
            len(mice_cols)
            + len(knn_cols)
            + len(reg_cols)
            + len(cluster_cols)
            + len(gmm_cols)
        )

        self._emit_stage_start("accuracy")
        emitter = Emitter(
            phase=self._PHASE,
            stage="accuracy",
            observer=self._observer,
            total=n_model_cols,
        )

        # Each unit writes only its own column keys, so the shared dict is never
        # the site of a concurrent collision (ADR-0056).
        accuracy: dict[str, AccuracyDiagnostic] = {}
        units: list[Callable[[], None]] = []

        if mice_cols:
            mice_model = models_by_id.get("mice")

            def _run_mice() -> None:
                if emitter is not None:
                    emitter.substep(
                        f"MICE accuracy over {len(mice_cols)} columns"
                    )
                scores = _score_mice_cv(
                    resolved, mice_cols, mice_model, config, emitter
                )
                for col in mice_cols:
                    r2_cv, rmse, mae = scores[col]
                    accuracy[col] = AccuracyDiagnostic(r2_cv=r2_cv, rmse=rmse, mae=mae)
                    emitter.item(col)

            units.append(_run_mice)

        if knn_cols:
            fitted_knn = models_by_id.get("knn")

            def _run_knn() -> None:
                if emitter is not None:
                    emitter.substep(
                        f"KNN accuracy over {len(knn_cols)} columns"
                    )
                scores = _score_knn_cv(
                    resolved, knn_cols, fitted_knn, config, emitter
                )
                for col in knn_cols:
                    r2_cv, rmse, mae = scores[col]
                    accuracy[col] = AccuracyDiagnostic(r2_cv=r2_cv, rmse=rmse, mae=mae)
                    emitter.item(col)

            units.append(_run_knn)

        def _run_regression_col(col: str) -> None:
            fitted_reg = models_by_id.get(f"regression:{col}")
            if fitted_reg is None:
                accuracy[col] = AccuracyDiagnostic(r2_cv=None, rmse=None, mae=None)
                emitter.item(col)
                return
            r2_cv, rmse, mae = _score_regression_cv(
                resolved, col, fitted_reg, config, emitter
            )
            accuracy[col] = AccuracyDiagnostic(r2_cv=r2_cv, rmse=rmse, mae=mae)
            emitter.item(col)

        for col in reg_cols:
            units.append(partial(_run_regression_col, col))

        def _run_cluster_col(col: str) -> None:
            fitted_cluster = models_by_id.get(f"cluster:{col}")
            r2_cv = rmse = mae = None
            # Only the feature-based branch (no grouping variable) has held-out
            # truth to score; the grouping-variable branch does not.
            if (
                fitted_cluster is not None
                and fitted_cluster.grouping_variable is None
                and fitted_cluster.feature_cols
            ):
                feat_cols = [
                    c for c in fitted_cluster.feature_cols if c in resolved.columns
                ]
                if feat_cols:
                    cp = profile.columns.get(col)
                    stats = (
                        cp.stats
                        if cp is not None and isinstance(cp.stats, NumericStats)
                        else None
                    )
                    r2_cv, rmse, mae = _score_cluster_cv(
                        resolved,
                        col,
                        feat_cols,
                        fitted_cluster.center1,
                        fitted_cluster.center2,
                        stats,
                        config,
                        emitter,
                    )
            accuracy[col] = AccuracyDiagnostic(r2_cv=r2_cv, rmse=rmse, mae=mae)
            emitter.item(col)

        for col in cluster_cols:
            units.append(partial(_run_cluster_col, col))

        def _run_gmm_col(col: str) -> None:
            # GMM sampling draws from a fitted mixture; there is no held-out
            # target to score, so accuracy fields stay ``None`` (ADR-0058).
            accuracy[col] = AccuracyDiagnostic(r2_cv=None, rmse=None, mae=None)
            emitter.item(col)

        for col in gmm_cols:
            units.append(partial(_run_gmm_col, col))

        # Independent units (strategy blocks and per-column work) fit
        # concurrently on threads, capped by ``max_workers`` (ADR-0056); a fixed
        # seed and fold split make the concurrent run identical to a sequential
        # one.
        workers = _resolve_fit_workers(config.max_workers, len(units))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(unit) for unit in units]
                for future in futures:
                    future.result()
        else:
            for unit in units:
                unit()

        self._emit_stage_end("accuracy")

        return AccuracyReport(columns=accuracy)
