"""
NumericImputer — Phase 2 sub-processor for SemanticType.Numeric columns.

Applies the Numeric Imputation Decision Priority (see issue #5) during fit()
and returns a _NumericFitBundle.  All fill values and models are computed
exclusively from train_df; the profile is used only for strategy routing.
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
from ..profiling._missingness_config import MissingnessFlag, MissingSeverity
from ..profiling._numeric_config import NonlinearityTag, NumericStats, SkewSeverity
from ._config import (
    ColumnImputationRecord,
    ImputationStrategy,
    NumericImputationConfig,
)
from ._regression_estimator_factory import RegressionEstimatorFactory
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
        internally.  Legacy entries serialised before Issue #141 may carry a
        ``(BayesianRidge, feat_means)`` tuple — the ``FittedImputer``
        migration path in ``from_dict()`` handles these transparently.
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

        # First pass: determine strategy + compute scalar fill values
        records: list[ColumnImputationRecord] = []
        for col in columns:
            cp = profile.columns.get(col)
            if cp is None:
                continue
            records.append(
                _fit_one(
                    train_df, col, cp, config, mnar_columns,
                    n_rows=n_rows, n_features=n_features, multi_mar=multi_mar,
                )
            )

        # Second pass: fit model-based strategies
        mice_cols = [r.column for r in records if r.strategy == ImputationStrategy.MICE]
        knn_cols = [r.column for r in records if r.strategy == ImputationStrategy.KNN]
        reg_cols = [r.column for r in records if r.strategy == ImputationStrategy.Regression]

        models: dict[str, Any] = {}
        model_cols: dict[str, list[str]] = {}

        if mice_cols:
            arr = _df_to_numpy(train_df, mice_cols)
            mice_model = IterativeImputer(random_state=0, max_iter=10)
            mice_model.fit(arr)
            models["mice"] = mice_model
            model_cols["mice"] = list(mice_cols)

        if knn_cols:
            arr = _df_to_numpy(train_df, knn_cols)
            knn_model = KNNImputer(n_neighbors=5)
            knn_model.fit(arr)
            models["knn"] = knn_model
            model_cols["knn"] = list(knn_cols)

        for col in reg_cols:
            feat_cols = [c for c in columns if c != col]
            cp = profile.columns.get(col)
            stats = cp.stats if cp is not None and isinstance(cp.stats, NumericStats) else None
            tag = (
                stats.nonlinearity_tag
                if stats is not None and stats.nonlinearity_tag is not None
                else NonlinearityTag.Linear
            )

            rec_idx = next(i for i, r in enumerate(records) if r.column == col)
            record = records[rec_idx]

            fitted = _fit_regression(train_df, col, feat_cols, tag, n_rows, config, stats)

            if fitted is None:
                if not feat_cols:
                    reason = "no feature columns available"
                elif tag == NonlinearityTag.Unpredictable:
                    reason = "nonlinearity_tag=Unpredictable: regression unsuitable"
                else:
                    reason = "insufficient target observations for regression"
                records[rec_idx] = _fallback_to_median(train_df, col, record, reason)
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
# Strategy selection per column
# ---------------------------------------------------------------------------


def _fit_one(
    train_df: pl.DataFrame,
    col: str,
    cp: ColumnProfile,
    config: NumericImputationConfig,
    mnar_columns: set[str],
    n_rows: int,
    n_features: int,
    multi_mar: bool,
) -> ColumnImputationRecord:
    missingness = cp.missingness
    signals: list[str] = []

    # Priority 1: DropCandidate — >50% missing
    if missingness and missingness.has_flag(MissingnessFlag.DropCandidate):
        signals.append(
            f"drop_candidate: {missingness.effective_null_ratio:.1%} effective missing"
        )
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Dropped, fill_value=None,
            indicator_added=False, signals=signals,
        )

    # Priority 2: MNAR declared by user
    if col in mnar_columns:
        signals.append("declared MNAR by user configuration")
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Constant,
            fill_value=float(config.mnar_constant_fill),
            indicator_added=True, signals=signals,
        )

    # No effective missingness → Passthrough
    if missingness is None or missingness.effective_null_count == 0:
        signals.append("no missing values in full-dataset profile")
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Passthrough, fill_value=None,
            indicator_added=False, signals=signals,
        )

    # Priority 3: BoundedDiscrete → Mode (unconditional — finite domain requires finite fill)
    # Fires before MAR routing: model-based predictions are not valid members of a fixed vocabulary.
    if cp.numeric_kind == NumericKind.BoundedDiscrete:
        signals.append("NumericKind.BoundedDiscrete: mode imputation")
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Mode,
            fill_value=_compute_mode(train_df, col),
            indicator_added=False, signals=signals,
        )

    # Priority 4: MARSuspect — full fallback chain
    if missingness.has_flag(MissingnessFlag.MARSuspect):
        corrs = missingness.correlated_with
        signals.append(f"mar_suspect: correlated missingness with {corrs}")
        strategy, signal = _mar_strategy(
            severity=missingness.severity,
            corrs=corrs,
            config=config,
            n_rows=n_rows,
            n_features=n_features,
            multi_mar=multi_mar,
        )
        signals.append(signal)
        fill_value = _compute_median(train_df, col) if strategy == ImputationStrategy.Median else None
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=strategy, fill_value=fill_value,
            indicator_added=False, signals=signals,
        )

    # Priority 5: MCAR routing by severity and skew
    severity = missingness.severity
    stats = cp.stats if isinstance(cp.stats, NumericStats) else None
    skew_sev = stats.skewness_severity if stats else None

    if severity in (MissingSeverity.High, MissingSeverity.Severe):
        strategy, signal = _mcar_model_strategy(
            severity=severity, config=config, n_rows=n_rows, n_features=n_features,
        )
        signals.append(signal)
        fill_value = _compute_median(train_df, col) if strategy == ImputationStrategy.Median else None
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=strategy, fill_value=fill_value,
            indicator_added=False, signals=signals,
        )

    # Minor + Normal skew → Mean
    if severity == MissingSeverity.Minor and skew_sev in (None, SkewSeverity.Normal):
        signals.append(f"mcar minor + skew={skew_sev or 'normal'}: mean imputation")
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Mean,
            fill_value=_compute_mean(train_df, col),
            indicator_added=False, signals=signals,
        )

    # Minor/Moderate + skew >= Moderate → Median
    signals.append(f"mcar {severity} + skew={skew_sev or 'unknown'}: median imputation")
    return ColumnImputationRecord(
        column=col, semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Median,
        fill_value=_compute_median(train_df, col),
        indicator_added=False, signals=signals,
    )


def _mar_strategy(
    severity: MissingSeverity | None,
    corrs: list[str],
    config: NumericImputationConfig,
    n_rows: int,
    n_features: int,
    multi_mar: bool,
) -> tuple[ImputationStrategy, str]:
    """Full fallback chain for MAR-suspect columns: MICE → Regression → KNN → Median."""
    # Multi-MAR or Severe → MICE
    if multi_mar:
        return ImputationStrategy.MICE, "mice: ≥2 MAR-suspect columns (multi-MAR)"
    if severity == MissingSeverity.Severe:
        return ImputationStrategy.MICE, "mice: MAR-suspect + severe missingness"

    # High with correlations → Regression → KNN → Median
    if severity == MissingSeverity.High and corrs:
        if n_rows >= config.regression_min_rows:
            return (
                ImputationStrategy.Regression,
                f"regression: MAR high + correlations, {n_rows:,} rows >= regression_min_rows={config.regression_min_rows:,}",
            )
        if n_rows <= config.knn_max_rows and n_features <= config.knn_max_features:
            return (
                ImputationStrategy.KNN,
                f"knn: regression size guard failed ({n_rows:,} rows < {config.regression_min_rows:,})",
            )
        return (
            ImputationStrategy.Median,
            f"median: all size guards failed (rows={n_rows:,}, features={n_features})",
        )

    return ImputationStrategy.Median, "median: MAR-suspect fallback (low severity or no correlations)"


def _mcar_model_strategy(
    severity: MissingSeverity,
    config: NumericImputationConfig,
    n_rows: int,
    n_features: int,
) -> tuple[ImputationStrategy, str]:
    """Full fallback chain for MCAR High/Severe: KNN → Regression → Median (High); MICE (Severe)."""
    if severity == MissingSeverity.Severe:
        return ImputationStrategy.MICE, "mice: MCAR severe missingness"

    # High: KNN → Regression → Median
    if n_rows <= config.knn_max_rows and n_features <= config.knn_max_features:
        return (
            ImputationStrategy.KNN,
            f"knn: MCAR high, rows={n_rows:,} <= {config.knn_max_rows:,}, features={n_features} <= {config.knn_max_features}",
        )
    if n_rows >= config.regression_min_rows:
        return (
            ImputationStrategy.Regression,
            f"regression: knn size guard failed, {n_rows:,} rows >= regression_min_rows={config.regression_min_rows:,}",
        )
    return (
        ImputationStrategy.Median,
        f"median: all size guards failed (rows={n_rows:,}, features={n_features})",
    )


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
        ``regression_base_max_iter``.
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

    signals: list[str] = [
        f"regression_estimator: {estimator_name} (tag={tag})"
    ]
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


# ---------------------------------------------------------------------------
# Dynamic IterativeImputer parameter helpers
# ---------------------------------------------------------------------------


def _compute_max_iter(
    tag: NonlinearityTag,
    feat_cols: list[str],
    arr: np.ndarray,
    stats: Optional[NumericStats],
    config: NumericImputationConfig,
) -> int:
    base = config.regression_base_max_iter

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
                        valid = ~(np.isnan(missing_feat_arr[:, i]) | np.isnan(missing_feat_arr[:, j]))
                        if valid.sum() >= 2:
                            c = np.corrcoef(missing_feat_arr[valid, i], missing_feat_arr[valid, j])[0, 1]
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
