"""
NumericImputer — Phase 2 sub-processor for SemanticType.Numeric columns.

Applies the Numeric Imputation Decision Priority (see issue #5) during fit()
and returns a _NumericFitBundle.  All fill values and models are computed
exclusively from train_df; the profile is used only for strategy routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

from ..config import SemanticType
from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag, MissingSeverity
from ..profiling._numeric_config import NumericStats, SkewSeverity
from ._config import (
    ColumnImputationRecord,
    ImputationStrategy,
    NumericImputationConfig,
)

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile, StructuralProfileResult


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
            _fit_regression(train_df, col, feat_cols, models, model_cols, records)

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

    # Priority 3: MARSuspect — full fallback chain
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

    # Priority 4: Discrete numeric → Mode
    if cp.numeric_kind == NumericKind.Discrete:
        signals.append("NumericKind.Discrete: mode imputation")
        return ColumnImputationRecord(
            column=col, semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Mode,
            fill_value=_compute_mode(train_df, col),
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
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
    records: list[ColumnImputationRecord],
) -> None:
    model_key = f"regression:{col}"

    if not feat_cols:
        _fallback_to_median(train_df, col, records, "no feature columns available")
        return

    arr = _df_to_numpy(train_df, [col] + feat_cols)
    complete_mask = ~np.isnan(arr).any(axis=1)

    if complete_mask.sum() < 2:
        _fallback_to_median(train_df, col, records, "insufficient complete rows for regression")
        return

    X_train = arr[complete_mask, 1:]
    y_train = arr[complete_mask, 0]
    feat_means = np.nanmean(arr[:, 1:], axis=0)

    reg = BayesianRidge()
    reg.fit(X_train, y_train)

    models[model_key] = (reg, feat_means)
    model_cols[model_key] = list(feat_cols)


def _fallback_to_median(
    train_df: pl.DataFrame,
    col: str,
    records: list[ColumnImputationRecord],
    reason: str,
) -> None:
    for rec in records:
        if rec.column == col:
            rec.strategy = ImputationStrategy.Median
            rec.fill_value = _compute_median(train_df, col)
            rec.signals.append(f"fallback_to_median: {reason}")
            break


# ---------------------------------------------------------------------------
# Scalar statistics helpers
# ---------------------------------------------------------------------------


def _df_to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract columns as float64 numpy array, converting Polars nulls to NaN."""
    return (
        df.select([pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in cols])
        .to_numpy()
        .astype(np.float64)
    )


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
