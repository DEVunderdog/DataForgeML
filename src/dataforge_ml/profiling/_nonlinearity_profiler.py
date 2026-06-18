"""
NonlinearityProfiler  –  Phase 1 extension: Nonlinearity signal computation.

Computes four signals per numeric column against all other numeric columns
(treated as predictors) and assigns a ``NonlinearityTag``:

1. **Spearman/Pearson discrepancy** — ``max |Spearman_r - Pearson_r|`` over
   predictors.  Reuses pre-computed matrices from ``CorrelationProfiler`` when
   available; computes them internally otherwise.

2. **Mutual information** — ``mutual_info_regression`` per predictor; store
   the mean.  High MI not explained by Pearson/Spearman indicates complex
   non-linear structure.

3. **R² gap** — 3-fold cross-validated R² for ``LinearRegression`` and a
   shallow ``RandomForestRegressor`` on a bootstrap sample of complete rows.
   ``R²_RF − R²_linear`` quantifies the benefit of switching estimators.
   Near-zero R²_RF → ``Unpredictable``.

4. **Breusch-Pagan heteroscedasticity** — tests linear model residuals for
   non-constant variance; implemented with ``scipy`` (no ``statsmodels``
   dependency).

Tag assignment (evaluated in priority order):
  - R²_RF < threshold                                       → Unpredictable
  - discrepancy ≥ threshold AND r2_gap < r2_gap_threshold   → MonotonicNonlinear
  - r2_gap ≥ r2_gap_threshold OR mi ≥ mi_threshold          → ComplexNonlinear
  - otherwise                                               → Linear

All four signals are always computed for every eligible column — none are staged.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl

from ._base import DatasetLevelProfiler
from ._numeric_config import (
    NonlinearityProfileConfig,
    NonlinearityProfileResult,
    NonlinearitySignals,
    NonlinearityTag,
)
from ..models._data_types import _NUMERIC_DTYPES

_CV_FOLDS = 3
_RF_N_ESTIMATORS = 50
_RF_MAX_DEPTH = 4
_MI_N_NEIGHBORS = 3


class NonlinearityProfiler(DatasetLevelProfiler[NonlinearityProfileResult]):
    """
    Computes four nonlinearity signals per numeric column and assigns a
    ``NonlinearityTag``.

    Each numeric column is treated as the target; all other numeric columns
    are its predictors.  All four signals are always computed for every
    eligible column (no staging or short-circuiting).

    Pre-computed Pearson and Spearman correlation matrices from
    ``CorrelationProfiler`` can be passed in to avoid recomputation.  When
    they are not provided, the matrices are computed internally.

    Parameters
    ----------
    numeric_columns : list[str]
        Columns to profile.  Columns absent from the DataFrame or with a
        non-numeric dtype are silently skipped.
    config : NonlinearityProfileConfig, optional
        Threshold configuration.  Defaults to ``NonlinearityProfileConfig()``.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        config: Optional[NonlinearityProfileConfig] = None,
    ) -> None:
        super().__init__()
        self._columns = numeric_columns
        self._config = config if config is not None else NonlinearityProfileConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        pearson_matrix: Optional[dict[str, dict[str, float]]] = None,
        spearman_matrix: Optional[dict[str, dict[str, float]]] = None,
        **kwargs,
    ) -> NonlinearityProfileResult:
        """
        Compute nonlinearity signals for each column in ``numeric_columns``.

        Parameters
        ----------
        data : pl.DataFrame
            Full dataset.
        pearson_matrix : dict[str, dict[str, float]], optional
            Pre-computed symmetric Pearson correlation matrix keyed by column
            name.  When provided, Signal 1 reuses these values instead of
            recomputing them.
        spearman_matrix : dict[str, dict[str, float]], optional
            Pre-computed Spearman matrix in the same shape as
            ``pearson_matrix``.  Must be provided together with
            ``pearson_matrix``; if either is ``None`` both are computed
            internally.

        Returns
        -------
        NonlinearityProfileResult
            Per-column signals and tags for all columns with enough complete
            rows to fit the models.
        """
        result = NonlinearityProfileResult()

        resolved = [
            c
            for c in self._columns
            if c in data.columns and data[c].dtype in _NUMERIC_DTYPES
        ]
        if len(resolved) < 2:
            return result

        if pearson_matrix is None or spearman_matrix is None:
            pearson_matrix, spearman_matrix = self._compute_correlation_matrices(
                data, resolved
            )

        rng = np.random.default_rng(self._config.random_state)

        for col in resolved:
            predictors = [c for c in resolved if c != col]
            if not predictors:
                continue

            relevant = [col] + predictors
            sub_df = (
                data.select([pl.col(c).cast(pl.Float64) for c in relevant])
                .drop_nulls()
            )
            n_complete = sub_df.height
            if n_complete < self._config.min_rows:
                continue

            y = sub_df[col].to_numpy()
            X = sub_df.select(predictors).to_numpy()

            disc = self._spearman_pearson_discrepancy(
                col, predictors, pearson_matrix, spearman_matrix
            )
            raw_mi, excess_mi = self._mutual_information_signals(
                X, y, predictors, col, pearson_matrix
            )

            sample_size = min(self._config.bootstrap_sample_size, n_complete)
            idx = rng.choice(n_complete, size=sample_size, replace=False)
            X_s, y_s = X[idx], y[idx]

            r2_linear, r2_rf = self._r2_scores(X_s, y_s)
            gap = r2_rf - r2_linear

            bp_pvalue = self._breusch_pagan_pvalue(X, y)

            tag = self._assign_tag(r2_rf, disc, bp_pvalue, gap, excess_mi)

            result.columns[col] = NonlinearitySignals(
                tag=tag,
                spearman_pearson_discrepancy=disc,
                mean_mutual_information=raw_mi,
                r2_gap=gap,
                heteroscedasticity_p_value=bp_pvalue,
            )
            result.analysed_columns.append(col)

        return result

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlation_matrices(
        df: pl.DataFrame,
        cols: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        from ._correlation_profiler import CorrelationProfiler

        return CorrelationProfiler._compute_matrices(df, cols)

    @staticmethod
    def _spearman_pearson_discrepancy(
        col: str,
        predictors: list[str],
        pearson_matrix: dict[str, dict[str, float]],
        spearman_matrix: dict[str, dict[str, float]],
    ) -> float:
        max_disc = 0.0
        for pred in predictors:
            p_r = pearson_matrix.get(col, {}).get(pred, 0.0)
            s_r = spearman_matrix.get(col, {}).get(pred, 0.0)
            disc = abs(s_r - p_r)
            if disc > max_disc:
                max_disc = disc
        return max_disc

    @staticmethod
    def _mutual_information_signals(
        X: np.ndarray,
        y: np.ndarray,
        predictors: list[str],
        col: str,
        pearson_matrix: dict[str, dict[str, float]],
    ) -> tuple[float, float]:
        """
        Compute raw mean MI and excess mean MI for all predictors.

        Raw mean MI is stored in ``NonlinearitySignals.mean_mutual_information``.
        Excess MI subtracts the Gaussian linear-implied MI
        ``−0.5 ln(1 − r²)`` for each predictor, isolating non-linear
        information content.  Excess MI is used in ``_assign_tag`` to avoid
        flagging tight linear relationships as ``ComplexNonlinear``.

        Returns
        -------
        tuple[float, float]
            ``(raw_mean_mi, excess_mean_mi)``
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            warnings.warn(
                "scikit-learn is required for mutual information in "
                "NonlinearityProfiler.  Install: pip install scikit-learn",
                stacklevel=4,
            )
            return 0.0, 0.0

        if X.shape[0] < _MI_N_NEIGHBORS + 1:
            return 0.0, 0.0

        try:
            scores = mutual_info_regression(
                X, y, n_neighbors=_MI_N_NEIGHBORS, random_state=42
            )
        except Exception as exc:
            warnings.warn(f"MI computation failed: {exc}", stacklevel=4)
            return 0.0, 0.0

        raw_mean = float(np.mean(scores))

        excess_vals: list[float] = []
        for i, pred in enumerate(predictors):
            mi = float(scores[i])
            r = pearson_matrix.get(col, {}).get(pred, 0.0)
            r_clamp = min(abs(r), 0.9999)
            linear_mi = -0.5 * float(np.log(1.0 - r_clamp ** 2))
            excess_vals.append(max(0.0, mi - linear_mi))

        excess_mean = float(np.mean(excess_vals)) if excess_vals else 0.0
        return raw_mean, excess_mean

    @staticmethod
    def _r2_scores(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Return ``(r2_linear, r2_rf)`` using 3-fold cross-validated R².

        Returns
        -------
        tuple[float, float]
            Cross-validated R² for LinearRegression and RandomForestRegressor.
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
        except ImportError:
            warnings.warn(
                "scikit-learn is required for R² gap in NonlinearityProfiler.  "
                "Install: pip install scikit-learn",
                stacklevel=4,
            )
            return 0.0, 0.0

        n = len(y)
        cv = min(_CV_FOLDS, n)
        if cv < 2:
            return 0.0, 0.0

        try:
            r2_linear = float(
                np.mean(
                    cross_val_score(
                        LinearRegression(), X, y, cv=cv, scoring="r2"
                    )
                )
            )
        except Exception:
            r2_linear = 0.0

        try:
            r2_rf = float(
                np.mean(
                    cross_val_score(
                        RandomForestRegressor(
                            n_estimators=_RF_N_ESTIMATORS,
                            max_depth=_RF_MAX_DEPTH,
                            random_state=42,
                        ),
                        X,
                        y,
                        cv=cv,
                        scoring="r2",
                    )
                )
            )
        except Exception:
            r2_rf = 0.0

        return r2_linear, r2_rf

    @staticmethod
    def _breusch_pagan_pvalue(X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Breusch-Pagan heteroscedasticity test p-value.

        Fits a linear model on ``X → y``, then regresses squared residuals
        (scaled by mean squared residual) on ``X`` with an intercept.  The LM
        statistic is ``n * R²`` of that auxiliary regression; under H₀ it
        follows ``χ²(k)`` where ``k`` is the number of columns in ``X``.

        Returns
        -------
        float
            p-value in [0, 1].  Returns 1.0 on degenerate input.
        """
        from scipy.stats import chi2

        n = len(y)
        try:
            X_aug = np.column_stack([np.ones(n), X])
            betas, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            residuals = y - X_aug @ betas
        except np.linalg.LinAlgError:
            return 1.0

        sigma2 = float(np.mean(residuals ** 2))
        if sigma2 == 0.0:
            return 1.0

        scaled_sq = residuals ** 2 / sigma2

        try:
            betas_aux, _, _, _ = np.linalg.lstsq(X_aug, scaled_sq, rcond=None)
            fitted_aux = X_aug @ betas_aux
        except np.linalg.LinAlgError:
            return 1.0

        ss_res = float(np.sum((scaled_sq - fitted_aux) ** 2))
        ss_tot = float(np.sum((scaled_sq - np.mean(scaled_sq)) ** 2))
        if ss_tot == 0.0:
            return 1.0

        r2_aux = max(0.0, 1.0 - ss_res / ss_tot)
        lm = n * r2_aux
        df = X.shape[1]
        return float(chi2.sf(lm, df=df))

    # ------------------------------------------------------------------
    # Tag assignment
    # ------------------------------------------------------------------

    def _assign_tag(
        self,
        r2_rf: float,
        discrepancy: float,
        bp_pvalue: float,
        r2_gap: float,
        mi: float,
    ) -> NonlinearityTag:
        cfg = self._config

        if r2_rf < cfg.r2_rf_unpredictable_threshold:
            return NonlinearityTag.Unpredictable

        # monotonic_signal = (
        #     discrepancy >= cfg.spearman_pearson_discrepancy_threshold
        #     or bp_pvalue < cfg.heteroscedasticity_p_value_threshold
        # )

        curvature_signal = (
            discrepancy >= cfg.spearman_pearson_discrepancy_threshold
        )
        # case when linear regression can approximates curvature in the graphs
        # but still the truth is graph is non-linear hence gap is consider smaller
        if curvature_signal and r2_gap < cfg.r2_gap_threshold:
            return NonlinearityTag.MonotonicNonlinear

        if (
            r2_gap >= cfg.r2_gap_threshold
            or mi >= cfg.mutual_information_threshold
        ):
            return NonlinearityTag.ComplexNonlinear

        return NonlinearityTag.Linear
