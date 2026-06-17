"""
RegressionEstimatorFactory — maps (NonlinearityTag, n_rows) to a fitted-ready
sklearn estimator for regression-based imputation.

Routing table:
  Linear              → Pipeline([StandardScaler, BayesianRidge(fit_intercept=True)])
  MonotonicNonlinear  → RandomForestRegressor
  ComplexNonlinear    → GradientBoostingRegressor (n_rows >= gradient_boost_min_rows)
                        RandomForestRegressor     (n_rows <  gradient_boost_min_rows)
  Unpredictable       → None  (caller routes to Median fallback)

The factory has no state and no side effects.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..profiling._numeric_config import NonlinearityTag
from ._config import NumericImputationConfig


class RegressionEstimatorFactory:
    """
    Stateless factory that returns a fitted-ready sklearn estimator based on
    a ``NonlinearityTag`` and row count.

    The factory has no instance state.  All public surface is a single static
    method so the caller can obtain the correct estimator with one call and
    proceed directly to ``fit``.
    """

    @staticmethod
    def build(
        tag: NonlinearityTag,
        n_rows: int,
        config: NumericImputationConfig,
    ) -> Optional[Any]:
        """
        Return a fitted-ready sklearn estimator for the given tag and dataset size.

        Parameters
        ----------
        tag : NonlinearityTag
            Nonlinearity classification for the target column, produced by
            ``NonlinearityProfiler``.
        n_rows : int
            Number of rows in the training dataset.  Used to choose between
            ``GradientBoostingRegressor`` and ``RandomForestRegressor`` for the
            ``ComplexNonlinear`` branch.
        config : NumericImputationConfig
            Imputation config supplying the ``gradient_boost_min_rows`` threshold.

        Returns
        -------
        sklearn estimator or None
            A freshly constructed, unfitted sklearn-compatible estimator, or
            ``None`` when ``tag`` is ``Unpredictable`` (signals the caller to
            route the column to a Median fallback instead).
        """
        if tag == NonlinearityTag.Linear:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(fit_intercept=True)),
            ])

        if tag == NonlinearityTag.MonotonicNonlinear:
            return RandomForestRegressor(random_state=0)

        if tag == NonlinearityTag.ComplexNonlinear:
            if n_rows >= config.gradient_boost_min_rows:
                return GradientBoostingRegressor(random_state=0)
            return RandomForestRegressor(random_state=0)

        # NonlinearityTag.Unpredictable
        return None
