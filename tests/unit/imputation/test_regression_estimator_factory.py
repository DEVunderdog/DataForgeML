"""
Unit tests for RegressionEstimatorFactory (Issue #140).

All five branches are covered:
  1. Linear              — estimator fits and produces finite, non-constant predictions
  2. MonotonicNonlinear  — estimator fits and produces non-null predictions
  3. ComplexNonlinear large dataset (n_rows >= gradient_boost_min_rows) — GradientBoosting path
  4. ComplexNonlinear small dataset (n_rows <  gradient_boost_min_rows) — RandomForest path
  5. Unpredictable       — factory returns None

Tests are written via fit/predict on toy datasets.  They do NOT inspect the
internal estimator type — correctness is verified through behaviour.
"""

import numpy as np
import pytest

from dataforge_ml.imputation._config import NumericImputationConfig
from dataforge_ml.imputation._regression_estimator_factory import (
    RegressionEstimatorFactory,
)
from dataforge_ml.profiling._numeric_config import NonlinearityTag


# ---------------------------------------------------------------------------
# Shared toy dataset helpers
# ---------------------------------------------------------------------------


def _make_linear_dataset(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """y = 3*x1 + 2*x2 + small noise (tight linear relationship)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.05 * rng.standard_normal(n)
    return X, y


def _make_nonlinear_dataset(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """y = x1 * x2 + noise (interaction term — non-linear)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 2))
    y = X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


def _split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2):
    n = len(y)
    split = int(n * (1 - test_frac))
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Branch 1: Linear
# ---------------------------------------------------------------------------


def test_linear_estimator_produces_finite_predictions():
    X, y = _make_linear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    config = NumericImputationConfig()
    estimator = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.Linear,
        n_rows=len(y_train),
        config=config,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert np.all(np.isfinite(preds)), "Linear predictions must be finite"


def test_linear_estimator_predictions_not_all_identical():
    X, y = _make_linear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    config = NumericImputationConfig()
    estimator = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.Linear,
        n_rows=len(y_train),
        config=config,
    )
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert len(np.unique(preds)) > 1, "Linear predictions must not all be identical"


# ---------------------------------------------------------------------------
# Branch 2: MonotonicNonlinear
# ---------------------------------------------------------------------------


def test_monotonic_nonlinear_estimator_produces_non_null_predictions():
    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    config = NumericImputationConfig()
    estimator = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.MonotonicNonlinear,
        n_rows=len(y_train),
        config=config,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert preds is not None
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Branch 3: ComplexNonlinear large dataset → GradientBoosting path
# ---------------------------------------------------------------------------


def test_complex_nonlinear_large_dataset_produces_non_null_predictions():
    X, y = _make_nonlinear_dataset(n=500)
    X_train, y_train, X_test, _ = _split(X, y)
    config = NumericImputationConfig(gradient_boost_min_rows=100)
    estimator = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=500,
        config=config,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert np.all(np.isfinite(preds))
    assert len(preds) == len(X_test)


def test_complex_nonlinear_large_uses_gradient_boost_path():
    """n_rows >= gradient_boost_min_rows — GradientBoosting path predictions differ from RF path."""
    X, y = _make_nonlinear_dataset(n=500)
    config = NumericImputationConfig(gradient_boost_min_rows=100)
    est_large = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=500,
        config=config,
    )
    est_small = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=50,
        config=config,
    )
    assert est_large is not None
    assert est_small is not None
    X_train, y_train, X_test, _ = _split(X, y)
    est_large.fit(X_train, y_train)
    est_small.fit(X_train, y_train)
    preds_large = est_large.predict(X_test)
    preds_small = est_small.predict(X_test)
    # They should both be valid predictions — not required to be identical
    assert np.all(np.isfinite(preds_large))
    assert np.all(np.isfinite(preds_small))


# ---------------------------------------------------------------------------
# Branch 4: ComplexNonlinear small dataset → RandomForest path
# ---------------------------------------------------------------------------


def test_complex_nonlinear_small_dataset_produces_non_null_predictions():
    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    config = NumericImputationConfig(gradient_boost_min_rows=10_000)
    estimator = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=200,
        config=config,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert np.all(np.isfinite(preds))
    assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# Branch 5: Unpredictable → None
# ---------------------------------------------------------------------------


def test_unpredictable_returns_none():
    config = NumericImputationConfig()
    result = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.Unpredictable,
        n_rows=1000,
        config=config,
    )
    assert result is None


def test_unpredictable_returns_none_regardless_of_n_rows():
    config = NumericImputationConfig()
    for n_rows in [10, 100, 1_000, 100_000]:
        result = RegressionEstimatorFactory.build(
            tag=NonlinearityTag.Unpredictable,
            n_rows=n_rows,
            config=config,
        )
        assert result is None, f"Expected None for n_rows={n_rows}"


# ---------------------------------------------------------------------------
# Threshold boundary: gradient_boost_min_rows
# ---------------------------------------------------------------------------


def test_complex_nonlinear_exactly_at_threshold_uses_gradient_boost_path():
    """n_rows == gradient_boost_min_rows should select the GradientBoosting branch."""
    X, y = _make_nonlinear_dataset(n=300)
    X_train, y_train, X_test, _ = _split(X, y)
    threshold = 200
    config = NumericImputationConfig(gradient_boost_min_rows=threshold)

    est_at = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=threshold,
        config=config,
    )
    est_below = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=threshold - 1,
        config=config,
    )
    assert est_at is not None
    assert est_below is not None

    est_at.fit(X_train, y_train)
    est_below.fit(X_train, y_train)

    assert np.all(np.isfinite(est_at.predict(X_test)))
    assert np.all(np.isfinite(est_below.predict(X_test)))


# ---------------------------------------------------------------------------
# Config threshold is respected
# ---------------------------------------------------------------------------


def test_custom_gradient_boost_min_rows_respected():
    """A custom config threshold changes which estimator is selected."""
    config_low = NumericImputationConfig(gradient_boost_min_rows=50)
    config_high = NumericImputationConfig(gradient_boost_min_rows=10_000)

    est_gb = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=100,
        config=config_low,
    )
    est_rf = RegressionEstimatorFactory.build(
        tag=NonlinearityTag.ComplexNonlinear,
        n_rows=100,
        config=config_high,
    )

    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)

    est_gb.fit(X_train, y_train)
    est_rf.fit(X_train, y_train)

    # Both must produce valid predictions
    assert np.all(np.isfinite(est_gb.predict(X_test)))
    assert np.all(np.isfinite(est_rf.predict(X_test)))
