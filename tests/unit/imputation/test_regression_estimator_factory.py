"""
Unit tests for RegressionEstimatorFactory — the Estimator Ladder (ADR-0094).

The ladder is a flat map, no bounds, no downgrade:
  Linear / Unpredictable                → BayesianRidge
  MonotonicNonlinear / ComplexNonlinear → RandomForestRegressor

``GradientBoostingRegressor`` is off the ladder entirely — reachable only
through ``with_model_choice`` — so ``resolve_choice``/``build_from_choice`` never
produce it. Tests are written via fit/predict on toy datasets.  They do NOT
inspect the internal estimator type — correctness is verified through
behaviour, except where the RandomForest branch's core-invariance (ADR-0069)
is itself the behaviour under test.
"""

import numpy as np
import pytest

from dataforge_ml.imputation._config import ModelChoice
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


def _build(tag: NonlinearityTag, n_jobs: int = 1):
    """The route-then-fit path the library takes: resolve the choice, then build it."""
    choice = RegressionEstimatorFactory.resolve_choice(tag)
    return RegressionEstimatorFactory.build_from_choice(choice, n_jobs=n_jobs)


def _split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2):
    n = len(y)
    split = int(n * (1 - test_frac))
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# resolve_choice is total over NonlinearityTag (ADR-0094)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,expected",
    [
        (NonlinearityTag.Linear, ModelChoice.BayesianRidge),
        (NonlinearityTag.Unpredictable, ModelChoice.BayesianRidge),
        (NonlinearityTag.MonotonicNonlinear, ModelChoice.RandomForestRegressor),
        (NonlinearityTag.ComplexNonlinear, ModelChoice.RandomForestRegressor),
    ],
)
def test_resolve_choice_is_total_and_matches_the_ladder(tag, expected):
    assert RegressionEstimatorFactory.resolve_choice(tag) == expected


def test_resolve_choice_never_returns_none():
    for tag in NonlinearityTag:
        choice = RegressionEstimatorFactory.resolve_choice(tag)
        assert choice is not None
        assert isinstance(choice, ModelChoice)


def test_resolve_choice_never_returns_gradient_boosting():
    """GradientBoosting is off the ladder — reachable only through with_model_choice."""
    for tag in NonlinearityTag:
        assert (
            RegressionEstimatorFactory.resolve_choice(tag)
            != ModelChoice.GradientBoostingRegressor
        )


# ---------------------------------------------------------------------------
# Linear / Unpredictable → BayesianRidge
# ---------------------------------------------------------------------------


def test_linear_estimator_produces_finite_predictions():
    X, y = _make_linear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    estimator = _build(
        tag=NonlinearityTag.Linear,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert np.all(np.isfinite(preds)), "Linear predictions must be finite"


def test_linear_estimator_predictions_not_all_identical():
    X, y = _make_linear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    estimator = _build(
        tag=NonlinearityTag.Linear,
    )
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert len(np.unique(preds)) > 1, "Linear predictions must not all be identical"


def test_unpredictable_builds_bayesian_ridge():
    """ADR-0094: Unpredictable maps to BayesianRidge, never None/skip."""
    choice = RegressionEstimatorFactory.resolve_choice(NonlinearityTag.Unpredictable)
    assert choice == ModelChoice.BayesianRidge
    estimator = _build(tag=NonlinearityTag.Unpredictable)
    assert estimator is not None
    X, y = _make_linear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    estimator.fit(X_train, y_train)
    assert np.all(np.isfinite(estimator.predict(X_test)))


# ---------------------------------------------------------------------------
# MonotonicNonlinear / ComplexNonlinear → RandomForestRegressor
# ---------------------------------------------------------------------------


def test_monotonic_nonlinear_estimator_produces_non_null_predictions():
    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    estimator = _build(
        tag=NonlinearityTag.MonotonicNonlinear,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert preds is not None
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


def test_complex_nonlinear_produces_non_null_predictions_at_any_size():
    """ADR-0097: no size-based downgrade — ComplexNonlinear is the forest at every size."""
    X, y = _make_nonlinear_dataset(n=500)
    X_train, y_train, X_test, _ = _split(X, y)
    estimator = _build(
        tag=NonlinearityTag.ComplexNonlinear,
    )
    assert estimator is not None
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    assert np.all(np.isfinite(preds))
    assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# build_from_choice: Custom raises, every real choice builds
# ---------------------------------------------------------------------------


def test_build_from_choice_custom_raises():
    with pytest.raises(ValueError, match="Custom"):
        RegressionEstimatorFactory.build_from_choice(ModelChoice.Custom)


def test_build_from_choice_gradient_boosting_still_buildable_explicitly():
    """Off the ladder, but with_model_choice can still ask for it directly."""
    estimator = RegressionEstimatorFactory.build_from_choice(
        ModelChoice.GradientBoostingRegressor
    )
    assert type(estimator).__name__ == "GradientBoostingRegressor"


# ---------------------------------------------------------------------------
# The RandomForest branch is core-invariant (ADR-0069)
#
# These are the exception to this module's "behaviour, never internal type"
# rule, and only just: what they assert is still behaviour -- that predictions
# do not move with n_jobs -- but the reason the branch has to be built as a
# wrapper at all is invisible from the outside, so `n_jobs` is passed
# explicitly rather than left to the caller that derives it.
# ---------------------------------------------------------------------------


def _forest(n_jobs):
    """The RandomForest branch of the factory, built at a given inner n_jobs."""
    return _build(
        tag=NonlinearityTag.MonotonicNonlinear,
        n_jobs=n_jobs,
    )


def test_forest_predictions_do_not_move_with_n_jobs():
    """Fitting wide must be bit-identical to fitting serially.

    Tree building is invariant across ``n_jobs`` -- each tree draws its seed
    sequentially -- so the whole of a forest's non-invariance lives in how
    ``predict`` accumulates them.  Pinning that accumulation is what lets the
    inner parallelism be spent without the derivation reaching the result.
    """
    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)

    serial = _forest(1)
    serial.fit(X_train, y_train)
    expected = serial.predict(X_test)

    for _ in range(3):
        wide = _forest(-1)
        wide.fit(X_train, y_train)
        # Exact, not approximate: the defect this guards is a ~1e-15 drift that
        # any tolerance-based assertion would wave through.
        assert np.array_equal(wide.predict(X_test), expected)


def test_forest_predictions_are_stable_against_themselves():
    """One fitted forest, one input, repeated predicts: the artifact must settle.

    A bare ``RandomForestRegressor`` holding ``n_jobs=-1`` fails this -- its
    trees are summed in thread-completion order on every call -- and ``n_jobs``
    pickles with the estimator, so it would fail for the life of the artifact.
    """
    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)

    forest = _forest(-1)
    forest.fit(X_train, y_train)

    first = forest.predict(X_test)
    for _ in range(5):
        assert np.array_equal(forest.predict(X_test), first)


def test_forest_survives_a_clone():
    """IterativeImputer clones its estimator per feature, so the branch must.

    A wrapper that dropped its params on ``clone`` would silently fit at the
    sklearn default rather than the derived ``n_jobs``.
    """
    from sklearn.base import clone

    forest = _forest(-1)
    cloned = clone(forest)
    assert cloned.get_params() == forest.get_params()

    X, y = _make_nonlinear_dataset()
    X_train, y_train, X_test, _ = _split(X, y)
    cloned.fit(X_train, y_train)
    assert np.all(np.isfinite(cloned.predict(X_test)))
