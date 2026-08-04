"""
RegressionEstimatorFactory — maps (NonlinearityTag, n_rows) to a fitted-ready
sklearn estimator for regression-based imputation.

Routing table:
  Linear              → Pipeline([StandardScaler, BayesianRidge(fit_intercept=True)])
  MonotonicNonlinear  → RandomForest
  ComplexNonlinear    → GradientBoostingRegressor (n_rows >= gradient_boost_min_rows)
                        RandomForest              (n_rows <  gradient_boost_min_rows)
  Unpredictable       → None  (caller routes to Median fallback)

The RandomForest branch is built as a ``_CoreInvariantRandomForest`` rather than
a bare ``RandomForestRegressor``, so ``n_jobs`` buys inner parallelism without
moving the numbers (ADR-0069).

The factory has no state and no side effects.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..profiling._numeric_config import NonlinearityTag
from ._config import ModelChoice, NumericImputationConfig


class _CoreInvariantRandomForest(BaseEstimator, RegressorMixin):
    """A RandomForest whose predictions do not depend on how many cores fit it.

    A bare ``RandomForestRegressor`` is not core-invariant, and the asymmetry is
    what this class exists to exploit (ADR-0069). Tree *building* is invariant:
    each tree draws its seed sequentially from ``random_state``, so ``n_jobs``
    only decides which core builds which tree, and the forests are byte-identical
    across the switch. *Prediction* is not: ``predict`` accumulates the trees into
    one shared array from a thread pool, so the additions land in completion
    order and the sum drifts at ~1e-15. That drift is not merely a difference
    between ``n_jobs=1`` and ``n_jobs=-1`` — repeated ``predict`` calls on a
    single wide model disagree with *each other*, and ``n_jobs`` is a fitted
    attribute that survives serialization, so a wide forest is an artifact that
    never settles.

    Fitting wide and predicting pinned is therefore bit-identical to fitting
    serially, which is what lets ``n_jobs`` stay derived from the outer degree
    without the derivation reaching the result. The pin must happen inside
    ``fit`` rather than at the call site because ``IterativeImputer`` predicts
    during its own round-robin, long before the caller sees an estimator back.
    """

    def __init__(self, n_jobs: int = 1, random_state: int = 0):
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestRegressor(
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        forest.fit(X, y)
        forest.n_jobs = 1
        self.forest_ = forest
        return self

    def predict(self, X):
        return self.forest_.predict(X)


class RegressionEstimatorFactory:
    """
    Stateless factory that returns a fitted-ready sklearn estimator based on
    a ``NonlinearityTag`` and row count.

    The factory has no instance state.  All public surface is a single static
    method so the caller can obtain the correct estimator with one call and
    proceed directly to ``fit``.
    """

    @staticmethod
    def resolve_choice(
        tag: NonlinearityTag,
        n_rows: int,
        config: NumericImputationConfig,
    ) -> Optional[ModelChoice]:
        """
        Resolve the concrete estimator family as a value-free ``ModelChoice`` label.

        The decide-time half of :meth:`build`: it selects the estimator family
        from ``(tag, n_rows, config)`` without constructing an estimator, so the
        assembler can stamp the choice on the plan (ADR-0060) and :meth:`build`
        can reuse the same mapping at fit-time.

        Parameters
        ----------
        tag : NonlinearityTag
            Nonlinearity classification for the target column.
        n_rows : int
            Number of rows the decision is made for.  Selects between
            ``GradientBoostingRegressor`` and ``RandomForestRegressor`` for the
            ``ComplexNonlinear`` branch via ``config.gradient_boost_min_rows``.
        config : NumericImputationConfig
            Imputation config supplying the ``gradient_boost_min_rows`` threshold.

        Returns
        -------
        ModelChoice or None
            The estimator family label, or ``None`` when ``tag`` is
            ``Unpredictable`` (the caller routes to a scalar fallback instead).
        """
        if tag == NonlinearityTag.Linear:
            return ModelChoice.BayesianRidge
        if tag == NonlinearityTag.MonotonicNonlinear:
            return ModelChoice.RandomForestRegressor
        if tag == NonlinearityTag.ComplexNonlinear:
            if n_rows >= config.gradient_boost_min_rows:
                return ModelChoice.GradientBoostingRegressor
            return ModelChoice.RandomForestRegressor
        return None

    @staticmethod
    def build(
        tag: NonlinearityTag,
        n_rows: int,
        config: NumericImputationConfig,
        n_jobs: int = 1,
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
        n_jobs : int, default 1
            ``n_jobs`` for estimators that support inner parallelism (the
            RandomForest branch).  Pinned to ``1`` when this fit is nested under
            the outer thread parallelism (ADR-0056) to avoid core
            oversubscription; a block running alone passes ``-1`` for full inner
            parallelism.  Estimators without an ``n_jobs`` parameter
            (``BayesianRidge``, ``GradientBoostingRegressor``) ignore it.  The
            value never reaches the result: the RandomForest branch is built as
            a :class:`_CoreInvariantRandomForest`, which spends the cores on tree
            building and predicts pinned regardless (ADR-0069).

        Returns
        -------
        sklearn estimator or None
            A freshly constructed, unfitted sklearn-compatible estimator, or
            ``None`` when ``tag`` is ``Unpredictable`` (signals the caller to
            route the column to a Median fallback instead).
        """
        choice = RegressionEstimatorFactory.resolve_choice(tag, n_rows, config)
        return RegressionEstimatorFactory.build_from_choice(choice, n_jobs=n_jobs)

    @staticmethod
    def build_from_choice(
        choice: Optional[ModelChoice],
        n_jobs: int = 1,
    ) -> Optional[Any]:
        """
        Return a fitted-ready sklearn estimator for an already-resolved choice.

        The fit-time counterpart of :meth:`resolve_choice`.  Where :meth:`build`
        re-derives the family from ``(tag, n_rows, config)``, this takes the
        ``ModelChoice`` the decision layer already stamped on the plan
        (ADR-0060), so the estimator that trains is the one the user inspected
        and could override, rather than a second, independent resolution.

        Parameters
        ----------
        choice : ModelChoice or None
            Estimator family carried on the plan.  ``None`` mirrors the
            ``Unpredictable`` branch of :meth:`resolve_choice`.
        n_jobs : int, default 1
            ``n_jobs`` for estimators that support inner parallelism
            (``RandomForestRegressor``); see :meth:`build`.

        Returns
        -------
        sklearn estimator or None
            A freshly constructed, unfitted sklearn-compatible estimator, or
            ``None`` when ``choice`` is ``None`` (signals the caller to route the
            column to a scalar fallback instead).

        Raises
        ------
        ValueError
            If ``choice`` is :attr:`~dataforge_ml.ModelChoice.Custom`.  That
            member names an estimator the library did not build, so there is
            nothing here to construct; the instance travels on the plan's
            ``custom_estimators`` map and the caller reads it from there
            (ADR-0083).  A plan reloaded from bytes has the slot empty — the
            estimator is never serialized — and this raise is what stops such a
            plan silently imputing with a library default instead.
        """
        if choice == ModelChoice.Custom:
            raise ValueError(
                "ModelChoice.Custom names a user-supplied estimator, which this "
                "factory did not build and cannot reconstruct. The instance is "
                "not serialized, so a plan reloaded from bytes carries the label "
                "with an empty slot; re-supply it with "
                "author(..., estimators={unit_id: estimator})."
            )
        if choice == ModelChoice.BayesianRidge:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(fit_intercept=True)),
            ])
        if choice == ModelChoice.RandomForestRegressor:
            return _CoreInvariantRandomForest(random_state=0, n_jobs=n_jobs)
        if choice == ModelChoice.GradientBoostingRegressor:
            return GradientBoostingRegressor(random_state=0)

        # NonlinearityTag.Unpredictable → no model choice → scalar fallback
        return None
