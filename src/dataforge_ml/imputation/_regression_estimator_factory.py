"""
RegressionEstimatorFactory — maps a ``NonlinearityTag`` to a fitted-ready
sklearn estimator for the MICE block, via the Estimator Ladder (ADR-0094).

Estimator Ladder — a flat map, no bounds, no downgrade:
  Linear / Unpredictable                  → BayesianRidge
  MonotonicNonlinear / ComplexNonlinear   → RandomForest

``GradientBoostingRegressor`` is off the ladder entirely; it is reachable only
through :meth:`~dataforge_ml.imputation.ImputationRouting.with_model_choice`
(an explicit user choice, ADR-0090). ``resolve_choice`` is total: every
``NonlinearityTag`` — including ``Unpredictable`` — resolves to a
:class:`~dataforge_ml.ModelChoice`, never ``None`` (ADR-0094 deleted the
``None``/skip branch the ``Unpredictable`` tag used to take).

The RandomForest branch is built as a ``_CoreInvariantRandomForest`` rather than
a bare ``RandomForestRegressor``, so ``n_jobs`` buys inner parallelism without
moving the numbers (ADR-0069).

The factory has no state and no side effects.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..profiling._numeric_config import NonlinearityTag
from ._config import ModelChoice


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
    a ``NonlinearityTag``, via the Estimator Ladder (ADR-0094).

    The factory has no instance state.  Its surface is two static methods:
    :meth:`resolve_choice` picks the family at route-time, and
    :meth:`build_from_choice` constructs it at fit-time.
    """

    @staticmethod
    def resolve_choice(tag: NonlinearityTag) -> ModelChoice:
        """
        Resolve the concrete estimator family as a value-free ``ModelChoice`` label.

        The route-time half of the factory: it selects the estimator family
        from the tag without constructing an estimator, so the router
        can stamp the choice on the routing (ADR-0088) and
        :meth:`build_from_choice` can construct it at fit-time. Total over ``NonlinearityTag``
        (ADR-0094): every tag, including ``Unpredictable``, resolves to a
        ``ModelChoice`` — a column only reaches MICE with that tag when the
        Signal Score found a signal the probe missed, so ``BayesianRidge`` is
        the honest, cheapest match.

        Parameters
        ----------
        tag : NonlinearityTag
            Nonlinearity classification for the target column.

        Returns
        -------
        ModelChoice
            The estimator family label: ``BayesianRidge`` for ``Linear`` and
            ``Unpredictable``; ``RandomForestRegressor`` for
            ``MonotonicNonlinear`` and ``ComplexNonlinear``.
            ``GradientBoostingRegressor`` is never returned — it is off the
            ladder, reachable only through ``with_model_choice``.
        """
        if tag in (NonlinearityTag.MonotonicNonlinear, NonlinearityTag.ComplexNonlinear):
            return ModelChoice.RandomForestRegressor
        return ModelChoice.BayesianRidge

    @staticmethod
    def build_from_choice(
        choice: ModelChoice,
        n_jobs: int = 1,
    ) -> Any:
        """
        Return a fitted-ready sklearn estimator for an already-resolved choice.

        The fit-time counterpart of :meth:`resolve_choice`, and the factory's
        sole fit-time entry.  It takes the ``ModelChoice`` the router already
        stamped on the routing (ADR-0088) rather than re-deriving the family
        from the tag, so the estimator that trains is the one the
        user inspected and could override (``with_model_choice``), rather than
        a second, independent resolution.

        Parameters
        ----------
        choice : ModelChoice
            Estimator family carried on the routing. Never ``None``:
            :meth:`resolve_choice` is total over every ``NonlinearityTag``
            (ADR-0094).
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
        sklearn estimator
            A freshly constructed, unfitted sklearn-compatible estimator.

        Raises
        ------
        ValueError
            If ``choice`` is :attr:`~dataforge_ml.ModelChoice.Custom`.  That
            member names an estimator the library did not build, so there is
            nothing here to construct; the instance travels on the routing's
            ``mice_estimator`` slot and the caller reads it from there
            (ADR-0090).  A routing reloaded from bytes has the slot empty — the
            estimator is never serialized — and this raise is what stops such a
            routing silently imputing with a library default instead.
        """
        if choice == ModelChoice.Custom:
            raise ValueError(
                "ModelChoice.Custom names a user-supplied estimator, which this "
                "factory did not build and cannot reconstruct. The instance is "
                "not serialized, so a routing reloaded from bytes carries the "
                "label with an empty slot; re-supply it with "
                "routing.with_model_choice(estimator)."
            )
        if choice == ModelChoice.BayesianRidge:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(fit_intercept=True)),
            ])
        if choice == ModelChoice.RandomForestRegressor:
            return _CoreInvariantRandomForest(random_state=0, n_jobs=n_jobs)
        return GradientBoostingRegressor(random_state=0)
