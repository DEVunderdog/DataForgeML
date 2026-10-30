"""
RegressionEstimatorFactory — maps a ``NonlinearityTag`` to a fitted-ready
sklearn estimator for the MICE block, via the Estimator Ladder (ADR-0094).

Estimator Ladder — a flat map, no bounds, no downgrade:
  Linear / Unpredictable                  → BayesianRidge
  MonotonicNonlinear / ComplexNonlinear   → RandomForest

``GradientBoostingRegressor`` is off the ladder entirely; it is reachable only
through the ``route`` Operation's ``model_choice`` argument (an explicit user
choice, ADR-0090). ``resolve_choice`` is total: every
``NonlinearityTag`` — including ``Unpredictable`` — resolves to a
:class:`~dataforge_ml.imputation.ModelChoice`, never ``None`` (ADR-0094 deleted the
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

from .._seed import DEFAULT_SEED
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

    # The forest parameters default to None, meaning "leave sklearn's default".
    def __init__(
        self,
        n_jobs: int = 1,
        random_state: int = DEFAULT_SEED,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        min_samples_split: int | None = None,
        min_samples_leaf: int | None = None,
        max_features: float | str | None = None,
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
        forest = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **{k: v for k, v in params.items() if v is not None},
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
            ladder, reachable only through ``route``'s ``model_choice``.
        """
        if tag in (NonlinearityTag.MonotonicNonlinear, NonlinearityTag.ComplexNonlinear):
            return ModelChoice.RandomForestRegressor
        return ModelChoice.BayesianRidge

    @staticmethod
    def build_from_choice(
        choice: ModelChoice,
        n_jobs: int = 1,
        params: dict[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> Any:
        """
        Return a fitted-ready sklearn estimator for an already-resolved choice.

        The fit-time counterpart of :meth:`resolve_choice`, and the factory's
        sole fit-time entry.  It takes the ``ModelChoice`` the router already
        stamped on the routing (ADR-0088) rather than re-deriving the family
        from the tag, so the estimator that trains is the one the
        user inspected and could override (``route``'s ``model_choice``), rather than
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
        params : dict, optional
            Constructor parameters for the estimator, as an estimator record's
            ``to_estimator()`` gives them. Omitted keys keep sklearn's
            defaults; ``random_state`` and ``n_jobs`` are never among them.
        seed : int, default DEFAULT_SEED
            Seed for the stochastic estimators (RandomForest and
            GradientBoosting), taken from the Operation's ``seed`` argument
            (ADR-0106). ``BayesianRidge`` is deterministic and ignores it.

        Returns
        -------
        sklearn estimator
            A freshly constructed, unfitted sklearn-compatible estimator.
        """
        params = params or {}
        if choice == ModelChoice.BayesianRidge:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(fit_intercept=True, **params)),
            ])
        if choice == ModelChoice.RandomForestRegressor:
            return _CoreInvariantRandomForest(random_state=seed, n_jobs=n_jobs, **params)
        return GradientBoostingRegressor(random_state=seed, **params)
