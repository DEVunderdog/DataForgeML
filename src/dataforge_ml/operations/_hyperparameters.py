"""Typed hyperparameter and estimator records, shared by the impute Operations (ADR-0101).

A unit's dials travel as one record per strategy, tagged by ``kind``, so an
unknown key or value fails in the record before any work. Each record converts
to the dial dict :class:`~dataforge_ml.imputation.ImputationRecipe` already
reads, and back.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import Field

from ..imputation._config import ImputationStrategy, ModelChoice
from ..records import WireRecord

Count = Annotated[int, Field(ge=1)]


class _DialRecord(WireRecord):
    """Base of the per-strategy hyperparameter records."""

    def to_hyperparameters(self) -> dict[str, Any]:
        """Return the set fields as the dial dict the recipe reads.

        Returns
        -------
        dict
            One entry per non-null field, ``kind`` left out. A partial record
            gives the sparse override delta; a full one the whole dial row.
            A value is ``None`` only where the recipe's dial means it, such as
            MICE's ``n_nearest_features`` set to ``all``.
        """
        return self.model_dump(exclude={"kind"}, exclude_none=True)

    @classmethod
    def _from_dials(cls, dials: dict[str, Any]) -> _DialRecord:
        (kind,) = get_args(cls.model_fields["kind"].annotation)
        return cls.model_validate({"kind": kind, **dials})


class MiceHyperparameters(_DialRecord):
    """Dials of the ``mice`` unit, the joint MICE block. Null keeps the recipe's resolved value."""

    kind: Literal["mice"]
    """Always ``mice``."""
    max_iter: Count | None = None
    """Most imputation rounds the block runs."""
    tol: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    """Convergence tolerance: the block stops once a round changes the fill by less than this."""
    initial_strategy: Literal["mean", "median"] | None = None
    """Which statistic fills missing cells before the first round."""
    n_nearest_features: Count | Literal["all"] | None = None
    """Most predictors each column's model reads, the most correlated first. ``all`` reads every active numeric column."""

    def to_hyperparameters(self) -> dict[str, Any]:
        dials = super().to_hyperparameters()
        if dials.get("n_nearest_features") == "all":
            dials["n_nearest_features"] = None
        return dials

    @classmethod
    def _from_dials(cls, dials: dict[str, Any]) -> _DialRecord:
        if "n_nearest_features" in dials and dials["n_nearest_features"] is None:
            dials = {**dials, "n_nearest_features": "all"}
        return super()._from_dials(dials)


class KnnHyperparameters(_DialRecord):
    """Dials of the ``knn`` unit, the joint KNN block. Null keeps the recipe's resolved value."""

    kind: Literal["knn"]
    """Always ``knn``."""
    n_neighbors: Count | None = None
    """Rows whose values fill a missing cell. A column with fewer donor rows uses them all."""
    weights: Literal["uniform", "distance"] | None = None
    """``distance`` weighs nearer rows more; ``uniform`` weighs every neighbour the same."""


class MnarHyperparameters(_DialRecord):
    """Dials of one ``mnar:<column>`` unit. Null keeps the recipe's resolved value."""

    kind: Literal["mnar"]
    """Always ``mnar``."""
    central_tendency: Literal["mean", "median", "mode"] | None = None
    """Which statistic the fill on record is. The column keeps its nulls either way."""


class ClusterConditionalHyperparameters(_DialRecord):
    """Dials of one ``cluster_conditional:<column>`` unit. Null keeps the recipe's resolved value."""

    kind: Literal["cluster_conditional"]
    """Always ``cluster_conditional``."""
    central_tendency: Literal["mean", "median"] | None = None
    """Which statistic of its group fills a missing cell."""


#: One unit's dials: the record of its strategy, picked by ``kind``.
Hyperparameters = Annotated[
    MiceHyperparameters | KnnHyperparameters | MnarHyperparameters | ClusterConditionalHyperparameters,
    Field(discriminator="kind"),
]

_RECORDS: dict[ImputationStrategy, type[_DialRecord]] = {
    ImputationStrategy.MICE: MiceHyperparameters,
    ImputationStrategy.KNN: KnnHyperparameters,
    ImputationStrategy.MNAR: MnarHyperparameters,
    ImputationStrategy.ClusterConditional: ClusterConditionalHyperparameters,
}


def hyperparameters_record(strategy: ImputationStrategy, dials: dict[str, Any]) -> _DialRecord:
    """Build the typed record for a strategy from the recipe's dial dict.

    Parameters
    ----------
    strategy : ImputationStrategy
        The unit's strategy.
    dials : dict
        A full dial row or a sparse override delta, as the recipe holds it.

    Returns
    -------
    _DialRecord
        The strategy's record, with every key absent from ``dials`` null.

    Raises
    ------
    KeyError
        If ``strategy`` has no dials, so no record.
    ValidationError
        If ``dials`` holds a key or value the record does not accept.
    """
    return _RECORDS[strategy]._from_dials(dials)


# --- Estimators -------------------------------------------------------------------

#: The estimators the MICE block can train with, by value.
EstimatorName = Literal["bayesian_ridge", "random_forest_regressor", "gradient_boosting_regressor"]

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SplitCount = Annotated[int, Field(ge=2)]
FeatureShare = Annotated[float, Field(gt=0, le=1)] | Literal["sqrt", "log2"]


class _EstimatorRecord(WireRecord):
    """Base of the per-estimator parameter records."""

    def to_estimator(self) -> tuple[ModelChoice, dict[str, Any]]:
        """Return the estimator as the factory builds it.

        Returns
        -------
        tuple of (ModelChoice, dict)
            The estimator family and its constructor parameters, one entry
            per non-null field. An empty dict builds the library's defaults.
        """
        params = self.model_dump(exclude={"kind"}, exclude_none=True)
        return ModelChoice(self.kind), params  # type: ignore[attr-defined]


class BayesianRidgeEstimator(_EstimatorRecord):
    """Bayesian ridge regression on standardised predictors. Null keeps the library's default."""

    kind: Literal["bayesian_ridge"]
    """Always ``bayesian_ridge``."""
    max_iter: Count | None = None
    """Most fitting iterations. Null means 300."""
    tol: NonNegative | None = None
    """Fitting stops once the weights change by less than this. Null means 0.001."""
    alpha_1: NonNegative | None = None
    """Shape of the Gamma prior on the noise precision. Null means 1e-06."""
    alpha_2: NonNegative | None = None
    """Rate of the Gamma prior on the noise precision. Null means 1e-06."""
    lambda_1: NonNegative | None = None
    """Shape of the Gamma prior on the weight precision. Null means 1e-06."""
    lambda_2: NonNegative | None = None
    """Rate of the Gamma prior on the weight precision. Null means 1e-06."""


class RandomForestEstimator(_EstimatorRecord):
    """A random forest of regression trees. Null keeps the library's default."""

    kind: Literal["random_forest_regressor"]
    """Always ``random_forest_regressor``."""
    n_estimators: Count | None = None
    """Number of trees. Null means 100."""
    max_depth: Count | None = None
    """Deepest a tree may grow. Null grows each tree until its leaves are pure or too small to split."""
    min_samples_split: SplitCount | None = None
    """Fewest rows a node needs to be split. Null means 2."""
    min_samples_leaf: Count | None = None
    """Fewest rows a leaf may hold. Null means 1."""
    max_features: FeatureShare | None = None
    """Predictors each split considers: a share of them, ``sqrt`` or ``log2`` of their count. Null means all of them."""


class GradientBoostingEstimator(_EstimatorRecord):
    """Gradient-boosted regression trees. Null keeps the library's default."""

    kind: Literal["gradient_boosting_regressor"]
    """Always ``gradient_boosting_regressor``."""
    n_estimators: Count | None = None
    """Number of boosting stages. Null means 100."""
    learning_rate: Positive | None = None
    """How much each stage's tree adds. Null means 0.1."""
    max_depth: Count | None = None
    """Deepest each tree may grow. Null means 3."""
    min_samples_split: SplitCount | None = None
    """Fewest rows a node needs to be split. Null means 2."""
    min_samples_leaf: Count | None = None
    """Fewest rows a leaf may hold. Null means 1."""
    subsample: Annotated[float, Field(gt=0, le=1)] | None = None
    """Share of rows each stage fits on; below 1 is stochastic boosting. Null means 1."""
    max_features: FeatureShare | None = None
    """Predictors each split considers: a share of them, ``sqrt`` or ``log2`` of their count. Null means all of them."""


#: The MICE block's estimator: the record of one named estimator, picked by ``kind``.
MiceEstimator = Annotated[
    BayesianRidgeEstimator | RandomForestEstimator | GradientBoostingEstimator,
    Field(discriminator="kind"),
]

_ESTIMATORS: dict[ModelChoice, type[_EstimatorRecord]] = {
    ModelChoice.BayesianRidge: BayesianRidgeEstimator,
    ModelChoice.RandomForestRegressor: RandomForestEstimator,
    ModelChoice.GradientBoostingRegressor: GradientBoostingEstimator,
}


def estimator_record(choice: ModelChoice, params: dict[str, Any]) -> _EstimatorRecord:
    """Build the typed record for an estimator from its constructor parameters.

    Parameters
    ----------
    choice : ModelChoice
        The estimator family.
    params : dict
        Constructor parameters, as :meth:`_EstimatorRecord.to_estimator` gives them.

    Returns
    -------
    _EstimatorRecord
        The estimator's record, with every parameter absent from ``params`` null.

    Raises
    ------
    ValidationError
        If ``params`` holds a key or value the record does not accept.
    """
    return _ESTIMATORS[choice].model_validate({"kind": choice.value, **params})
