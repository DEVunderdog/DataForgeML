"""The ``route`` Operation: route every column of a profile to an imputation strategy."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._operation import operation
from .._boundary._route import route_profile
from ..records import ColumnKeyedResult, Table, WireRecord
from ._hyperparameters import EstimatorName

#: Every strategy a column can end up with, by value.
StrategyName = Literal[
    "mean",
    "median",
    "mode",
    "knn",
    "mice",
    "mnar",
    "constant",
    "dropped",
    "passthrough",
    "cluster_conditional",
    "gmm_sampling",
    "indicator",
]
#: The strategies you may declare: every one except ``indicator``.
DeclarableStrategy = Literal[
    "mean",
    "median",
    "mode",
    "knn",
    "mice",
    "mnar",
    "constant",
    "dropped",
    "passthrough",
    "cluster_conditional",
    "gmm_sampling",
]
#: The strategies ``default_strategy`` may stamp: every declarable one except ``constant``.
DefaultStrategy = Literal[
    "mean",
    "median",
    "mode",
    "knn",
    "mice",
    "mnar",
    "dropped",
    "passthrough",
    "cluster_conditional",
    "gmm_sampling",
]
ReasonName = Literal[
    "drop_candidate",
    "constant_declared",
    "strategy_forced",
    "mnar_declared",
    "no_missing_values",
    "soft_excluded",
    "not_imputed_type",
    "authored",
    "mcar_minor_unskewed",
    "mcar_skewed",
    "mar_unescalated",
    "bounded_discrete_scalar",
    "bimodal_grouping_declared",
    "bimodal_few_correlated",
    "bimodal_uncorrelated",
    "signal_tier_pick",
    "bottom_tier_fill",
    "no_feasible_model",
    "no_predictor_correlation",
]
SemanticTypeName = Literal["numeric", "categorical", "datetime", "boolean", "text", "identifier"]

Ratio = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=1)]


# --- Declarations -----------------------------------------------------------------


class StrategyDeclaration(WireRecord):
    """One column's strategy, set by you instead of routed."""

    column: str
    """Column name, exactly as in the data."""
    strategy: DeclarableStrategy
    """The strategy. A column that is not numeric, or that profiling skipped, takes only ``passthrough`` or ``dropped``. ``indicator`` is never declared: ``mnar`` adds the ``<column>_missing`` indicator itself."""
    constant_fill: Annotated[float, Field(allow_inf_nan=False)] | None = None
    """The fill value of a ``constant`` column, which needs one. Null for every other strategy."""
    grouping_variable: str | None = None
    """For ``cluster_conditional`` only: the column whose groups the fill is taken within. Null takes it per mixture component instead."""


# --- Dials ------------------------------------------------------------------------


class RoutingDials(WireRecord):
    """Dials routing reads for numeric columns. Null keeps every default in this group."""

    knn_max_rows: Count | None = None
    """KNN's Resource Ceiling: most rows a frame may have for a column to take ``knn``. Null means 50000."""
    mice_min_rows_per_predictor: Annotated[float, Field(ge=0)] | None = None
    """The Feasibility Floor: fewest non-missing rows per predictor a column needs to take ``mice``. Null means 2."""
    mice_max_nearest_features: Count | None = None
    """Most predictors the Feasibility Floor counts for ``mice``. Null means 20."""
    mice_correlation_threshold: Ratio | None = None
    """Least absolute correlation for a column to count as a predictor in the Signal Score's breadth. Null means 0.1."""
    signal_score_breadth_weight: Ratio | None = None
    """How much the count of predictors may add to the Signal Score. Null means 0.1."""
    signal_score_latent_weight: Ratio | None = None
    """How far mutual information alone can raise the Signal Score. Null means 0.3."""
    signal_score_middle_tier_min: Ratio | None = None
    """Signal Score at or above which a column may take a model (``knn`` first). Must not exceed ``signal_score_top_tier_min``. Null means 0.3."""
    signal_score_top_tier_min: Ratio | None = None
    """Signal Score at or above which a column takes the richest feasible model (``mice`` first). Null means 0.6."""
    bimodal_min_correlated_features: Count | None = None
    """Fewest correlated columns a bimodal column needs to try a model instead of ``cluster_conditional``. Null means 3."""
    bimodal_correlation_threshold: Ratio | None = None
    """Least absolute correlation for a column to count toward ``bimodal_min_correlated_features``. Null means 0.2."""


class RouteThresholds(WireRecord):
    """Every routing dial, grouped. Null keeps every default."""

    numeric: RoutingDials | None = None
    """The Feasibility Floor, the Signal Score and the bimodal rules."""


# --- Result -----------------------------------------------------------------------


class RoutedColumn(WireRecord):
    """One column's strategy and the rule that fixed it."""

    column: str
    """Column name. A ``<column>_missing`` row is the indicator an ``mnar`` column adds."""
    semantic_type: SemanticTypeName | None
    """The profile's semantic type; null when profiling skipped the column. An indicator row is ``boolean``."""
    strategy: StrategyName
    """The strategy the column takes."""
    reason: ReasonName
    """The Routing Reason: the one rule that fixed ``strategy``. ``authored`` when you declared it. ``explain_routing`` gives the evidence behind it."""


class StrategyCount(WireRecord):
    """How many columns take one strategy."""

    strategy: StrategyName
    """The strategy."""
    count: int
    """Number of ``columns`` rows, indicators included, that take it."""


class EstimatorReason(WireRecord):
    """Why the MICE block trains with its estimator."""

    source: Literal["ladder", "argument"]
    """``argument`` when ``model_choice`` set it, ``ladder`` when the most complex nonlinearity among the block's columns did."""
    tag: Literal["linear", "monotonic_nonlinear", "complex_nonlinear", "unpredictable"]
    """The block's most complex nonlinearity tag. The ladder takes ``random_forest_regressor`` for a nonlinear tag, else ``bayesian_ridge``."""
    column: str
    """The first MICE column carrying ``tag``."""


class RouteResult(ColumnKeyedResult):
    """The routing Handle and each column's strategy. ``result.get(name)`` finds one column."""

    routing: str
    """The routing Handle, such as ``routing_7f3a9c2e``. It stores every column's Routing Evidence."""
    profile: str
    """The profile Handle that was routed."""
    mice_estimator: EstimatorName | None
    """The estimator the MICE block trains with; null when no column takes ``mice``."""
    estimator_reason: EstimatorReason | None
    """Why ``mice_estimator`` was chosen; null when it is null."""
    census: Table[StrategyCount]
    """Columns per strategy, over every column, most first."""
    columns: Table[RoutedColumn]
    """Every column but the excluded ones: those that fill or drop values first, then ``passthrough``, each group in dataset order. At most 100; past 100, ``omitted`` counts the rest and names ``explain_routing``."""


@operation(effect="create", phase="impute", scales_with_data=False)
def route(
    profile: str,
    strategies: list[StrategyDeclaration] | None = None,
    default_strategy: DefaultStrategy | None = None,
    model_choice: EstimatorName | None = None,
    exclude_columns: list[str] | None = None,
    skip_columns: list[str] | None = None,
    thresholds: RouteThresholds | None = None,
) -> RouteResult:
    """Route every column of a profile to an imputation strategy, into a routing Handle.

    Use when
    --------
    Call this after ``profile`` to decide how each column's missing values
    are filled. Routing reads only the profile, never the rows. Each column
    gets a strategy and a Routing Reason: the one rule that fixed it.
    Columns you list in ``strategies`` take your strategy with reason
    ``authored``; every other column is routed from its profile, or takes
    ``default_strategy`` when you set one. A column you force onto ``knn``,
    ``mice``, ``cluster_conditional`` or ``gmm_sampling`` past the Feasibility
    Floor still trains, and the ``forced_past_feasibility`` warning names it.
    Non-numeric columns keep their missing values. To change a routing, call
    ``route`` again; that makes a new routing Handle and leaves the old one
    as it was.

    Parameters
    ----------
    profile : str
        A profile Handle, such as ``profile_7f3a9c2e``.
    strategies : list of StrategyDeclaration or None
        Strategies you set, one entry per column. Null means none.
    default_strategy : str or None
        The strategy every unlisted numeric column takes. Unlisted columns
        that are not numeric stay ``passthrough``. Null routes every unlisted
        column from its profile.
    model_choice : str or None
        The estimator the MICE block trains with. Null lets the Estimator
        Ladder pick it from the block's nonlinearity. Setting it with no
        ``mice`` column fails.
    exclude_columns : list of str or None
        Columns to leave out of this and every later phase, added to what the
        profile already excludes. Null means none.
    skip_columns : list of str or None
        Columns routing leaves as they are: ``passthrough`` with reason
        ``soft_excluded``, and never a predictor. Null means none.
    thresholds : RouteThresholds or None
        Dials, grouped by what they judge, each with its default and range.
        Null keeps every default.

    Returns
    -------
    RouteResult
        ``routing`` and ``profile`` Handles, ``mice_estimator`` and
        ``estimator_reason``, ``census`` (columns per strategy) and
        ``columns`` (at most 100). ``warnings`` lists non-numeric columns
        that keep their missing values (``non_numeric_missing_kept``) and
        columns forced past the Feasibility Floor
        (``forced_past_feasibility``).

    Errors
    ------
    unknown_handle
        ``profile`` names no profile Handle in this Workspace. ``candidates``
        lists profile Handles, newest first.
    wrong_handle_kind
        ``profile`` is a Handle of another kind.
    unknown_column
        An argument names a column the profile's frame lacks. ``candidates``
        lists close matches; ``issues`` lists every problem argument.
    duplicate_key
        ``strategies`` names the same column twice.
    declaration_conflict
        Two declarations contradict: a listed or grouping column that is
        excluded or skipped, or an excluded target.
    strategy_not_declarable
        A column that is not numeric, or that profiling skipped, is given a
        strategy other than ``passthrough`` or ``dropped``.
    constant_fill_missing
        A ``constant`` entry has no ``constant_fill``.
    invalid_argument
        An argument has the wrong type or is out of its range, or an entry
        sets ``constant_fill`` or ``grouping_variable`` its strategy does not
        read.
    invalid_dial_value
        ``signal_score_middle_tier_min`` is above ``signal_score_top_tier_min``.
    no_mice_block
        ``model_choice`` is set but no column takes ``mice``.
    incompatible_artifact
        Another library format version wrote ``profile``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    profile : Make the profile Handle to route.
    list_handles : Find routing Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge route --profile profile_1b2c3d4e --strategies '[{"column":"income","strategy":"median"}]'
    {...,"routing":"routing_5a6b7c8d","profile":"profile_1b2c3d4e",...}
    """
    return RouteResult.model_validate(
        route_profile(
            profile=profile,
            strategies=strategies,
            default_strategy=default_strategy,
            model_choice=model_choice,
            exclude_columns=exclude_columns,
            skip_columns=skip_columns,
            thresholds=thresholds,
        )
    )
