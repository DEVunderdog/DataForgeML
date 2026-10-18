"""The ``explain_routing`` Operation: the stored Routing Evidence behind each column's strategy."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .._boundary._explain_routing import EXPLAIN_CAP, explain_routing_columns
from .._boundary._operation import operation
from ..records import ColumnKeyedResult, DialPointer, Table, WireRecord
from .route import ReasonName, StrategyName

EntryCauseName = Literal[
    "bounded_discrete",
    "mar_multi",
    "mar_severe",
    "mar_high_correlated",
    "mar_high_uncorrelated",
    "mar_shape_escalation",
    "mcar_high_severity",
    "mcar_minor_leptokurtic",
    "mcar_moderate_shape",
    "bimodal_many_correlated",
]
TermName = Literal[
    "rows_per_predictor",
    "predictors",
    "row_count",
    "bimodal_flag",
    "correlated_features",
]


class SignalScore(WireRecord):
    """The Signal Score and the terms it was built from."""

    score: float
    """The Signal Score, from 0 to 1. ``tier`` says which Signal Tier it fell in."""
    base: float
    """Explainable Variance: the larger of the best R² and the squared largest absolute correlation."""
    latent: float
    """Latent Structure, from the largest mutual information."""
    breadth: float
    """Signal Breadth: ``predictors / (predictors + 1)``."""
    predictors: int
    """Columns whose absolute correlation with this one is above ``thresholds.numeric.mice_correlation_threshold`` of ``route``."""
    degraded: bool
    """True when the R² pair was missing, so ``base`` is the squared largest absolute correlation alone."""


class FailedTerm(WireRecord):
    """One Feasibility term or condition a strategy failed."""

    term: TermName
    """The term: ``rows_per_predictor`` and ``predictors`` (``mice``), ``row_count`` (``knn``), ``bimodal_flag`` and ``correlated_features`` (a declared ``cluster_conditional`` or ``gmm_sampling``)."""
    value: bool | int | float | None
    """The column's value for the term; null when undefined."""
    limit: bool | int | float | None
    """The bound ``value`` failed; null when the term has none."""
    dial: DialPointer | None
    """The dial that set ``limit``, and whether you set it; null when no dial changes this outcome."""


class FeasibilityCheck(WireRecord):
    """One strategy's Feasibility Floor and Resource Ceiling verdict."""

    strategy: Literal["mice", "knn", "cluster_conditional", "gmm_sampling"]
    """The strategy checked."""
    passed: bool
    """True when it passed every term."""
    failed_terms: Table[FailedTerm]
    """Each term it failed, with the dial behind it. Empty when ``passed`` is true."""


class ColumnEvidence(WireRecord):
    """One column's strategy, its Routing Reason, and the Routing Evidence behind it."""

    column: str
    """Column name. A ``<column>_missing`` row is the indicator an ``mnar`` column adds."""
    strategy: StrategyName
    """The strategy the column takes."""
    reason: ReasonName
    """The Routing Reason: the one rule that fixed ``strategy``."""
    entry: list[EntryCauseName]
    """Why the column reached the Signal Score check, outermost rule first. Empty when it never did."""
    signal_score: SignalScore | None
    """The Signal Score; null when the column never reached the check or has no predictor correlation."""
    tier: Literal["bottom", "middle", "top"] | None
    """The Signal Tier, set by ``thresholds.numeric.signal_score_middle_tier_min`` and ``signal_score_top_tier_min`` of ``route``; null when not reached."""
    feasibility: Table[FeasibilityCheck]
    """Each model strategy checked: ``mice`` then ``knn`` for a routed column, or the one strategy you declared. Empty when none was checked."""


class ExplainRoutingResult(ColumnKeyedResult):
    """One page of Routing Evidence. ``result.get(name)`` finds one column."""

    routing: str
    """The routing Handle read."""
    profile: str
    """The profile Handle the routing was made from."""
    total: int
    """How many columns match ``columns``."""
    offset: int
    """How many matching columns this page skipped."""
    columns: Table[ColumnEvidence]
    """Matching columns in ``route``'s order: those that fill or drop first, then ``passthrough``. At most 15; when more remain, ``omitted`` counts them and names ``explain_routing``."""


@operation(effect="read", phase="impute", scales_with_data=False)
def explain_routing(
    routing: str,
    columns: list[str] | None = None,
    offset: Annotated[int, Field(ge=0)] | None = None,
    limit: Annotated[int, Field(ge=1, le=EXPLAIN_CAP)] | None = None,
) -> ExplainRoutingResult:
    """Read the stored Routing Evidence behind each column's strategy.

    Use when
    --------
    Call this after ``route`` to see how close a column came to another
    strategy. Its Routing Evidence is the facts behind its Routing Reason:
    why it reached the Signal Score check, the Signal Score (how well other
    columns predict it, from 0 to 1) and its parts, the Signal Tier the score
    fell in (which picks a model or a plain fill), and each Feasibility check
    (whether the data can train ``mice`` or ``knn``) with its failed terms
    and the dial that would change them. For each column's strategy and
    reason alone, ``route``'s own result already has them; do not call this.
    It reads only what the routing Handle stores, so it works on a routing
    made in another process or by another Agent, and never recomputes. A
    column you declared in ``strategies`` has empty evidence, except one
    ``feasibility`` check when its strategy is ``knn``, ``mice``,
    ``cluster_conditional`` or ``gmm_sampling``. Filter with ``columns``,
    then page with ``offset`` and ``limit``. The same arguments always
    return the same page.

    Parameters
    ----------
    routing : str
        A routing Handle, such as ``routing_7f3a9c2e``.
    columns : list of str or None
        Only these columns. Null means every routed column.
    offset : int or None
        How many matching columns to skip, for the next page. Null means 0.
    limit : int or None
        Most columns to return, from 1 to 15. Null means 15.

    Returns
    -------
    ExplainRoutingResult
        ``routing`` and ``profile`` Handles, ``total`` matching columns and
        ``columns``: this page, each with its ``strategy``, ``reason``,
        ``entry``, ``signal_score``, ``tier`` and ``feasibility``. When more
        remain, ``omitted`` counts them; call again with ``offset`` raised by
        the page size.

    Errors
    ------
    unknown_handle
        ``routing`` names no routing Handle in this Workspace. ``candidates``
        lists routing Handles, newest first.
    wrong_handle_kind
        ``routing`` is a Handle of another kind.
    unknown_column
        A name in ``columns`` is not a routed column; excluded columns are
        not routed. ``candidates`` lists close matches; ``issues`` lists every
        unknown name.
    invalid_argument
        An argument has the wrong type or is out of its range.
    incompatible_artifact
        Another library format version wrote ``routing``.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    route : Make the routing Handle and each column's Routing Reason.
    list_handles : Find routing Handles made earlier.

    Example
    -------
    $ dataforge load --path train.csv
    {...,"frame":"frame_7f3a9c2e",...}
    $ dataforge profile --frame frame_7f3a9c2e
    {...,"profile":"profile_1b2c3d4e",...}
    $ dataforge route --profile profile_1b2c3d4e
    {...,"routing":"routing_5a6b7c8d",...}
    $ dataforge explain-routing --routing routing_5a6b7c8d --columns income
    {"warnings":...,"omitted":...,"routing":"routing_5a6b7c8d","profile":"profile_1b2c3d4e","total":1,"offset":0,...}
    """
    return ExplainRoutingResult.model_validate(
        explain_routing_columns(
            routing=routing, columns=columns, offset=offset, limit=limit
        )
    )
