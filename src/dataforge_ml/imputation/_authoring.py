"""The manual authoring door — a person writes the routing ``route()`` would derive.

:func:`author` builds a real :class:`~dataforge_ml.imputation.ImputationRouting`
from a column→strategy map, and the routing it returns is indistinguishable to
everything downstream from one :func:`~dataforge_ml.imputation.route` produced
(ADR-0090): one recipe layer, one execution layer, two authors.

The door is route-time, so it never touches data or config — ``profile``
supplies the column universe and each column's semantic type (names and types
only, never statistics), and ``base`` supplies a routing to re-author. It
checks exactly what it can see from those (an unknown map key, ``MICE``
declared for one column, a non-numeric column declared anything other than
``Passthrough``/``Dropped``); a strategy/dtype mismatch or a size guard needs
data and stays a fit-time ``UnitNotTrainableError``. A missing bimodal
estimate is not checked here either — :func:`~dataforge_ml.imputation.resolve_recipe`
gives every bimodal column a :class:`~dataforge_ml.imputation.ColumnEstimates`
entry, and a still-missing centre is refused by
:func:`~dataforge_ml.imputation.fit_unit`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import SemanticType
from ._config import (
    ColumnRouting,
    ImputationRouting,
    ImputationStrategy,
    ModelChoice,
)

if TYPE_CHECKING:
    from ..profiling._config import StructuralProfileResult

__all__ = ["AuthoredColumn", "author"]


@dataclass(frozen=True)
class AuthoredColumn:
    """One column's hand-written imputation declaration.

    The richer half of :func:`author`'s input: a bare
    :class:`~dataforge_ml.imputation.ImputationStrategy` says *what happens* to a
    column, and this says it alongside the two config declarations that
    complete a strategy. The profile-derived estimates (the bimodal centres,
    ``feature_cols``, ``domain_snap_bounds``) are not here — they are set on
    the recipe with
    :meth:`~dataforge_ml.imputation.ImputationRecipe.with_estimates`
    (ADR-0090).

    Parameters
    ----------
    strategy : ImputationStrategy
        The strategy this column takes. Every strategy that answers "what
        happens to this column?" is allowed; ``Indicator`` is refused, being the
        label of a column that arrives as a consequence of ``MNAR`` rather than
        as a choice.
    constant_fill : float, optional
        The fill value a ``Constant`` column takes. Required by that strategy.
    grouping_variable : str, optional
        Column whose groups the ``ClusterConditional`` group-wise branch
        aggregates within. When absent, the recipe's centroid branch estimates
        apply instead.
    excluded : bool, default False
        Declares the column Imputation-soft-excluded, as
        ``PipelineConfig.add_phase_exclusion`` does for
        :func:`~dataforge_ml.imputation.route`: its missing values ride
        through untouched and it is never counted as an MICE or KNN
        predictor. Only valid with ``Passthrough``.
    """

    strategy: ImputationStrategy
    constant_fill: float | None = None
    grouping_variable: str | None = None
    excluded: bool = False


def _coerce(column: str, value) -> AuthoredColumn:
    """Normalise one map entry to an ``AuthoredColumn``.

    Accepts an ``AuthoredColumn``, an ``ImputationStrategy``, or the strategy's
    string value (``ImputationStrategy`` is a ``StrEnum``, so ``"median"`` is a
    spelling of one).
    """
    if isinstance(value, AuthoredColumn):
        return value
    if isinstance(value, (ImputationStrategy, str)):
        try:
            return AuthoredColumn(strategy=ImputationStrategy(value))
        except ValueError:
            raise ValueError(
                f"Column '{column}': '{value}' is not an ImputationStrategy."
            ) from None
    raise TypeError(
        f"Column '{column}': expected an ImputationStrategy or an AuthoredColumn, "
        f"got {type(value).__name__}."
    )


def _validate(column: str, authored: AuthoredColumn) -> None:
    """Refuse everything the door can see is wrong from the names alone.

    Data-dependent failures — a strategy the column's dtype cannot take, a
    missing bimodal estimate, a block too small to train — are not visible
    here and stay fit-time raises (``UnitNotTrainableError``).
    """
    strategy = authored.strategy
    if strategy == ImputationStrategy.Indicator:
        raise ValueError(
            f"Column '{column}': 'Indicator' cannot be authored. It labels the "
            f"'{column}_missing' column an MNAR declaration appends, which the "
            f"door registers on its own."
        )
    if strategy == ImputationStrategy.Constant and authored.constant_fill is None:
        raise ValueError(
            f"Column '{column}': strategy is 'Constant' but no fill value was "
            f"provided. Set AuthoredColumn.constant_fill."
        )
    if authored.excluded and strategy != ImputationStrategy.Passthrough:
        raise ValueError(
            f"Column '{column}': an excluded column rides through untouched, so "
            f"its strategy must be 'Passthrough', not '{strategy}'."
        )


def _to_routing(
    column: str,
    authored: AuthoredColumn,
    semantic_type: SemanticType,
) -> ColumnRouting:
    """Project one validated ``AuthoredColumn`` onto its routing entry.

    ``indicator_flag``, ``mnar`` and ``drop`` are derived from the strategy
    rather than accepted; ``excluded`` is carried from the declaration.
    """
    strategy = authored.strategy
    return ColumnRouting(
        column=column,
        semantic_type=semantic_type,
        strategy=strategy,
        constant_fill=authored.constant_fill,
        grouping_variable=authored.grouping_variable,
        indicator_flag=strategy == ImputationStrategy.MNAR,
        mnar=strategy == ImputationStrategy.MNAR,
        drop=strategy == ImputationStrategy.Dropped,
        excluded=authored.excluded,
    )


def author(
    columns_map: Mapping[str, ImputationStrategy | AuthoredColumn],
    *,
    profile: StructuralProfileResult | None = None,
    base: ImputationRouting | None = None,
    default: ImputationStrategy = ImputationStrategy.Passthrough,
) -> ImputationRouting:
    """Build an imputation routing by hand, from column names and declared strategies.

    The manual end of the automated↔manual spectrum (ADR-0090), sibling to
    :func:`~dataforge_ml.imputation.route`: both take a description and return
    an :class:`~dataforge_ml.imputation.ImputationRouting`, one from a profile and
    one from a person. The returned routing reaches a recipe the one way any
    routing does — :func:`~dataforge_ml.imputation.resolve_recipe` — so a
    hand-written routing gets the same profile-computed dials and estimates as
    a routed one.

    No statistics are touched: ``profile`` supplies only the column universe
    and each column's semantic type. What the door derives rather than
    accepts: ``indicator_flag`` and ``mnar`` (from ``strategy == MNAR``),
    ``drop`` (from ``strategy == Dropped``), and an ``{col}_missing``
    ``Indicator`` entry per MNAR column. ``excluded`` is taken from the
    :class:`AuthoredColumn` declaration, since there is no config to read it
    from. ``signals`` is empty on an authored column — there was no routing
    to explain.

    **The door resolves no MICE model choice for fresh authoring.** ``MICE``
    may still be declared for two or more columns — the door refuses only a
    single-column block — but every fresh routing :func:`author` returns carries
    ``mice_model_choice=None`` and ``mice_estimator=None`` unless
    ``with_model_choice`` sets one explicitly. When re-authoring with
    ``base=``, MICE membership carries/creates/dissolves the estimator
    (ADR-0090): if MICE columns remain, ``mice_model_choice`` and the estimator
    carry; if the edit creates the block, it is stamped ``BayesianRidge``; if
    the edit dissolves it, both go. The Feasibility Floor, Signal Score, and
    Estimator Ladder (ADR-0091, ADR-0092, ADR-0094) are computed only in
    :func:`~dataforge_ml.imputation.route`: forcing a strategy by hand is
    informed consent, the same rule ADR-0090 set for a missing profile
    estimate.

    **Re-authoring.** ``base=`` hands the door a routing instead of a column
    universe, so a user who ran :func:`~dataforge_ml.imputation.route` and
    disagrees with two of forty columns changes those two and keeps the
    thirty-eight the router got right. A column named in ``columns_map`` is
    rebuilt from the map and takes only its semantic type from the base; every
    other column of the base's universe keeps its
    :class:`~dataforge_ml.imputation.ColumnRouting` verbatim, signals included.
    If MICE columns remain, ``mice_model_choice`` and the estimator carry; if
    the edit creates the block, it is stamped ``BayesianRidge``; if the edit
    dissolves it, both go.

    Parameters
    ----------
    columns_map : Mapping[str, ImputationStrategy or AuthoredColumn]
        The columns the author has an opinion about, keyed by column name. A
        bare strategy declares what happens; an :class:`AuthoredColumn` declares
        that plus the two config declarations its strategy reads. Every key
        must appear in the column universe (``profile`` or ``base``).
    profile : StructuralProfileResult, optional
        Supplies the column universe and each column's semantic type. Exactly
        one of ``profile`` and ``base`` must be given.
    base : ImputationRouting, optional
        A routing to re-author. Its columns are the universe. Exactly one of
        ``profile`` and ``base`` must be given.
    default : ImputationStrategy, default ``ImputationStrategy.Passthrough``
        Strategy stamped on every column of the universe not named in
        ``columns_map``, when ``profile`` is given. Ignored under ``base``,
        where an unnamed column keeps the base's routing instead.

    Returns
    -------
    ImputationRouting
        The immutable routing: one routing entry per column of the universe,
        plus the derived MNAR indicator entries.

    Raises
    ------
    ValueError
        If neither or both of ``profile``/``base`` are given; if a
        ``columns_map`` key is absent from the column universe; if a
        non-numeric column is declared any strategy other than ``Passthrough``
        or ``Dropped``; if ``Indicator`` is declared; if ``Constant`` carries no
        fill; if ``excluded`` is declared with a strategy other than
        ``Passthrough``; or if ``MICE`` is declared for exactly one column,
        which leaves the block a target and no predictor.
    TypeError
        If a ``columns_map`` value is neither an ``ImputationStrategy`` nor an
        ``AuthoredColumn``.
    """
    if (profile is None) == (base is None):
        raise ValueError(
            "author() needs exactly one of 'profile' (a column universe to "
            "route fresh) or 'base' (a routing to re-author)."
        )

    if base is not None:
        # The carried universe deliberately excludes the base's derived
        # '{col}_missing' entries: they are a consequence of an MNAR
        # declaration, so the indicator pass below re-derives them from
        # whatever survives the edit.>
        carried: dict[str, ColumnRouting] = {
            col: routing
            for col, routing in base.column_routings.items()
            if routing.strategy != ImputationStrategy.Indicator
        }
        ordered = list(carried)
        universe_semantic = {col: r.semantic_type for col, r in carried.items()}
    else:
        carried = {}
        ordered = list(profile.columns)
        universe_semantic = {
            col: cp.semantic_type for col, cp in profile.columns.items()
        }

    known = set(ordered)
    unknown = [col for col in columns_map if col not in known]
    if unknown:
        names = ", ".join(f"'{c}'" for c in sorted(unknown))
        source = "the base routing" if base is not None else "'profile'"
        raise ValueError(
            f"Column(s) {names} are named in the strategy map but absent from "
            f"{source}. Every planned column must be part of the column "
            f"universe."
        )

    routings: dict[str, ColumnRouting] = {}
    for col in ordered:
        if col not in columns_map and col in carried:
            routings[col] = carried[col]
            continue
        entry = _coerce(col, columns_map.get(col, default))
        semantic_type = universe_semantic.get(col, SemanticType.Numeric)
        if semantic_type != SemanticType.Numeric and entry.strategy not in (
            ImputationStrategy.Passthrough,
            ImputationStrategy.Dropped,
        ):
            raise ValueError(
                f"Column '{col}': semantic type is '{semantic_type}', so only "
                f"'Passthrough' or 'Dropped' may be declared for it, not "
                f"'{entry.strategy}'."
            )
        _validate(col, entry)
        routings[col] = _to_routing(col, entry, semantic_type)

    mice_cols = [
        c for c, r in routings.items() if r.strategy == ImputationStrategy.MICE
    ]
    if len(mice_cols) == 1:
        raise ValueError(
            f"Column '{mice_cols[0]}': MICE is a joint block over several "
            f"columns and was declared for this one alone, leaving it a target "
            f"with no predictor. Declare MICE for at least two columns."
        )

    # Indicator pass: pre-register the {col}_missing columns the MNAR
    # mechanism appends at transform time, exactly as route() does.
    for col, routing in list(routings.items()):
        if not routing.indicator_flag:
            continue
        indicator_col = f"{col}_missing"
        routings[indicator_col] = ColumnRouting(
            column=indicator_col,
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
        )

    mice_model_choice: ModelChoice | None = None
    mice_estimator: Any = None
    if base is not None:
        base_has_mice = any(
            r.strategy == ImputationStrategy.MICE
            for r in base.column_routings.values()
        )
        if mice_cols:
            if base_has_mice:
                # MICE columns remain: carry over mice_model_choice and mice_estimator
                mice_model_choice = base.mice_model_choice
                mice_estimator = base.mice_estimator
            else:
                # The edit creates the block: stamp BayesianRidge
                mice_model_choice = ModelChoice.BayesianRidge
                mice_estimator = None
        else:
            # The edit dissolves it: both go
            mice_model_choice = None
            mice_estimator = None

    return ImputationRouting(
        column_routings=routings,
        mice_model_choice=mice_model_choice,
        mice_estimator=mice_estimator,
    )
