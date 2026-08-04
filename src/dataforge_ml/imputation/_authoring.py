"""The manual authoring door — a person writes the plan ``decide()`` would derive.

:func:`author` builds a real :class:`~dataforge_ml.imputation.ImputationDecision`
from a column→strategy map, and the plan it returns is indistinguishable to
everything downstream from one :func:`~dataforge_ml.imputation.decide` produced
(ADR-0083): one execution layer, one artifact, one evaluation surface, two
authors.

The door is decide-time, so it never touches data — ``columns`` is an ordered
sequence of *names*, which lets a plan be built in CI from a schema before any
frame exists. It checks exactly what it can see from those names (an unknown map
key, MICE declared for one column, a bimodal strategy with no centres); a
strategy/dtype mismatch or a size guard needs data and stays a fit-time
``UnitNotTrainableError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..config import SemanticType
from ._config import (
    ColumnImputationDecision,
    ImputationDecision,
    ImputationStrategy,
    ImputationUnit,
    ModelChoice,
    _derive_units,
    _dial_defaults,
)

__all__ = ["AuthoredColumn", "author"]


@dataclass(frozen=True)
class AuthoredColumn:
    """One column's hand-written imputation declaration.

    The richer half of :func:`author`'s input: a bare
    :class:`~dataforge_ml.imputation.ImputationStrategy` says *what happens* to a
    column, and this says it alongside the facts about the data that the
    strategy's fitter reads — the bimodal centres, the centroid features, the
    grouping variable, the constant fill, the domain-snap bounds (ADR-0083).

    Deliberately not a :class:`~dataforge_ml.ColumnImputationDecision`: there is
    no ``column`` to restate (the map key is the column), no ``semantic_type``
    (the door stamps ``Numeric``), no ``drop``, ``mnar`` or ``indicator_flag``
    (all derived from ``strategy``) and no ``signals`` (a hand-authored decision
    explains no routing). Dials are not here either — they keep their one
    grammar, :meth:`~dataforge_ml.ImputationDecision.with_hyperparameters`,
    whichever author made the plan.

    Parameters
    ----------
    strategy : ImputationStrategy
        The strategy this column takes. Every strategy that answers "what
        happens to this column?" is allowed; ``Indicator`` is refused, being the
        label of a column that arrives as a consequence of ``MNAR`` rather than
        as a choice.
    center1 : float, optional
        First of the two mode centres the ``GMMSampling`` and
        ``ClusterConditional`` strategies split on. Required by both.
    center2 : float, optional
        Second mode centre. See ``center1``.
    feature_cols : tuple[str, ...], optional
        Columns the ``ClusterConditional`` centroid branch measures its
        per-cluster centroids over. Either this or ``grouping_variable`` must be
        supplied for that strategy.
    grouping_variable : str, optional
        Column whose groups the ``ClusterConditional`` group-wise branch
        aggregates within. Selects that branch over the centroid one.
    constant_fill : float, optional
        The fill value a ``Constant`` column takes. Required by that strategy.
    domain_snap_bounds : tuple[float, float], optional
        ``(min, max)`` a model-based strategy's predictions are rounded and
        clipped into, for a column with a bounded discrete domain.
    """

    strategy: ImputationStrategy
    center1: float | None = None
    center2: float | None = None
    feature_cols: tuple[str, ...] | None = None
    grouping_variable: str | None = None
    constant_fill: float | None = None
    domain_snap_bounds: tuple[float, float] | None = None


_BIMODAL_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {ImputationStrategy.GMMSampling, ImputationStrategy.ClusterConditional}
)


def _coerce(column: str, value: Any) -> AuthoredColumn:
    """Normalise one map entry to an ``AuthoredColumn``.

    Accepts an ``AuthoredColumn``, an ``ImputationStrategy``, or the strategy's
    string value (``ImputationStrategy`` is a ``StrEnum``, so ``"median"`` is a
    spelling of one). Anything carrying a ``semantic_type`` other than
    ``Numeric`` is refused here rather than silently restamped: the field
    chooses the MICE block's predictor set (ADR-0079), so it is not decoration.
    """
    semantic_type = getattr(value, "semantic_type", None)
    if semantic_type is not None and semantic_type != SemanticType.Numeric:
        raise ValueError(
            f"Column '{column}': the authoring door plans numeric columns only, "
            f"but this entry declares semantic type '{semantic_type}'."
        )
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

    Data-dependent failures — a strategy the column's dtype cannot take, a block
    too small to train — are not visible here and stay fit-time raises
    (``UnitNotTrainableError``, ADR-0083).
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
    if strategy in _BIMODAL_STRATEGIES and (
        authored.center1 is None or authored.center2 is None
    ):
        raise ValueError(
            f"Column '{column}': strategy is '{strategy}' but the two mode "
            f"centres are not both declared. Set AuthoredColumn.center1 and "
            f"center2 — the fitter splits the column on them."
        )
    if strategy == ImputationStrategy.ClusterConditional and not (
        authored.grouping_variable or authored.feature_cols
    ):
        raise ValueError(
            f"Column '{column}': strategy is 'ClusterConditional' but neither a "
            f"grouping_variable nor a non-empty feature_cols was declared; the "
            f"unit would fit cleanly and fill nothing. Declare one of them."
        )


_ESTIMATOR_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {ImputationStrategy.MICE}
)


def _resolve_estimators(
    estimators: Mapping[str, Any],
    units: tuple[ImputationUnit, ...],
) -> dict[str, Any]:
    """Check the ``estimators`` map against the plan's units and copy it out.

    Unit-keyed, so the checks are unit-shaped: the id must name a unit of this
    plan, and that unit's strategy must have an estimator slot — MICE is the
    only one, ``build_from_choice`` being called once in the whole fitter file
    (ADR-0083). The mapping is copied; the instances inside it are not.
    """
    known = {u.unit_id: u for u in units}
    resolved: dict[str, Any] = {}
    for unit_id, estimator in estimators.items():
        unit = known.get(unit_id)
        if unit is None:
            names = ", ".join(f"'{u}'" for u in known) or "(none)"
            raise ValueError(
                f"Unit '{unit_id}' is named in 'estimators' but this plan has no "
                f"such unit. Known units: {names}."
            )
        if unit.strategy not in _ESTIMATOR_STRATEGIES:
            raise ValueError(
                f"Unit '{unit_id}' has strategy '{unit.strategy}', which trains "
                f"no estimator. Only MICE takes one."
            )
        if estimator is None:
            raise ValueError(
                f"Unit '{unit_id}' is named in 'estimators' with no estimator. "
                f"Drop the key to let the library choose the family."
            )
        resolved[unit_id] = estimator
    return resolved


def _to_decision(
    column: str,
    authored: AuthoredColumn,
    semantic_type: SemanticType,
) -> ColumnImputationDecision:
    """Project one validated ``AuthoredColumn`` onto its plan entry.

    ``indicator_flag``, ``mnar`` and ``drop`` are derived from the strategy
    rather than accepted, and ``model_choice`` takes ``decide``'s own neutral
    branch for MICE so the plainest authorable plan trains (ADR-0083).
    """
    strategy = authored.strategy
    return ColumnImputationDecision(
        column=column,
        semantic_type=semantic_type,
        strategy=strategy,
        model_choice=(
            ModelChoice.BayesianRidge if strategy == ImputationStrategy.MICE else None
        ),
        domain_snap_bounds=authored.domain_snap_bounds,
        center1=authored.center1,
        center2=authored.center2,
        feature_cols=(
            tuple(authored.feature_cols) if authored.feature_cols is not None else None
        ),
        grouping_variable=authored.grouping_variable,
        constant_fill=authored.constant_fill,
        indicator_flag=strategy == ImputationStrategy.MNAR,
        mnar=strategy == ImputationStrategy.MNAR,
        drop=strategy == ImputationStrategy.Dropped,
    )


def author(
    columns_map: Mapping[str, ImputationStrategy | AuthoredColumn],
    *,
    columns: Sequence[str] | None = None,
    default: ImputationStrategy | None = None,
    estimators: Mapping[str, Any] | None = None,
    numeric_sentinels: Mapping[str, Sequence[float]] | None = None,
    string_sentinels: Mapping[str, Sequence[str]] | None = None,
    base: ImputationDecision | None = None,
) -> ImputationDecision:
    """Build an imputation plan by hand, from column names and declared strategies.

    The manual end of the automated↔manual spectrum (ADR-0083), sibling to
    :func:`~dataforge_ml.imputation.decide`: both take a description and return
    an :class:`~dataforge_ml.ImputationDecision`, one from a profile and one from
    a person. The returned plan carries a complete decided hyperparameter base
    written from the same dial table ``decide`` reads, so
    :meth:`~dataforge_ml.ImputationDecision.with_hyperparameters`,
    :func:`~dataforge_ml.fit_unit`,
    :meth:`~dataforge_ml.FittedImputer.compose`, ``transform``,
    :func:`~dataforge_ml.core_budget` and persistence all behave exactly as they
    do for a decided plan.

    No data is touched: ``columns`` is a sequence of *names*, so a plan can be
    authored in CI against a schema before any frame exists. What the door
    derives rather than accepts: ``semantic_type`` (``Numeric``, or the base's
    own value when re-authoring),
    ``indicator_flag`` and ``mnar`` (from ``strategy == MNAR``), ``drop`` (from
    ``strategy == Dropped``), and an ``{col}_missing`` ``Indicator`` entry per
    MNAR column. ``signals`` is empty on an authored column — there was no
    routing to explain — and ``config_snapshot`` is empty, which truthfully
    reads back as library defaults. Both come from the base instead when one is
    given.

    A MICE column is stamped with :attr:`~dataforge_ml.ModelChoice.BayesianRidge`
    — ``decide``'s own neutral branch — so the plainest plan an author can write
    actually trains; override it with
    :meth:`~dataforge_ml.ImputationDecision.with_model_choice`, or hand the block
    your own tuned estimator through ``estimators``.

    **Estimators enter here; dials do not.** An estimator has no other way onto a
    plan — ``with_model_choice`` takes a *label*, and there is no
    ``with_estimator`` — whereas dials keep their one grammar,
    :meth:`~dataforge_ml.ImputationDecision.with_hyperparameters`, identical
    whichever author made the plan (ADR-0083).

    **Re-authoring.** ``base=`` hands the door a plan instead of a column
    universe, so a user who ran ``decide`` and disagrees with two of forty
    columns changes those two and keeps the thirty-eight the router got right.
    This amends ADR-0082 rather than reopening it: a structural edit still
    invalidates what is derived downstream of the unit list, but the door
    *re-derives* that state — ``with_strategy`` stays deleted because it did
    not. See ``base`` for the field-by-field carry-over rules.

    **What the door cannot check.** It sees names, never data, so a
    strategy/dtype mismatch and every size guard stay fit-time failures. That
    leaves one accepted asymmetry: a hand-authored unit too small or too typed
    to train raises a hard
    :class:`~dataforge_ml.imputation.UnitNotTrainableError`, where the same unit
    *forced* onto the automatic path is merely reported through
    :class:`~dataforge_ml.imputation.FitSignals` as a warning. Parity would mean
    handing a decide-time door a frame for the sake of one diagnostic. Which
    library guarantees narrow for a hand-authored plan and which do not is
    tabulated under "The guarantee boundary" in the imputation API guide;
    briefly, everything below the plan is unchanged, while routing rationale,
    pre-fit size warnings, and — for a user-supplied estimator — the ``n_jobs``
    determinism pinning, the ``core_budget`` count and dial tuning do not apply.

    Parameters
    ----------
    columns_map : Mapping[str, ImputationStrategy or AuthoredColumn]
        The columns the author has an opinion about, keyed by column name. A
        bare strategy declares what happens; an :class:`AuthoredColumn` declares
        that plus the facts its fitter reads. Every key must appear in
        ``columns``.
    columns : Sequence[str], optional
        Every column of the frames this plan will be fitted and applied to, in
        order. Required unless ``base`` is given, and keyword-only:
        ``transform`` rejects a frame column the plan never described, so a
        partial map without the full universe would build an imputer that
        refuses its own author's frame.
    default : ImputationStrategy, optional, default ``ImputationStrategy.Passthrough``
        Strategy stamped on every column not named in ``columns_map``. Note that
        it flips which case is silent: under ``default=Median`` a forgotten
        column is quietly imputed rather than left alone. Not accepted with
        ``base``, where an unnamed column keeps the base's decision instead.
    estimators : Mapping[str, Any], optional
        Sklearn-compatible estimator instances keyed by **unit id** — in
        practice ``{"mice": my_model}``, MICE being the only strategy with an
        estimator slot. Every column of the named unit is stamped
        :attr:`~dataforge_ml.ModelChoice.Custom` and the instance is carried on
        the plan's ``custom_estimators`` map, unchanged and uncloned. Keyed by
        unit rather than by column because a per-column spelling would let two
        estimators be named for one joint block. A pre-fitted estimator is
        accepted; its state is inert, since ``IterativeImputer`` clones and
        refits from scratch. Two consequences worth knowing: the object is
        **shared, not copied**, so mutating it after authoring silently moves
        every plan holding it; and it is **never serialized**, so a plan saved
        and reloaded keeps the ``Custom`` label with an empty slot and raises at
        fit time until the estimator is supplied again. With ``base``, the
        base's estimators are carried and this argument overwrites the entry it
        names.
    numeric_sentinels : Mapping[str, Sequence[float]], optional
        Declared numeric sentinel values per column (e.g. ``-999``), normalised
        to nulls before any fit or transform (ADR-0068). Plan-level rather than
        per-column, so a sentinel-bearing column the author has no opinion about
        is still normalised. With ``base``, the base's map is carried and an
        entry given here overwrites that column's.
    string_sentinels : Mapping[str, Sequence[str]], optional
        Declared string sentinel values per column. See ``numeric_sentinels``.
    base : ImputationDecision, optional
        A plan to re-author. Its columns are the universe, so ``columns`` and
        ``default`` are refused alongside it — two sources for one fact, and
        nothing left to default. Carry-over runs field by field: a column absent
        from ``columns_map`` keeps its decision **verbatim** (semantic type,
        model choice, routing signals and all), while a named one is rebuilt
        from the map and takes only its semantic type from the base;
        ``config_snapshot`` and both sentinel maps are carried verbatim, since a
        re-authoring user may no longer hold the profile or the config;
        ``custom_estimators`` is carried; the decided hyperparameter base is
        **gap-filled, never overwritten**, because ``decide``'s dials are
        profile-*computed* and blanket recalculation would be a silent quality
        regression rather than a loud crash; and tuning deltas are carried for
        surviving units. Everything belonging to a unit the edit dissolved — a
        delta, an estimator, a decided base — is **removed**, not left inert: a
        dead entry survives ``to_dict`` and a save/load and would spring back to
        life when a column re-enters that unit later. Knowingly kept: two
        provenances of settings in one plan, and a stale ``n_neighbors`` when a
        block's membership changes, since a structural edit hands dial
        responsibility to the author. Nothing records the mix — mixed provenance
        gets no field. Re-authoring a *deserialized* plan is allowed and
        unguarded, but the units saved beside it were fitted under the old
        strategies: the result is a plan to **fit fresh**, never one to pair
        with those units.

    Returns
    -------
    ImputationDecision
        The immutable plan: one decision per column of the universe (``columns``
        or, when re-authoring, the base's own), the derived MNAR indicator
        entries, the materialised units, the complete decided hyperparameter
        base, any supplied estimators, and the sentinel maps.

    Raises
    ------
    ValueError
        If neither ``columns`` nor ``base`` is given, or if ``base`` is combined
        with ``columns`` or ``default``; if a ``columns_map`` key is absent from
        the column universe; if an entry declares
        a semantic type other than ``Numeric``; if ``Indicator`` is declared; if
        ``Constant`` carries no fill; if a bimodal strategy carries fewer than
        two centres; if ``ClusterConditional`` carries neither a grouping
        variable nor a non-empty ``feature_cols``; if ``MICE`` is declared for
        exactly one column, which leaves the block a target and no predictor; or
        if an ``estimators`` key names no unit of the plan, names a unit whose
        strategy has no estimator slot, or carries ``None``.
    TypeError
        If a ``columns_map`` value is neither an ``ImputationStrategy`` nor an
        ``AuthoredColumn``.
    """
    if base is not None:
        supplied = [
            name
            for name, given in (("columns", columns), ("default", default))
            if given is not None
        ]
        if supplied:
            names = " and ".join(f"'{n}'" for n in supplied)
            raise ValueError(
                f"'base' cannot be combined with {names}. A base plan already "
                f"names every column and already decides the ones the strategy "
                f"map leaves out."
            )
    elif columns is None:
        raise ValueError(
            "author() needs a column universe: pass 'columns' with every column "
            "of the frames this plan will see, or 'base' with a plan to re-author."
        )

    # The carried universe deliberately excludes the base's derived
    # '{col}_missing' entries: they are a consequence of an MNAR declaration, so
    # the indicator pass below re-derives them from whatever survives the edit.
    carried: dict[str, ColumnImputationDecision] = (
        {
            col: decision
            for col, decision in base.column_decisions.items()
            if decision.strategy != ImputationStrategy.Indicator
        }
        if base is not None
        else {}
    )
    ordered = list(carried) if base is not None else list(columns or ())
    known = set(ordered)
    unknown = [col for col in columns_map if col not in known]
    if unknown:
        names = ", ".join(f"'{c}'" for c in sorted(unknown))
        source = "the base plan" if base is not None else "'columns'"
        raise ValueError(
            f"Column(s) {names} are named in the strategy map but absent from "
            f"{source}. Every planned column must be part of the frame's "
            f"column universe."
        )

    stamp = default if default is not None else ImputationStrategy.Passthrough

    decisions: dict[str, ColumnImputationDecision] = {}
    for col in ordered:
        if col not in columns_map and col in carried:
            decisions[col] = carried[col]
            continue
        entry = _coerce(
            col,
            columns_map.get(col, stamp),
        )
        _validate(col, entry)
        # A re-authored column keeps the base's semantic type rather than the
        # door's Numeric stamp: the field chooses MICE's predictor set
        # (ADR-0079), and where a base exists it is the one that measured it.
        prior = carried.get(col)
        decisions[col] = _to_decision(
            col,
            entry,
            prior.semantic_type if prior is not None else SemanticType.Numeric,
        )

    mice_cols = [
        c for c, d in decisions.items() if d.strategy == ImputationStrategy.MICE
    ]
    if len(mice_cols) == 1:
        raise ValueError(
            f"Column '{mice_cols[0]}': MICE is a joint block over several "
            f"columns and was declared for this one alone, leaving it a target "
            f"with no predictor. Declare MICE for at least two columns."
        )

    # Indicator pass: pre-register the {col}_missing columns the MNAR mechanism
    # appends at transform time, exactly as decide() does.
    for col, decision in list(decisions.items()):
        if not decision.indicator_flag:
            continue
        indicator_col = f"{col}_missing"
        decisions[indicator_col] = ColumnImputationDecision(
            column=indicator_col,
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
        )

    units = _derive_units(decisions)

    live = {unit.unit_id for unit in units}

    # The estimators channel: the author's own instance, stamped as Custom on
    # every column of the unit it was named for (ADR-0083). Held by identity —
    # never cloned, here or on any derived copy. A base's estimator survives
    # only as long as its unit does.
    custom_estimators: dict[str, Any] = (
        {
            unit_id: estimator
            for unit_id, estimator in base.custom_estimators.items()
            if unit_id in live
        }
        if base is not None
        else {}
    )
    custom_estimators.update(_resolve_estimators(estimators or {}, units))
    for unit_id in custom_estimators:
        unit = next(u for u in units if u.unit_id == unit_id)
        for col in unit.columns:
            decisions[col] = replace(decisions[col], model_choice=ModelChoice.Custom)

    # The decided base is written eagerly from the one dial table decide() reads,
    # so with_hyperparameters works on a hand-authored plan exactly as on a
    # decided one (ADR-0083). A strategy with no dial row gets no entry. Under
    # a base the table only *gap-fills*: a router-computed value outranks the
    # neutral one, and a unit the edit dissolved leaves nothing behind.
    decided_hyperparameters: dict[str, tuple[tuple[str, Any], ...]] = {}
    for unit in units:
        dials = _dial_defaults(unit.strategy)
        if base is not None:
            dials.update(dict(base.decided_hyperparameters.get(unit.unit_id, ())))
        if dials:
            decided_hyperparameters[unit.unit_id] = tuple(dials.items())

    # Tuning deltas ride along for surviving units; an orphan is removed rather
    # than left inert, since a dead delta survives a save/load and springs back
    # to life when a column re-enters that unit later.
    override_hyperparameters: dict[str, tuple[tuple[str, Any], ...]] = (
        {
            unit_id: delta
            for unit_id, delta in base.override_hyperparameters.items()
            if unit_id in live
        }
        if base is not None
        else {}
    )

    numeric: dict[str, list[float]] = (
        {col: list(vals) for col, vals in base.numeric_sentinels.items()}
        if base is not None
        else {}
    )
    numeric.update(
        {
            col: [float(v) for v in vals]
            for col, vals in (numeric_sentinels or {}).items()
        }
    )
    strings: dict[str, list[str]] = (
        {col: list(vals) for col, vals in base.string_sentinels.items()}
        if base is not None
        else {}
    )
    strings.update(
        {col: [str(v) for v in vals] for col, vals in (string_sentinels or {}).items()}
    )

    return ImputationDecision(
        column_decisions=decisions,
        config_snapshot=dict(base.config_snapshot) if base is not None else {},
        decided_hyperparameters=decided_hyperparameters,
        override_hyperparameters=override_hyperparameters,
        custom_estimators=custom_estimators,
        numeric_sentinels=numeric,
        string_sentinels=strings,
    )
