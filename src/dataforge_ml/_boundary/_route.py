"""Internal code behind the ``route`` Operation (ADR-0105, ADR-0106, ADR-0110).

The Operation file holds the records; this module checks every argument before
any work, routes the profile once, lays the Agent's declarations over the
routed columns, writes the routing Handle and builds the result.

``author`` folds into ``route`` here (ADR-0110): a column listed in
``strategies``, or stamped by ``default_strategy``, takes that strategy with
the Routing Reason ``authored``. It skips every routing rule, the drop rule
included, so the Agent's declaration always wins. A declared strategy that has
a Feasibility Floor still gets its one ``feasibility`` fact, and a failed one
is reported once as ``forced_past_feasibility`` (ADR-0105).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._serialization import serialize
from ..records import DataForgeError, ResultWarning
from ._error_codes import close_matches, entry_for
from ._profile import _Issue
from ._workspace import InheritedDeclarations, read_profile, write_handle

#: Most column rows the result holds; ``explain_routing`` pages through the rest.
COLUMN_CAP = 100
#: Most columns one Result Warning names; the rest are counted in ``omitted``.
WARNING_COLUMNS_CAP = 20
#: The Operation dial paths are relative to.
OPERATION = "route"
#: The only dial group, and the ``NumericImputationConfig`` fields it holds.
DIAL_GROUP = "numeric"
ROUTING_DIALS: tuple[str, ...] = (
    "knn_max_rows",
    "mice_min_rows_per_predictor",
    "mice_max_nearest_features",
    "mice_correlation_threshold",
    "signal_score_breadth_weight",
    "signal_score_latent_weight",
    "signal_score_middle_tier_min",
    "signal_score_top_tier_min",
    "bimodal_min_correlated_features",
    "bimodal_correlation_threshold",
)
#: Declared strategies that have a Feasibility Floor or conditions to check.
_FLOORED = ("knn", "mice", "cluster_conditional", "gmm_sampling")
#: What a column that is not numeric, or that profiling skipped, may declare.
_NON_NUMERIC_ALLOWED = ("passthrough", "dropped")


def dial_defaults() -> dict[str, Any]:
    """Return every routing dial's default, keyed by its path under ``thresholds``.

    Returns
    -------
    dict
        Path such as ``numeric.knn_max_rows`` to the library default.
    """
    from ..imputation._config import NumericImputationConfig

    defaults = NumericImputationConfig()
    return {f"{DIAL_GROUP}.{name}": getattr(defaults, name) for name in ROUTING_DIALS}


@dataclass(frozen=True)
class _Dials:
    """Every routing dial's resolved value, and which ones the caller set."""

    values: dict[str, Any]
    set_by_caller: frozenset[str]

    def numeric(self) -> dict[str, Any]:
        prefix = f"{DIAL_GROUP}."
        return {p[len(prefix) :]: v for p, v in self.values.items()}


def _resolve_dials(thresholds: Any) -> _Dials:
    values = dial_defaults()
    given: set[str] = set()
    group = None if thresholds is None else thresholds.numeric
    if group is not None:
        for name, value in group:
            if value is not None:
                values[f"{DIAL_GROUP}.{name}"] = value
                given.add(f"{DIAL_GROUP}.{name}")
    return _Dials(values=values, set_by_caller=frozenset(given))


@dataclass(frozen=True)
class _Universe:
    """What the profile Handle says about the frame's columns."""

    ordered: list[str]  # profiled and profile-skipped columns, dataset order
    excluded: list[str]  # hard exclusions the profile inherited or declared
    profile_skipped: set[str]
    numeric: set[str]
    targets: list[str]

    @property
    def known(self) -> list[str]:
        return [*self.ordered, *self.excluded]


def _check_arguments(
    universe: _Universe,
    *,
    strategies: Sequence[Any],
    exclude_columns: Sequence[str],
    skip_columns: Sequence[str],
    dials: _Dials,
) -> list[_Issue]:
    """Check every argument against the profile and each other."""
    known = set(universe.known)
    excluded = set(universe.excluded) | set(exclude_columns)
    skipped = set(skip_columns)
    issues: list[_Issue] = []

    def unknown(param: str, column: str) -> None:
        issues.append(
            _Issue(
                "unknown_column",
                param,
                f"no column `{column}` in the profile's frame",
                column=column,
                unknown=column,
            )
        )

    def conflict(param: str, column: str, problem: str) -> None:
        issues.append(_Issue("declaration_conflict", param, problem, column=column))

    seen: set[str] = set()
    for i, entry in enumerate(strategies):
        param, column, strategy = f"strategies[{i}]", entry.column, entry.strategy
        if column not in known:
            unknown(f"{param}.column", column)
            continue
        if column in seen:
            issues.append(
                _Issue(
                    "duplicate_key",
                    f"{param}.column",
                    f"`{column}` is listed twice",
                    column=column,
                )
            )
            continue
        seen.add(column)
        if column in excluded:
            conflict(f"{param}.column", column, f"`{column}` is excluded, so it is not routed")
            continue
        if column in skipped:
            conflict(f"{param}.column", column, f"`{column}` is in `skip_columns`")
            continue
        if column not in universe.numeric and strategy not in _NON_NUMERIC_ALLOWED:
            why = (
                "profiling skipped it"
                if column in universe.profile_skipped
                else "it is not numeric"
            )
            issues.append(
                _Issue(
                    "strategy_not_declarable",
                    f"{param}.strategy",
                    f"`{column}` may take only `passthrough` or `dropped`, because {why}",
                    column=column,
                )
            )
        if strategy == "constant" and entry.constant_fill is None:
            issues.append(
                _Issue(
                    "constant_fill_missing",
                    f"{param}.constant_fill",
                    f"`{column}` is `constant` but has no fill value",
                    column=column,
                )
            )
        if strategy != "constant" and entry.constant_fill is not None:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.constant_fill",
                    f"only a `constant` column takes a fill value, not `{strategy}`",
                    column=column,
                )
            )
        grouping = entry.grouping_variable
        if grouping is None:
            continue
        if strategy != "cluster_conditional":
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.grouping_variable",
                    f"only a `cluster_conditional` column takes a grouping variable, not `{strategy}`",
                    column=column,
                )
            )
        elif grouping not in known:
            unknown(f"{param}.grouping_variable", grouping)
        elif grouping == column:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.grouping_variable",
                    f"`{column}` cannot be grouped by itself",
                    column=column,
                )
            )
        elif grouping in excluded:
            conflict(
                f"{param}.grouping_variable",
                grouping,
                f"grouping variable `{grouping}` is excluded",
            )

    for name, names in (("exclude_columns", exclude_columns), ("skip_columns", skip_columns)):
        for i, column in enumerate(names):
            param = f"{name}[{i}]"
            if column not in known:
                unknown(param, column)
            elif column in universe.targets:
                conflict(param, column, f"`{column}` is a target the profile declares")
            elif name == "skip_columns" and column in excluded:
                conflict(param, column, f"`{column}` is excluded, so it cannot also be skipped")

    middle = dials.values[f"{DIAL_GROUP}.signal_score_middle_tier_min"]
    top = dials.values[f"{DIAL_GROUP}.signal_score_top_tier_min"]
    if middle > top:
        upper = f"{DIAL_GROUP}.signal_score_top_tier_min"
        at = upper if upper in dials.set_by_caller else f"{DIAL_GROUP}.signal_score_middle_tier_min"
        issues.append(
            _Issue(
                "invalid_dial_value",
                f"thresholds.{at}",
                f"`thresholds.{DIAL_GROUP}.signal_score_middle_tier_min` ({middle}) must be "
                f"at most `thresholds.{upper}` ({top})",
            )
        )
    return issues


def _issues_error(issues: list[_Issue], columns: Sequence[str]) -> DataForgeError:
    # The first issue's code names the error; the rest ride along in ``issues``.
    first = issues[0]
    params = list(dict.fromkeys(i.param for i in issues))
    named = list(dict.fromkeys(i.column for i in issues if i.column is not None))
    unknown = [i.unknown for i in issues if i.unknown is not None]
    if len(issues) == 1:
        message = f"`{first.param}`: {first.problem}."
    else:
        message = (
            f"{len(issues)} problems in the arguments, first `{first.param}`: "
            f"{first.problem}. See `issues` for all of them."
        )
    fields: dict[str, Any] = {
        "param": first.param if len(params) == 1 else None,
        "params": params,
        "columns": named or None,
        "issues": [{"param": i.param, "problem": i.problem} for i in issues],
    }
    if first.code == "unknown_column":
        return DataForgeError.from_code(
            "unknown_column", message, candidates=close_matches(unknown, columns), **fields
        )
    if first.code == "duplicate_key":
        return DataForgeError.from_code("duplicate_key", message, **fields)
    if first.code == "declaration_conflict":
        return DataForgeError.from_code("declaration_conflict", message, **fields)
    if first.code == "strategy_not_declarable":
        return DataForgeError.from_code(
            "strategy_not_declarable", message, allowed=list(_NON_NUMERIC_ALLOWED), **fields
        )
    if first.code == "constant_fill_missing":
        return DataForgeError.from_code("constant_fill_missing", message, **fields)
    if first.code == "invalid_argument":
        return DataForgeError.from_code("invalid_argument", message, **fields)
    return DataForgeError.from_code("invalid_dial_value", message, **fields)


def _build_config(
    dials: _Dials,
    *,
    exclude_columns: Sequence[str],
    skip_columns: Sequence[str],
    declared: dict[str, Any],
) -> Any:
    from ..config import PipelineConfig, PipelinePhase
    from ..imputation._config import ImputationConfig, NumericImputationConfig

    numeric = NumericImputationConfig(**dials.numeric())
    for column, entry in declared.items():
        if entry.strategy == "cluster_conditional" and entry.grouping_variable is not None:
            numeric.set_bimodal_grouping_variable(column, entry.grouping_variable)
    imputation = ImputationConfig(numeric=numeric)
    # MNAR columns count out of the multi-MAR rule, as a routed MNAR column does.
    mnar = [c for c, e in declared.items() if e.strategy == "mnar"]
    if mnar:
        imputation.add_mnar_column(mnar)
    config = PipelineConfig(imputation=imputation)
    if exclude_columns:
        config.add_exclusion(list(exclude_columns))
    if skip_columns:
        config.add_phase_exclusion(PipelinePhase.Imputation, list(skip_columns))
    return config


def _with_dial_sources(evidence: tuple[Any, ...], dials: _Dials) -> tuple[Any, ...]:
    # The escalation seam can tell only "equal to the default"; the caller's
    # own record says whether the Agent set the dial.
    from ..imputation._evidence import FeasibilityFact

    out = []
    for fact in evidence:
        if isinstance(fact, FeasibilityFact) and any(t.dial for t in fact.failed_terms):
            terms = []
            for term in fact.failed_terms:
                if term.dial is not None:
                    path = term.dial.dial.removeprefix("thresholds.")
                    source = "argument" if path in dials.set_by_caller else "default"
                    term = term.model_copy(
                        update={"dial": term.dial.model_copy(update={"source": source})}
                    )
                terms.append(term)
            fact = fact.model_copy(update={"failed_terms": terms})
        out.append(fact)
    return tuple(out)


def ordered_columns(routings: dict[str, Any]) -> list[str]:
    """Return routed columns in result order: those that fill or drop first, then ``passthrough``.

    Each group keeps the routing's own order: dataset order, then indicators.

    Parameters
    ----------
    routings : dict
        Column name to ``ColumnRouting``, in routing order.

    Returns
    -------
    list of str
        The column names, in the order ``route`` and ``explain_routing`` list them.
    """
    return sorted(routings, key=lambda c: routings[c].strategy.value == "passthrough")


def _capped_warning(
    warning: ResultWarning, columns: list[str], index: int
) -> tuple[ResultWarning, list[dict[str, Any]]]:
    omitted = []
    if len(columns) > WARNING_COLUMNS_CAP:
        omitted.append(
            {
                "field": f"warnings[{index}].columns",
                "shown": WARNING_COLUMNS_CAP,
                "total": len(columns),
                "operation": None,
            }
        )
    return warning.model_copy(update={"columns": columns[:WARNING_COLUMNS_CAP]}), omitted


def route_profile(
    *,
    profile: str,
    strategies: Sequence[Any] | None,
    default_strategy: str | None,
    model_choice: str | None,
    exclude_columns: Sequence[str] | None,
    skip_columns: Sequence[str] | None,
    thresholds: Any,
) -> dict[str, Any]:
    """Route a profile Handle into a new routing Handle and build the ``route`` result.

    Every argument is checked before any work; all problems come back in one
    Library Error, and no Handle is written.

    Parameters
    ----------
    profile : str
        The profile Handle.
    strategies : sequence or None
        The ``StrategyDeclaration`` records, as validated by the args model.
    default_strategy : str or None
        The strategy unlisted numeric columns take; None routes them.
    model_choice : str or None
        The MICE block's estimator, by value; None lets the ladder pick.
    exclude_columns, skip_columns : sequence of str or None
        Column names.
    thresholds : RouteThresholds or None
        The dials record.

    Returns
    -------
    dict
        The fields of the ``route`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column``, ``duplicate_key``, ``declaration_conflict``,
        ``strategy_not_declarable``, ``constant_fill_missing``,
        ``invalid_argument`` or ``invalid_dial_value`` naming every problem
        argument; ``no_mice_block`` when ``model_choice`` has no block; plus
        the profile Handle's own errors.
    """
    from ..config import PipelinePhase, SemanticType
    from ..imputation._config import (
        ColumnRouting,
        ImputationRouting,
        ImputationStrategy,
        ModelChoice,
    )
    from ..imputation._escalation import (
        _column_stats,
        _forced_feasibility,
        _mice_winning_tag,
        _resolve_nonlinearity_tag,
    )
    from ..imputation._evidence import FeasibilityFact, RoutingReason
    from ..imputation._regression_estimator_factory import RegressionEstimatorFactory
    from ..imputation._router import _filter_feature_correlation
    from ..imputation._router import route as route_columns

    result, record = read_profile(profile, param="profile")
    inherited = InheritedDeclarations.from_dict(record["declarations"])
    universe = _Universe(
        ordered=list(record["columns"]),
        excluded=list(inherited.exclusions),
        profile_skipped=set(record["skip_columns"]),
        numeric={
            c
            for c, cp in result.columns.items()
            if cp.semantic_type == SemanticType.Numeric
        },
        targets=list(inherited.targets),
    )
    strategies = list(strategies or [])
    dials = _resolve_dials(thresholds)
    issues = _check_arguments(
        universe,
        strategies=strategies,
        exclude_columns=list(exclude_columns or []),
        skip_columns=list(skip_columns or []),
        dials=dials,
    )
    if issues:
        raise _issues_error(issues, universe.known)
    # Excluding an already-excluded column again changes nothing.
    exclude_columns = [
        c for c in dict.fromkeys(exclude_columns or []) if c not in universe.excluded
    ]
    skip_columns = list(dict.fromkeys(skip_columns or []))

    excluded = set(exclude_columns)
    skipped = set(skip_columns)
    declared = {entry.column: entry for entry in strategies}
    config = _build_config(
        dials, exclude_columns=exclude_columns, skip_columns=skip_columns, declared=declared
    )
    numeric_cfg = config.imputation.numeric
    routed = route_columns(result, config).column_routings

    # The same row count, predictor count and predictor view the router read.
    n_rows = result.dataset.row_count
    active = set(config.resolve_active_columns(PipelinePhase.Imputation, list(result.columns)))
    numeric_active = [c for c in result.columns if c in universe.numeric and c in active]
    feature_correlation = result.dataset.feature_correlation
    if feature_correlation is not None and (excluded or skipped):
        feature_correlation = _filter_feature_correlation(feature_correlation, set(numeric_active))

    def authored(column: str, strategy: str, entry: Any = None) -> ColumnRouting:
        chosen = ImputationStrategy(strategy)
        cp = result.columns[column]
        evidence: tuple[Any, ...] = ()
        if strategy in _FLOORED:
            fact = _forced_feasibility(
                column,
                cp,
                numeric_cfg,
                n_rows,
                len(numeric_active),
                chosen,
                feature_correlation=feature_correlation,
            )
            evidence = (fact,) if fact is not None else ()
        return ColumnRouting(
            column=column,
            # Profiling skipped the column, so it has no type; ``Text`` leaves
            # its dtype untouched and keeps it out of every predictor pool.
            semantic_type=cp.semantic_type or SemanticType.Text,
            strategy=chosen,
            reason=RoutingReason.authored,
            evidence=evidence,
            constant_fill=None if entry is None else entry.constant_fill,
            grouping_variable=None if entry is None else entry.grouping_variable,
            indicator_flag=chosen == ImputationStrategy.MNAR,
            mnar=chosen == ImputationStrategy.MNAR,
            drop=chosen == ImputationStrategy.Dropped,
        )

    routings: dict[str, ColumnRouting] = {}
    for column, cp in result.columns.items():
        if column in excluded:
            continue
        if column in declared:
            routings[column] = authored(column, declared[column].strategy, declared[column])
        elif (
            default_strategy is not None
            and column in universe.numeric
            and column not in skipped
        ):
            routings[column] = authored(column, default_strategy)
        elif column in routed:
            routings[column] = routed[column]
        else:  # profiling skipped it, so the router could not see it
            routings[column] = ColumnRouting(
                column=column,
                semantic_type=SemanticType.Text,
                strategy=ImputationStrategy.Passthrough,
                reason=RoutingReason.soft_excluded,
                excluded=True,
            )
        evidence = _with_dial_sources(routings[column].evidence, dials)
        if evidence != routings[column].evidence:
            routings[column] = dataclasses.replace(routings[column], evidence=evidence)

    # Indicator pass: the ``<column>_missing`` each MNAR column appends.
    for column, routing in list(routings.items()):
        if routing.indicator_flag:
            routings[f"{column}_missing"] = ColumnRouting(
                column=f"{column}_missing",
                semantic_type=SemanticType.Boolean,
                strategy=ImputationStrategy.Indicator,
                reason=routing.reason,
            )

    mice_columns = [c for c, r in routings.items() if r.strategy == ImputationStrategy.MICE]
    if model_choice is not None and not mice_columns:
        raise DataForgeError.from_code(
            "no_mice_block",
            "`model_choice` sets the MICE block's estimator, but no column takes `mice`. "
            "Leave `model_choice` null, or route at least one column to `mice`.",
            param="model_choice",
        )
    mice_choice = None
    estimator_reason = None
    if mice_columns:
        tags = {c: _resolve_nonlinearity_tag(_column_stats(result.columns[c])) for c in mice_columns}
        winning = _mice_winning_tag(list(tags.values()))
        mice_choice = (
            ModelChoice(model_choice)
            if model_choice is not None
            else RegressionEstimatorFactory.resolve_choice(winning)
        )
        estimator_reason = {
            "source": "ladder" if model_choice is None else "argument",
            "tag": winning.value,
            "column": next(c for c in mice_columns if tags[c] == winning),
        }

    routing = ImputationRouting(column_routings=routings, mice_model_choice=mice_choice)
    handle = write_handle(
        "routing",
        size={"columns": len(routings)},
        payload_name="routing.bin",
        write_payload=lambda tmp: tmp.write_bytes(serialize(routing)),
        obj=routing,
        declarations=InheritedDeclarations(
            targets=inherited.targets,
            type_overrides=inherited.type_overrides,
            sentinels=inherited.sentinels,
            exclusions=(*inherited.exclusions, *exclude_columns),
        ),
        extra={
            "profile": profile,
            "frame": record["frame"],
            "skip_columns": skip_columns,
            "estimator_reason": estimator_reason,
            "dials": {
                "values": dials.values,
                "set_by_caller": sorted(dials.set_by_caller),
            },
        },
    )

    warnings: list[ResultWarning] = []
    omitted: list[dict[str, Any]] = []
    forced = [
        c
        for c, r in routings.items()
        if r.reason == RoutingReason.authored
        and any(isinstance(f, FeasibilityFact) and not f.passed for f in r.evidence)
    ]
    if forced:
        warning, cut = _capped_warning(
            ResultWarning(
                code="forced_past_feasibility",
                message=(
                    f"{len(forced)} column(s) take a strategy they fail the Feasibility "
                    f"Floor or the strategy's conditions for; they will still train. "
                    f"`explain_routing` shows each failed term and its dial."
                ),
                param="strategies",
                hint=entry_for("forced_past_feasibility").hint,
            ),
            forced,
            len(warnings),
        )
        warnings.append(warning)
        omitted += cut
    kept = [
        c
        for c, r in routings.items()
        if r.reason == RoutingReason.not_imputed_type
        and result.columns[c].missingness is not None
        and result.columns[c].missingness.effective_null_count > 0
    ]
    if kept:
        warning, cut = _capped_warning(
            ResultWarning(
                code="non_numeric_missing_kept",
                message=(
                    f"{len(kept)} non-numeric column(s) have missing values that routing "
                    f"keeps: only numeric columns are imputed."
                ),
                hint=entry_for("non_numeric_missing_kept").hint,
            ),
            kept,
            len(warnings),
        )
        warnings.append(warning)
        omitted += cut

    rows = []
    for column in ordered_columns(routings):
        r = routings[column]
        cp = result.columns.get(column)
        semantic_type = (
            r.semantic_type if cp is None else cp.semantic_type
        )  # an indicator has no profile
        rows.append(
            {
                "column": column,
                "semantic_type": None if semantic_type is None else semantic_type.value,
                "strategy": r.strategy.value,
                "reason": r.reason.value,
            }
        )
    census: dict[str, int] = {}
    for row in rows:
        census[row["strategy"]] = census.get(row["strategy"], 0) + 1
    if len(rows) > COLUMN_CAP:
        omitted.insert(
            0,
            {
                "field": "columns",
                "shown": COLUMN_CAP,
                "total": len(rows),
                "operation": "explain_routing",
            },
        )

    return {
        "warnings": warnings,
        "omitted": omitted,
        "routing": handle,
        "profile": profile,
        "mice_estimator": None if mice_choice is None else mice_choice.value,
        "estimator_reason": estimator_reason,
        "census": [
            {"strategy": s, "count": n}
            for s, n in sorted(census.items(), key=lambda item: -item[1])
        ],
        "columns": rows[:COLUMN_CAP],
    }
