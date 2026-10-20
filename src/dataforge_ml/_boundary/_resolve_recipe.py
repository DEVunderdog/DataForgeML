"""Internal code behind the ``resolve_recipe`` Operation (ADR-0089, ADR-0110).

It reads the routing Handle and the profile Handle the routing was made from,
resolves the recipe once, writes the recipe Handle and lists its units and
column estimates by value, so ``derive_units`` is not an Operation.

The routing's own dials feed the recipe: the correlation thresholds and the
predictor cap that routing judged a column by are the ones its recipe resolves
with. Every other recipe dial takes its library default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._serialization import serialize
from ..records import DataForgeError
from ._detail import page
from ._error_codes import close_matches
from ._profile import _Issue
from ._route import numeric_dials
from ._workspace import InheritedDeclarations, read_profile, read_routing, write_handle

#: Most units the result lists by value; ``explain_routing`` pages through the columns behind the rest.
#: Half the other pages, so a full page of units and of estimates fit one Result Budget together.
UNIT_CAP = 50
#: Most columns one unit lists by value.
UNIT_COLUMNS_CAP = 100
#: Most columns the result lists estimates for.
ESTIMATES_CAP = 15
#: Most ``feature_cols`` one column's estimates list.
FEATURE_COLS_CAP = 10


def _recipe_config(routing_dials: dict[str, Any]) -> Any:
    # A config holding the routing's own dials; every other field is the default.
    from ..config import PipelineConfig
    from ..imputation._config import ImputationConfig, NumericImputationConfig

    numeric = NumericImputationConfig(**numeric_dials(routing_dials))
    return PipelineConfig(imputation=ImputationConfig(numeric=numeric))


def _check_hyperparameters(entries: Sequence[Any], units: Sequence[Any]) -> list[_Issue]:
    """Check every ``hyperparameters`` entry against the routing's units."""
    from ..operations._hyperparameters import _RECORDS

    strategies = {u.unit_id: u.strategy for u in units}
    issues: list[_Issue] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        param, unit_id = f"hyperparameters[{i}]", entry.unit_id
        if unit_id not in strategies:
            issues.append(
                _Issue(
                    "unknown_unit",
                    f"{param}.unit_id",
                    f"the routing has no unit `{unit_id}`",
                    unknown=unit_id,
                )
            )
            continue
        if unit_id in seen:
            issues.append(
                _Issue("duplicate_key", f"{param}.unit_id", f"`{unit_id}` is listed twice")
            )
            continue
        seen.add(unit_id)
        strategy = strategies[unit_id]
        if strategy not in _RECORDS:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.unit_id",
                    f"`{unit_id}` is a `{strategy.value}` unit, which has no hyperparameters",
                )
            )
        elif entry.hyperparameters.kind != strategy.value:
            issues.append(
                _Issue(
                    "invalid_argument",
                    f"{param}.hyperparameters.kind",
                    f"`{unit_id}` is a `{strategy.value}` unit, so its record's `kind` "
                    f"is `{strategy.value}`, not `{entry.hyperparameters.kind}`",
                    unknown=strategy.value,
                )
            )
    return issues


def _check_estimates(entries: Sequence[Any], routing: Any, profile: Any) -> list[_Issue]:
    """Check every ``estimates`` entry against the routing, by the rules the recipe resolves with."""
    from ..config import SemanticType
    from ..imputation._config import ImputationStrategy
    from ..imputation._recipe import (
        _BIMODAL_STRATEGIES,
        _bimodal_centers,
        _resolve_domain_snap_bounds,
    )

    columns = routing.column_routings
    issues: list[_Issue] = []
    seen: set[str] = set()

    def issue(code: str, param: str, problem: str, column: str, **kw: Any) -> None:
        issues.append(_Issue(code, param, problem, column=column, **kw))

    for i, entry in enumerate(entries):
        param, column = f"estimates[{i}]", entry.column
        if column not in columns:
            issue(
                "unknown_column",
                f"{param}.column",
                f"no column `{column}` in the routing",
                column,
                unknown=column,
            )
            continue
        if column in seen:
            issue("duplicate_key", f"{param}.column", f"`{column}` is listed twice", column)
            continue
        seen.add(column)
        strategy = columns[column].strategy
        bimodal = strategy in _BIMODAL_STRATEGIES
        bounded = _resolve_domain_snap_bounds(profile.columns[column], strategy) is not None
        if not (bimodal or bounded):
            issue(
                "estimates_not_applicable",
                f"{param}.column",
                f"`{column}` is a `{strategy.value}` column, which reads no estimates",
                column,
            )
            continue
        not_read = []
        if not bimodal:
            not_read += [f for f in ("center1", "center2") if getattr(entry, f) is not None]
        if strategy != ImputationStrategy.ClusterConditional and entry.feature_cols is not None:
            not_read.append("feature_cols")
        if not bounded and entry.domain_snap_bounds is not None:
            not_read.append("domain_snap_bounds")
        for field in not_read:
            issue(
                "estimates_not_applicable",
                f"{param}.{field}",
                f"`{column}` is a `{strategy.value}` column, which does not read `{field}`",
                column,
            )
        # Setting one centre over a missing pair leaves a column that cannot fit.
        one_centre = (entry.center1 is None) != (entry.center2 is None)
        if bimodal and one_centre and None in _bimodal_centers(profile.columns[column], strategy):
            unset = "center2" if entry.center2 is None else "center1"
            issue(
                "invalid_argument",
                f"{param}.{unset}",
                f"`{column}` has no resolved centres, so set both",
                column,
            )
        for j, feature in enumerate(entry.feature_cols or []):
            at = f"{param}.feature_cols[{j}]"
            if feature not in columns:
                issue(
                    "unknown_column", at, f"no column `{feature}` in the routing", feature,
                    unknown=feature,
                )
            elif feature == column:
                issue("invalid_argument", at, f"`{column}` cannot be its own feature", column)
            elif columns[feature].semantic_type != SemanticType.Numeric:
                issue("invalid_argument", at, f"`{feature}` is not numeric", feature)
        bounds = entry.domain_snap_bounds
        if bounds is not None and bounds.low > bounds.high:
            issue(
                "invalid_argument",
                f"{param}.domain_snap_bounds",
                f"`low` ({bounds.low}) is above `high` ({bounds.high})",
                column,
            )
    return issues


def _apply_estimates(recipe: Any, entries: Sequence[Any]) -> Any:
    # Each entry's set fields overwrite the resolved ones; a null field keeps its value.
    for entry in entries:
        fields = entry.model_dump(exclude={"column"}, exclude_none=True)
        if entry.feature_cols is not None:
            fields["feature_cols"] = tuple(entry.feature_cols)
        if entry.domain_snap_bounds is not None:
            fields["domain_snap_bounds"] = (
                entry.domain_snap_bounds.low,
                entry.domain_snap_bounds.high,
            )
        if fields:
            recipe = recipe.with_estimates(entry.column, **fields)
    return recipe


def _issues_error(
    issues: list[_Issue], unit_ids: Sequence[str], columns: Sequence[str]
) -> DataForgeError:
    # The first issue's code names the error; the rest ride along in ``issues``.
    first = issues[0]
    params = list(dict.fromkeys(i.param for i in issues))
    named = list(dict.fromkeys(i.column for i in issues if i.column is not None))
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
        "issues": [{"param": i.param, "problem": i.problem} for i in issues],
    }
    if first.code == "unknown_unit":
        unknown = [i.unknown for i in issues if i.code == "unknown_unit"]
        return DataForgeError.from_code(
            "unknown_unit", message, candidates=close_matches(unknown, unit_ids), **fields
        )
    if first.code == "unknown_column":
        unknown = [i.unknown for i in issues if i.code == "unknown_column"]
        return DataForgeError.from_code(
            "unknown_column",
            message,
            candidates=close_matches(unknown, columns) or list(columns[:5]),
            columns=named or None,
            **fields,
        )
    if first.code == "duplicate_key":
        return DataForgeError.from_code("duplicate_key", message, columns=named or None, **fields)
    if first.code == "estimates_not_applicable":
        return DataForgeError.from_code(
            "estimates_not_applicable", message, columns=named or None, **fields
        )
    # A kind mismatch names the one kind its unit takes.
    allowed = [first.unknown] if first.unknown is not None else None
    return DataForgeError.from_code("invalid_argument", message, allowed=allowed, **fields)


def _shown_hyperparameters(recipe: Any, unit: Any) -> dict[str, Any] | None:
    # The unit's merged dials as its strategy's record; None for a strategy with none.
    from ..operations._hyperparameters import _RECORDS, hyperparameters_record

    if unit.strategy not in _RECORDS:
        return None
    record = hyperparameters_record(unit.strategy, recipe.hyperparameters(unit.unit_id))
    return record.model_dump()


def _shown_estimates(recipe: Any, column: str) -> dict[str, Any]:
    # One column's estimates as a ``RecipeEstimates`` row, ``feature_cols`` not yet capped.
    est = recipe.column_estimates[column]
    bounds = est.domain_snap_bounds
    return {
        "column": column,
        "center1": est.center1,
        "center2": est.center2,
        "feature_cols": list(est.feature_cols) if est.feature_cols is not None else None,
        "domain_snap_bounds": None if bounds is None else {"low": bounds[0], "high": bounds[1]},
    }


def resolve_routing_recipe(
    *,
    routing: str,
    hyperparameters: Sequence[Any] | None,
    estimates: Sequence[Any] | None,
) -> dict[str, Any]:
    """Resolve a routing Handle into a new recipe Handle and build the ``resolve_recipe`` result.

    Parameters
    ----------
    routing : str
        The routing Handle.
    hyperparameters : sequence or None
        The ``UnitHyperparameters`` records, as validated by the args model.
    estimates : sequence or None
        The ``ColumnEstimatesOverride`` records, as validated by the args model.

    Returns
    -------
    dict
        The fields of the ``resolve_recipe`` result.

    Raises
    ------
    DataForgeError
        ``unknown_unit``, ``unknown_column``, ``duplicate_key``,
        ``estimates_not_applicable`` or ``invalid_argument`` naming every
        problem ``hyperparameters`` and ``estimates`` entry, raised before
        any work; the routing Handle's own errors, and the
        profile Handle's when the routing's profile can no longer be read.
    """
    from ..imputation._recipe import resolve_recipe
    from ..imputation._units import derive_units

    stored, record = read_routing(routing, param="routing")
    units = derive_units(stored)
    profile = record["profile"]
    # The profile is read first: which estimates a column reads depends on its numeric kind.
    result, _ = read_profile(profile, param="routing")
    issues = _check_hyperparameters(hyperparameters or [], units)
    issues += _check_estimates(estimates or [], stored, result)
    if issues:
        raise _issues_error(issues, [u.unit_id for u in units], list(stored.column_routings))
    recipe = resolve_recipe(stored, result, _recipe_config(record["dials"]["values"]))
    for entry in hyperparameters or []:
        recipe = recipe.with_hyperparameters(
            entry.unit_id, entry.hyperparameters.to_hyperparameters()
        )
    recipe = _apply_estimates(recipe, estimates or [])

    handle = write_handle(
        "recipe",
        size={"units": len(units)},
        payload_name="recipe.bin",
        write_payload=lambda tmp: tmp.write_bytes(serialize(recipe)),
        obj=recipe,
        declarations=InheritedDeclarations.from_dict(record["declarations"]),
        extra={"routing": routing, "profile": profile, "frame": record["frame"]},
    )

    shown, _, omitted = page(
        list(units), None, None, UNIT_CAP, field="units", operation="explain_routing"
    )
    rows = []
    for i, unit in enumerate(shown):
        columns, _, cut = page(
            list(unit.columns),
            None,
            None,
            UNIT_COLUMNS_CAP,
            field=f"units[{i}].columns",
            operation="explain_routing",
        )
        omitted += cut
        rows.append(
            {
                "unit_id": unit.unit_id,
                "strategy": unit.strategy.value,
                "columns": columns,
                "hyperparameters": _shown_hyperparameters(recipe, unit),
            }
        )
    shown_estimates, _, cut = page(
        list(recipe.column_estimates), None, None, ESTIMATES_CAP, field="estimates",
        operation=None,
    )
    omitted += cut
    estimate_rows = []
    for i, column in enumerate(shown_estimates):
        row = _shown_estimates(recipe, column)
        if row["feature_cols"] is not None:
            row["feature_cols"], _, cut = page(
                row["feature_cols"],
                None,
                None,
                FEATURE_COLS_CAP,
                field=f"estimates[{i}].feature_cols",
                operation=None,
            )
            omitted += cut
        estimate_rows.append(row)
    return {
        "warnings": [],
        "omitted": omitted,
        "recipe": handle,
        "routing": routing,
        "profile": profile,
        "units": rows,
        "estimates": estimate_rows,
    }
