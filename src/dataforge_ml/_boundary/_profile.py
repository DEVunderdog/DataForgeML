"""Internal code behind the ``profile`` Operation (ADR-0104, ADR-0106).

The Operation file holds the records; this module checks every argument before
any work, builds the profiling config from the declarations and dials, runs
the profiler once, writes the profile Handle and builds the Dataset Digest.

Every profiling flag a threshold decides maps to the dials its rule reads in
:data:`FLAG_DIALS`, so the digest can point each flag at the knob behind it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._serialization import serialize
from ..records import DataForgeError, ResultWarning
from ._error_codes import close_matches, entry_for
from ._operation import progress_observer
from ._workspace import InheritedDeclarations, read_frame, write_handle

#: Most column rows the Dataset Digest holds (ADR-0104).
COLUMN_CAP = 100
#: Most columns one Result Warning names; the rest are counted in ``omitted``.
WARNING_COLUMNS_CAP = 20
#: The Operation dial paths are relative to.
OPERATION = "profile"

#: Each digest flag, as ``<group>.<flag>``, to the dials its rule reads, as
#: paths under ``thresholds``. A flag with no dials is decided by definition.
FLAG_DIALS: dict[str, tuple[str, ...]] = {
    "missingness.fully_null": (),
    "missingness.drop_candidate": ("missingness.col_drop_threshold",),
    "missingness.mar_suspect": ("missingness.mar_correlation_threshold",),
    "numeric.near_constant": ("numeric.near_constant_threshold",),
    "numeric.scale_anomaly": ("numeric.scale_orders_of_magnitude",),
    "numeric.bimodal": (
        "numeric.bimodal_dip_p_value_threshold",
        "numeric.bimodal_min_separation_threshold",
        "numeric.bimodal_min_component_weight",
    ),
    "numeric.high_outlier_density": (
        "numeric.high_outlier_density_threshold",
        "numeric.outlier_sigma_threshold",
    ),
    "numeric.format_mismatch": (),
    "categorical.near_constant": ("categorical.near_constant_threshold",),
    "categorical.mixed_type": ("categorical.mixed_type_min_minor_pct",),
    "datetime.mnar_suspected": ("datetime.mnar_null_ratio_threshold",),
    "datetime.high_gap_variance": ("datetime.high_gap_cv_threshold",),
    "datetime.recent_date_missing": ("datetime.recent_window_fraction",),
    "datetime.future_dates": (),
    "datetime.format_mismatch": (),
    "boolean.format_mismatch": (),
}

#: Dial pairs that must stay in order: (lower, upper, strictly below).
_ORDERED_DIALS: tuple[tuple[str, str, bool], ...] = (
    ("missingness.severity_minor", "missingness.severity_moderate", False),
    ("missingness.severity_moderate", "missingness.severity_high", False),
    ("numeric.skew_normal", "numeric.skew_moderate", False),
    ("numeric.skew_moderate", "numeric.skew_high", False),
    ("numeric.kurt_platykurtic_upper", "numeric.kurt_leptokurtic_lower", True),
    (
        "numeric.tail_asymmetry_left_share_threshold",
        "numeric.tail_asymmetry_right_share_threshold",
        True,
    ),
)

#: Fallback warning codes, most actionable first.
_FALLBACK_ORDER: tuple[str, ...] = (
    "column_profiling_failed",
    "nonlinearity_probe_failed",
    "nonlinearity_too_few_rows",
    "association_failed",
    "mutual_information_too_few_rows",
)


def _config_classes() -> dict[str, tuple[str, type]]:
    # thresholds group -> (ProfileConfig attribute, sub-config class)
    from ..profiling._categorical_config import CategoricalProfileConfig
    from ..profiling._correlation_config import CorrelationProfileConfig
    from ..profiling._datetime_config import DatetimeProfileConfig
    from ..profiling._missingness_config import MissingnessProfileConfig
    from ..profiling._numeric_config import (
        NonlinearityProfileConfig,
        NumericProfileConfig,
    )
    from ..profiling._type_detection_config import TypeDetectionConfig

    return {
        "missingness": ("missingness", MissingnessProfileConfig),
        "type_detection": ("type_detection", TypeDetectionConfig),
        "numeric": ("numeric", NumericProfileConfig),
        "categorical": ("categorical", CategoricalProfileConfig),
        "correlation": ("correlation", CorrelationProfileConfig),
        "datetime": ("datetime_", DatetimeProfileConfig),
        "nonlinearity": ("nonlinearity", NonlinearityProfileConfig),
    }


def dial_defaults() -> dict[str, Any]:
    """Return every profiling dial's default, keyed by its path under ``thresholds``.

    Returns
    -------
    dict
        Path such as ``numeric.skew_high`` to the library default.
    """
    from ..profiling._config import ProfileConfig

    defaults: dict[str, Any] = {}
    for group, (_, cls) in _config_classes().items():
        for f in dataclasses.fields(cls):
            defaults[f"{group}.{f.name}"] = getattr(cls(), f.name)
    defaults["missingness.row_drop_threshold"] = ProfileConfig().row_drop_threshold
    return defaults


@dataclass(frozen=True)
class _Dials:
    """Every dial's resolved value, and which ones the caller set."""

    values: dict[str, Any]
    set_by_caller: frozenset[str]

    def pointer(self, path: str) -> dict[str, Any]:
        return {
            "operation": OPERATION,
            "dial": f"thresholds.{path}",
            "value": self.values[path],
            "source": "argument" if path in self.set_by_caller else "default",
        }

    def group(self, name: str) -> dict[str, Any]:
        prefix = f"{name}."
        return {
            p[len(prefix) :]: v for p, v in self.values.items() if p.startswith(prefix)
        }


def _resolve_dials(thresholds: Any) -> _Dials:
    values = dial_defaults()
    given: set[str] = set()
    if thresholds is not None:
        for group, record in thresholds:
            if record is None:
                continue
            for name, value in record:
                if value is not None:
                    values[f"{group}.{name}"] = value
                    given.add(f"{group}.{name}")
    return _Dials(values=values, set_by_caller=frozenset(given))


@dataclass(frozen=True)
class _Issue:
    code: str
    param: str
    problem: str
    column: str | None = None
    unknown: str | None = None


@dataclass
class _Declared:
    """Declarations merged from the frame and the call, in that order."""

    semantic_types: dict[str, str] = field(default_factory=dict)
    numeric_kinds: dict[str, str] = field(default_factory=dict)
    sentinels: dict[str, list[Any]] = field(default_factory=dict)
    datetime_formats: dict[str, tuple[str | None, str | None]] = field(
        default_factory=dict
    )
    targets: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    # Carried as they are: no profile argument declares or reads them.
    soft_exclusions: list[str] = field(default_factory=list)

    @classmethod
    def from_inherited(cls, inherited: InheritedDeclarations) -> _Declared:
        out = cls(
            targets=list(inherited.targets),
            exclusions=list(inherited.exclusions),
            soft_exclusions=list(inherited.soft_exclusions),
        )
        for entry in inherited.type_overrides:
            column = entry["column"]
            if entry.get("semantic_type") is not None:
                out.semantic_types[column] = entry["semantic_type"]
            if entry.get("numeric_kind") is not None:
                out.numeric_kinds[column] = entry["numeric_kind"]
            if entry.get("format") is not None or entry.get("epoch_unit") is not None:
                out.datetime_formats[column] = (
                    entry.get("format"),
                    entry.get("epoch_unit"),
                )
        for entry in inherited.sentinels:
            out.sentinels[entry["column"]] = list(entry["values"])
        return out

    def to_inherited(self) -> InheritedDeclarations:
        columns = list(
            dict.fromkeys(
                [*self.semantic_types, *self.numeric_kinds, *self.datetime_formats]
            )
        )
        overrides = []
        for column in columns:
            fmt, unit = self.datetime_formats.get(column, (None, None))
            overrides.append(
                {
                    "column": column,
                    "semantic_type": self.semantic_types.get(column),
                    "numeric_kind": self.numeric_kinds.get(column),
                    "format": fmt,
                    "epoch_unit": unit,
                }
            )
        return InheritedDeclarations(
            targets=tuple(self.targets),
            type_overrides=tuple(overrides),
            sentinels=tuple(
                {"column": c, "values": v} for c, v in self.sentinels.items()
            ),
            exclusions=tuple(self.exclusions),
            soft_exclusions=tuple(self.soft_exclusions),
        )


def _check_arguments(
    columns: Sequence[str],
    inherited: _Declared,
    *,
    semantic_types: Sequence[Any],
    numeric_kinds: Sequence[Any],
    sentinels: Sequence[Any],
    datetime_formats: Sequence[Any],
    targets: Sequence[str],
    exclude_columns: Sequence[str],
    skip_columns: Sequence[str],
    dials: _Dials,
) -> tuple[_Declared, list[_Issue]]:
    """Check every argument against the frame, the inherited declarations and each other."""
    known = set(columns)
    issues: list[_Issue] = []
    merged = _Declared(
        semantic_types=dict(inherited.semantic_types),
        numeric_kinds=dict(inherited.numeric_kinds),
        sentinels=dict(inherited.sentinels),
        datetime_formats=dict(inherited.datetime_formats),
        targets=list(inherited.targets),
        exclusions=list(inherited.exclusions),
        soft_exclusions=list(inherited.soft_exclusions),
    )

    def keyed(
        name: str,
        entries: Sequence[Any],
        value_of: Callable[[Any], Any],
        inherited_map: dict[str, Any],
        merged_map: dict[str, Any],
        what: str,
    ) -> None:
        seen: set[str] = set()
        for i, entry in enumerate(entries):
            param, column = f"{name}[{i}].column", entry.column
            if column not in known:
                issues.append(
                    _Issue(
                        "unknown_column",
                        param,
                        f"no column `{column}` in the frame",
                        column=column,
                        unknown=column,
                    )
                )
            elif column in seen:
                issues.append(
                    _Issue(
                        "duplicate_key",
                        param,
                        f"`{column}` is listed twice",
                        column=column,
                    )
                )
            elif column in inherited_map and inherited_map[column] != value_of(entry):
                issues.append(
                    _Issue(
                        "declaration_conflict",
                        param,
                        f"the frame already declares a different {what} for `{column}`",
                        column=column,
                    )
                )
            else:
                merged_map[column] = value_of(entry)
            seen.add(column)

    keyed(
        "semantic_types",
        semantic_types,
        lambda e: e.semantic_type,
        inherited.semantic_types,
        merged.semantic_types,
        "semantic type",
    )
    keyed(
        "numeric_kinds",
        numeric_kinds,
        lambda e: e.numeric_kind,
        inherited.numeric_kinds,
        merged.numeric_kinds,
        "numeric kind",
    )
    for i, entry in enumerate(numeric_kinds):
        declared = merged.semantic_types.get(entry.column)
        if entry.column in known and declared not in (None, "numeric"):
            issues.append(
                _Issue(
                    "declaration_conflict",
                    f"numeric_kinds[{i}].column",
                    f"`{entry.column}` is declared `{declared}`; a numeric kind needs `numeric`",
                    column=entry.column,
                )
            )
    keyed(
        "sentinels",
        sentinels,
        lambda e: list(e.values),
        inherited.sentinels,
        merged.sentinels,
        "sentinel list",
    )
    keyed(
        "datetime_formats",
        datetime_formats,
        lambda e: (e.format, e.epoch_unit),
        inherited.datetime_formats,
        merged.datetime_formats,
        "datetime format",
    )

    excluded = set(inherited.exclusions) | set(exclude_columns)
    skipped = set(skip_columns)
    seen_targets: set[str] = set()
    for i, target in enumerate(targets):
        param = f"targets[{i}]"
        if target not in known:
            issues.append(
                _Issue(
                    "unknown_column",
                    param,
                    f"no column `{target}` in the frame",
                    column=target,
                    unknown=target,
                )
            )
        elif target in seen_targets:
            issues.append(
                _Issue(
                    "duplicate_key", param, f"`{target}` is listed twice", column=target
                )
            )
        elif target in excluded or target in skipped:
            issues.append(
                _Issue(
                    "declaration_conflict",
                    param,
                    f"target `{target}` is also {'excluded' if target in excluded else 'skipped'}",
                    column=target,
                )
            )
        elif target not in merged.targets:
            merged.targets.append(target)
        seen_targets.add(target)

    for name, names in (
        ("exclude_columns", exclude_columns),
        ("skip_columns", skip_columns),
    ):
        for i, column in enumerate(names):
            param = f"{name}[{i}]"
            if column not in known:
                issues.append(
                    _Issue(
                        "unknown_column",
                        param,
                        f"no column `{column}` in the frame",
                        column=column,
                        unknown=column,
                    )
                )
            elif column in inherited.targets:
                issues.append(
                    _Issue(
                        "declaration_conflict",
                        param,
                        f"`{column}` is a target the frame inherited",
                        column=column,
                    )
                )
            elif name == "exclude_columns" and column not in merged.exclusions:
                merged.exclusions.append(column)

    for lower, upper, strict in _ORDERED_DIALS:
        low, high = dials.values[lower], dials.values[upper]
        if low > high or (strict and low == high):
            at = upper if upper in dials.set_by_caller else lower
            relation = "below" if strict else "at most"
            issues.append(
                _Issue(
                    "invalid_dial_value",
                    f"thresholds.{at}",
                    f"`thresholds.{lower}` ({low}) must be {relation} "
                    f"`thresholds.{upper}` ({high})",
                )
            )
    return merged, issues


def _raise_issues(issues: list[_Issue], columns: Sequence[str]) -> None:
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
        "candidates": close_matches(unknown, columns) if unknown else None,
        "issues": [{"param": i.param, "problem": i.problem} for i in issues],
    }
    raise _issues_error(first.code, message, fields)


def _issues_error(code: str, message: str, fields: dict[str, Any]) -> DataForgeError:
    # The first issue's code names the error; the rest ride along in ``issues``.
    if code == "unknown_column":
        return DataForgeError.from_code("unknown_column", message, **fields)
    if code == "duplicate_key":
        return DataForgeError.from_code("duplicate_key", message, **fields)
    if code == "declaration_conflict":
        return DataForgeError.from_code("declaration_conflict", message, **fields)
    return DataForgeError.from_code("invalid_dial_value", message, **fields)


def _build_config(
    declared: _Declared, skip_columns: Sequence[str], dials: _Dials
) -> Any:
    from ..config import PipelineConfig, PipelinePhase
    from ..profiling._config import ProfileConfig

    groups = {
        attribute: cls(
            **{k: v for k, v in dials.group(group).items() if k != "row_drop_threshold"}
        )
        for group, (attribute, cls) in _config_classes().items()
    }
    numeric_sentinels = {
        c: [float(v) for v in values]
        for c, values in declared.sentinels.items()
        if all(isinstance(v, (int, float)) for v in values)
    }
    string_sentinels = {
        c: list(values)
        for c, values in declared.sentinels.items()
        if c not in numeric_sentinels
    }
    profiling = ProfileConfig(
        target_columns=list(declared.targets),
        # Always on (ADR-0104): routing reads both, so no switch may weaken it.
        compute_correlation=True,
        compute_nonlinearity=True,
        row_drop_threshold=dials.values["missingness.row_drop_threshold"],
        numeric_sentinels=numeric_sentinels,
        string_sentinels=string_sentinels,
        datetime_formats={c: f for c, (f, _) in declared.datetime_formats.items() if f},
        datetime_epoch_units={
            c: u for c, (_, u) in declared.datetime_formats.items() if u
        },
        **groups,
    )
    config = PipelineConfig(profiling=profiling)
    for column, semantic_type in declared.semantic_types.items():
        config.set_column_type(column, semantic_type)
    for column, kind in declared.numeric_kinds.items():
        config.set_numeric_kind(column, kind)
    if declared.exclusions:
        config.add_exclusion(list(declared.exclusions))
    if skip_columns:
        config.add_phase_exclusion(PipelinePhase.Profiling, list(skip_columns))
    return config


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _cardinality(frame: Any, columns: list[str], declared: _Declared) -> dict[str, int]:
    import polars as pl

    from ..utils._null_normalization import _resolve_effective_nulls

    if not columns:
        return {}
    numeric = {
        c: v
        for c, v in declared.sentinels.items()
        if c in columns and all(isinstance(x, (int, float)) for x in v)
    }
    strings = {
        c: v for c, v in declared.sentinels.items() if c in columns and c not in numeric
    }
    resolved = _resolve_effective_nulls(
        frame.select(columns), numeric_sentinels=numeric, string_sentinels=strings
    )
    row = resolved.select(
        [pl.col(c).drop_nulls().n_unique().alias(c) for c in columns]
    ).row(0, named=True)
    return {c: int(n) for c, n in row.items()}


def _flags(column_profile: Any) -> list[str]:
    from ..profiling._boolean_config import BooleanStats
    from ..profiling._categorical_config import CategoricalStats
    from ..profiling._datetime_config import DatetimeStats
    from ..profiling._numeric_config import NumericStats

    flags: list[str] = []
    if column_profile.missingness is not None:
        flags += [f"missingness.{f}" for f in column_profile.missingness.flags]
    stats = column_profile.stats
    for kind, group in (
        (NumericStats, "numeric"),
        (CategoricalStats, "categorical"),
        (DatetimeStats, "datetime"),
        (BooleanStats, "boolean"),
    ):
        if isinstance(stats, kind):
            flags += [f"{group}.{f}" for f in stats.flags]
    return flags


def _digest_row(
    name: str,
    column_profile: Any,
    *,
    skipped: bool,
    targets: set[str],
    cardinality: dict[str, int],
) -> dict[str, Any]:
    from ..profiling._numeric_config import NumericStats

    stats = column_profile.stats
    numeric = stats if isinstance(stats, NumericStats) else None
    missingness = column_profile.missingness
    severity = missingness.severity if missingness is not None else None
    semantic_type = column_profile.semantic_type
    return {
        "column": name,
        "role": "target" if name in targets else "skipped" if skipped else "feature",
        "semantic_type": None if semantic_type is None else semantic_type.value,
        "numeric_kind": (
            column_profile.numeric_kind.value
            if column_profile.numeric_kind is not None
            and semantic_type is not None
            and semantic_type.value == "numeric"
            else None
        ),
        "null_ratio": (
            None if missingness is None else _finite(missingness.effective_null_ratio)
        ),
        "severity": None if severity is None else severity.value,
        "cardinality": cardinality.get(name),
        "skew": None if numeric is None else _finite(numeric.skewness),
        "outlier_density": (
            None if numeric is None else _finite(numeric.outlier_density)
        ),
        "flags": _flags(column_profile),
    }


def _fallback_warning(code: str, columns: list[str], dials: _Dials) -> ResultWarning:
    count = f"{len(columns)} column(s)"
    if code == "column_profiling_failed":
        return ResultWarning(
            code="column_profiling_failed",
            message=f"Profiling failed for {count}; they keep no statistics or flags.",
            columns=columns,
            hint=entry_for("column_profiling_failed").hint,
        )
    if code == "nonlinearity_probe_failed":
        return ResultWarning(
            code="nonlinearity_probe_failed",
            message=f"A nonlinearity probe failed for {count}; their probe readings are unset.",
            columns=columns,
            hint=entry_for("nonlinearity_probe_failed").hint,
        )
    if code == "nonlinearity_too_few_rows":
        dial = "thresholds.nonlinearity.min_rows"
        return ResultWarning(
            code="nonlinearity_too_few_rows",
            message=(
                f"{count} had fewer complete rows than `{dial}` "
                f"({dials.values['nonlinearity.min_rows']}), so their nonlinearity was not probed."
            ),
            columns=columns,
            param=dial,
            hint=entry_for("nonlinearity_too_few_rows").hint,
        )
    if code == "association_failed":
        return ResultWarning(
            code="association_failed",
            message=f"An association measure failed for pairs involving {count}; those are unset.",
            columns=columns,
            hint=entry_for("association_failed").hint,
        )
    dial = "thresholds.correlation.mi_min_rows"
    return ResultWarning(
        code="mutual_information_too_few_rows",
        message=(
            f"{count} had fewer rows with the target present than `{dial}` "
            f"({dials.values['correlation.mi_min_rows']}), so their mutual information "
            f"with the target was not estimated."
        ),
        columns=columns,
        param=dial,
        hint=entry_for("mutual_information_too_few_rows").hint,
    )


def _warnings(
    fallbacks: Sequence[Any], dials: _Dials
) -> tuple[list[ResultWarning], list[dict[str, Any]]]:
    by_code: dict[str, list[str]] = {}
    for fallback in fallbacks:
        columns = by_code.setdefault(fallback.code, [])
        if fallback.column is not None and fallback.column not in columns:
            columns.append(fallback.column)
    warnings: list[ResultWarning] = []
    omitted: list[dict[str, Any]] = []
    for code in _FALLBACK_ORDER:
        if code not in by_code:
            continue
        columns = by_code[code]
        if len(columns) > WARNING_COLUMNS_CAP:
            omitted.append(
                {
                    "field": f"warnings[{len(warnings)}].columns",
                    "shown": WARNING_COLUMNS_CAP,
                    "total": len(columns),
                    "operation": None,
                }
            )
        warning = _fallback_warning(code, columns, dials)
        warnings.append(
            warning.model_copy(update={"columns": columns[:WARNING_COLUMNS_CAP]})
        )
    return warnings, omitted


def profile_frame(
    *,
    frame: str,
    semantic_types: Sequence[Any] | None,
    numeric_kinds: Sequence[Any] | None,
    sentinels: Sequence[Any] | None,
    datetime_formats: Sequence[Any] | None,
    targets: Sequence[str] | None,
    exclude_columns: Sequence[str] | None,
    skip_columns: Sequence[str] | None,
    seed: int | None,
    thresholds: Any,
) -> dict[str, Any]:
    """Profile a frame Handle into a new profile Handle and build its Dataset Digest.

    Every argument is checked before any work; all problems come back in one
    Library Error, and no Handle is written.

    Parameters
    ----------
    frame : str
        The frame Handle.
    semantic_types, numeric_kinds, sentinels, datetime_formats : sequence or None
        The keyed declaration records, as validated by the args model.
    targets, exclude_columns, skip_columns : sequence of str or None
        Column names.
    seed : int or None
        The profiling seed; None means the fixed default.
    thresholds : ProfileThresholds or None
        The dials record.

    Returns
    -------
    dict
        The fields of the ``profile`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column``, ``duplicate_key``, ``declaration_conflict`` or
        ``invalid_dial_value`` naming every problem argument, plus the frame
        Handle's own errors and ``override_coercion_failed``.
    """
    from ..profiling._fallbacks import collect_fallbacks
    from ..profiling.orchestrator import StructuralProfiler

    data, inherited = read_frame(frame, param="frame")
    skip_columns = list(skip_columns or [])
    dials = _resolve_dials(thresholds)
    declared, issues = _check_arguments(
        list(data.columns),
        _Declared.from_inherited(inherited),
        semantic_types=semantic_types or [],
        numeric_kinds=numeric_kinds or [],
        sentinels=sentinels or [],
        datetime_formats=datetime_formats or [],
        targets=targets or [],
        exclude_columns=exclude_columns or [],
        skip_columns=skip_columns,
        dials=dials,
    )
    if issues:
        _raise_issues(issues, list(data.columns))

    config = _build_config(declared, skip_columns, dials)
    with collect_fallbacks() as fallbacks:
        result = StructuralProfiler(
            config, observer=progress_observer(), seed=seed
        ).profile(data)

    skipped = set(skip_columns)
    ordered = [c for c in data.columns if c in result.columns]
    cardinality = _cardinality(data, [c for c in ordered if c not in skipped], declared)
    handle = write_handle(
        "profile",
        size={"rows": data.height, "columns": len(ordered)},
        payload_name="profile.bin",
        write_payload=lambda tmp: tmp.write_bytes(serialize(result)),
        obj=result,
        declarations=declared.to_inherited(),
        extra={
            "frame": frame,
            "skip_columns": skip_columns,
            "cardinality": cardinality,
            # What detail Operations need without reading the frame again.
            "columns": ordered,
            "dials": {
                "values": dials.values,
                "set_by_caller": sorted(dials.set_by_caller),
            },
        },
    )

    targets_set = set(declared.targets)
    rows = [
        _digest_row(
            c,
            result.columns[c],
            skipped=c in skipped,
            targets=targets_set,
            cardinality=cardinality,
        )
        for c in ordered
    ]
    rows.sort(
        key=lambda row: not row["flags"]
    )  # stable: flagged first, then dataset order
    flag_dials: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for flag in row["flags"]:
            if flag not in flag_dials:
                flag_dials[flag] = [dials.pointer(p) for p in FLAG_DIALS.get(flag, ())]

    warnings, omitted = _warnings(fallbacks, dials)
    if len(rows) > COLUMN_CAP:
        omitted.insert(
            0,
            {
                "field": "columns",
                "shown": COLUMN_CAP,
                "total": len(rows),
                "operation": "describe_columns",
            },
        )

    dataset = result.dataset
    return {
        "warnings": warnings,
        "omitted": omitted,
        "profile": handle,
        "frame": frame,
        "n_rows": data.height,
        "n_columns": data.width,
        "duplicate_rows": dataset.duplicate_count,
        "duplicate_ratio": _finite(dataset.duplicate_ratio) or 0.0,
        "sparsity": _finite(dataset.overall_sparsity) or 0.0,
        "targets": [
            {
                "column": t,
                "problem_type": result.targets[t].problem_type.value,
                "missing_ratio": result.targets[t].missing_ratio,
            }
            for t in declared.targets
            if t in result.targets
        ],
        "declarations": {
            "semantic_types": [
                {"column": c, "semantic_type": v}
                for c, v in declared.semantic_types.items()
            ],
            "numeric_kinds": [
                {"column": c, "numeric_kind": v}
                for c, v in declared.numeric_kinds.items()
            ],
            "sentinels": [
                {"column": c, "values": v} for c, v in declared.sentinels.items()
            ],
            "datetime_formats": [
                {"column": c, "format": f, "epoch_unit": u}
                for c, (f, u) in declared.datetime_formats.items()
            ],
            "targets": list(declared.targets),
            "exclude_columns": list(declared.exclusions),
            "skip_columns": skip_columns,
        },
        "columns": rows[:COLUMN_CAP],
        "flag_dials": [{"flag": f, "dials": d} for f, d in flag_dials.items()],
    }
