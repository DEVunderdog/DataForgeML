"""Internal code behind the ``split`` Operation (ADR-0106, ADR-0110).

The Operation file holds the records; this module checks every argument
before any work, runs the one ``DataSplitter`` method the scheme names on the
profile's frame, writes every partition as a frame Handle that inherits the
profile's declarations, and builds the result.

A fallback the profile-stratified schemes take is reported with Dial
Pointers and counts, never with a label value (ADR-0106, ADR-0107). The
viability floor is set by ``scheme.test_size`` or ``scheme.k``, so a pointer
may name a ``scheme`` argument as well as a ``thresholds`` dial.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..records import DataForgeError, ResultWarning
from ._error_codes import close_matches, entry_for
from ._profile import _Issue
from ._workspace import InheritedDeclarations, read_frame, read_profile, write_handle

#: The Operation dial paths are relative to.
OPERATION = "split"
#: Most folds one split may make; the ``folds`` list is never cut.
FOLD_CAP = 100
#: The only dial group, and the ``SplitConfig`` fields it holds.
DIAL_GROUP = "stratification"
SPLIT_DIALS: tuple[str, ...] = (
    "max_stratification_signals",
    "rows_per_signal",
    "boolean_minority_threshold",
    "redundancy_correlation_threshold",
)
#: The library defaults of the ``scheme`` fields a null leaves unset.
SCHEME_DEFAULTS: dict[str, Any] = {"test_size": 0.2, "k": 5, "repeats": 10}
_PROFILE_STRATIFIED = ("profile_stratified", "profile_stratified_kfold")
_STRATIFIABLE = ("random", "kfold", "group_kfold", "repeated_kfold", "holdout_cv")


def dial_defaults() -> dict[str, Any]:
    """Return every split dial's default, keyed by its path under ``thresholds``.

    Returns
    -------
    dict
        Path such as ``stratification.rows_per_signal`` to the library default.
    """
    from ..splitting._config import SplitConfig

    defaults = SplitConfig()
    return {f"{DIAL_GROUP}.{name}": getattr(defaults, name) for name in SPLIT_DIALS}


@dataclass(frozen=True)
class _Resolved:
    """The scheme and dials with every null resolved, and what the caller set."""

    kind: str
    fields: dict[str, Any]  # scheme fields, nulls replaced by defaults
    dials: dict[str, Any]
    set_by_caller: frozenset[str]  # argument paths: ``scheme.k``, ``thresholds.…``

    def pointer(self, path: str) -> dict[str, Any]:
        if path.startswith("scheme."):
            value = self.fields[path.removeprefix("scheme.")]
        else:
            value = self.dials[path.removeprefix("thresholds.")]
        return {
            "operation": OPERATION,
            "dial": path,
            "value": value,
            "source": "argument" if path in self.set_by_caller else "default",
        }

    def config(self) -> Any:
        from ..splitting._config import SplitConfig

        prefix = f"{DIAL_GROUP}."
        return SplitConfig(**{p[len(prefix) :]: v for p, v in self.dials.items()})


def _resolve(scheme: Any, thresholds: Any) -> _Resolved:
    given: set[str] = set()
    fields: dict[str, Any] = {}
    for name, value in scheme:
        if name == "kind":
            continue
        if value is None:
            fields[name] = SCHEME_DEFAULTS.get(name)
        else:
            fields[name] = value
            given.add(f"scheme.{name}")
    if scheme.kind == "time" and fields.get("cutoff") is not None:
        fields["test_size"] = None  # the cutoff splits; no default size applies
    dials = dial_defaults()
    group = None if thresholds is None else thresholds.stratification
    if group is not None:
        for name, value in group:
            if value is not None:
                dials[f"{DIAL_GROUP}.{name}"] = value
                given.add(f"thresholds.{DIAL_GROUP}.{name}")
    return _Resolved(
        kind=scheme.kind, fields=fields, dials=dials, set_by_caller=frozenset(given)
    )


def _is_datetime(dtype: Any) -> bool:
    import polars as pl

    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def _time_key(data: Any, column: str, result: Any) -> Any:
    """The time column as the split orders it: sentinels nulled, strings parsed."""
    from ..utils._dtype_floor import _apply_dtype_floor
    from ..utils._null_normalization import _resolve_effective_nulls

    cp = result.columns.get(column)
    frame = _apply_dtype_floor(
        _resolve_effective_nulls(
            data.select(column),
            numeric_sentinels=result.numeric_sentinels,
            string_sentinels=result.string_sentinels,
        ),
        {column: None if cp is None else cp.semantic_type},
    )
    return frame[column]


def _comparable(key: Any, cutoff: Any) -> tuple[Any, Any]:
    """Align a datetime cutoff with a Date or zoned Datetime key."""
    import polars as pl

    if key.dtype == pl.Date:
        key = key.cast(pl.Datetime("us"))
    zone = getattr(key.dtype, "time_zone", None)
    if zone is not None and cutoff.tzinfo is None:
        from zoneinfo import ZoneInfo

        cutoff = cutoff.replace(tzinfo=ZoneInfo(zone))
    elif zone is None and cutoff.tzinfo is not None:
        cutoff = cutoff.astimezone(UTC).replace(tzinfo=None)
    return key, cutoff


@dataclass(frozen=True)
class _Plan:
    """What the checks settled, for the run."""

    target: str | None  # the target handed to the splitter
    stratify: bool | None  # resolved for the plain schemes; None otherwise
    time_key: Any = None


def _check_arguments(
    data: Any,
    result: Any,
    inherited: InheritedDeclarations,
    resolved: _Resolved,
) -> tuple[_Plan, list[_Issue]]:
    """Check the scheme against the frame, the profile and itself."""
    kind, fields = resolved.kind, resolved.fields
    columns = set(data.columns)
    excluded = set(inherited.exclusions)
    targets = list(inherited.targets)
    n_rows = data.height
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

    def invalid(param: str, problem: str, column: str | None = None) -> None:
        issues.append(_Issue("invalid_argument", param, problem, column=column))

    def usable_column(name: str) -> bool:
        column = fields[name]
        if column not in columns:
            unknown(f"scheme.{name}", column)
            return False
        if column in excluded:
            issues.append(
                _Issue(
                    "declaration_conflict",
                    f"scheme.{name}",
                    f"`{column}` is excluded, so no phase may read it",
                    column=column,
                )
            )
            return False
        return True

    # The target, and whether to stratify on it.
    target = fields.get("target")
    stratify = fields.get("stratify")
    if target is not None:
        if target not in columns:
            unknown("scheme.target", target)
        elif target not in targets:
            invalid(
                "scheme.target",
                f"`{target}` is not a target the profile declares; declare it on `profile`",
                column=target,
            )
        if stratify is False:
            invalid("scheme.target", "`target` is read only when `stratify` is not false")
    if kind in _STRATIFIABLE and stratify is None:
        stratify = bool(targets)
    if kind in _STRATIFIABLE and stratify and not targets:
        issues.append(
            _Issue(
                "stratify_without_target",
                "scheme.stratify",
                "stratifying needs a target, and the profile declares none",
            )
        )
    wants_target = kind in _PROFILE_STRATIFIED or (kind in _STRATIFIABLE and stratify)
    if target is None and wants_target and len(targets) > 1:
        invalid(
            "scheme.target",
            f"the profile declares {len(targets)} targets; name the one to stratify on",
        )
    if target is None and wants_target and len(targets) == 1:
        target = targets[0]

    # Fold counts against the rows and groups there are.
    k = fields.get("k")
    if k is not None:
        rows_for_folds = n_rows
        if kind == "holdout_cv":
            rows_for_folds = n_rows - math.ceil(n_rows * fields["test_size"])
        if k > rows_for_folds:
            invalid("scheme.k", f"`k` ({k}) is above the {rows_for_folds} rows to fold")
    if kind == "repeated_kfold" and k * fields["repeats"] > FOLD_CAP:
        invalid(
            "scheme.repeats",
            f"`k * repeats` ({k * fields['repeats']}) is above {FOLD_CAP} folds",
        )
    if kind in ("group", "group_kfold") and usable_column("group_column"):
        groups = data[fields["group_column"]].n_unique()
        need = 2 if kind == "group" else k
        if groups < need:
            param = "scheme.group_column" if kind == "group" else "scheme.k"
            invalid(param, f"the split needs at least {need} groups, and there are {groups}")

    time_key = None
    if kind == "time":
        cutoff = fields["cutoff"]
        if cutoff is not None and "scheme.test_size" in resolved.set_by_caller:
            invalid("scheme.cutoff", "set `test_size` or `cutoff`, not both")
        if usable_column("time_column"):
            column = fields["time_column"]
            time_key = _time_key(data, column, result)
            is_datetime = _is_datetime(time_key.dtype)
            if not (is_datetime or time_key.dtype.is_numeric()):
                invalid(
                    "scheme.time_column",
                    f"`{column}` is neither a datetime nor a numeric column",
                    column=column,
                )
            elif cutoff is not None and is_datetime != isinstance(cutoff, datetime):
                want = "a datetime" if is_datetime else "a number"
                invalid("scheme.cutoff", f"`{column}` needs {want} as `cutoff`", column=column)

    return _Plan(target=target, stratify=stratify, time_key=time_key), issues


def _issues_error(issues: list[_Issue], columns: Sequence[str]) -> DataForgeError:
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
        "columns": named or None,
        "issues": [{"param": i.param, "problem": i.problem} for i in issues],
    }
    if first.code == "unknown_column":
        unknown = [i.unknown for i in issues if i.unknown is not None]
        return DataForgeError.from_code(
            "unknown_column", message, candidates=close_matches(unknown, columns), **fields
        )
    if first.code == "declaration_conflict":
        return DataForgeError.from_code("declaration_conflict", message, **fields)
    if first.code == "stratify_without_target":
        return DataForgeError.from_code(
            "stratify_without_target", message, param=first.param
        )
    return DataForgeError.from_code("invalid_argument", message, **fields)


def _run(splitter: Any, resolved: _Resolved, plan: _Plan, result: Any, report: Any):
    """Run the scheme; return the test frame (or None) and ``(index, repeat, train, validation)``."""
    kind, f = resolved.kind, resolved.fields
    stratify = plan.stratify
    if kind == "random":
        out = splitter.random_split(f["test_size"], stratify=stratify)
        return out.test, [(0, 0, out.train, None)]
    if kind == "time":
        out = splitter.time_split(
            f["time_column"], test_size=f["test_size"], cutoff=f["cutoff"], order=plan.time_key
        )
        return out.test, [(0, 0, out.train, None)]
    if kind == "group":
        out = splitter.group_split(f["test_size"], f["group_column"])
        return out.test, [(0, 0, out.train, None)]
    if kind == "profile_stratified":
        out = splitter.profile_stratified_split(result, f["test_size"], report=report)
        return out.test, [(0, 0, out.train, None)]
    if kind == "holdout_cv":
        out = splitter.holdout_cv(f["test_size"], f["k"], stratify=stratify)
        test, folds = out.test, out.folds
    elif kind == "kfold":
        test, folds = None, splitter.kfold(f["k"], stratify=stratify)
    elif kind == "group_kfold":
        test, folds = None, splitter.group_kfold(f["k"], f["group_column"], stratify=stratify)
    elif kind == "repeated_kfold":
        test, folds = None, splitter.repeated_kfold(f["k"], f["repeats"], stratify=stratify)
    else:  # profile_stratified_kfold
        test, folds = None, splitter.profile_stratified_kfold(result, f["k"], report=report)
    return test, [(fold.fold_index, fold.repeat_index, fold.train, fold.val) for fold in folds]


def _fallbacks(
    report: Any, resolved: _Resolved, n_rows: int
) -> tuple[list[dict[str, Any]], list[ResultWarning]]:
    """The fallbacks a profile-stratified scheme took, and one warning each."""
    floor_arg = "scheme.k" if resolved.kind == "profile_stratified_kfold" else "scheme.test_size"
    caps = [f"thresholds.{DIAL_GROUP}.{d}" for d in ("max_stratification_signals", "rows_per_signal")]
    fallbacks: list[dict[str, Any]] = []
    warnings: list[ResultWarning] = []

    def add(fallback: str, count: int, dials: list[str], warning: ResultWarning) -> None:
        fallbacks.append(
            {
                "fallback": fallback,
                "count": count,
                "dials": [resolved.pointer(d) for d in dials],
            }
        )
        warnings.append(warning)

    if report.kept == 0:
        if report.target_evicted:
            dials, why = caps, f"the signal cap ({report.cap}) cannot hold every target signal"
        elif report.candidates:
            dials, why = (
                [floor_arg],
                f"none of {report.candidates} signal(s) has {report.floor} row(s) on each side",
            )
        else:
            dials, why = [], "the profile flags nothing to balance"
        add(
            "unstratified",
            n_rows,
            dials,
            ResultWarning(
                code="split_unstratified",
                message=f"All {n_rows} rows were split at random: {why}.",
                param=dials[0] if dials else None,
                hint=entry_for("split_unstratified").hint,
            ),
        )
    elif report.kept < report.non_redundant:
        left_out = report.non_redundant - report.kept
        add(
            "signals_capped",
            left_out,
            caps,
            ResultWarning(
                code="stratification_signals_capped",
                message=(
                    f"{left_out} of {report.non_redundant} signal(s) were not balanced: "
                    f"the signal cap is {report.cap}."
                ),
                param=caps[0],
                hint=entry_for("stratification_signals_capped").hint,
            ),
        )
    if report.rows_to_train:
        add(
            "rows_to_train",
            report.rows_to_train,
            [floor_arg],
            ResultWarning(
                code="rare_rows_to_train",
                message=(
                    f"{report.rows_to_train} row(s) of {report.labels_to_train} target "
                    f"class(es) or rare value(s) below the floor of {report.floor} row(s) "
                    f"all went to train."
                ),
                param=floor_arg,
                hint=entry_for("rare_rows_to_train").hint,
            ),
        )
    return fallbacks, warnings


def split_profile(
    *,
    profile: str,
    scheme: Any,
    seed: int | None,
    thresholds: Any,
) -> dict[str, Any]:
    """Split a profile Handle's frame into new frame Handles and build the ``split`` result.

    Every argument is checked before any work; all problems come back in one
    Library Error, and no Handle is written.

    Parameters
    ----------
    profile : str
        The profile Handle.
    scheme : scheme record or None
        One of the ``split`` Operation's scheme records; None means
        ``profile_stratified`` with its defaults.
    seed : int or None
        The shuffle seed; None means the fixed default.
    thresholds : SplitThresholds or None
        The dials record.

    Returns
    -------
    dict
        The fields of the ``split`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column``, ``declaration_conflict``, ``stratify_without_target``
        or ``invalid_argument`` naming every problem argument;
        ``stratify_on_continuous_target`` when the target cannot be stratified
        on; plus the profile Handle's own errors.
    """
    from .._seed import DEFAULT_SEED
    from ..operations.split import ProfileStratifiedScheme
    from ..splitting._profile_signals import SignalReport
    from ..splitting._splitter import DataSplitter

    result, record = read_profile(profile, param="profile")
    inherited = InheritedDeclarations.from_dict(record["declarations"])
    data, _ = read_frame(record["frame"], param="profile")
    if scheme is None:
        scheme = ProfileStratifiedScheme(kind="profile_stratified")
    resolved = _resolve(scheme, thresholds)
    plan, issues = _check_arguments(data, result, inherited, resolved)
    if issues:
        raise _issues_error(issues, list(data.columns))

    seed = DEFAULT_SEED if seed is None else seed
    report = SignalReport()
    splitter = DataSplitter(data, target=plan.target, seed=seed, config=resolved.config())
    test, folds = _run(splitter, resolved, plan, result, report)

    declarations = inherited  # every partition keeps the profile's declarations

    def write(frame: Any, partition: str, index: int | None, repeat: int | None) -> str:
        return write_handle(
            "frame",
            size={"rows": frame.height, "columns": frame.width},
            payload_name="frame.parquet",
            write_payload=lambda tmp: frame.write_parquet(tmp),
            obj=frame,
            declarations=declarations,
            extra={
                "profile": profile,
                "source_frame": record["frame"],
                "partition": partition,
                "fold": index,
                "repeat": repeat,
            },
        )

    test_handle = None if test is None else write(test, "test", None, None)
    fold_rows = []
    for index, repeat, train, validation in folds:
        fold_rows.append(
            {
                "index": index,
                "repeat": repeat,
                "train": write(train, "train", index, repeat),
                "validation": None
                if validation is None
                else write(validation, "validation", index, repeat),
                "train_rows": train.height,
                "validation_rows": None if validation is None else validation.height,
            }
        )

    fallbacks: list[dict[str, Any]] = []
    warnings: list[ResultWarning] = []
    stratification = None
    if resolved.kind in _PROFILE_STRATIFIED:
        fallbacks, warnings = _fallbacks(report, resolved, data.height)
        stratification = {
            "floor": report.floor,
            "candidates": report.candidates,
            "viable": report.viable,
            "non_redundant": report.non_redundant,
            "cap": report.cap,
            "kept": report.kept,
            "rows_to_train": report.rows_to_train,
            "labels_to_train": report.labels_to_train,
        }
    if resolved.kind == "time" and plan.time_key.null_count():
        nulls = plan.time_key.null_count()
        warnings.append(
            ResultWarning(
                code="null_time_rows_to_train",
                message=f"{nulls} row(s) have no `{resolved.fields['time_column']}`, so they all went to train.",
                columns=[resolved.fields["time_column"]],
                param="scheme.time_column",
                hint=entry_for("null_time_rows_to_train").hint,
            )
        )

    if resolved.kind in _PROFILE_STRATIFIED:
        stratified_on = plan.target if report.target_kept else None
    else:
        stratified_on = plan.target if plan.stratify else None
    return {
        "warnings": warnings,
        "omitted": [],
        "profile": profile,
        "frame": record["frame"],
        "scheme": resolved.kind,
        "seed": seed,
        "stratified_on": stratified_on,
        "test": test_handle,
        "test_rows": None if test is None else test.height,
        "folds": fold_rows,
        "stratification": stratification,
        "fallbacks": fallbacks,
    }
