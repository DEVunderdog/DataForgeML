"""Internal code behind the ``impute`` Operation (ADR-0110).

It reads an imputer Handle, composes the imputer again (ADR-0103), and fills
a frame Handle with it into a new frame Handle that is ready to model on:
hard-excluded and ``Dropped`` columns are left out, indicator columns are
added, and an ``MNAR`` column keeps its nulls with its fill on record
(ADR-0098). Drop Candidates stay in, listed with a Dial Pointer to the
threshold that flagged them (ADR-0106). The new frame inherits the source
frame's declarations, plus the exclusions the imputer's routing added, and
records its indicator columns as Soft Exclusions for later phases (ADR-0110).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

from ..records import DataForgeError
from ._detail import page
from ._profile import FLAG_DIALS, _Dials
from ._profile_detail import _scalar
from ._route import ordered_columns
from ._workspace import (
    InheritedDeclarations,
    read_frame,
    read_imputer,
    read_record,
    write_handle,
)

#: Most per-column records one ``impute`` result lists (ADR-0102).
COLUMNS_CAP = 100


def _union(first: tuple[Any, ...], second: tuple[Any, ...]) -> tuple[Any, ...]:
    # ``first`` in order, then each item of ``second`` it lacks.
    return (*first, *(item for item in second if item not in first))


def _merged(frame: InheritedDeclarations, imputer: InheritedDeclarations) -> InheritedDeclarations:
    # The frame's declarations first, then each one the imputer's chain added.
    return InheritedDeclarations(
        targets=_union(frame.targets, imputer.targets),
        type_overrides=_union(frame.type_overrides, imputer.type_overrides),
        sentinels=_union(frame.sentinels, imputer.sentinels),
        exclusions=_union(frame.exclusions, imputer.exclusions),
        soft_exclusions=_union(frame.soft_exclusions, imputer.soft_exclusions),
    )


def _cell(value: Any) -> bool | int | float | str | None:
    # A fill value as a JSON scalar; a numpy scalar unwraps first.
    return _scalar(value.item() if hasattr(value, "item") else value)


def impute_frame(*, imputer: str, frame: str) -> dict[str, Any]:
    """Fill a frame Handle with an imputer Handle and build the ``impute`` result.

    Parameters
    ----------
    imputer : str
        The imputer Handle.
    frame : str
        The frame Handle to fill.

    Returns
    -------
    dict
        The fields of the ``impute`` result.

    Raises
    ------
    DataForgeError
        ``unseen_column`` naming every column of ``frame`` the imputer was
        never fitted on and no declaration excludes; ``missing_fitted_columns``
        naming every column the imputer fills or leaves as it is that
        ``frame`` lacks; the imputer and frame Handles' own errors.
    """
    from ..imputation._config import ImputationStrategy
    from ..imputation._evidence import RoutingReason
    from ..imputation._fitted_imputer import DroppedColumnAbsentWarning
    from ..utils._null_normalization import _resolve_effective_nulls

    composed, imputer_record = read_imputer(imputer, param="imputer")
    data, frame_declarations = read_frame(frame, param="frame")
    declarations = _merged(
        frame_declarations, InheritedDeclarations.from_dict(imputer_record["declarations"])
    )

    # Hard exclusions leave this and every later phase, so they never reach the imputer.
    excluded = set(declarations.exclusions)
    data = data.drop([c for c in data.columns if c in excluded])

    unseen = [c for c in data.columns if c not in composed.records]
    if unseen:
        names = ", ".join(f"`{c}`" for c in unseen[:5])
        more = f" and {len(unseen) - 5} more" if len(unseen) > 5 else ""
        raise DataForgeError.from_code(
            "unseen_column",
            f"`frame`: column(s) {names}{more} were not in the frame the imputer was "
            "fitted on.",
            columns=unseen,
        )
    absent_exempt = {ImputationStrategy.Dropped, ImputationStrategy.Indicator}
    missing = [
        c
        for c, rec in composed.records.items()
        if rec.decision.strategy not in absent_exempt and c not in data.columns
    ]
    if missing:
        names = ", ".join(f"`{c}`" for c in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise DataForgeError.from_code(
            "missing_fitted_columns",
            f"`frame`: lacks column(s) {names}{more} the imputer fills or leaves as it is.",
            columns=missing,
        )

    with warnings.catch_warnings():
        # A Dropped column the frame already lacks is dropped all the same.
        warnings.simplefilter("ignore", DroppedColumnAbsentWarning)
        out = composed.transform(data).dataframe

    resolved = _resolve_effective_nulls(
        data,
        numeric_sentinels=composed.numeric_sentinels,
        string_sentinels=composed.string_sentinels,
    )
    before = resolved.null_count()
    after = out.null_count()

    # Drop Candidates are reported, never applied (ADR-0106): each goes back
    # in its source place, sentinels read as missing like every other column.
    candidates = [
        c
        for c in data.columns
        if c in composed.records
        and composed.records[c].decision.strategy == ImputationStrategy.Dropped
        and composed.records[c].decision.reason == RoutingReason.drop_candidate
    ]
    if candidates:
        kept = set(out.columns) | set(candidates)
        order = [c for c in data.columns if c in kept]
        order += [c for c in out.columns if c not in data.columns]
        out = out.with_columns(resolved.select(candidates)).select(order)

    # Indicator columns stay in the frame, but Phases 3–6 skip them (ADR-0110).
    indicators = tuple(
        c
        for c, rec in composed.records.items()
        if rec.decision.strategy == ImputationStrategy.Indicator and c in out.columns
    )
    declarations = replace(
        declarations, soft_exclusions=_union(declarations.soft_exclusions, indicators)
    )

    handle = write_handle(
        "frame",
        size={"rows": out.height, "columns": out.width},
        payload_name="frame.parquet",
        write_payload=lambda tmp: out.write_parquet(tmp),
        obj=out,
        declarations=declarations,
        extra={"imputer": imputer, "source_frame": frame},
    )

    rows = []
    for column in ordered_columns({c: r.decision for c, r in composed.records.items()}):
        rec = composed.records[column]
        strategy = rec.decision.strategy
        n_missing = before[column].item() if column in before.columns else 0
        n_left = after[column].item() if column in after.columns else n_missing
        rows.append(
            {
                "column": column,
                "strategy": strategy.value,
                "missing": n_missing,
                "filled": 0 if strategy == ImputationStrategy.Dropped else n_missing - n_left,
                "fill_value": _cell(rec.fill_value),
            }
        )
    shown, _, omitted = page(
        rows, None, None, COLUMNS_CAP, field="columns", operation=None
    )
    drop_candidates = None
    if candidates:
        recipe_record = read_record(imputer_record["recipe"], "recipe", param="imputer")
        profile_record = read_record(recipe_record["profile"], "profile", param="imputer")
        listed, _, cut = page(
            candidates,
            None,
            None,
            COLUMNS_CAP,
            field="drop_candidates.columns",
            operation=None,
        )
        omitted += cut
        (path,) = FLAG_DIALS["missingness.drop_candidate"]
        drop_candidates = {
            "columns": listed,
            "dial": _Dials.from_record(profile_record).pointer(path),
        }
    return {
        "warnings": [],
        "omitted": omitted,
        "frame": handle,
        "imputer": imputer,
        "source_frame": frame,
        "rows": out.height,
        "columns": shown,
        "drop_candidates": drop_candidates,
    }
