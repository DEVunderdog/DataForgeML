"""Internal code behind the ``list_handles`` Operation."""

from __future__ import annotations

from typing import Any

from ..records import ResultWarning
from ._error_codes import entry_for
from ._operation import canonical_json
from ._workspace import list_records

#: Most Handles one ``list_handles`` page holds, and the default page size.
PAGE_CAP = 20
#: Most Lineage inputs listed per Handle; the rest are counted in ``omitted``.
INPUTS_CAP = 10
#: Longest argument value, as JSON text, before it is cut.
ARGUMENT_VALUE_CAP = 80


def _argument(name: str, value: Any) -> dict[str, Any]:
    text = canonical_json(value)
    cut = len(text) > ARGUMENT_VALUE_CAP
    if cut:
        text = text[: ARGUMENT_VALUE_CAP - 1] + "…"
    return {"name": name, "value": text, "value_cut": cut}


def list_workspace_handles(
    kind: str | None, offset: int | None, limit: int | None
) -> dict[str, Any]:
    """Return one page of the Workspace's Handles, newest first.

    Parameters
    ----------
    kind : str or None
        A Handle kind, or None for every kind.
    offset : int or None
        Handles to skip from the newest; None means 0.
    limit : int or None
        Page size, at most :data:`PAGE_CAP`; None means the cap.

    Returns
    -------
    dict
        The fields of the ``list_handles`` result.
    """
    offset = 0 if offset is None else offset
    limit = PAGE_CAP if limit is None else limit
    listing = list_records(kind)
    page = listing.records[offset : offset + limit]

    handles: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for index, record in enumerate(page):
        lineage = record["lineage"]
        inputs = list(lineage["inputs"])
        if len(inputs) > INPUTS_CAP:
            omitted.append(
                {
                    "field": f"handles[{index}].inputs",
                    "shown": INPUTS_CAP,
                    "total": len(inputs),
                    "operation": None,
                }
            )
            inputs = inputs[:INPUTS_CAP]
        handles.append(
            {
                "handle": record["handle"],
                "kind": record["kind"],
                "operation": lineage["operation"],
                "arguments": [_argument(k, v) for k, v in lineage["arguments"].items()],
                "inputs": inputs,
                "size": [{"measure": k, "value": v} for k, v in record["size"].items()],
            }
        )

    total = len(listing.records)
    if offset + len(page) < total:
        omitted.insert(
            0,
            {
                "field": "handles",
                "shown": len(page),
                "total": total,
                "operation": "list_handles",
            },
        )
    warnings = []
    if listing.unreadable:
        warnings.append(
            ResultWarning(
                code="handles_unreadable",
                message=(
                    f"{listing.unreadable} Handle(s) in this Workspace could not "
                    f"be read and are not listed."
                ),
                hint=entry_for("handles_unreadable").hint,
            )
        )
    return {
        "total": total,
        "offset": offset,
        "handles": handles,
        "warnings": warnings,
        "omitted": omitted,
    }
