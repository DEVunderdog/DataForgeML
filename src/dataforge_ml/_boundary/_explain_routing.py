"""Internal code behind the ``explain_routing`` Operation (ADR-0105).

It reads only the routing Handle: the stored routing, whose columns carry
their Routing Evidence, and the Handle record. It never reads the profile or
the frame, so it explains a routing made in any process.

The stored evidence is an ordered list of tagged facts whose order is fixed
by construction: entry causes, the Signal Score, the Signal Tier, then the
feasibility checks. The wire spells each kind as its own field, so every
list in the result is one record type in Table Layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..records import DataForgeError
from ._detail import page as _page
from ._detail import unknown_columns
from ._route import ordered_columns
from ._workspace import read_routing

#: Most columns one ``explain_routing`` page holds (ADR-0105).
EXPLAIN_CAP = 15


def _evidence_fields(evidence: Sequence[Any]) -> dict[str, Any]:
    entry: list[str] = []
    signal_score = None
    tier = None
    feasibility: list[dict[str, Any]] = []
    for fact in evidence:
        if fact.kind == "entry":
            entry.append(fact.cause.value)
        elif fact.kind == "signal_score":
            signal_score = fact.to_dict()
            del signal_score["kind"]
        elif fact.kind == "tier":
            tier = fact.tier.value
        elif fact.kind == "feasibility":
            feasibility.append(
                {
                    "strategy": fact.strategy,
                    "passed": fact.passed,
                    "failed_terms": [t.to_dict() for t in fact.failed_terms],
                }
            )
        else:  # a new fact kind needs its own wire field first
            raise DataForgeError.from_code(
                "unexpected", f"Routing Evidence fact kind {fact.kind!r} has no wire field."
            )
    return {
        "entry": entry,
        "signal_score": signal_score,
        "tier": tier,
        "feasibility": feasibility,
    }


def explain_routing_columns(
    *,
    routing: str,
    columns: Sequence[str] | None,
    offset: int | None,
    limit: int | None,
) -> dict[str, Any]:
    """Return one page of stored Routing Evidence, in ``route``'s column order.

    Parameters
    ----------
    routing : str
        The routing Handle.
    columns : sequence of str or None
        Only these columns; None means every routed column.
    offset, limit : int or None
        The page; None means 0 and :data:`EXPLAIN_CAP`.

    Returns
    -------
    dict
        The fields of the ``explain_routing`` result.

    Raises
    ------
    DataForgeError
        ``unknown_column`` naming every name the routing lacks, plus the
        routing Handle's own errors.
    """
    stored, record = read_routing(routing, param="routing")
    routings = stored.column_routings
    order = ordered_columns(routings)
    names = list(dict.fromkeys(columns or []))
    unknown = [(f"columns[{i}]", n) for i, n in enumerate(names) if n not in routings]
    if unknown:
        excluded = frozenset(record.get("declarations", {}).get("exclusions", ()))
        raise unknown_columns(unknown, pool=order, excluded=excluded, holder="routing")

    chosen = [c for c in order if not names or c in names]
    page, offset, omitted = _page(
        chosen,
        offset,
        limit,
        EXPLAIN_CAP,
        field="columns",
        operation="explain_routing",
    )
    return {
        "warnings": [],
        "omitted": omitted,
        "routing": routing,
        "profile": record["profile"],
        "total": len(chosen),
        "offset": offset,
        "columns": [
            {
                "column": c,
                "strategy": routings[c].strategy.value,
                "reason": routings[c].reason.value,
                **_evidence_fields(routings[c].evidence),
            }
            for c in page
        ],
    }
