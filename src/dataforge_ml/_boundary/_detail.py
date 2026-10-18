"""What every detail Operation shares: one page cut and one ``unknown_column`` error.

A detail Operation reads one stored Handle, filters, sorts by a stated rule,
then cuts a page with ``offset``/``limit`` (``limit`` at most the cap). When
more remain, ``omitted`` names the Operation that returns the rest
(ADR-0102). A name the Handle lacks is one ``unknown_column`` error naming
every name at fault.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from ..records import DataForgeError
from ._error_codes import close_matches


def page(
    items: list[Any],
    offset: int | None,
    limit: int | None,
    cap: int,
    *,
    field: str,
    operation: str,
) -> tuple[list[Any], int, list[dict[str, Any]]]:
    """Cut one page out of a filtered, sorted list; count it when more remain.

    Parameters
    ----------
    items : list
        Every matching item, already sorted.
    offset, limit : int or None
        The page; None means 0 and ``cap``.
    cap : int
        The Operation's item cap.
    field : str
        The result field the page fills, for the ``omitted`` entry.
    operation : str
        The Operation that returns the rest.

    Returns
    -------
    tuple of (list, int, list of dict)
        The page, the resolved offset, and the ``omitted`` entries.
    """
    offset = 0 if offset is None else offset
    limit = cap if limit is None else limit
    cut = items[offset : offset + limit]
    omitted = []
    if offset + len(cut) < len(items):  # more remain past this page
        omitted.append(
            {
                "field": field,
                "shown": len(cut),
                "total": len(items),
                "operation": operation,
            }
        )
    return cut, offset, omitted


def unknown_columns(
    unknown: Sequence[tuple[str, str]],
    *,
    pool: Sequence[str],
    excluded: Collection[str],
    holder: str,
    what: str | None = None,
) -> DataForgeError:
    """Return one ``unknown_column`` error naming every ``(param, name)`` at fault.

    Parameters
    ----------
    unknown : sequence of (str, str)
        Each argument path and the name it gave.
    pool : sequence of str
        The names the Handle holds, for ``candidates``.
    excluded : collection of str
        Excluded columns, which get their own problem text.
    holder : str
        What the Handle is, such as ``profile`` or ``routing``.
    what : str, optional
        What a name should have been; defaults to ``column in the <holder>``.

    Returns
    -------
    DataForgeError
        The ``unknown_column`` error, with ``candidates`` and ``issues``.
    """
    what = what or f"column in the {holder}"

    def problem(name: str) -> str:
        if name in excluded:
            return f"`{name}` was excluded, so the {holder} holds nothing on it"
        return f"no {what} `{name}`"

    first_param, first_name = unknown[0]
    if len(unknown) == 1:
        message = f"`{first_param}`: {problem(first_name)}."
    else:
        message = (
            f"{len(unknown)} unknown columns, first `{first_param}`: "
            f"{problem(first_name)}. See `issues` for all of them."
        )
    params = [p for p, _ in unknown]
    return DataForgeError.from_code(
        "unknown_column",
        message,
        param=first_param if len(params) == 1 else None,
        params=params,
        columns=list(dict.fromkeys(n for _, n in unknown)),
        candidates=close_matches([n for _, n in unknown], pool) or list(pool[:5]),
        issues=[{"param": p, "problem": problem(n)} for p, n in unknown],
    )
