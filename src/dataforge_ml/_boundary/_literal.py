"""The Literal Policy's ``deny`` side: withhold every Literal Field (ADR-0107).

Under ``deny`` the boundary rewrites each result once, after the Operation
returns and before either delivery prints it. Every Literal Field that holds
a value becomes ``null``; each path withheld gets one ``omitted`` entry with
reason ``literal_policy``; and the result gains one ``literal_policy`` Result
Warning. Fields that are not Literal Fields, such as counts, shares and the
declarations the Agent supplied, are left as they are.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..records import LiteralFieldMarker, Omission, OutputRecord, ResultWarning, Table


def _is_literal(record: type[BaseModel], name: str) -> bool:
    return any(isinstance(m, LiteralFieldMarker) for m in record.model_fields[name].metadata)


def withhold_literals(result: OutputRecord) -> OutputRecord:
    """Return ``result`` with every Literal Field nulled and the withholding reported.

    Parameters
    ----------
    result : OutputRecord
        An Operation's result, as it returned it.

    Returns
    -------
    OutputRecord
        The same record when it holds no literal value; else a copy whose
        Literal Fields are null, with one ``omitted`` entry per withheld path
        and one ``literal_policy`` warning appended.
    """
    withheld: dict[str, int] = {}

    def scrub_record(record: BaseModel, prefix: str) -> BaseModel:
        update: dict[str, Any] = {}
        for name in type(record).model_fields:
            value = getattr(record, name)
            path = f"{prefix}{name}"
            if _is_literal(type(record), name):
                if value is not None:
                    withheld[path] = withheld.get(path, 0) + 1
                    update[name] = None
                continue
            scrubbed = scrub(value, path)
            if scrubbed is not value:
                update[name] = scrubbed
        return record.model_copy(update=update) if update else record

    def scrub(value: Any, path: str) -> Any:
        if isinstance(value, BaseModel):
            return scrub_record(value, f"{path}.")
        if isinstance(value, (Table, list, tuple)):
            items = [scrub(item, f"{path}[]") for item in value]
            if all(new is old for new, old in zip(items, value)):
                return value
            return Table(items) if isinstance(value, Table) else type(value)(items)
        return value

    scrubbed = scrub_record(result, "")
    if not withheld:
        return result
    total = sum(withheld.values())
    return scrubbed.model_copy(
        update={
            "omitted": Table(
                [
                    *scrubbed.omitted,
                    *(
                        Omission(
                            field=path,
                            shown=0,
                            total=count,
                            operation=None,
                            reason="literal_policy",
                        )
                        for path, count in withheld.items()
                    ),
                ]
            ),
            "warnings": Table(
                [
                    *scrubbed.warnings,
                    ResultWarning(
                        code="literal_policy",
                        message=(
                            f"The Literal Policy is `deny`, so {total} Literal Field "
                            f"value(s) at {len(withheld)} path(s) are null. `omitted` "
                            "lists each path with reason `literal_policy`."
                        ),
                    ),
                ]
            ),
        }
    )
