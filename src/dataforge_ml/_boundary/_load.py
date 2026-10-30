"""Internal code behind the ``load`` Operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..records import DataForgeError
from ..utils.data_loader import _load_csv
from ._workspace import InheritedDeclarations, declared_inputs, write_handle

#: Most columns ``load`` lists by name; the rest are counted in ``omitted``.
COLUMN_CAP = 100


def load_declared_csv(path: str) -> dict[str, Any]:
    """Load a CSV into a new frame Handle, if the Declared Inputs allow it.

    Declared Inputs are opt-in (ADR-0112): with none declared, any path loads.

    Parameters
    ----------
    path : str
        The path the Agent passed, compared to the Declared Inputs after
        resolving symlinks and relative segments.

    Returns
    -------
    dict
        The fields of the ``load`` result: ``frame``, ``n_rows``,
        ``n_columns``, ``columns`` (at most :data:`COLUMN_CAP`, in header
        order) and ``omitted``.

    Raises
    ------
    DataForgeError
        ``input_not_declared`` when Declared Inputs are listed and ``path`` is
        not one of them; ``input_not_found`` when no file is at ``path``.
    """
    resolved = Path(path).resolve()
    declared = declared_inputs()
    if declared and resolved not in declared:
        raise DataForgeError.from_code(
            "input_not_declared",
            "`path` is not a Declared Input, so it cannot be loaded.",
            param="path",
        )
    if not resolved.is_file():
        raise DataForgeError.from_code(
            "input_not_found", "No file exists at `path`.", param="path"
        )
    frame = _load_csv(resolved.read_bytes())
    handle = write_handle(
        "frame",
        size={"rows": frame.height, "columns": frame.width},
        payload_name="frame.parquet",
        write_payload=lambda tmp: frame.write_parquet(tmp),
        obj=frame,
        declarations=InheritedDeclarations(),
    )
    columns = [
        {"column": name, "dtype": str(dtype)}
        for name, dtype in list(frame.schema.items())[:COLUMN_CAP]
    ]
    omitted = []
    if frame.width > COLUMN_CAP:
        omitted.append(
            {
                "field": "columns",
                "shown": COLUMN_CAP,
                "total": frame.width,
                # describe_columns (#594) will return the rest; until it
                # lands, no command does.
                "operation": None,
            }
        )
    return {
        "frame": handle,
        "n_rows": frame.height,
        "n_columns": frame.width,
        "columns": columns,
        "omitted": omitted,
    }
