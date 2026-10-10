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
    """Load a Declared Input CSV into a new frame Handle.

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
        ``input_not_declared`` when ``path`` is not a Declared Input.
    """
    resolved = Path(path).resolve()
    if resolved not in declared_inputs():
        raise DataForgeError.from_code(
            "input_not_declared",
            "`path` is not a Declared Input, so it cannot be loaded.",
            param="path",
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
                "operation": "describe_columns",
            }
        )
    return {
        "frame": handle,
        "n_rows": frame.height,
        "n_columns": frame.width,
        "columns": columns,
        "omitted": omitted,
    }
