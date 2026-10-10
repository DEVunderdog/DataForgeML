"""Internal code behind the ``export_frame`` Operation."""

from __future__ import annotations

from typing import Any

from ._workspace import _atomic_write, exports_dir, read_frame


def export_frame_csv(frame: str) -> dict[str, Any]:
    """Write a frame Handle's data as CSV into the Workspace's ``exports/``.

    The file is named after the Handle. A frame never changes, so a second
    export of the same Handle finds the file already there and writes nothing.
    Two processes exporting the same Handle at once each write a temporary
    file and rename it into place: the renamed file is always whole, and both
    hold the same bytes.

    Parameters
    ----------
    frame : str
        The frame Handle the Agent passed.

    Returns
    -------
    dict
        The fields of the ``export_frame`` result: ``path``, absolute.

    Raises
    ------
    DataForgeError
        As :func:`~._workspace.read_frame`.
    """
    data, _ = read_frame(frame, param="frame")
    path = exports_dir() / f"{frame}.csv"
    if not path.exists():
        _atomic_write(path, lambda tmp: data.write_csv(tmp))
    return {"path": str(path)}
