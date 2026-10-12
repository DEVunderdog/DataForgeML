"""The ``export_frame`` Operation: write a frame Handle out as CSV."""

from __future__ import annotations

from .._boundary._export_frame import export_frame_csv
from .._boundary._operation import operation
from ..records import OutputRecord


class ExportFrameResult(OutputRecord):
    """Where the exported CSV is."""

    path: str
    """Absolute path of the CSV, inside the Workspace's ``exports/`` folder."""


@operation(effect="export", phase="workspace", scales_with_data=True)
def export_frame(frame: str) -> ExportFrameResult:
    """Write a frame Handle's data as a CSV file in the Workspace and return its path.

    Use when
    --------
    Call this to hand a frame on as a file: a split fold, an imputed frame,
    or a loaded one. The file goes into the Workspace's ``exports/`` folder,
    named after the Handle; you cannot choose where it goes. Exporting the
    same Handle again returns the same path and writes nothing, because a
    frame never changes. A CSV keeps no dtypes, so reading it back may infer
    types again. Exporting never changes the Handle: to change the data,
    change the declarations and call the Operation that made it again.

    Parameters
    ----------
    frame : str
        A frame Handle, such as ``frame_7f3a9c2e``.

    Returns
    -------
    ExportFrameResult
        ``path``: the absolute path of the CSV, with a header row.

    Errors
    ------
    unknown_handle
        ``frame`` names no frame Handle in this Workspace. ``candidates``
        lists up to 5 frame Handles, newest first.
    wrong_handle_kind
        ``frame`` is a Handle of another kind. ``allowed`` names ``frame``.
    incompatible_artifact
        Another library format version wrote ``frame``. Load it again.
    invalid_argument
        ``frame`` is missing or not a string.
    invalid_setting
        A Workspace Setting is invalid. The message names it and its source.

    See Also
    --------
    list_handles : Find a frame Handle to export.
    load : Make a frame Handle from a CSV file.

    Example
    -------
    >>> from dataforge_ml import export_frame, load
    >>> frame = load(path="train.csv").frame
    >>> result = export_frame(frame=frame)
    >>> result.path.endswith(f"exports/{frame}.csv")
    True
    >>> open(result.path).readline().strip()
    'age,city,income'
    """
    return ExportFrameResult.model_validate(export_frame_csv(frame))
