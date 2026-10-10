"""The Workspace: write-once Handles on disk, cached in memory (ADR-0103).

The Harness sets the Workspace directory and the Declared Inputs through
environment variables, outside the Agent's reach. Every Handle is a directory
``handles/<kind>_<id>/`` holding ``record.json`` and an optional payload. The
handle id is claimed with an atomic ``mkdir``, so two processes can never write
the same Handle, and ``record.json`` is renamed into place last, so a Handle
exists exactly when its record does. Nothing here ever deletes.

The library reads payloads only from inside its own Workspace: every path is
built from a validated handle id and checked to resolve under the root.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._serialization import FORMAT_SCHEMA_VERSION, library_version
from ..records import DataForgeError

#: Environment variable naming the Workspace directory.
WORKSPACE_ENV = "DATAFORGE_ML_WORKSPACE"
#: Environment variable listing Declared Input paths, ``os.pathsep``-separated.
DECLARED_INPUTS_ENV = "DATAFORGE_ML_DECLARED_INPUTS"

#: The closed list of Handle kinds (ADR-0110).
HANDLE_KINDS: tuple[str, ...] = (
    "frame",
    "profile",
    "routing",
    "recipe",
    "unit",
    "imputer",
    "c2st",
)

_ID_HEX = 8
_HANDLE_RE = re.compile(rf"^({'|'.join(HANDLE_KINDS)})_([0-9a-f]{{{_ID_HEX}}})$")
_RECORD = "record.json"
_CANDIDATES = 5


@dataclass(frozen=True)
class InheritedDeclarations:
    """The declarations a frame Handle carries down the Handle chain.

    Empty on a loaded frame. Later Operations may add to them, never remove
    them (ADR-0110).
    """

    targets: tuple[str, ...] = ()
    type_overrides: tuple[dict[str, Any], ...] = ()
    sentinels: tuple[dict[str, Any], ...] = ()
    exclusions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "type_overrides": list(self.type_overrides),
            "sentinels": list(self.sentinels),
            "exclusions": list(self.exclusions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InheritedDeclarations:
        return cls(
            targets=tuple(data["targets"]),
            type_overrides=tuple(data["type_overrides"]),
            sentinels=tuple(data["sentinels"]),
            exclusions=tuple(data["exclusions"]),
        )


@dataclass(frozen=True)
class CallContext:
    """The Operation call in progress, from which every Handle's Lineage is taken."""

    operation: str
    arguments: dict[str, Any]
    inputs: list[str] = field(default_factory=list)


_CALL: ContextVar[CallContext | None] = ContextVar("dataforge_ml_call", default=None)

# In-memory cache of loaded objects, keyed by (workspace root, handle).
_CACHE: dict[tuple[str, str], Any] = {}


def current_call() -> CallContext:
    """Return the Operation call in progress.

    Raises
    ------
    RuntimeError
        If no Operation is running; Handles are written only by Operations.
    """
    call = _CALL.get()
    if call is None:
        raise RuntimeError("Handles are written only inside an Operation call")
    return call


def workspace_root() -> Path:
    """Return the Harness-set Workspace directory, creating ``handles/`` in it.

    Raises
    ------
    DataForgeError
        ``workspace_not_set`` when the Harness set no Workspace.
    """
    raw = os.environ.get(WORKSPACE_ENV)
    if not raw:
        raise DataForgeError.from_code(
            "workspace_not_set",
            f"No Workspace is set: the Harness must set {WORKSPACE_ENV}.",
        )
    root = Path(raw).resolve()
    (root / "handles").mkdir(parents=True, exist_ok=True)
    return root


def declared_inputs() -> tuple[Path, ...]:
    """Return the Harness-declared input paths, resolved."""
    raw = os.environ.get(DECLARED_INPUTS_ENV, "")
    return tuple(Path(p).resolve() for p in raw.split(os.pathsep) if p)


def _handle_dir(root: Path, handle: str) -> Path:
    # The id is validated against _HANDLE_RE before it reaches here, and the
    # resolved path is checked to sit under the Workspace: the trust boundary
    # is that nothing outside the Workspace is ever read as a payload.
    path = (root / "handles" / handle).resolve()
    if path.parent != (root / "handles").resolve():
        raise RuntimeError(f"Handle path escapes the Workspace: {handle!r}")
    return path


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_handle(
    kind: str,
    *,
    size: dict[str, int],
    payload_name: str | None = None,
    write_payload: Callable[[Path], None] | None = None,
    obj: Any = None,
    declarations: InheritedDeclarations | None = None,
) -> str:
    """Write a new Handle with the current call's Lineage and return it.

    Parameters
    ----------
    kind : str
        One of :data:`HANDLE_KINDS`.
    size : dict
        A short size summary, such as ``{"rows": 10, "columns": 3}``.
    payload_name : str, optional
        File name of the payload inside the Handle directory.
    write_payload : callable, optional
        Writes the payload to the given temporary path.
    obj : object, optional
        The live object, cached in memory for later reads in this process.
    declarations : InheritedDeclarations, optional
        Inherited Declarations, for a frame Handle.

    Returns
    -------
    str
        The new Handle, such as ``frame_7f3a9c2e``.
    """
    if kind not in HANDLE_KINDS:
        raise ValueError(f"unknown Handle kind {kind!r}")
    call = current_call()
    root = workspace_root()
    while True:
        handle = f"{kind}_{secrets.token_hex(_ID_HEX // 2)}"
        directory = _handle_dir(root, handle)
        try:
            directory.mkdir()
            break
        except FileExistsError:
            continue

    if write_payload is not None:
        assert payload_name is not None
        _atomic_write(directory / payload_name, write_payload)

    record = {
        "handle": handle,
        "kind": kind,
        "format_schema_version": FORMAT_SCHEMA_VERSION,
        "library_version": library_version(),
        "lineage": {
            "operation": call.operation,
            "arguments": call.arguments,
            "inputs": list(call.inputs),
        },
        "size": size,
        "payload": payload_name,
    }
    if declarations is not None:
        record["declarations"] = declarations.to_dict()

    def _write_record(tmp: Path) -> None:
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")

    _atomic_write(directory / _RECORD, _write_record)
    if obj is not None:
        _CACHE[(str(root), handle)] = obj
    return handle


def _existing(root: Path, kind: str) -> list[str]:
    """Existing Handles of ``kind``, newest first."""
    found = []
    for directory in (root / "handles").iterdir():
        record = directory / _RECORD
        if directory.name.startswith(f"{kind}_") and record.exists():
            found.append((record.stat().st_mtime_ns, directory.name))
    return [name for _, name in sorted(found, reverse=True)]


def read_record(handle: str, kind: str, *, param: str) -> dict[str, Any]:
    """Read a Handle's record, checking its kind, existence and format.

    Parameters
    ----------
    handle : str
        The Handle string the Agent passed.
    kind : str
        The kind the argument expects.
    param : str
        The argument path, for the error record.

    Returns
    -------
    dict
        The Handle's stored record, Lineage included.

    Raises
    ------
    DataForgeError
        ``wrong_handle_kind`` when the prefix names another kind,
        ``unknown_handle`` when no such Handle exists in the Workspace, and
        ``incompatible_artifact`` when another format version wrote it.
    """
    root = workspace_root()
    given_kind = handle.split("_", 1)[0] if "_" in handle else None
    if given_kind in HANDLE_KINDS and given_kind != kind:
        raise DataForgeError.from_code(
            "wrong_handle_kind",
            f"`{param}` takes a `{kind}` Handle but got a `{given_kind}` Handle.",
            param=param,
            allowed=[kind],
        )
    record_path = None
    if _HANDLE_RE.match(handle):
        record_path = _handle_dir(root, handle) / _RECORD
    if record_path is None or not record_path.exists():
        raise DataForgeError.from_code(
            "unknown_handle",
            f"`{param}` names no `{kind}` Handle in this Workspace.",
            param=param,
            candidates=_existing(root, kind)[:_CANDIDATES],
        )
    try:
        record = json.loads(record_path.read_text("utf-8"))
    except ValueError:
        record = {}
    if record.get("format_schema_version") != FORMAT_SCHEMA_VERSION:
        raise DataForgeError.from_code(
            "incompatible_artifact",
            f"`{param}` was written by a different Workspace format version "
            f"and cannot be read.",
            param=param,
        )
    return record


def payload_path(handle: str, record: dict[str, Any]) -> Path:
    """Return the path of a Handle's payload, inside the Workspace."""
    return _handle_dir(workspace_root(), handle) / record["payload"]


def cached(handle: str) -> Any:
    """Return the in-memory object for ``handle``, or None."""
    return _CACHE.get((str(workspace_root()), handle))


def cache(handle: str, obj: Any) -> None:
    """Cache a loaded object for later reads in this process."""
    _CACHE[(str(workspace_root()), handle)] = obj


def read_frame(handle: str, *, param: str) -> tuple[Any, InheritedDeclarations]:
    """Read a frame Handle and record it as an input of the current call.

    Parameters
    ----------
    handle : str
        The frame Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    tuple of (polars.DataFrame, InheritedDeclarations)
        The frame, exactly as loaded, and the declarations it carries.

    Raises
    ------
    DataForgeError
        As :func:`read_record`.
    """
    import polars as pl

    record = read_record(handle, "frame", param=param)
    call = _CALL.get()
    if call is not None and handle not in call.inputs:
        call.inputs.append(handle)
    frame = cached(handle)
    if frame is None:
        frame = pl.read_parquet(payload_path(handle, record))
        cache(handle, frame)
    return frame, InheritedDeclarations.from_dict(record["declarations"])
