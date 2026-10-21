"""The Workspace: write-once Handles on disk, cached in memory (ADR-0103).

The Workspace directory and the Declared Inputs are Workspace Settings
(``_settings``): ``.dataforge/`` and none by default. Nothing is created until
the first write. Every Handle is a directory
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
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._serialization import FORMAT_SCHEMA_VERSION, library_version
from ..records import DataForgeError
from ._settings import WorkspaceSettings, resolve_settings

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
    them (ADR-0110). ``soft_exclusions`` are the columns Phases 3–6 skip but
    keep: the indicator columns ``impute`` adds.
    """

    targets: tuple[str, ...] = ()
    type_overrides: tuple[dict[str, Any], ...] = ()
    sentinels: tuple[dict[str, Any], ...] = ()
    exclusions: tuple[str, ...] = ()
    soft_exclusions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "type_overrides": list(self.type_overrides),
            "sentinels": list(self.sentinels),
            "exclusions": list(self.exclusions),
            "soft_exclusions": list(self.soft_exclusions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InheritedDeclarations:
        return cls(
            targets=tuple(data["targets"]),
            type_overrides=tuple(data["type_overrides"]),
            sentinels=tuple(data["sentinels"]),
            exclusions=tuple(data["exclusions"]),
            soft_exclusions=tuple(data["soft_exclusions"]),
        )


@dataclass(frozen=True)
class CallContext:
    """The Operation call in progress, from which every Handle's Lineage is taken."""

    operation: str
    arguments: dict[str, Any]
    settings: WorkspaceSettings
    inputs: list[str] = field(default_factory=list)


_CALL: ContextVar[CallContext | None] = ContextVar("dataforge_ml_call", default=None)

# In-memory cache of loaded objects, keyed by (workspace root, handle).
_CACHE: dict[tuple[str, str], Any] = {}


def current_call() -> CallContext:
    """Return the Operation call in progress.

    Raises
    ------
    Unexpected
        ``unexpected`` if no Operation is running; Handles are written only by
        Operations.
    """
    call = _CALL.get()
    if call is None:
        raise DataForgeError.from_code(
            "unexpected", "Handles are written only inside an Operation call."
        )
    return call


def settings() -> WorkspaceSettings:
    """Return the Workspace Settings: the running Operation's, or resolved now.

    Raises
    ------
    DataForgeError
        ``invalid_setting`` as :func:`~._settings.resolve_settings`.
    """
    call = _CALL.get()
    return call.settings if call is not None else resolve_settings()


def workspace_root() -> Path:
    """Return the Workspace directory. It may not exist yet: reads never create it."""
    return settings().workspace


def exports_dir() -> Path:
    """Return the Workspace's ``exports/`` folder, creating it (and the Workspace) if needed."""
    directory = workspace_root() / "exports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def declared_inputs() -> tuple[Path, ...]:
    """Return the Declared Input paths, resolved. Empty means any path may be loaded."""
    return settings().declared_inputs


def _handle_dir(root: Path, handle: str) -> Path:
    # The id is validated against _HANDLE_RE before it reaches here, and the
    # resolved path is checked to sit under the Workspace: the trust boundary
    # is that nothing outside the Workspace is ever read as a payload.
    path = (root / "handles" / handle).resolve()
    if path.parent != (root / "handles").resolve():
        raise DataForgeError.from_code(
            "unexpected", f"Handle path escapes the Workspace: {handle!r}."
        )
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
    extra: dict[str, Any] | None = None,
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
        Inherited Declarations: a frame's own, or those a profile records
        for every frame made from it.
    extra : dict, optional
        Kind-specific JSON fields stored on the record, such as the frame a
        profile Handle carries. They may not replace a standard field.

    Returns
    -------
    str
        The new Handle, such as ``frame_7f3a9c2e``.
    """
    if kind not in HANDLE_KINDS:
        raise DataForgeError.from_code("unexpected", f"Unknown Handle kind {kind!r}.")
    call = current_call()
    root = workspace_root()
    (root / "handles").mkdir(parents=True, exist_ok=True)
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
        "created_ns": time.time_ns(),
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
    for key, value in (extra or {}).items():
        if key in record:
            raise DataForgeError.from_code(
                "unexpected", f"Handle record field {key!r} is reserved."
            )
        record[key] = value

    def _write_record(tmp: Path) -> None:
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")

    _atomic_write(directory / _RECORD, _write_record)
    if obj is not None:
        _CACHE[(str(root), handle)] = obj
    return handle


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class HandleListing:
    """Every readable Handle of the asked kinds, newest first.

    ``unreadable`` counts Handles whose record is damaged or was written by
    another format version: they exist on disk but no Operation can read them.
    """

    records: list[dict[str, Any]]
    unreadable: int


def list_records(kind: str | None = None) -> HandleListing:
    """List the Workspace's Handle records, newest first.

    A Handle whose directory exists but whose record is not yet in place is
    still being written by another process and is skipped. Order is creation
    time, then the Handle string, so it is the same in every process.

    Parameters
    ----------
    kind : str, optional
        One of :data:`HANDLE_KINDS`; None lists every kind.

    Returns
    -------
    HandleListing
        The records, Lineage included, and the count of unreadable ones.

    """
    handles = workspace_root() / "handles"
    found: list[tuple[int, str, dict[str, Any]]] = []
    unreadable = 0
    for directory in handles.iterdir() if handles.is_dir() else ():
        match = _HANDLE_RE.match(directory.name)
        if match is None or (kind is not None and match.group(1) != kind):
            continue
        path = directory / _RECORD
        if not path.exists():
            continue
        record = _read_json(path)
        if (
            record is None
            or record.get("format_schema_version") != FORMAT_SCHEMA_VERSION
        ):
            unreadable += 1
            continue
        created = record.get("created_ns")
        if not isinstance(created, int):
            created = path.stat().st_mtime_ns
        found.append((created, directory.name, record))
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return HandleListing(
        records=[record for _, _, record in found], unreadable=unreadable
    )


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
            candidates=[r["handle"] for r in list_records(kind).records[:_CANDIDATES]],
        )
    record = _read_json(record_path) or {}
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


def _record_input(handle: str) -> None:
    # A Handle read during an Operation call joins that call's Lineage inputs, once.
    call = _CALL.get()
    if call is not None and handle not in call.inputs:
        call.inputs.append(handle)


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
    _record_input(handle)
    frame = cached(handle)
    if frame is None:
        frame = pl.read_parquet(payload_path(handle, record))
        cache(handle, frame)
    return frame, InheritedDeclarations.from_dict(record["declarations"])


def _read_serialized(handle: str, kind: str, *, param: str) -> tuple[Any, dict[str, Any]]:
    # One serialized library object, recorded as an input of the current call.
    from .._serialization import deserialize

    record = read_record(handle, kind, param=param)
    _record_input(handle)
    obj = cached(handle)
    if obj is None:
        obj = deserialize(payload_path(handle, record).read_bytes())
        cache(handle, obj)
    return obj, record


def read_profile(handle: str, *, param: str) -> tuple[Any, dict[str, Any]]:
    """Read a profile Handle and record it as an input of the current call.

    Parameters
    ----------
    handle : str
        The profile Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    tuple of (StructuralProfileResult, dict)
        The stored profile and its Handle record, which holds the column
        order, the resolved dials and the declarations.

    Raises
    ------
    DataForgeError
        As :func:`read_record`.
    """
    return _read_serialized(handle, "profile", param=param)


def read_routing(handle: str, *, param: str) -> tuple[Any, dict[str, Any]]:
    """Read a routing Handle and record it as an input of the current call.

    Parameters
    ----------
    handle : str
        The routing Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    tuple of (ImputationRouting, dict)
        The stored routing, with every column's Routing Evidence, and its
        Handle record, which holds the profile it was routed from and the
        declarations.

    Raises
    ------
    DataForgeError
        As :func:`read_record`.
    """
    return _read_serialized(handle, "routing", param=param)


def read_recipe(handle: str, *, param: str) -> tuple[Any, dict[str, Any]]:
    """Read a recipe Handle and record it as an input of the current call.

    Parameters
    ----------
    handle : str
        The recipe Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    tuple of (ImputationRecipe, dict)
        The stored recipe, with its own copy of the routing, and its Handle
        record, which holds the routing, profile and frame it was resolved
        from, and the declarations.

    Raises
    ------
    DataForgeError
        As :func:`read_record`.
    """
    return _read_serialized(handle, "recipe", param=param)


def read_unit_record(handle: str, *, param: str) -> dict[str, Any]:
    """Read a unit Handle's record, not its payload, and record it as an input.

    Parameters
    ----------
    handle : str
        The unit Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    dict
        The unit's Handle record, which holds the recipe, frame, unit id and
        strategy it was fitted from.

    Raises
    ------
    DataForgeError
        As :func:`read_record`.
    """
    record = read_record(handle, "unit", param=param)
    _record_input(handle)
    return record


def read_imputer(handle: str, *, param: str) -> tuple[Any, dict[str, Any]]:
    """Read an imputer Handle, composing its recipe and units again (ADR-0103).

    The imputer Handle is a manifest: nothing aggregate is stored, so the
    imputer is composed from the recipe and unit Handles it names, once per
    process.

    Parameters
    ----------
    handle : str
        The imputer Handle the Agent passed.
    param : str
        The argument path, for error records.

    Returns
    -------
    tuple of (FittedImputer, dict)
        The composed imputer and its Handle record, which holds the recipe
        and the unit Handles by unit id.

    Raises
    ------
    DataForgeError
        As :func:`read_record`, for the imputer and every Handle it names.
    """
    from ..imputation._fitted_imputer import FittedImputer

    record = read_record(handle, "imputer", param=param)
    _record_input(handle)
    imputer = cached(handle)
    if imputer is None:
        recipe, _ = _read_serialized(record["recipe"], "recipe", param=param)
        units = {
            entry["unit_id"]: _read_serialized(entry["unit"], "unit", param=param)[0]
            for entry in record["units"]
        }
        imputer = FittedImputer.compose(recipe, units)
        cache(handle, imputer)
    return imputer, record
