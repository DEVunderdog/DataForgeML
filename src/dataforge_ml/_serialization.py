"""Bare-bytes persistence: ``serialize`` / ``deserialize`` / ``inspect`` (ADR-0072, ADR-0096).

Persistence is the user's plain business — "give me bytes, take bytes back". A
persistable object (a fitted unit, an imputation routing, an imputation recipe,
or a structural profile) crosses the boundary as a single opaque ``bytes`` value
the user stores wherever they like; there is no store framework, no
content-addressed key, and no data-fingerprint validation (all retired with
cross-process resume, ADR-0071 / ADR-0072).

Wire format
-----------
Every serialized object is a **JSON header line** — a compact, newline-free JSON
object carrying a ``kind`` tag, the format-schema version, and the installed
``library_version`` — optionally followed by ``b"\\n"`` and an **opaque joblib
payload**. Only the :class:`~dataforge_ml.FittedUnit` has a payload (its learned
state); a routing, a recipe, and a profile are pure data structures whose whole
``to_dict()`` rides inside the header under ``"data"``. Because ``json.dumps``
escapes newlines inside strings, the first raw ``b"\\n"`` byte unambiguously ends
the header, which is what lets :func:`inspect` read provenance and compatibility
**without** unpickling the payload.

Trust boundary (ADR-0063, unchanged)
------------------------------------
Reconstructing a fitted unit unpickles its joblib payload, which is **arbitrary
code execution the moment the bytes are loaded** — so :func:`deserialize` is only
safe on bytes you produced or otherwise trust. The payload SHA-256 verified
before unpickling is an **integrity** guard (accidental corruption, tampering in
transit), never an **authenticity** guard: an attacker who rewrites the payload
can also rewrite the recorded checksum. Cryptographic signing is the authenticity
tier and is deferred to a future service.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import joblib

from .imputation import ImputationRecipe, ImputationRouting
from .profiling import StructuralProfileResult

# Bumped only when the JSON layout of a stamped document changes in a way
# that breaks reading older documents.
FORMAT_SCHEMA_VERSION = 2

# The ``kind`` tags a serialized envelope carries so :func:`deserialize`
# dispatches to the right object type without the caller declaring it.
KIND_FITTED_UNIT = "fitted_unit"
KIND_ROUTING = "routing"
KIND_RECIPE = "recipe"
KIND_PROFILE = "profile"


class ArtifactPythonVersionWarning(UserWarning):
    """Emitted when an artifact loads under a different Python version.

    Signals the one lenient branch of the ADR-0063 load policy: the artifact
    was saved under a Python version other than the one running now. CPython's
    pickle protocol is stable across minor versions, so the load proceeds — this
    warning is the only signal. Unlike a scikit-learn / numpy / joblib mismatch
    (which refuses with :class:`IncompatibleArtifactError`), a Python drift is
    tolerated. Suppress with
    ``warnings.filterwarnings("ignore", category=ArtifactPythonVersionWarning)``.
    """


class IncompatibleArtifactError(Exception):
    """Raised when a stored artifact cannot be *reconstructed* safely.

    Signals a reconstruction/correctness failure (ADR-0063): the saved bytes
    cannot be rebuilt faithfully under the format schema and libraries
    installed right now — a newer format-schema major, a mismatched
    DataForgeML / scikit-learn / numpy / joblib version, or a payload whose
    SHA-256 checksum does not match its header. With content-addressed identity
    and load-for-reuse validation retired (ADR-0072), this is the only load-time
    failure the library raises; relevance is no longer checked at deserialize.
    """


def library_version() -> str:
    """Return the installed DataForgeML version, or a dev placeholder.

    Returns
    -------
    str
        The ``dataforge-ml`` distribution version, or ``"0.0.0-dev"`` when
        the package metadata is unavailable (e.g. running from a source
        checkout with no installed distribution).
    """
    try:
        return version("dataforge-ml")
    except PackageNotFoundError:
        return "0.0.0-dev"


def _distribution_version(name: str) -> str:
    """Return an installed distribution's version, or ``"unknown"``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def produced_with() -> dict:
    """Capture the six-version reconstruction stamp for a fitted artifact.

    The exact facts needed to judge, at load time, whether a joblib payload
    will unpickle faithfully (ADR-0063): the format-schema version plus the
    installed DataForgeML, scikit-learn, numpy, joblib, and Python versions.
    Captured at serialize time so the artifact is self-describing for diagnosis.

    Returns
    -------
    dict
        Mapping with keys ``format_schema_version``, ``dataforge_ml``,
        ``scikit_learn``, ``numpy``, ``joblib``, and ``python``.
    """
    return {
        "format_schema_version": FORMAT_SCHEMA_VERSION,
        "dataforge_ml": library_version(),
        "scikit_learn": _distribution_version("scikit-learn"),
        "numpy": _distribution_version("numpy"),
        "joblib": _distribution_version("joblib"),
        "python": platform.python_version(),
    }


def check_reconstructable(stamp_block: dict) -> None:
    """Refuse to reconstruct an artifact stamped under an incompatible env.

    Enforces the asymmetric load policy of ADR-0063: a newer format-schema
    major, or any mismatch of DataForgeML / scikit-learn / numpy / joblib
    against the versions installed now, refuses loudly; a Python version drift
    is tolerated but *warns* (pickle protocol is stable across CPython minors),
    and an ``"unknown"`` stamped value is treated as "cannot verify → allow" so
    a source checkout can still round-trip its own artifacts within one process.

    Parameters
    ----------
    stamp_block : dict
        A ``produced_with`` block captured by :func:`produced_with`.

    Warns
    -----
    ArtifactPythonVersionWarning
        If the artifact was produced under a different Python version than the
        one running now. The load still proceeds.

    Raises
    ------
    IncompatibleArtifactError
        If the format-schema major exceeds this code's, or any
        reconstruction-critical dependency version differs from the installed
        one.
    """
    saved_schema = stamp_block.get("format_schema_version", 0)
    if saved_schema < FORMAT_SCHEMA_VERSION:
        raise IncompatibleArtifactError(
            f"Artifact format-schema version {saved_schema} is obsolete "
            f"(version {FORMAT_SCHEMA_VERSION} is required); profile, route, "
            f"resolve, or fit again."
        )
    if saved_schema > FORMAT_SCHEMA_VERSION:
        raise IncompatibleArtifactError(
            f"Artifact format-schema version {saved_schema} is newer than the "
            f"maximum this library supports ({FORMAT_SCHEMA_VERSION}); it cannot "
            f"be read. Regenerate the artifact with this DataForgeML version."
        )

    current = produced_with()
    strict = ("dataforge_ml", "scikit_learn", "numpy", "joblib")
    for key in strict:
        saved = stamp_block.get(key, "unknown")
        if saved in ("unknown", current[key]):
            continue
        raise IncompatibleArtifactError(
            f"Artifact was produced with {key}=={saved} but {current[key]} is "
            f"installed. A cross-version load risks silent numerical drift and "
            f"is refused; retrain the unit under the current environment."
        )

    saved_python = stamp_block.get("python", "unknown")
    if saved_python not in ("unknown", current["python"]):
        warnings.warn(
            f"Artifact was produced with Python {saved_python} but "
            f"{current['python']} is running. CPython's pickle protocol is "
            f"stable across minor versions, so the load proceeds; retrain the "
            f"unit if you observe anomalies.",
            ArtifactPythonVersionWarning,
            stacklevel=2,
        )


def _encode_envelope(header: dict, payload: bytes | None = None) -> bytes:
    """Frame a JSON header (and optional opaque payload) into wire bytes."""
    line = json.dumps(header).encode("utf-8")
    if payload is None:
        return line
    return line + b"\n" + payload


def _decode_envelope(data: bytes) -> tuple[dict, bytes]:
    """Split wire bytes into their JSON header and trailing opaque payload."""
    newline = data.find(b"\n")
    if newline == -1:
        return json.loads(data.decode("utf-8")), b""
    return json.loads(data[:newline].decode("utf-8")), data[newline + 1 :]


def _dumps(unit: Any) -> bytes:
    """Serialise a fitted unit's full learned state to opaque joblib bytes."""
    buf = io.BytesIO()
    joblib.dump(unit, buf)
    return buf.getvalue()


def _loads(payload: bytes) -> Any:
    """Reconstruct a fitted unit from its opaque joblib bytes."""
    return joblib.load(io.BytesIO(payload))


def _encode_fitted_unit(unit: Any) -> bytes:
    """Encode a fitted unit as a JSON header plus an opaque joblib payload.

    The header carries the ``produced_with`` reconstruction stamp and the
    payload's SHA-256 so :func:`_decode_fitted_unit` can gate the unpickle, and
    readable metadata (``unit_type``, ``columns``) for :func:`inspect`.
    """
    payload = _dumps(unit)
    header = {
        "kind": KIND_FITTED_UNIT,
        "format_schema_version": FORMAT_SCHEMA_VERSION,
        "library_version": library_version(),
        "produced_with": produced_with(),
        "unit_type": type(unit).__name__,
        "columns": list(unit.target_columns),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return _encode_envelope(header, payload)


def _decode_fitted_unit(header: dict, payload: bytes) -> Any:
    """Reconstruct a fitted unit from an envelope header and its joblib payload.

    Runs the ADR-0063 reconstruction gates before trusting any bytes: the
    header's ``produced_with`` stamp is checked against the installed
    environment, and the payload's SHA-256 is verified against the checksum
    recorded in the header. The reconstructed unit's ``transform`` is
    bit-identical to the original's on equal input.

    Reconstruction unpickles the payload, which executes **arbitrary code** —
    only decode bytes you produced or trust. The checksum verified first is an
    integrity guard (corruption, tampering-in-transit), not an authenticity
    guard; see the module-level trust-boundary note.
    """
    check_reconstructable(header.get("produced_with", {}))

    actual = hashlib.sha256(payload).hexdigest()
    expected = header.get("payload_sha256")
    if actual != expected:
        raise IncompatibleArtifactError(
            f"Payload checksum mismatch for unit "
            f"'{header.get('unit_type', 'unknown')}': header recorded {expected} "
            f"but the payload hashes to {actual}. The artifact is corrupt or was "
            f"tampered with; it will not be loaded."
        )
    return _loads(payload)


def serialize(obj: Any) -> bytes:
    """Serialize a fitted unit, an imputation routing, a recipe, or a profile to bytes.

    The single write door of the persistence boundary (ADR-0072, ADR-0096). It dispatches
    on the object's type and produces a self-describing JSON-header envelope the
    user stores anywhere; a fitted unit additionally carries an opaque joblib
    payload holding its learned state.

    Parameters
    ----------
    obj : FittedUnit or ImputationRouting or ImputationRecipe or StructuralProfileResult
        The object to persist. Any object satisfying the fitted-unit contract
        (a ``target_columns`` property) is serialized through the unit path.

    Returns
    -------
    bytes
        The serialized envelope, consumed by :func:`deserialize` and
        :func:`inspect`.
    """
    if isinstance(obj, ImputationRouting):
        return _encode_envelope(
            {
                "kind": KIND_ROUTING,
                "format_schema_version": FORMAT_SCHEMA_VERSION,
                "library_version": library_version(),
                "data": obj.to_dict(),
            }
        )
    if isinstance(obj, ImputationRecipe):
        return _encode_envelope(
            {
                "kind": KIND_RECIPE,
                "format_schema_version": FORMAT_SCHEMA_VERSION,
                "library_version": library_version(),
                "data": obj.to_dict(),
            }
        )
    if isinstance(obj, StructuralProfileResult):
        return _encode_envelope(
            {
                "kind": KIND_PROFILE,
                "format_schema_version": FORMAT_SCHEMA_VERSION,
                "library_version": library_version(),
                "data": obj.to_dict(),
            }
        )

    return _encode_fitted_unit(obj)


def deserialize(obj: bytes) -> Any:
    """Reconstruct the original object from bytes produced by :func:`serialize`.

    Polymorphic (ADR-0072, ADR-0096): the envelope's ``kind`` tag selects the
    object type, so the caller never declares what it is deserializing. A
    routing, a recipe, and a profile rebuild from their JSON ``data``; a fitted
    unit runs the reconstruction gates — the ``produced_with`` version check and
    the payload SHA-256 — **before** its joblib payload is unpickled, so a unit
    produced under an incompatible library/estimator version refuses here rather
    than transforming to subtly wrong numbers later.

    Parameters
    ----------
    obj : bytes
        Bytes previously returned by :func:`serialize`.

    Returns
    -------
    FittedUnit or ImputationRouting or ImputationRecipe or StructuralProfileResult
        The reconstructed object; a fitted unit's ``transform`` is bit-identical
        to the original's on equal input.

    Raises
    ------
    IncompatibleArtifactError
        If the envelope's ``kind`` is unrecognised, or a v1-schema / obsolete
        decision payload is encountered, or a fitted unit's format schema or
        reconstruction-critical dependency version is incompatible, or its
        payload checksum does not match.
    ValueError
        If a recipe payload fails the strict load: a missing or unknown dial
        row or key, or decided-base unit ids that mismatch
        ``derive_units(routing)``.

    Warnings
    --------
    Deserializing a fitted unit unpickles its payload, which executes arbitrary
    code — only deserialize bytes you produced or trust. See the module-level
    trust-boundary note.
    """
    header, payload = _decode_envelope(obj)
    kind = header.get("kind")

    if kind == "decision":
        raise IncompatibleArtifactError(
            "Artifact kind 'decision' is obsolete (schema v1); profile, route, "
            "resolve, or fit again."
        )

    check_reconstructable(header)

    if kind == KIND_ROUTING:
        return ImputationRouting.from_dict(header["data"])
    if kind == KIND_RECIPE:
        return ImputationRecipe.from_dict(header["data"])
    if kind == KIND_PROFILE:
        return StructuralProfileResult.from_dict(header["data"])
    if kind == KIND_FITTED_UNIT:
        return _decode_fitted_unit(header, payload)

    raise IncompatibleArtifactError(
        f"Unrecognised artifact kind {kind!r}; the bytes were not produced by "
        f"this library's serialize() or use an unsupported format."
    )


def inspect(obj: bytes) -> dict:
    """Read a serialized envelope's header without unpickling its payload.

    The pre-deserialize probe (ADR-0072): it parses only the JSON header, so a
    caller can check an artifact's ``kind``, format-schema/library versions, and
    — for a fitted unit — its ``produced_with`` provenance stamp *before*
    trusting untrusted-transport bytes into an unpickling :func:`deserialize`.
    The bulky ``data`` payload of a routing or profile envelope is omitted; the
    fitted unit's opaque joblib tail is never touched.

    Parameters
    ----------
    obj : bytes
        Bytes previously returned by :func:`serialize`.

    Returns
    -------
    dict
        The envelope header: always ``kind``, ``format_schema_version``, and
        ``library_version``; for a fitted unit also ``produced_with``,
        ``payload_sha256``, ``unit_type``, and ``columns``.
    """
    header, _ = _decode_envelope(obj)
    return {k: v for k, v in header.items() if k != "data"}
