"""Fitted-unit envelope encode/decode over an opaque joblib payload (ADR-0072).

A :class:`~dataforge_ml.FittedUnit` crosses the persistence boundary as a JSON
header (unit type, owned columns, the ``produced_with`` six-version stamp, and
the payload's SHA-256 checksum) followed by an opaque joblib payload holding the
unit's learned state — *not* base64-inlined into the header. The header is fully
inspectable without touching the payload; the payload never bloats a JSON field.

These two helpers are the surviving core of the old manifest+payload persistence
(ADR-0063): the content-addressed ``save``/``load`` pair and the store port they
talked to are gone (ADR-0072), and the encode/decode is now reached only through
:func:`dataforge_ml.serialize` / :func:`dataforge_ml.deserialize`.

Trust boundary (ADR-0063)
-------------------------
Reconstructing a unit unpickles its joblib payload, which is **arbitrary code
execution the moment the bytes are loaded** — so the decode path is only safe
against artifacts you produced or otherwise trust. The SHA-256 checksum verified
before deserialising is an **integrity** guard: it catches accidental corruption
and tampering-in-transit and refuses loudly on a mismatch. It is *not* an
**authenticity** guard: an attacker who can rewrite the payload can also rewrite
the recorded checksum. Cryptographic signing is the authenticity tier and is
deferred to a future service.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import joblib

from .._serialization import (
    FORMAT_SCHEMA_VERSION,
    KIND_FITTED_UNIT,
    IncompatibleArtifactError,
    _encode_envelope,
    check_reconstructable,
    library_version,
    produced_with,
)


def _dumps(unit: Any) -> bytes:
    """Serialise a fitted unit's full learned state to opaque joblib bytes."""
    buf = io.BytesIO()
    joblib.dump(unit, buf)
    return buf.getvalue()


def _loads(payload: bytes) -> Any:
    """Reconstruct a fitted unit from its opaque joblib bytes."""
    return joblib.load(io.BytesIO(payload))


def encode_fitted_unit(unit: Any) -> bytes:
    """Encode a fitted unit as a JSON header plus an opaque joblib payload.

    The header carries the ``produced_with`` reconstruction stamp and the
    payload's SHA-256 so :func:`decode_fitted_unit` can gate the unpickle, and
    readable metadata (``unit_type``, ``columns``) for :func:`dataforge_ml.inspect`.

    Parameters
    ----------
    unit : Any
        The fitted unit to persist. Its concrete class must be importable for
        :func:`decode_fitted_unit` to reconstruct it.

    Returns
    -------
    bytes
        The serialized envelope.
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


def decode_fitted_unit(header: dict, payload: bytes) -> Any:
    """Reconstruct a fitted unit from an envelope header and its joblib payload.

    Runs the ADR-0063 reconstruction gates before trusting any bytes: the
    header's ``produced_with`` stamp is checked against the installed
    environment, and the payload's SHA-256 is verified against the checksum
    recorded in the header. The reconstructed unit's ``transform`` is
    bit-identical to the original's on equal input.

    Parameters
    ----------
    header : dict
        The parsed JSON header produced by :func:`encode_fitted_unit`.
    payload : bytes
        The opaque joblib payload trailing the header.

    Returns
    -------
    Any
        The reconstructed fitted unit.

    Raises
    ------
    IncompatibleArtifactError
        If the format schema or a reconstruction-critical dependency version
        is incompatible, or the payload checksum does not match the header.

    Warnings
    --------
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
