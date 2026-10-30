"""The installed DataForgeML version, read without importing the library.

This module imports nothing but the standard library, so the CLI can read the
version on every command without loading pandas or scikit-learn.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


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
