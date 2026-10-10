"""The Error Code registry: one entry per code (ADR-0100, ADR-0102).

Every Library Error and every Result Warning the library can produce has one
entry here. The entry holds the code's category (or ``warning``), the hint the
library writes for it, and the optional record fields it fills. The catalogue
and the Package Skill list codes generated from this table, and a contract
test fails any raise site whose code is missing from it.

Renaming or removing a code is a breaking change. Adding one is not.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The six Library Error categories, plus ``warning`` for Result Warnings.
CATEGORIES: tuple[str, ...] = (
    "InvalidInput",
    "UnknownName",
    "Conflict",
    "Infeasible",
    "IncompatibleArtifact",
    "Unexpected",
    "warning",
)


@dataclass(frozen=True)
class ErrorCodeEntry:
    """One registered Error Code.

    Parameters
    ----------
    code : str
        The stable snake_case code an Agent branches on.
    category : str
        One of :data:`CATEGORIES`.
    hint : str
        The library-written advice attached to every error with this code.
    fields : tuple of str
        The optional record fields this code fills, beyond ``code`` and
        ``message``.
    """

    code: str
    category: str
    hint: str
    fields: tuple[str, ...] = ()


_ENTRIES: tuple[ErrorCodeEntry, ...] = (
    ErrorCodeEntry(
        code="invalid_argument",
        category="InvalidInput",
        hint=(
            "Fix every argument named in `params` and call again. Values are "
            "checked strictly: a quoted number is a string, not a number."
        ),
        fields=("param", "params", "allowed", "issues", "omitted"),
    ),
    ErrorCodeEntry(
        code="input_not_declared",
        category="InvalidInput",
        hint=(
            "Only paths the Harness declared can be loaded. Use one of the "
            "declared paths exactly as given to you; an edited copy cannot be "
            "loaded."
        ),
        fields=("param",),
    ),
    ErrorCodeEntry(
        code="unknown_operation",
        category="UnknownName",
        hint="Call one of the Operations in `candidates`, or list the tools again.",
        fields=("candidates",),
    ),
    ErrorCodeEntry(
        code="unknown_handle",
        category="UnknownName",
        hint=(
            "Use a Handle returned by an earlier Operation in this Workspace. "
            "`candidates` lists existing Handles of the same kind, newest first."
        ),
        fields=("param", "candidates"),
    ),
    ErrorCodeEntry(
        code="wrong_handle_kind",
        category="InvalidInput",
        hint=(
            "Pass a Handle of the kind named in `allowed`. The kind is the "
            "prefix before the underscore, such as `frame` in `frame_7f3a9c2e`."
        ),
        fields=("param", "allowed"),
    ),
    ErrorCodeEntry(
        code="incompatible_artifact",
        category="IncompatibleArtifact",
        hint=(
            "This Handle was written by a different library or format version "
            "and cannot be read. Run the Operation that made it again."
        ),
        fields=("param",),
    ),
    ErrorCodeEntry(
        code="workspace_not_set",
        category="Unexpected",
        hint=(
            "The Harness did not provide a Workspace. Stop and report this to "
            "the Harness; no argument can fix it."
        ),
    ),
    ErrorCodeEntry(
        code="unexpected",
        category="Unexpected",
        hint=(
            "An internal failure the library did not anticipate. Stop and "
            "report the code and message; retrying the same call will fail "
            "the same way."
        ),
    ),
)

REGISTRY: dict[str, ErrorCodeEntry] = {entry.code: entry for entry in _ENTRIES}


def entry_for(code: str) -> ErrorCodeEntry:
    """Return the registry entry for ``code``.

    Parameters
    ----------
    code : str
        A registered Error Code.

    Returns
    -------
    ErrorCodeEntry
        The code's registry entry.

    Raises
    ------
    KeyError
        If ``code`` has no registry entry. This is a library bug: every raise
        site must register its code first.
    """
    try:
        return REGISTRY[code]
    except KeyError:
        raise KeyError(f"Error Code {code!r} has no registry entry") from None
