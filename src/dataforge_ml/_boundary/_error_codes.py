"""The Error Code registry: one entry per code (ADR-0100, ADR-0102).

Every Library Error and every Result Warning the library can produce has one
entry here. The entry holds the code's category (or ``warning``), the hint the
library writes for it, and the optional record fields it fills. The catalogue
and the Package Skill list codes generated from this table, and a contract
test fails any raise site whose code is missing from it.

Renaming or removing a code is a breaking change. Adding one is not.
"""

from __future__ import annotations

from collections.abc import Iterable
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
        code="handles_unreadable",
        category="warning",
        hint=(
            "Some Handles were written by another format version or are "
            "damaged. Run the Operations that made them again if you need them."
        ),
    ),
    ErrorCodeEntry(
        code="invalid_dial_value",
        category="InvalidInput",
        hint=(
            "Set the dial named in `param` to a value inside its documented "
            "range and call again."
        ),
        fields=("param", "params", "allowed"),
    ),
    ErrorCodeEntry(
        code="retired_dial",
        category="InvalidInput",
        hint=(
            "The dial in `params` no longer exists. The message names what "
            "replaced it; remove the dial and set its replacement instead."
        ),
        fields=("params",),
    ),
    ErrorCodeEntry(
        code="input_not_found",
        category="InvalidInput",
        hint="No file exists at the path in `param`. Check the path and call again.",
        fields=("param",),
    ),
    ErrorCodeEntry(
        code="unsupported_format",
        category="InvalidInput",
        hint="Pass a file whose extension is one of `allowed`.",
        fields=("param", "allowed"),
    ),
    ErrorCodeEntry(
        code="unsupported_artifact_object",
        category="InvalidInput",
        hint=(
            "Only the object kinds in `allowed` can be serialized. Persist the "
            "parts an aggregate is made of, and compose it again after loading."
        ),
        fields=("allowed",),
    ),
    ErrorCodeEntry(
        code="strategy_not_declarable",
        category="InvalidInput",
        hint=(
            "The strategy is output-only or not valid for these columns. "
            "Declare one of `allowed`, or use the declaration the message names."
        ),
        fields=("columns", "allowed"),
    ),
    ErrorCodeEntry(
        code="constant_fill_missing",
        category="InvalidInput",
        hint="Give every `Constant` column in `columns` a fill value, then declare the strategy.",
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="mice_block_too_small",
        category="InvalidInput",
        hint=(
            "MICE fits a joint block, so it needs at least two columns. Declare "
            "MICE for another column too, or pick a single-column strategy."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="estimates_not_applicable",
        category="InvalidInput",
        hint=(
            "Only columns whose strategy reads estimates carry them: the bimodal "
            "strategies and a BoundedDiscrete model-based one."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="stratify_without_target",
        category="InvalidInput",
        hint="Declare a target column, or turn stratification off.",
        fields=("param",),
    ),
    ErrorCodeEntry(
        code="unseen_column",
        category="InvalidInput",
        hint=(
            "The frame has columns the imputer was never fitted on. Drop the "
            "columns in `columns`, or fit again on a frame that has them."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="missing_fitted_columns",
        category="InvalidInput",
        hint=(
            "The frame lacks columns the imputer or unit was fitted to fill. "
            "Pass a frame with every column in `columns`, or fit again on the "
            "columns this frame has."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="frames_not_aligned",
        category="InvalidInput",
        hint=(
            "The frames compared must have the same rows or the same columns "
            "in the same order. Pass frames derived from the same source."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="unit_coverage_mismatch",
        category="InvalidInput",
        hint=(
            "Compose exactly the units the recipe's routing derives: fit any "
            "missing unit, and leave out units from another routing."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="unknown_column",
        category="UnknownName",
        hint=(
            "Use a column name exactly as it appears in the data. "
            "`candidates` lists close matches, best first."
        ),
        fields=("param", "columns", "candidates"),
    ),
    ErrorCodeEntry(
        code="unknown_enum_value",
        category="UnknownName",
        hint="Pass one of the values in `allowed`. `candidates` lists close matches.",
        fields=("param", "allowed", "candidates"),
    ),
    ErrorCodeEntry(
        code="unknown_dial",
        category="UnknownName",
        hint=(
            "Name only dials this strategy or record takes, listed in "
            "`allowed`. `candidates` lists close matches."
        ),
        fields=("params", "allowed", "candidates"),
    ),
    ErrorCodeEntry(
        code="unknown_unit",
        category="UnknownName",
        hint=(
            "Use a unit id the recipe's routing derives. `candidates` lists "
            "close matches."
        ),
        fields=("param", "candidates"),
    ),
    ErrorCodeEntry(
        code="declaration_conflict",
        category="Conflict",
        hint=(
            "Two declarations on the columns in `columns` contradict each "
            "other. Remove or change one of them."
        ),
        fields=("columns", "params"),
    ),
    ErrorCodeEntry(
        code="unit_routing_mismatch",
        category="Conflict",
        hint=(
            "The unit was derived from a different routing than the recipe's. "
            "Derive the units from the recipe's own routing and fit those."
        ),
        fields=("param", "columns"),
    ),
    ErrorCodeEntry(
        code="unit_not_trainable",
        category="Infeasible",
        hint=(
            "This unit cannot train on this data. Route its columns to another "
            "strategy, or fit on a frame with more usable rows."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="override_coercion_failed",
        category="Infeasible",
        hint=(
            "No value of the column converts to the declared type. Declare "
            "another type, or for a datetime column declare its format."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="stratify_on_continuous_target",
        category="Infeasible",
        hint=(
            "A continuous target has too few rows per value to stratify on. "
            "Turn stratification off, or use the profile-stratified split, "
            "which bins a continuous target first."
        ),
        fields=("param", "columns"),
    ),
    ErrorCodeEntry(
        code="no_mice_block",
        category="Infeasible",
        hint=(
            "An estimator applies only to a MICE block, and this routing has "
            "none. Route at least two columns to MICE first."
        ),
    ),
    ErrorCodeEntry(
        code="custom_estimator_missing",
        category="Infeasible",
        hint=(
            "A custom estimator is never stored, so a reloaded routing has none. "
            "Set the estimator on the routing again before fitting."
        ),
    ),
    ErrorCodeEntry(
        code="c2st_string_column",
        category="Infeasible",
        hint=(
            "The C2ST classifier cannot read string columns. Cast the columns "
            "in `columns` to categorical, or leave them out of both samples."
        ),
        fields=("columns",),
    ),
    ErrorCodeEntry(
        code="c2st_sample_floor",
        category="Infeasible",
        hint=(
            "One sample is below the C2ST size floor. Use larger samples, or "
            "lower the floor dial if you accept a weaker test."
        ),
        fields=("param",),
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

#: Most close-match candidates an ``UnknownName`` error offers.
_CANDIDATES_CAP = 5


def close_matches(names: Iterable[str], pool: Iterable[str]) -> list[str]:
    """Return up to five names from ``pool`` closest to any of ``names``.

    Parameters
    ----------
    names : iterable of str
        The unknown names the caller supplied.
    pool : iterable of str
        The names that exist.

    Returns
    -------
    list of str
        Close matches, best first, without repeats.
    """
    import difflib

    pool = [str(p) for p in pool]
    found: list[str] = []
    for name in names:
        for match in difflib.get_close_matches(str(name), pool, n=_CANDIDATES_CAP, cutoff=0.5):
            if match not in found:
                found.append(match)
    return found[:_CANDIDATES_CAP]


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
