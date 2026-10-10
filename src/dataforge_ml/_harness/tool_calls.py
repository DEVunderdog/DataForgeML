"""The Tool Calls delivery: one tool call in, one MCP ``CallToolResult``-shaped dict out.

Generated from the same Operation as Code Mode, so both deliveries return the
same JSON (ADR-0099). A Library Error comes back as its JSON record with
``isError: true`` instead of being raised (ADR-0100). ``MemoryError`` and
``KeyboardInterrupt`` still propagate: resources belong to the Harness.

The Harness sets the Workspace and Declared Inputs in the environment before
calling, exactly as for Code Mode.
"""

from __future__ import annotations

import json
from typing import Any

from .._boundary._operation import canonical_json, invoke_json, operations
from ..records import DataForgeError


def tool_names() -> list[str]:
    """Return the name of every Operation offered as a tool.

    Returns
    -------
    list of str
        Operation names, sorted.
    """
    return sorted(operations())


def call_tool(name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    """Run one Tool Call and return its result.

    Parameters
    ----------
    name : str
        The Operation name.
    arguments : dict or str or None
        The tool input object, parsed or as JSON text. None means ``{}``.

    Returns
    -------
    dict
        ``{"content": [{"type": "text", "text": <JSON>}], "structuredContent":
        <JSON object>, "isError": <bool>}``. The JSON is the Wire Record's
        ``to_dict()`` on success and the Library Error's on failure.
    """
    try:
        spec = operations().get(name)
        if spec is None:
            raise DataForgeError.from_code(
                "unknown_operation",
                f"No Operation is named `{name}`.",
                candidates=_close(name),
            )
        if arguments is None:
            text = "{}"
        elif isinstance(arguments, str):
            text = arguments
        else:
            text = json.dumps(arguments)
        payload, is_error = invoke_json(spec, text).to_dict(), False
    except DataForgeError as err:
        payload, is_error = err.to_dict(), True
    return {
        "content": [{"type": "text", "text": canonical_json(payload)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _close(name: str) -> list[str]:
    import difflib

    return difflib.get_close_matches(name, tool_names(), n=5, cutoff=0.5)
