"""A small numpy-docstring section parser (ADR-0101, ADR-0108).

Operation docstrings are the single source of parameter descriptions, the
``Errors`` code list and the ``See Also`` names. This parser splits a docstring
into its summary and its underlined sections, and reads the ``name : type``
entry lists inside them.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

#: The seven parts of an Operation docstring, in order (ADR-0108). The summary
#: is the first part and has no header.
OPERATION_SECTIONS: tuple[str, ...] = (
    "Use when",
    "Parameters",
    "Returns",
    "Errors",
    "See Also",
    "Example",
)

_UNDERLINE = re.compile(r"^-{3,}$")


@dataclass(frozen=True)
class ParsedDocstring:
    """A docstring split into its summary and its sections, in order."""

    summary: str
    sections: dict[str, str]

    def entries(self, section: str) -> dict[str, str]:
        """Return the ``name : type`` entries of a section, mapped to their text.

        Parameters
        ----------
        section : str
            A section header, such as ``Parameters`` or ``Errors``.

        Returns
        -------
        dict
            Entry name to its indented description, joined into one
            paragraph. An entry's type after `` : `` is dropped.
        """
        out: dict[str, str] = {}
        name: str | None = None
        lines: list[str] = []
        for line in self.sections.get(section, "").splitlines():
            if line and not line[0].isspace():
                if name is not None:
                    out[name] = " ".join(lines).strip()
                name, lines = line.split(" : ", 1)[0].strip(), []
            elif name is not None and line.strip():
                lines.append(line.strip())
        if name is not None:
            out[name] = " ".join(lines).strip()
        return out

    def see_also(self) -> dict[str, str]:
        """Return the ``See Also`` entries, each Operation name mapped to its text.

        Returns
        -------
        dict
            Operation name to the text after its `` : ``, joined with any
            indented lines below it into one paragraph.
        """
        out: dict[str, str] = {}
        name: str | None = None
        for line in self.sections.get("See Also", "").splitlines():
            if line and not line[0].isspace():
                name, _, text = line.partition(" : ")
                name = name.strip()
                out[name] = text.strip()
            elif name is not None and line.strip():
                out[name] = f"{out[name]} {line.strip()}".strip()
        return out


def parse(docstring: str | None) -> ParsedDocstring:
    """Parse a numpy docstring.

    Parameters
    ----------
    docstring : str or None
        The raw ``__doc__``.

    Returns
    -------
    ParsedDocstring
        The first line as summary, and every underlined section's body,
        dedented, keyed by its header in document order.
    """
    lines = inspect.cleandoc(docstring or "").splitlines()
    summary = lines[0].strip() if lines else ""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    i = 1
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and _UNDERLINE.match(lines[i + 1].strip()):
            current = sections.setdefault(lines[i].strip(), [])
            i += 2
            continue
        if current is not None:
            current.append(lines[i])
        i += 1
    return ParsedDocstring(
        summary=summary,
        sections={k: "\n".join(v).strip("\n") for k, v in sections.items()},
    )
