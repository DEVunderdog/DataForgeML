"""Read the Agent glossary the ``dataforge glossary`` command prints (ADR-0113).

The entries live in the package Skill, one markdown file per pipeline phase
plus ``common.md`` for the Common Terms, so an Agent whose lookup misses reads
the same files the command parses. An entry is a ``## Term`` heading, then
``Aliases:``, ``Related:`` and ``Commands:`` lines, then the definition. Text
above the first heading is the file's own introduction.

This module imports nothing but the standard library, so a lookup costs no
heavy import.
"""

from __future__ import annotations

import difflib
import functools
from dataclasses import dataclass
from pathlib import Path

#: The CLI command that looks terms up. It is no Operation.
GLOSSARY_COMMAND = "glossary"
GLOSSARY_USAGE = """\
usage: dataforge glossary [--phase <phase> | --common | <term...>]

Look up a term the dataforge commands use, one layer at a time.
  dataforge glossary                 each phase and how many terms it has
  dataforge glossary --phase impute  one phase's term names
  dataforge glossary --common        the names of the terms every phase uses
  dataforge glossary signal score    one term: its definition, aliases,
                                     related terms and commands
A term may be several words, unquoted. Case and plurals do not matter, and an
alias finds its term.
"""
#: The glossary files, shipped inside the package Skill.
GLOSSARY_DIR = Path(__file__).with_name("skill") / "glossary"
#: The pipeline phases with a glossary file, in pipeline order.
PHASES = ("load", "profile", "split", "impute", "evaluate")
#: The group of the Common Terms, which is not a phase.
COMMON = "common"
#: The flags ``dataforge glossary`` takes.
GLOSSARY_FLAGS = ("--phase", "--common")

_FIELDS = ("Aliases:", "Related:", "Commands:")


@dataclass(frozen=True)
class Entry:
    """One glossary entry.

    Parameters
    ----------
    term : str
        The term, as its heading spells it.
    group : str
        Its phase, or ``common``.
    definition : str
        The definition, its wrapped lines joined and its paragraphs split
        by a blank line.
    aliases, related, commands : tuple of str
        Other names that find it, the entries it relates to, and the
        commands whose output uses it.
    """

    term: str
    group: str
    definition: str
    aliases: tuple[str, ...]
    related: tuple[str, ...]
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the entry as the command prints it.

        Returns
        -------
        dict
            The term, its phase (or ``common``), definition, aliases, related
            terms and commands.
        """
        return {
            "term": self.term,
            "phase": self.group,
            "definition": self.definition,
            "aliases": list(self.aliases),
            "related": list(self.related),
            "commands": list(self.commands),
        }


def _items(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _entry(group: str, term: str, lines: list[str]) -> Entry:
    fields = dict.fromkeys(_FIELDS, "")
    body = list(lines)
    while body and (not body[0].strip() or body[0].startswith(_FIELDS)):
        line = body.pop(0)
        if line.strip():
            label, _, value = line.partition(":")
            fields[f"{label}:"] = value
    paragraphs = "\n".join(body).strip().split("\n\n")
    return Entry(
        term=term,
        group=group,
        definition="\n\n".join(" ".join(p.split()) for p in paragraphs),
        aliases=_items(fields["Aliases:"]),
        related=_items(fields["Related:"]),
        commands=_items(fields["Commands:"]),
    )


@functools.cache
def entries(group: str) -> tuple[Entry, ...]:
    """Return a phase's entries, or the Common Terms, in file order.

    Parameters
    ----------
    group : str
        One of :data:`PHASES`, or :data:`COMMON`.

    Returns
    -------
    tuple of Entry
        Empty when the file holds no entry yet.
    """
    found: list[Entry] = []
    term: str | None = None
    lines: list[str] = []
    for line in (GLOSSARY_DIR / f"{group}.md").read_text("utf-8").splitlines():
        if line.startswith("## "):
            if term is not None:
                found.append(_entry(group, term, lines))
            term, lines = line[3:].strip(), []
        elif term is not None:
            lines.append(line)
    if term is not None:
        found.append(_entry(group, term, lines))
    return tuple(found)


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _key(text: str) -> str:
    # Case and the last word's plural are ignored: "signal scores" finds "Signal Score".
    words = text.lower().split()
    return " ".join([*words[:-1], _singular(words[-1])]) if words else ""


@functools.cache
def _names() -> dict[str, Entry]:
    names: dict[str, Entry] = {}
    for group in (*PHASES, COMMON):
        for entry in entries(group):
            for name in (entry.term, *entry.aliases):
                names.setdefault(_key(name), entry)
    return names


def lookup(words: list[str]) -> Entry | None:
    """Return the entry a term or alias names, or None when none does.

    Parameters
    ----------
    words : list of str
        The term, as one string or as several words.

    Returns
    -------
    Entry or None
        The entry, matched ignoring case and plurals.
    """
    return _names().get(_key(" ".join(words)))


def close_matches(words: list[str]) -> list[str]:
    """Return the terms whose name or alias is close to a missed term, best first.

    Parameters
    ----------
    words : list of str
        The term that found no entry.

    Returns
    -------
    list of str
        At most five entry terms, each once.
    """
    names = _names()
    close = difflib.get_close_matches(_key(" ".join(words)), list(names), n=10, cutoff=0.6)
    return list(dict.fromkeys(names[name].term for name in close))[:5]
