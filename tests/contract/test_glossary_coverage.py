"""The glossary covers every term the Agent-facing text uses, and nothing else (ADR-0113).

A term is found the way this codebase marks one, by capitalising it. The
scanned text is ``SKILL.md``, every Skill reference, every generated ``--help``
and schema page, and the glossary itself. Every term found must match exactly
one entry, and every entry must be used. A new capitalised term fails here
until it has an entry in ``skill/glossary/``, or a reason on the allow-list.

The check reads text in four rules:

1. Drop code spans, link targets, headings, lines that end in ``:`` and an
   entry's field lines.
2. Join wrapped lines that share an indent. A change of indent, a list item, a
   table cell and a gap of two spaces inside a line each start a new sentence.
3. Collect runs of Title-case words that sit mid-sentence: after a lowercase
   word, a digit, ``,``, ``;``, ``)`` or a possessive. All-caps acronyms and
   CamelCase names are not terms, nor is a lone letter such as Cramér's V, and
   an allowed name splits a run.
4. A run of two or more words at the start of a sentence still counts, after
   dropping a leading determiner such as "The" or "Each".
"""

from __future__ import annotations

import re

import tiktoken

from dataforge_ml import _glossary, _help
from dataforge_ml._boundary._operation import operations
from dataforge_ml._boundary._skill_build import SKILL_FILE
from dataforge_ml._cli import kebab
from dataforge_ml._skill_install import INSTALL_COMMAND

#: Capitalised names that are not terms, each with its reason.
ALLOWED = {
    "Python": "a proper name",
    "Claude Code": "a proper name",
    "Codex": "a proper name",
    "Gemini": "a proper name",
    "Cursor": "a proper name",
    "Pearson": "a proper name",
    "Spearman": "a proper name",
    "Cramér": "a proper name",
    "Breusch-Pagan": "a proper name",
    "Conflict": "an error category; the Skill's Errors table defines it",
    "Infeasible": "an error category; the Skill's Errors table defines it",
    "Unexpected": "an error category; the Skill's Errors table defines it",
}
#: Words that open a sentence without being part of its term, as "Each" in "Each Handle".
LEADING_DETERMINERS = {
    "A", "An", "The", "Each", "Every", "Its", "This", "These", "That", "Those",
    "No", "Only", "Most", "Any", "All", "Some",
}  # fmt: skip
ENTRY_TOKEN_CAP = 120

_CODE = re.compile(r"`[^`]*`")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_LIST_MARKER = re.compile(r"(?:[-*]|\d+\.)\s+")
_FIELD_LINE = re.compile(r"(?:Aliases|Related|Commands):")
_CELL_GAP = re.compile(r"\S {2,}(?=\S)")
_TOKEN = re.compile(r"[^\W\d_][\w'’-]*|\d+|[^\w\s]")
_TITLE = re.compile(r"[A-Z][a-zà-ÿ]*(?:-[A-Z][a-zà-ÿ]+)*")
_POSSESSIVE = re.compile(r"['’]s$")
_SENTENCE_END = {".", "?", "!", ":", "|"}
_MID_SENTENCE_AFTER = {",", ";", ")"}


def _scanned() -> dict[str, str]:
    texts = {"SKILL.md": SKILL_FILE.read_text("utf-8"), "--help": _help.overview()}
    skill = SKILL_FILE.parent
    for path in sorted([*skill.glob("references/*.md"), *skill.glob("glossary/*.md")]):
        texts[str(path.relative_to(skill))] = path.read_text("utf-8")
    for command in _help.commands():
        texts[f"{command} --help"] = _help.command_page(command) or ""
        for path in _help.schema_paths(command):
            texts[f"schema {command} {path}"] = _help.schema_page(command, path) or ""
    return texts


def _sentences(text: str) -> list[str]:
    # Rules 1 and 2 of the module docstring.
    sentences: list[str] = []
    current: list[str] = []
    indent: int | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if (
            not stripped
            or stripped.startswith(("#", "<!--", "---"))
            or stripped.endswith(":")
            or _FIELD_LINE.match(stripped)
        ):
            sentences.append(" ".join(current))
            current, indent = [], None
            continue
        depth = len(line) - len(stripped)
        marker = _LIST_MARKER.match(stripped)
        if marker or stripped.startswith("|") or depth != indent:
            sentences.append(" ".join(current))
            current = []
            stripped = stripped[marker.end() :] if marker else stripped
        indent = depth
        stripped = _CODE.sub(" code ", _LINK_TARGET.sub("", stripped))
        stripped = stripped.replace("[", "").replace("]", "")
        current.append(_CELL_GAP.sub(lambda m: m.group()[0] + " | ", stripped))
    sentences.append(" ".join(current))
    return [s for s in sentences if s]


def _split_at_allowed(run: list[str]) -> list[list[str]]:
    pieces: list[list[str]] = [[]]
    i = 0
    while i < len(run):
        for name in ALLOWED:
            words = name.split()
            if run[i : i + len(words)] == words:
                pieces.append([])
                i += len(words)
                break
        else:
            pieces[-1].append(run[i])
            i += 1
    return pieces


def terms(text: str) -> list[str]:
    """Return the terms a text uses, by rules 3 and 4 of the module docstring."""
    found: list[str] = []
    for sentence in _sentences(text):
        tokens = _TOKEN.findall(sentence)
        before: str | None = None
        i = 0
        while i < len(tokens):
            if not _TITLE.fullmatch(_POSSESSIVE.sub("", tokens[i])):
                before = None if tokens[i] in _SENTENCE_END else tokens[i]
                i += 1
                continue
            run: list[str] = []
            while i < len(tokens) and _TITLE.fullmatch(word := _POSSESSIVE.sub("", tokens[i])):
                run.append(word)
                i += 1
                if word != tokens[i - 1]:
                    break  # a possessive ends its term
            mid = before is not None and (
                before[0].islower()
                or before.isdigit()
                or before in _MID_SENTENCE_AFTER
                or bool(_POSSESSIVE.search(before))
            )
            for n, piece in enumerate(_split_at_allowed(run)):
                if mid or n:
                    if piece and piece != [piece[0][:1]]:  # a lone letter names a statistic
                        found.append(" ".join(piece))
                    continue
                if piece and piece[0] in LEADING_DETERMINERS:
                    piece = piece[1:]
                if len(piece) >= 2:
                    found.append(" ".join(piece))
            before = tokens[i - 1]
    return found


def _all_entries() -> list[_glossary.Entry]:
    return [e for g in (*_glossary.PHASES, _glossary.COMMON) for e in _glossary.entries(g)]


def _entries_by_name() -> dict[str, list[str]]:
    # Each name or alias, keyed the way `dataforge glossary` matches it, to its entries.
    found: dict[str, list[str]] = {}
    for entry in _all_entries():
        for key in {_glossary._key(name) for name in (entry.term, *entry.aliases)}:
            found.setdefault(key, []).append(entry.term)
    return found


def test_every_term_used_matches_exactly_one_entry() -> None:
    names = _entries_by_name()
    misses: dict[str, set[str]] = {}
    for source, text in _scanned().items():
        for term in terms(text):
            if len(names.get(_glossary._key(term), [])) != 1:
                misses.setdefault(term, set()).add(source)
    assert misses == {}, (
        "each term needs exactly one entry in skill/glossary/, or a reason in ALLOWED. "
        "A sentence-start run such as 'Frame Handle, ...' is no term: reword it, "
        "as 'The frame Handle, ...': " + repr({t: sorted(s)[:3] for t, s in misses.items()})
    )


def test_no_two_entries_share_a_name_or_alias() -> None:
    shared = {key: terms_ for key, terms_ in _entries_by_name().items() if len(terms_) > 1}
    assert shared == {}


def test_every_entry_is_used_somewhere() -> None:
    # An entry's own definition does not count; another entry's does.
    pages = [_CODE.sub(" ", t) for s, t in _scanned().items() if not s.startswith("glossary/")]
    pages_text = " ".join(pages).lower()
    unused = []
    for entry in _all_entries():
        others = " ".join(_CODE.sub(" ", e.definition) for e in _all_entries() if e != entry)
        text = f"{pages_text} {others.lower()}"
        if not any(re.search(_plural_pattern(n), text) for n in (entry.term, *entry.aliases)):
            unused.append(entry.term)
    assert unused == []


def _plural_pattern(name: str) -> str:
    # The singular or plural of a name's last word, as `_glossary._key` folds them.
    words = name.lower()
    if words.endswith("y"):
        return rf"\b{re.escape(words[:-1])}(?:y|ies)\b"
    return rf"\b{re.escape(words)}(?:e?s)?\b"


def test_related_terms_are_entries_and_commands_are_real() -> None:
    entry_terms = {e.term for e in _all_entries()}
    commands = {kebab(name) for name in operations()}
    commands |= {_glossary.GLOSSARY_COMMAND, _help.SCHEMA_COMMAND, INSTALL_COMMAND}
    for entry in _all_entries():
        assert set(entry.related) <= entry_terms - {entry.term}, entry.term
        assert set(entry.commands) <= commands, entry.term


def test_each_definition_is_within_the_token_cap() -> None:
    encoding = tiktoken.get_encoding("o200k_base")
    for entry in _all_entries():
        assert len(encoding.encode(entry.definition)) <= ENTRY_TOKEN_CAP, entry.term


def test_the_check_finds_terms_the_way_the_codebase_marks_them() -> None:
    text = (
        "# A Heading Here\n"
        "Every Handle has a Routing Reason, unlike `Code Span` or the [Link](Some Target).\n"
        "The Signal Score is read; MICE and RouteResult are not terms. It reads the\n"
        "Lineage of the\n"
        "  Workspace. Absolute Pearson correlation, then Dataset Digest, then 2 Conflict.\n"
        "It reads the Harness's Skill and Cramér's V.\n"
        "Lines like This One:\n"
        "  load     Read a CSV file.\n"
    )
    assert terms(text) == [
        "Routing Reason",
        "Link",
        "Signal Score",
        "Lineage",
        "Dataset Digest",
        "Harness",
        "Skill",
    ]
