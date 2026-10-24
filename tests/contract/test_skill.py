"""The package Skill, ``skill/SKILL.md``: shipped, fresh and small (ADR-0108, ADR-0112).

The command index between the markers is generated from the Operation
registry, so an Operation added later shows up in the Skill with no change
here. Refresh it after an intended change with
``uv run python -m dataforge_ml._boundary._skill_build`` (or
``DATAFORGE_ML_UPDATE_SNAPSHOTS=1 uv run pytest -n 0 tests/contract``).
"""

from __future__ import annotations

import importlib.resources
import os
import re

import pytest
import tiktoken

from dataforge_ml import _help
from dataforge_ml._boundary._operation import operations
from dataforge_ml._boundary._skill_build import (
    BACKGROUND_MARK,
    BEGIN_MARKER,
    END_MARKER,
    SKILL_BUDGET_TOKENS,
    SKILL_FILE,
    command_index,
    skill_text,
)
from dataforge_ml._cli import kebab

OPERATIONS = operations()


def _skill() -> str:
    return SKILL_FILE.read_text("utf-8")


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md opens with a frontmatter block"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines())


def test_the_skill_ships_inside_the_package() -> None:
    shipped = importlib.resources.files("dataforge_ml") / "skill" / "SKILL.md"
    assert shipped.is_file()
    assert shipped.read_text("utf-8") == _skill()


def test_the_frontmatter_names_the_skill_and_its_tools() -> None:
    front = _frontmatter(_skill())
    assert front["name"] == "dataforge"
    assert front["allowed-tools"] == "Bash(dataforge:*)"
    assert front["description"].strip()


def test_the_command_index_is_fresh() -> None:
    current = skill_text(_skill())
    if os.environ.get("DATAFORGE_ML_UPDATE_SNAPSHOTS"):
        SKILL_FILE.write_text(current, "utf-8")
    assert _skill() == current, (
        "SKILL.md's command index is stale; run "
        "`uv run python -m dataforge_ml._boundary._skill_build`"
    )


def test_the_index_has_one_usage_line_per_command() -> None:
    usage = re.compile(r"- `dataforge ([a-z0-9-]+)((?: --[a-z-]+ <\w+>)*)`( \(\w+\))?: (.+)")
    lines = {m.group(1): m for m in map(usage.fullmatch, command_index().splitlines()) if m}
    assert sorted(lines) == sorted(kebab(name) for name in OPERATIONS)
    for name, spec in OPERATIONS.items():
        _, flags, mark, summary = lines[kebab(name)].groups()
        assert summary == spec.docstring.summary
        assert (mark == f" {BACKGROUND_MARK}") == spec.scales_with_data, name
        # Required flags only: the full signature lives in `--help`.
        required = [kebab(a) for a, f in spec.args_model.model_fields.items() if f.is_required()]
        assert re.findall(r"--([a-z-]+)", flags) == required, name


def test_the_skill_is_about_2k_tokens() -> None:
    tokens = len(tiktoken.get_encoding("o200k_base").encode(_skill()))
    # Over budget: move reference into `references/` or `--help`. Never cut the index.
    assert tokens <= SKILL_BUDGET_TOKENS, tokens


def test_every_command_the_skill_names_is_real() -> None:
    named = set(re.findall(r"dataforge ([a-z][a-z0-9-]*)", _skill()))
    assert named - {_help.SCHEMA_COMMAND} <= set(_help.commands())


def test_regenerating_needs_both_markers() -> None:
    with pytest.raises(ValueError):
        skill_text("no markers here")
    with pytest.raises(ValueError):
        skill_text(f"{END_MARKER}\n{BEGIN_MARKER}")
    assert skill_text(f"before\n{BEGIN_MARKER}\nold\n{END_MARKER}\nafter") == (
        f"before\n{command_index()}\nafter"
    )
