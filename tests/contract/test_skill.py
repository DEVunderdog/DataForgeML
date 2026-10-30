"""The package Skill, ``skill/SKILL.md`` and ``skill/references/``: shipped, fresh and small (ADR-0108, ADR-0112).

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

from dataforge_ml import _glossary, _help
from dataforge_ml._boundary._help_build import PHASE_ORDER
from dataforge_ml._boundary._operation import operations
from dataforge_ml._boundary._skill_build import (
    BACKGROUND_MARK,
    BEGIN_MARKER,
    END_MARKER,
    SKILL_BUDGET_BYTES,
    SKILL_FILE,
    command_index,
    skill_text,
)
from dataforge_ml._cli import kebab
from dataforge_ml._skill_install import INSTALL_COMMAND

from .conftest import Harness

OPERATIONS = operations()
#: The Skill's reference files: one per phase, ``python.md``, ``harness.md`` and ``glossary.md``.
REFERENCES_DIR = SKILL_FILE.parent / "references"


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


def test_the_skill_fits_before_the_codex_cut() -> None:
    size = len(_skill().encode("utf-8"))
    # Over budget: move reference into `references/` or `--help`. Never cut the index.
    assert size <= SKILL_BUDGET_BYTES, size


def test_the_references_line_comes_before_the_commands() -> None:
    # A Harness that cuts the file cuts its end, so the links sit above the index.
    text = _skill()
    assert text.index("](references/glossary.md)") < text.index(BEGIN_MARKER)


def test_every_command_the_skill_names_is_real() -> None:
    named = set(re.findall(r"dataforge ([a-z][a-z0-9-]*)", _skill()))
    helpers = {_help.SCHEMA_COMMAND, INSTALL_COMMAND, _glossary.GLOSSARY_COMMAND}
    assert named - helpers <= set(_help.commands())


def _references() -> dict[str, str]:
    return {path.name: path.read_text("utf-8") for path in REFERENCES_DIR.glob("*.md")}


def test_references_ship_one_file_per_phase_plus_python_harness_and_glossary() -> None:
    expected = {f"{phase}.md" for phase in PHASE_ORDER} | {"python.md", "harness.md", "glossary.md"}
    assert set(_references()) == expected
    shipped = importlib.resources.files("dataforge_ml") / "skill" / "references"
    assert {p.name for p in shipped.iterdir() if p.name.endswith(".md")} == expected


def test_the_skill_links_every_reference() -> None:
    linked = set(re.findall(r"\]\(references/([a-z]+\.md)\)", _skill()))
    assert linked == set(_references())


def test_every_command_the_references_name_is_real() -> None:
    real = set(_help.commands()) | {
        _help.SCHEMA_COMMAND,
        INSTALL_COMMAND,
        _glossary.GLOSSARY_COMMAND,
    }
    for name, text in _references().items():
        named = set(re.findall(r"dataforge ([a-z][a-z0-9-]*)", text))
        assert named <= real, (name, named - real)


def test_the_harness_contract_states_the_guardrails() -> None:
    contract = _references()["harness.md"]
    # What the Harness provides (ADR-0112).
    for term in ("Python 3.11", "`dataforge` on the PATH", "This Skill installed"):
        assert term in contract, term
    # Workspace Settings are guardrails, `deny` holds only under Harness
    # permission rules, and `query_frame` still leaks through counts.
    for term in ("guardrails, not a boundary", "permission rules", "query-frame", "counts"):
        assert term in contract, term
    # SKILL.md reaches the contract from its Workspace Settings section.
    settings = _skill().split("## Workspace Settings", 1)[1]
    assert "(references/harness.md)" in settings.split("\n## ", 1)[0]


def test_the_glossary_reference_teaches_the_layers_and_the_fallback() -> None:
    guide = _references()["glossary.md"]
    # The three layers of `dataforge glossary` (ADR-0113).
    for layer in ("`dataforge glossary`", "--phase", "--common", "`dataforge glossary <term>`"):
        assert layer in guide, layer
    # The fallback: one phase's glossary file plus the Common Terms' file.
    assert "unknown_term" in guide
    assert "glossary/<phase>.md" in guide and "glossary/common.md" in guide


def test_the_python_reference_example_runs(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    # A frame with gaps for the example to fill, under the name it loads.
    lines = ["a,b,c,city"]
    for i in range(150):
        a = (i * 13) % 47 + 0.5
        b = "" if i % 3 == 1 else f"{2 * a + (i % 5)}"
        c = "" if i % 7 == 3 else f"{a / 2 + (i % 11)}"
        city = "" if i % 10 == 7 else "nse"[i % 3]
        lines.append(f"{a},{b},{c},{city}")
    (harness.inputs / "train.csv").write_text("\n".join(lines) + "\n")
    blocks = re.findall(r"```python\n(.*?)```", _references()["python.md"], re.DOTALL)
    assert len(blocks) == 1
    exec(compile(blocks[0], "python.md", "exec"), {})  # noqa: S102 - runs the shipped example
    printed = capsys.readouterr().out.splitlines()
    # One census line per fold, then the caught error's code and candidates.
    assert [line.split()[0] for line in printed[:3]] == ["0", "1", "2"]
    assert printed[3].startswith("unknown_handle ")


def test_regenerating_needs_both_markers() -> None:
    with pytest.raises(ValueError):
        skill_text("no markers here")
    with pytest.raises(ValueError):
        skill_text(f"{END_MARKER}\n{BEGIN_MARKER}")
    assert skill_text(f"before\n{BEGIN_MARKER}\nold\n{END_MARKER}\nafter") == (
        f"before\n{command_index()}\nafter"
    )
