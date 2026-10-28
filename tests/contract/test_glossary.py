"""``dataforge glossary``: the three layers, the misses, and no heavy import (ADR-0113).

Each layer is checked through the CLI, against literal expected values from the
shipped glossary files, never recomputed through the parser.
"""

from __future__ import annotations

import importlib.resources
import json
import subprocess
import sys

from dataforge_ml import _glossary
from dataforge_ml._boundary._operation import PHASES, operations
from dataforge_ml._cli import kebab

from .conftest import DATAFORGE, run_dataforge

_HEAVY = ("pandas", "polars", "sklearn", "scipy", "numpy")

IMPUTE_TERMS = [
    "Routing Reason",
    "Routing Evidence",
    "Signal Score",
    "Signal Tier",
    "Feasibility Floor",
    "Resource Ceiling",
    "Estimator Ladder",
    "Drop Candidate",
    "Soft Exclusion",
]


def test_no_arguments_counts_each_phase_and_the_common_terms_apart() -> None:
    out = run_dataforge(["glossary"])

    assert (out.exit_code, out.stderr) == (0, "")
    assert out.json == {
        "phases": {
            "fields": ["phase", "terms"],
            "rows": [["load", 0], ["profile", 0], ["split", 0], ["impute", 9], ["evaluate", 0]],
        },
        "common": 0,
    }


def test_phase_lists_its_term_names_only() -> None:
    out = run_dataforge(["glossary", "--phase", "impute"])

    assert out.exit_code == 0
    assert out.json == {"phase": "impute", "terms": IMPUTE_TERMS}
    assert run_dataforge(["glossary", "--phase=impute"]).json == out.json


def test_common_lists_the_common_terms_names() -> None:
    out = run_dataforge(["glossary", "--common"])

    assert out.exit_code == 0
    assert out.json == {"phase": "common", "terms": []}


SIGNAL_SCORE = {
    "term": "Signal Score",
    "phase": "impute",
    "definition": (
        "How strongly a column's other columns predict it, from 0 to 1: is a model worth "
        "it over a plain fill? It is null when no other column correlates with it at all. "
        "Routing reads it after the Feasibility Floor, and its Signal Tier picks the "
        "strategy. The `signal_score_*` dials of `route` weight its parts and set its tiers."
    ),
    "aliases": ["Signal"],
    "related": ["Signal Tier", "Feasibility Floor"],
    "commands": ["route", "explain-routing"],
}


def test_one_term_returns_its_entry() -> None:
    out = run_dataforge(["glossary", "signal", "score"])

    assert (out.exit_code, out.stderr) == (0, "")
    assert out.json == SIGNAL_SCORE


def test_a_term_resolves_whatever_its_case_plural_alias_or_quoting() -> None:
    for argv in (
        ["Signal Score"],
        ["SIGNAL", "SCORES"],
        ["signal scores"],
        ["Signal"],
        ["signals"],
    ):
        out = run_dataforge(["glossary", *argv])
        assert (out.exit_code, out.json) == (0, SIGNAL_SCORE), argv
    feasibility = run_dataforge(["glossary", "feasibility", "checks"])
    assert feasibility.json["term"] == "Feasibility Floor"
    assert feasibility.json["aliases"] == ["Feasibility", "Feasibility check"]


def test_a_missing_term_is_unknown_term_with_candidates_and_the_fallback() -> None:
    out = run_dataforge(["glossary", "signal", "scroe"])

    assert out.exit_code == 3
    assert out.json["code"] == "unknown_term"
    assert out.json["candidates"] == ["Signal Score", "Signal Tier"]
    assert "glossary/<phase>.md" in out.json["hint"]
    assert "glossary/common.md" in out.json["hint"]


def test_an_unknown_phase_is_invalid_argument_and_common_is_no_phase() -> None:
    for phase in ("imputation", "common"):
        out = run_dataforge(["glossary", "--phase", phase])
        assert out.exit_code == 2, phase
        assert out.json["code"] == "invalid_argument"
        assert out.json["allowed"] == ["load", "profile", "split", "impute", "evaluate"]


def test_mixed_or_unknown_flags_are_invalid_argument() -> None:
    for argv in (["--phase"], ["--phase", "impute", "signal"], ["--common", "signal"], ["--all"]):
        out = run_dataforge(["glossary", *argv])
        assert (out.exit_code, out.json["code"]) == (2, "invalid_argument"), argv


def test_help_prints_the_usage() -> None:
    done = subprocess.run(
        [str(DATAFORGE), "glossary", "--help"], capture_output=True, text=True, check=False
    )

    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.startswith("usage: dataforge glossary [--phase <phase> | --common | <term...>]")


def test_no_layer_imports_a_heavy_dependency() -> None:
    argvs = [[], ["--phase", "impute"], ["--common"], ["signal", "score"], ["--help"]]
    script = (
        "import io, json, sys\n"
        "from dataforge_ml._cli import run\n"
        f"for argv in {argvs!r}:\n"
        "    assert run(['glossary', *argv], stdout=io.StringIO()) == 0, argv\n"
        "assert run(['glossary', 'nothing here'], stdout=io.StringIO()) == 3\n"
        f"print(json.dumps(sorted(m for m in {_HEAVY!r} if m in sys.modules)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


def test_the_glossary_phases_are_the_pipeline_phases_and_no_operation_is_named_glossary() -> None:
    # Workspace terms are Common Terms, so `workspace` has no glossary file.
    assert [p for p in PHASES if p != "workspace"] == list(_glossary.PHASES)
    assert "glossary" not in {kebab(name) for name in operations()}


def test_the_glossary_files_ship_inside_the_package() -> None:
    shipped = importlib.resources.files("dataforge_ml") / "skill" / "glossary"
    names = {"load", "profile", "split", "impute", "evaluate", "common"}
    assert {f"{name}.md" for name in names} <= {p.name for p in shipped.iterdir()}
