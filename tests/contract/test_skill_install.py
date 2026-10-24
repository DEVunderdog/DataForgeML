"""``dataforge install --skills``: both skills folders, the version stamp, the stale warning (ADR-0112).

The Skill is copied into ``.claude/skills/dataforge/`` and
``.agents/skills/dataforge/``, under the working directory or, with
``--global``, the home folder, and stamped with the library version. Any other
command warns once on stderr when an installed stamp differs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dataforge_ml._skill_install import SKILL_DIRS, SKILL_SOURCE, STAMP_FILE
from dataforge_ml._version import library_version

from .conftest import Harness, run_dataforge


def _files(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def _assert_installed(root: Path) -> None:
    shipped = _files(SKILL_SOURCE)
    assert "SKILL.md" in shipped and "references/harness.md" in shipped
    for relative in SKILL_DIRS:
        folder = root / relative
        installed = _files(folder)
        assert installed.pop(STAMP_FILE).decode().strip() == library_version()
        assert installed == shipped


def test_install_writes_both_folders_in_the_working_directory_stamped(
    harness: Harness, home: Path
) -> None:
    out = run_dataforge(["install", "--skills"])

    assert (out.exit_code, out.stderr) == (0, "")
    assert out.json == {
        "library_version": library_version(),
        "installed": [str(harness.inputs / relative) for relative in SKILL_DIRS],
    }
    _assert_installed(harness.inputs)
    assert not (home / ".claude").exists()
    assert harness.handles() == []


def test_install_global_writes_both_folders_in_the_home_folder(
    harness: Harness, home: Path
) -> None:
    out = run_dataforge(["install", "--skills", "--global"])

    assert out.exit_code == 0
    assert out.json["installed"] == [str(home / relative) for relative in SKILL_DIRS]
    _assert_installed(home)
    assert not (harness.inputs / ".claude").exists()


def test_install_replaces_an_earlier_copy_whole(harness: Harness) -> None:
    run_dataforge(["install", "--skills"])
    leftover = harness.inputs / SKILL_DIRS[0] / "references" / "dropped.md"
    leftover.write_text("from an older Skill")

    assert run_dataforge(["install", "--skills"]).exit_code == 0

    assert not leftover.exists()
    _assert_installed(harness.inputs)


def test_a_mismatched_stamp_warns_once(harness: Harness) -> None:
    run_dataforge(["install", "--skills"])
    stale = harness.inputs / SKILL_DIRS[1]
    (stale / STAMP_FILE).write_text("0.0.1\n")

    first = run_dataforge(["list-handles"])
    second = run_dataforge(["list-handles"])

    assert first.exit_code == second.exit_code == 0
    [line] = first.stderr.splitlines()
    assert str(stale) in line and "0.0.1" in line and library_version() in line
    assert "dataforge install --skills" in line
    assert str(harness.inputs / SKILL_DIRS[0]) not in line  # its stamp is current
    assert second.stderr == ""

    # Another stale stamp is news, so it warns once more.
    (stale / STAMP_FILE).write_text("0.0.3\n")
    assert "0.0.3" in run_dataforge(["list-handles"]).stderr
    assert run_dataforge(["list-handles"]).stderr == ""


def test_a_stale_global_copy_warns_too_and_reinstalling_clears_it(
    harness: Harness, home: Path
) -> None:
    run_dataforge(["install", "--skills", "--global"])
    (home / SKILL_DIRS[0] / STAMP_FILE).write_text("0.0.1\n")

    # Any command warns, even one that names no Operation.
    warned = run_dataforge(["lod"])
    assert warned.json["code"] == "unknown_operation"
    assert str(home / SKILL_DIRS[0]) in warned.stderr

    run_dataforge(["install", "--skills", "--global"])
    (home / SKILL_DIRS[0] / STAMP_FILE).write_text("0.0.2\n")
    assert "0.0.2" in run_dataforge(["list-handles"]).stderr


def test_a_current_or_missing_stamp_stays_quiet(harness: Harness) -> None:
    assert run_dataforge(["list-handles"]).stderr == ""
    run_dataforge(["install", "--skills"])
    assert run_dataforge(["list-handles"]).stderr == ""


def test_install_needs_skills_and_takes_no_other_flag(harness: Harness) -> None:
    for argv in (["install"], ["install", "--skills", "--force"], ["install", "skills"]):
        out = run_dataforge(argv)
        assert (out.exit_code, out.json["code"]) == (2, "invalid_argument"), argv
        assert out.json["allowed"] == ["--skills", "--global"]
    assert not (harness.inputs / ".claude").exists()


def test_install_help_prints_its_usage(harness: Harness) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "dataforge_ml", "install", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.startswith("usage: dataforge install --skills [--global]")
    assert not (harness.inputs / ".claude").exists()
