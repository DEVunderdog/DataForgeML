"""Install the package Skill into the Harnesses' skills folders (ADR-0112).

``dataforge install --skills`` copies the package's ``skill/`` folder into each
of :data:`SKILL_DIRS`, under the working directory or, with ``--global``, under
the home folder. Claude Code reads ``.claude/skills/``; Codex, Gemini CLI and
Cursor read ``.agents/skills/``. Each copy is stamped with the library version
in :data:`STAMP_FILE`.

Every other command checks the copies under both roots. When a stamp differs
from the library version, it prints one line to stderr, then notes the version
it warned about in :data:`WARNED_FILE` so the next command stays quiet. The
warning comes back after the next upgrade, and a fresh install clears it.

This module imports nothing but the standard library, so the check costs a
command no heavy import.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ._version import library_version

#: The command that installs the Skill. It is no Operation.
INSTALL_COMMAND = "install"
INSTALL_USAGE = """\
usage: dataforge install --skills [--global]

Copy the dataforge Skill into .claude/skills/dataforge/ and
.agents/skills/dataforge/ under the working directory, stamped with the library
version. --global copies it under the home folder instead. Run it again after
upgrading the library; until then every command's first run warns on stderr.
"""
#: The package Skill, shipped inside the package.
SKILL_SOURCE = Path(__file__).with_name("skill")
#: Where the Skill goes, relative to the working directory or the home folder.
SKILL_DIRS = (Path(".claude/skills/dataforge"), Path(".agents/skills/dataforge"))
#: The file in an installed copy that holds the library version it came from.
STAMP_FILE = ".dataforge-version"
#: The file in an installed copy that holds the library version last warned about.
WARNED_FILE = ".dataforge-version-warned"


def install_skills(root: Path) -> list[Path]:
    """Copy the package Skill into every skills folder under ``root``, stamped.

    An earlier copy is replaced whole, so a file the new Skill dropped does
    not linger.

    Parameters
    ----------
    root : Path
        The working directory, or the home folder for ``--global``.

    Returns
    -------
    list of Path
        The installed Skill folders, in :data:`SKILL_DIRS` order.
    """
    current = library_version()
    installed = []
    for relative in SKILL_DIRS:
        folder = root / relative
        if folder.is_symlink() or folder.is_file():
            folder.unlink()
        elif folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(SKILL_SOURCE, folder, ignore=shutil.ignore_patterns("__pycache__"))
        (folder / STAMP_FILE).write_text(current + "\n", "utf-8")
        installed.append(folder)
    return installed


def _read(path: Path) -> str | None:
    try:
        return path.read_text("utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def stale_skill_warning(roots: list[Path] | None = None) -> str | None:
    """Return the one-line warning for installed Skills stamped another version.

    A copy already warned about for this library version is skipped, and each
    copy warned about now is marked so, so the warning prints once.

    Parameters
    ----------
    roots : list of Path, optional
        Where to look for :data:`SKILL_DIRS`; the working directory and the
        home folder by default.

    Returns
    -------
    str or None
        The warning, or None when every installed copy is current, already
        warned about, or there is none.
    """
    current = library_version()
    if roots is None:
        roots = [Path.cwd()]
        try:
            roots.append(Path.home())
        except RuntimeError:  # no home folder to look in
            pass
    stale: list[tuple[Path, str]] = []
    for root in dict.fromkeys(roots):
        for relative in SKILL_DIRS:
            folder = root / relative
            stamp = _read(folder / STAMP_FILE)
            if stamp is None or stamp == current or _read(folder / WARNED_FILE) == current:
                continue
            stale.append((folder, stamp))
            try:
                (folder / WARNED_FILE).write_text(current + "\n", "utf-8")
            except OSError:
                pass  # cannot mark it: it warns again next time
    if not stale:
        return None
    copies = ", ".join(f"{folder} (stamped {stamp})" for folder, stamp in stale)
    return (
        f"dataforge: warning: the installed Skill does not match library version "
        f"{current}: {copies}. Run `dataforge install --skills` again, with "
        "`--global` for a copy in the home folder."
    )
