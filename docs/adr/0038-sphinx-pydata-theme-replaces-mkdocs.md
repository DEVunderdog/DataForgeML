# ADR 0038: Sphinx + PyData Sphinx Theme replaces MkDocs as the docs build toolchain

Supersedes ADR-0034 (toolchain only — docstring format and scope rules unchanged).

## Context

ADR-0034 selected MkDocs 1.x + mkdocstrings as the documentation build toolchain on the grounds that the existing ADRs are Markdown files and Sphinx was rst-first. That decision is no longer tenable:

- **MkDocs 1.x is abandoned upstream.** The project has received no maintenance releases.
- **MkDocs 2.0 removes the plugin system entirely.** The migration path eliminates both mkdocstrings and mkdocs-material with no supported replacement. Upgrading to 2.0 would require rebuilding the entire toolchain from scratch at unknown future cost.

These two facts together mean the toolchain selected in ADR-0034 has no viable upgrade path. Staying on 1.x pins us to an abandoned dependency indefinitely.

## Decision

Migrate the docs build toolchain from **MkDocs 1.x + mkdocstrings** to **Sphinx + PyData Sphinx Theme + MyST-Parser + `sphinx.ext.napoleon`**.

- `mkdocs.yml` is replaced by `docs/conf.py`.
- `mkdocstrings :::` directives are replaced by Sphinx `autoclass` / `autofunction` directives.
- `mkdocs serve` / `mkdocs build` are replaced by `sphinx-autobuild` / `sphinx-build`.
- MyST-Parser allows all existing Markdown files (ADRs, index pages) to be consumed by Sphinx as-is — no rewrite to rst is required.

## Rationale

- **Sphinx is the Python ecosystem standard.** It is governed by NumFOCUS and is the toolchain used by NumPy, pandas, and scikit-learn — the same ecosystem DataForgeML targets. Users of those libraries arrive already familiar with the output format.
- **PyData Sphinx Theme matches the visual language users expect.** NumPy, pandas, and scikit-learn all use it. A DataForgeML user moving between those docs and ours encounters a consistent reading experience.
- **`sphinx.ext.napoleon` reads numpy-style docstrings natively.** No docstring rewrites are required. The format mandated by ADR-0034 is exactly what `napoleon` expects.
- **MyST-Parser eliminates the Markdown migration cost.** The original rationale for preferring MkDocs over Sphinx was the cost of converting 34+ Markdown ADRs to rst. MyST-Parser removes that cost entirely — Markdown files are first-class Sphinx source.
- **Long-term maintenance.** Sphinx has a stable, actively maintained plugin ecosystem. There is no equivalent abandonment risk.

## What stays the same

**Docstring format and scope rules from ADR-0034 are fully in force and unchanged.**

- All in-scope symbols (Public API, Phase Orchestrators, Config dataclasses) must have numpy-style docstrings before a task is closed.
- The minimum contract (one-line summary, `Parameters`, `Returns`, `Raises` sections as applicable) is unchanged.
- `_`-prefixed private methods and functions remain exempt.
- `CLAUDE.md` enforcement language is unchanged.

`sphinx.ext.napoleon` reads the numpy-style format natively; no docstring rewrites are required as part of this toolchain migration.

## What changes

Toolchain only:

| Before (ADR-0034) | After (ADR-0038) |
|---|---|
| `mkdocs.yml` | `docs/conf.py` |
| `mkdocstrings :::` directives | `autoclass` / `autofunction` directives |
| `mkdocs serve` | `sphinx-autobuild` |
| `mkdocs build` | `sphinx-build` |
| mkdocstrings plugin | `sphinx.ext.autodoc` + `sphinx.ext.napoleon` |

## Consequences

- `mkdocs.yml` and any `mkdocstrings`-specific directives in `docs/api/` are replaced.
- `docs/conf.py` is added as the Sphinx configuration entry point.
- Build dependencies shift from `mkdocs`, `mkdocstrings[python]`, `mkdocs-material` to `sphinx`, `pydata-sphinx-theme`, `myst-parser`, `sphinx-autobuild`.
- All existing Markdown files in `docs/` (ADRs, index pages) continue to work without modification via MyST-Parser.
- The GitHub Actions `deploy-docs.yml` workflow (planned in ADR-0034, not yet merged) targets `sphinx-build` output instead of `mkdocs build` output.
