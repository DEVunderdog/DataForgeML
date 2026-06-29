# ADR 0034: Documentation strategy — numpy-style docstrings as API contract, mkdocs + mkdocstrings for generation

**Status:** Superseded by ADR-0038 (toolchain only — docstring rules unchanged)

## Context

DataForgeML had two living documentation systems: `CONTEXT.md` (domain glossary) and `docs/adr/` (architectural decisions). Neither addressed the API contract layer — what `ImputationOrchestrator.fit()` accepts, returns, and raises was not captured anywhere outside the source code itself.

Three documentation gaps existed simultaneously:

1. **Library-consumer gap** — a user installing DataForgeML could not understand the API without reading source.
2. **Code-generation gap** — no rule enforced docstring coverage, so documentation degraded silently as new code was written.
3. **Contributor gap** — the *what/how* of each module was not written down; only the *why* (ADRs) and *what is this concept* (CONTEXT.md) were.

A unified documentation strategy was needed that covered all three layers without adding unbounded maintenance burden.

## Decision

1. **Docstrings are the API contract layer.** `CONTEXT.md` remains the domain glossary. `docs/adr/` remains the decision record. Docstrings carry the API contract: parameters, return types, and exceptions. They do not repeat domain definitions from `CONTEXT.md` — they reference those terms by name only.

2. **Scope: Public API + Phase Orchestrators + Config classes.** All classes and their public methods that fall into one of these categories must have a compliant docstring:
   - Everything exported from `dataforge_ml.__init__`
   - Phase Orchestrators (`StructuralProfiler`, `ImputationOrchestrator`, and equivalents in future phases)
   - All Config dataclasses (`PipelineConfig`, `ProfileConfig`, `ImputationConfig`, Phase Sub-Configs, `SplitConfig`)
   - All standalone public functions in the Public API
   - `_`-prefixed (private) methods and functions are exempt.

3. **Format: numpy-style, minimum contract.** Every compliant docstring must contain:
   - A one-line summary on the opening line.
   - `Parameters` section for any argument beyond `self`.
   - `Returns` section for any non-`None` return value.
   - `Raises` section for any exception the method explicitly raises.
   - A longer prose explanation is optional and added only when behaviour is non-obvious.

4. **Tooling: mkdocs + mkdocstrings.** The `mkdocstrings` plugin pulls docstrings from source and renders them as API reference pages. `mkdocs` serves both the API reference and the existing `docs/adr/` ADRs in a single flat site, requiring no migration of existing Markdown files.

5. **Site structure: flat.** API reference pages (`docs/api/`) and ADRs (`docs/adr/`) are siblings under one `mkdocs.yml` nav. No user/contributor split at this stage.

6. **Hosting: local now, GitHub Pages after retroactive pass.** During the retroactive docstring pass, the site is local-only (`mkdocs serve`). Once coverage over all in-scope files is complete, a `deploy-docs.yml` GitHub Actions workflow is added to publish to GitHub Pages on every push to `main`.

7. **Retroactive pass: big-bang, one dedicated session.** All in-scope files that currently lack compliant docstrings are updated in a single focused pass before any other feature work proceeds. Partial coverage is not shipped.

## Rationale

- **Docstrings over hand-authored API docs** — a hand-authored API page in `docs/` requires two edits on every API change (code + docs). Docstrings are the single source of truth; drift is structurally impossible.
- **mkdocs + mkdocstrings over Sphinx** — the existing `docs/adr/` are Markdown. Sphinx is rst-first; migrating 33 ADRs to rst adds cost with no benefit. mkdocs plugs them in as-is.
- **mkdocs + mkdocstrings over pdoc** — pdoc is zero-config but code-only; the ADRs cannot be included in the same site. mkdocs gives one unified site for both layers at modest config cost.
- **Numpy-style over Google-style** — DataForgeML sits alongside NumPy, pandas, and sklearn; all use numpy-style. Users reading those docs arrive already fluent in the format. Existing partial docstrings in the codebase already use numpy-style — switching incurs migration cost with no gain.
- **Scope limited to Public API + orchestrators + configs** — pure private helpers (`_classify_numeric_kind`, `_resolve_effective_nulls`) are self-describing by name and surrounding context. Documenting them adds maintenance burden without serving users or contributors in any meaningful way.
- **Big-bang retroactive pass over touch-as-you-go** — carrying partial coverage forward means new code builds on undocumented foundations. A single focused pass eliminates documentation debt before it compounds.
- **Local-first, GitHub Pages after pass** — publishing a half-complete API reference would expose gaps publicly. The phase gate (docstrings complete → publish) keeps the public site accurate.

## Consequences

- A `mkdocs.yml` and `docs/api/` structure are added to the repo root.
- `CLAUDE.md` is created with a hard rule: all in-scope code generated or modified must have compliant numpy-style docstrings before the task is closed.
- A dedicated retroactive pass adds docstrings to all in-scope files that currently lack them.
- Once the retroactive pass is complete, a `deploy-docs.yml` GitHub Actions workflow is added to publish to GitHub Pages.
- `CONTEXT.md` and `docs/adr/` are unaffected — they continue to serve their existing roles unchanged.
