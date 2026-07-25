# uv owns the whole development toolchain, and the capped pickle-critical dependencies are bumped by a manual runbook

ADR-0065 settled dependency *policy*: ranges published to consumers, upper caps on the numeric/pickle-critical deps, a committed lockfile for CI, and an automated freshness pipeline. It left the *toolchain* mostly alone ("build backend unchanged") and assumed Dependabot would cover both routine and security updates. Implementing it (issues #365/#366/#367) surfaced one factual error in that assumption and several places where a half-migration would leave determinism holes. This ADR amends ADR-0065 on those points and records the decisions that carry it into code. **ADR-0065's policy is unchanged and remains in force; what changes is the machinery and the update ritual.**

## The error being corrected

ADR-0065 states that Dependabot "covers both jobs" — routine version PRs and security PRs. **That is true only against open ranges.** Dependabot never proposes a version outside a declared constraint, it [updates `uv.lock` but not `pyproject.toml` constraints](https://github.com/dependabot/dependabot-core/issues/12788), and [`versioning-strategy` is unsupported for the uv ecosystem](https://github.com/dependabot/dependabot-core/issues/12162). So the moment the caps land, `scikit-learn`, `numpy`, `scipy`, and `joblib` become **invisible** to Dependabot — permanently and silently.

Executed literally and in order, #365 then #367 would therefore produce a repository in which **the four most critical dependencies are the only ones nobody is watching** — the precise inverse of the ticket's intent. The caps and the automation are not composable; one of them has to give.

## Locked decisions

- **uv owns the entire development toolchain, including the build backend.** `uv_build` replaces `setuptools.build_meta`; `uv build` and `uv publish` replace `python -m build` and the PyPI upload action. This **amends ADR-0065's "build backend unchanged"**. The *published dependency contract is untouched* — the wheel's `Requires-Dist` still carries ranges, so co-installability and the `pip install dataforge-ml` experience are exactly as ADR-0065 specified. This is a decision about the factory, not the product. Shipping pinned dependencies to consumers was reconsidered and re-rejected for the reasons ADR-0065 already gives: it makes the library un-co-installable and still fails to lock the transitive graph.

- **uv owns the interpreter too; `actions/setup-python` is removed.** A `.python-version` file names the interpreter, `astral-sh/setup-uv` provides uv, and uv fetches the Python. `setup-python` was silently floating the patch version from whatever the runner image shipped that week — locking every package while leaving the interpreter free is locking the door and leaving the window open. The cost is accepted: uv-managed Pythons are Astral's `python-build-standalone` builds, not the distributions consumers will typically run, and the project takes on a supply-chain dependency on Astral.

- **`requires-python` becomes `>=3.11`; the matrix is 3.11/3.12/3.13.** The previous `">3.10"` was a typo that excluded 3.10.0 while admitting 3.10.1, and CI had never actually run 3.10 despite `CONTRIBUTING.md` promising it. Support for 3.10 is dropped rather than fixed, because a `scikit-learn>=1.9` cap cannot be satisfied on it.

- **The lock stays universal (forked per interpreter) rather than collapsed to one environment.** `uv.lock` resolves e.g. numpy 2.4.6 on 3.11 and 2.5.1 on 3.12+. Determinism is therefore *per Python version*, not absolute — two contributors on the same interpreter get byte-identical environments, and the forks are exactly what the matrix exists to exercise. The absolute form was available (`requires-python = ">=3.12"`) and rejected as too expensive for the guarantee it adds, because **ADR-0063's runtime `produced_with` refuse is the real enforcement** and is unaffected by lock forking. A consequence to keep in view: a cap may legitimately span two minors (`numpy>=2.4,<2.6`) because the tested band itself spans two.

- **Dev dependencies move from the published `dev` extra to PEP 735 `[dependency-groups]`, split `test` / `docs`.** As an extra they were shipped in the wheel's metadata, advertising the maintainer's Sphinx toolchain to every consumer; and CI installed the whole documentation stack in order to run pytest, which meant a Sphinx release could redden the test matrix. Groups fix both. **This makes uv mandatory for contributors** — pip cannot install dependency groups — which is accepted and stated in `CONTRIBUTING.md`.

- **Standalone tools run ephemerally but version-pinned:** `uvx pip-audit@2.10.1`, never bare `uvx pip-audit`. An unpinned scanner reintroduces the exact failure ADR-0065 set out to kill — a pip-audit release turning the build red on a commit that touched a docstring. The asymmetry is deliberate: the **scanner binary** is pinned, while the **CVE database** is fetched fresh on every run. Keeping the scanner out of the lock also stops Dependabot from opening PRs against the project's own vulnerability tooling.

- **`joblib` is declared as a direct dependency.** It is imported directly (`_serialization.py`) and recorded in the `produced_with` stamp, yet reached the environment only transitively via scikit-learn. A future scikit-learn that vendors or drops joblib would have broken serialization silently. This is a correctness fix that is independent of the capping decision.

- **`scipy` is capped but deliberately *not* added to the `produced_with` stamp.** ADR-0065 calls scipy "pickle-critical" while `produced_with()` records only `scikit_learn`, `numpy`, `joblib`, `python`, and the schema version. The contradiction is resolved in favour of the code: adding a seventh stamp field is a format-schema change with artifact-invalidation consequences and must be its own ticket, not a passenger on a dependency PR. **The inconsistency is recorded here rather than silently settled.**

- **The four capped dependencies are bumped by a manual, on-demand runbook** documented in `CONTRIBUTING.md`; Dependabot is configured for everything else (uncapped deps, the transitive graph, GitHub Actions). Discovery is an explicit step that queries PyPI directly, because the caps blind every automated path. The runbook's load-bearing detail: **the cap must be raised before `uv lock --upgrade-package`**, since uv obeys the cap as a hard constraint and would otherwise report a no-op that reads as "nothing changed upstream."

## Status

accepted

Amends ADR-0065 (build backend; the "Dependabot covers both jobs" claim). ADR-0065's policy decisions — ranges published, caps on the pickle-critical four, lockfile for CI only, retrain-only recovery, pickle-critical bumps forcing a new library version — all stand.

## Considered Options

- **Keep `setuptools` and migrate only the dev environment.** The conservative sibling; leaves `publish.yml` resolving dependencies its own way. Rejected as exactly the residue that makes a migration feel half-done — one workflow with its own install path is where drift re-enters. The risk accepted in exchange: `uv_build` is younger than setuptools, a broken build backend breaks *releases*, and the release path is only exercised on tag pushes, so a regression surfaces late.
- **Cap to the next major (`scikit-learn>=1.9,<2`) instead of the next minor.** Would have restored full Dependabot automation and cut release churn substantially. Rejected because a `<2` cap admits many untested minors, which drains the caps of the "this is the tested band" meaning that justified them in ADR-0065.
- **Drop the caps entirely and rely on the runtime stamp.** Coherent, and closer to mainstream packaging advice — upper caps propagate into consumers' resolvers and cannot be retracted from an already-published wheel. Rejected because the stamp protects *artifacts*, not the fresh installer who has no artifacts yet and would silently run untested numerics.
- **An automated cap sweep** — a scheduled job checking PyPI for new minors of the capped four and opening the cap-raise PR. Rejected for now as custom machinery the project would own and have to maintain, with a failure mode (the sweep breaks silently) that returns you to manual without telling you. Revisit if manual discovery proves unreliable in practice.
- **Pinning `pip-audit` inside the lock as a `tools` group.** Rejected: it makes the vulnerability scanner a dependency Dependabot manages, which is circular and noisy.

## Consequences

- **Contributors must install uv**, and can no longer use `pip install -e ".[dev]"`. Setup collapses to `git clone && uv sync`, with no interpreter prerequisite and no venv activation.
- **`requirements.txt` is deleted** (it was a `pip freeze` containing an `-e git+ssh://` self-reference and a dead mkdocs toolchain left over from the ADR-0034 → ADR-0038 migration). The stale `site/` mkdocs output directory goes with it.
- **CI gains a `pip-audit` job** and asserts `uv sync --locked`, so a PR whose `pyproject.toml` and `uv.lock` disagree fails rather than merging a silent drift.
- **Nobody is automatically notified of new scikit-learn/numpy/scipy/joblib releases.** This is the accepted cost of caps-plus-manual, and it is the single most likely way this design decays. Step 0 of the runbook exists solely to counteract it; if it is skipped for long enough, the caps become a slow-acting freeze.
- **A CVE whose fix sits above a cap puts the manual runbook on the critical path for a security fix.** `pip-audit` will fail CI continuously until the cap is raised, so the condition is loud rather than silent — but the latency is real.
- Python 3.10 users are dropped, which is a compatibility break to note in release notes.
