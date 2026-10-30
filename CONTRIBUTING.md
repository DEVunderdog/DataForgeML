# Contributing to DataForgeML

Thank you for your interest in contributing. This document covers everything you need to get started.

## Ground rules

- All changes go through a pull request — direct pushes to `main` are blocked.
- Every PR must be approved by the maintainer ([@DEVunderdog](https://github.com/DEVunderdog)) before it can merge.
- CI must be green (pytest on Python 3.11, 3.12, 3.13) before approval is given.
- **Version changes and dependency management are maintainer-only.** Anything that touches
  the DataForgeML version, the dependency declarations in `pyproject.toml`, or `uv.lock` is
  handled exclusively by the maintainer ([@DEVunderdog](https://github.com/DEVunderdog)).
  Those changes are made **directly on `main`** — the maintainer is the only person with
  that access, and nobody else has it. Do not open PRs for them.

---

## Setting up your development environment

DataForgeML uses [uv](https://docs.astral.sh/uv/) for everything: the virtual environment,
the dependency lock, the Python interpreter itself, and the release build. **uv is required
— there is no pip fallback**, because the dev dependencies live in PEP 735 dependency
groups, which pip cannot install.

Install uv ([instructions](https://docs.astral.sh/uv/getting-started/installation/)), then:

```bash
git clone https://github.com/DEVunderdog/DataForgeML.git
cd DataForgeML
uv sync
```

That is the whole setup. You do **not** need Python installed beforehand and you do **not**
activate anything — `uv sync` reads `.python-version`, fetches that interpreter if it is
missing, creates `.venv`, and installs the exact versions pinned in `uv.lock`. Everyone,
including CI, gets a byte-identical environment.

Run things with `uv run`, which syncs first if anything is stale:

```bash
uv run pytest                                    # the suite, in parallel
uv run ruff check                                # lint; CI and the hook hold it at zero
uv run pytest tests/unit/imputation -q           # one directory
uv run pytest -n 0 tests/contract/test_load.py   # serially, to debug one test
uv run --python 3.11 pytest                      # a different interpreter from the matrix
```

The suite runs in parallel by default: `pyproject.toml` sets `addopts = "-n auto"`, so
pytest-xdist starts one worker per physical core. Pass `-n 0` to run in one process, where
`pdb`, `-s` and print debugging behave normally, and to refresh snapshots
(`DATAFORGE_ML_UPDATE_SNAPSHOTS=1`), which rewrites files other workers read. A test must
not depend on another test's leftovers, because the tests that share a worker change from
run to run.

`uv.lock` is committed and CI runs `uv sync --locked`, which **fails if the lock disagrees
with `pyproject.toml`**. If you change a dependency, commit the regenerated lock in the
same PR.

### Install the pre-commit hook

```bash
git config core.hooksPath .githooks
```

Run it once per clone. Every commit then runs `.githooks/pre-commit`. The hook does three
things. It runs `ruff check` over the repo. Then it imports every module under
`dataforge_ml.operations` and `dataforge_ml.splitting`, and names any import that fails.
Then it runs `tests/unit`. The whole hook takes under a
minute. A failure refuses the commit. Contract and integration tests stay out of the hook
because they are slow; CI runs them.

The hook checks your working tree, not only what you staged. Unstaged edits count.

---

## Dependency groups

Dev dependencies live in a dependency group, so a job installs only what it uses:

| Group  | Contains                          | Sync it with                                 |
|--------|-----------------------------------|----------------------------------------------|
| `test` | pytest, pytest-xdist, tiktoken    | `uv sync --no-default-groups --group test`   |

A bare `uv sync` gives you `test`. Groups are **not** published in the wheel.

---

## Updating a dependency

> **Maintainer-only — reference material.** This section documents how the maintainer
> ([@DEVunderdog](https://github.com/DEVunderdog)) bumps dependencies and versions. It is
> written down so the process is auditable, **not** as an invitation to run it. These
> changes land directly on `main`, which only the maintainer can push to. If you are a
> contributor, read it for context and stop there — a PR that edits dependency pins,
> `uv.lock`, or the project version will be closed.

Most dependencies are handled by Dependabot. **Four are not**: `scikit-learn`, `numpy`,
`scipy`, and `joblib` carry upper caps in `pyproject.toml` (ADR-0065), and Dependabot
cannot see past a cap — it will never open a PR for them, and it will not warn you that
it isn't. Those four are bumped by hand, in this order.

**0. Find out what exists.** Nothing will tell you. Ask PyPI directly, which bypasses the
caps that the resolver would otherwise obey:

```bash
for p in scikit-learn numpy scipy joblib; do
  printf "%-14s latest=%s\n" "$p" \
    "$(curl -s https://pypi.org/pypi/$p/json | jq -r .info.version)"
done
uv pip list --outdated       # the uncapped dependencies
```

**1. Raise the cap *first*, in `pyproject.toml`.** This step is easy to get backwards: uv
treats the cap as a hard resolution constraint, so `uv lock --upgrade` is a **no-op** while
the old cap stands, and you will wrongly conclude nothing has changed upstream.

```toml
"scikit-learn>=1.10,<1.11",   # was >=1.9,<1.10
```

**2. Re-resolve just that package** — targeted, not blanket, so the diff stays reviewable:

```bash
uv lock --upgrade-package scikit-learn
git diff uv.lock
```

**3. Test every interpreter.** `uv.lock` resolves different versions per Python version, so
local green is not matrix green:

```bash
uv run --python 3.11 pytest && uv run --python 3.12 pytest && uv run --python 3.13 pytest
```

**4. Bump the DataForgeML version.** dependency to ship *as* a new library version

**5. One PR, both files.** `pyproject.toml` and `uv.lock` always travel together — a lock
that disagrees with the caps is exactly the drift this design exists to prevent.

**6. Tag and release.** `publish.yml` builds and uploads.

If a CVE fix sits above a cap, this runbook is what unblocks it — `pip-audit` will be
failing CI until you run it.

---

## Picking an issue to work on

Browse the [open issues](https://github.com/DEVunderdog/DataForgeML/issues). Before starting work:

1. Comment on the issue to let the maintainer know you're picking it up.
2. Wait for confirmation — this avoids duplicate effort.
3. Do not start work on an issue that is already assigned to someone.

---

## Branch workflow

Always branch off a fresh `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/<issue-number>-short-description
```

Examples:
- `feature/90-knn-hyperparameter-selection`
- `feature/103-per-column-strategy`
- `feature/108-numpy-to-df-utils`

Keep the description short — 3 to 5 words.

---

## Commit message convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Every commit message must start with one of these prefixes:

| Prefix | Use for |
|---|---|
| `feat:` | new behaviour or capability |
| `fix:` | correcting a bug |
| `refactor:` | restructuring without changing behaviour |
| `test:` | adding or fixing tests |
| `docs:` | docstrings, README, ADRs |
| `chore:` | dependency updates, config changes |

Examples:

```
feat: add adaptive k selection for KNNImputer
fix: prevent Int64 overflow in _numpy_to_df sentinel handling
test: add unit tests for BoundedDiscrete mode imputation
docs: add numpy docstrings to ImputationExecutor
```

---

## Documentation requirement

Every class, every public method, and every exported standalone function you add or modify **must** have a numpy-style docstring. This is enforced as a project rule — PRs that add undocumented public symbols will not be approved.

Required structure:

```python
def method(self, param: Type) -> ReturnType:
    """One-line summary of what this does.

    Parameters
    ----------
    param : Type
        Description of param.

    Returns
    -------
    ReturnType
        Description of return value.

    Raises
    ------
    ErrorType
        When this condition occurs.
    """
```

- `Parameters`: required for any argument beyond `self`.
- `Returns`: required for any non-`None` return value.
- `Raises`: required for any exception the method explicitly raises.
- Private (`_`-prefixed) methods are exempt.

---

## Adding an Operation

An Operation touches more files than its own. Most of the spots below are enforced by a
contract test, but you only learn that after a long run. Work down this list instead.
`<name>` is the Operation's snake_case name, for example `explain_routing`.

**Write the Operation**

| Where | What goes there |
|---|---|
| `src/dataforge_ml/operations/<name>.py` | The thin Operation file: the function under `@operation(effect=..., phase=..., scales_with_data=..., result_budget_tokens=...)`, the Wire Records only it returns, and one call into internal code. |
| The same file, the docstring | Written for Agents: a summary line plus the sections in `OPERATION_SECTIONS` (`_boundary/_docstring.py`). The `Example` must run as shell commands; `test_help.py` checks it. |
| `src/dataforge_ml/_boundary/` | The internal boundary module the Operation calls into, where its logic lives. Usually a new `_<name>.py`. Related Operations may share one: the profile detail reads all live in `_profile_detail.py`. |
| `src/dataforge_ml/_boundary/_error_codes.py` | One `ErrorCodeEntry` (code, category, hint, and `fields` if any) for every new error or warning code the Operation raises. `test_error_registry.py` fails on a raise with no registered code. |
| `CONTEXT.md` | Any new domain term the Operation introduces. Skip this if there is none. |
| `src/dataforge_ml/skill/glossary/<phase>.md` | A glossary entry, written for Agents, for every new capitalised term the docstring or reference uses. A term every phase uses goes in `common.md`. `test_glossary_coverage.py` fails on a term with no entry. |
| `_boundary/_operation.py` `PHASES` and `_boundary/_help_build.py` `PHASE_ORDER` | Only when the Operation needs a phase that does not exist yet. Add it to both. `test_help.py` fails when a phase is missing from `PHASE_ORDER`. |
| `_glossary.py` `PHASES` and `src/dataforge_ml/skill/glossary/<phase>.md` | The same new phase, with its glossary file. `test_glossary.py` fails when the two phase lists differ. |
| `src/dataforge_ml/skill/references/<phase>.md` | When to reach for the new command, and any judgement call `--help` leaves out. A new phase needs its own file, linked from `SKILL.md`; `test_skill.py` fails without both. |

**Register it**

| Where | What goes there |
|---|---|
| `src/dataforge_ml/operations/__init__.py` | The module name, in the alphabetical import list. Importing the package registers every Operation. |
| `src/dataforge_ml/__init__.py`, `__all__` | `"<name>",`, in alphabetical order. The package root exports the Operations and nothing else; `test_operation_contract.py` fails when the list and the registered Operations differ. |

**Join the contract tests**

| Where | What goes there |
|---|---|
| `tests/contract/test_<name>.py` | The Operation's own contract tests. |
| `tests/contract/test_operation_contract.py`, `WORST_CASE` | A function that makes the Operation's largest result on the supported fixtures, plus its entry in the table. The test checks that result is under the Result Budget. |
| `tests/contract/test_operation_contract.py`, `PENDING_OPERATIONS` | Delete the Operation's entry if it has one. A pending entry for a registered Operation fails a test. |
| `tests/contract/test_literal_policy.py`, `CALLS` | Canary calls for the Operation: at least one that succeeds and one that fails. Most existing failures misspell a fixture column, such as `"citty"`. |

**Regenerate the derived files**

All three are committed, and a contract test fails when any is stale. Run them after the
Operation's signature and docstring are final.

```bash
# The committed CLI help, src/dataforge_ml/_help.json
uv run python -m dataforge_ml._boundary._help_build

# The command index in the package Skill, src/dataforge_ml/skill/SKILL.md
uv run python -m dataforge_ml._boundary._skill_build

# The schema snapshots, tests/contract/snapshots/schemas/<name>.json (also refreshes the help and the Skill index)
DATAFORGE_ML_UPDATE_SNAPSHOTS=1 uv run pytest -n 0 tests/contract
```

The snapshot refresh needs `-n 0`; see "Setting up your development environment" above.
Read the diff of both before you commit. A change to another Operation's snapshot means
you changed a shared record.

---

## Opening a pull request

When your work is ready:

```bash
git push origin feature/<issue-number>-short-description
```

On GitHub, open a PR against `main`:

- **Title**: follow Conventional Commits — e.g. `feat(scope/90): adaptive k selection for KNNImputer`
- **Body**: include `Closes #<issue-number>` so the issue closes automatically on merge
- Describe what you changed and why, not just what the code does

CI runs on every push to a `feature/**` branch, before any PR is open, and again on the PR
itself. Fix any CI failures before requesting review.

---

## What happens after you open a PR

1. CI runs pytest on Python 3.11, 3.12, and 3.13, plus a `pip-audit` CVE scan.
2. The maintainer reviews your PR — expect feedback or questions.
3. Address review comments with new commits on the same branch (do not force-push).
4. Once approved and CI is green, the maintainer merges.

---

## What not to do

- Do not open a PR without a linked issue.
- Do not combine multiple unrelated issues in one PR.
- Do not modify the `pyproject.toml` version — versioning and releases are maintainer-only
  and land directly on `main`.
- Do not modify dependency declarations, version caps, or `uv.lock` — dependency management
  is maintainer-only and lands directly on `main`.
- Do not add dependencies without prior discussion in the issue thread.
