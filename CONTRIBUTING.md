# Contributing to DataForgeML

Thank you for your interest in contributing. This document covers everything you need to get started.

## Ground rules

- All changes go through a pull request — direct pushes to `main` are blocked.
- Every PR must be approved by the maintainer ([@DEVunderdog](https://github.com/DEVunderdog)) before it can merge.
- CI must be green (pytest on Python 3.11, 3.12, 3.13) before approval is given.

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
uv run pytest                                    # the suite
uv run pytest tests/unit/imputation -q           # one directory
uv run --python 3.11 pytest                      # a different interpreter from the matrix
uv run sphinx-build -W -b html docs/ docs/_build/html   # the docs
```

`uv.lock` is committed and CI runs `uv sync --locked`, which **fails if the lock disagrees
with `pyproject.toml`**. If you change a dependency, commit the regenerated lock in the
same PR.

---

## Dependency groups

Dev dependencies are split so that a job installs only what it uses — a Sphinx release
cannot redden the test matrix:

| Group  | Contains                | Sync it with                                        |
|--------|-------------------------|-----------------------------------------------------|
| `test` | pytest                  | `uv sync --no-default-groups --group test`          |
| `docs` | Sphinx toolchain        | `uv sync --no-default-groups --group docs`          |

A bare `uv sync` gives you both. These groups are **not** published in the wheel.

---

## Updating a dependency

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

**4. Bump the DataForgeML version.** ADR-0065 requires a bump to a pickle-critical
dependency to ship *as* a new library version — two environments must never wear the same
version label while carrying different scikit-learn/numpy/joblib. This **invalidates every
previously-saved `FittedUnit`**; users must retrain. Say so in the release notes.

**5. One PR, both files.** `pyproject.toml` and `uv.lock` always travel together — a lock
that disagrees with the caps is exactly the drift this design exists to prevent.

**6. Tag and release.** `publish.yml` builds and uploads; `deploy-docs.yml` follows.

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

## Opening a pull request

When your work is ready:

```bash
git push origin feature/<issue-number>-short-description
```

On GitHub, open a PR against `main`:

- **Title**: follow Conventional Commits — e.g. `feat(scope/90): adaptive k selection for KNNImputer`
- **Body**: include `Closes #<issue-number>` so the issue closes automatically on merge
- Describe what you changed and why, not just what the code does

CI will run automatically. Fix any failing tests before requesting review.

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
- Do not modify `pyproject.toml` version — releases are managed by the maintainer.
- Do not add dependencies without prior discussion in the issue thread.
