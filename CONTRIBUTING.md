# Contributing to DataForgeML

Thank you for your interest in contributing. This document covers everything you need to get started.

## Ground rules

- All changes go through a pull request — direct pushes to `main` are blocked.
- Every PR must be approved by the maintainer ([@DEVunderdog](https://github.com/DEVunderdog)) before it can merge.
- CI must be green (pytest on Python 3.10, 3.11, 3.12) before approval is given.

---

## Setting up your development environment

```bash
git clone https://github.com/DEVunderdog/DataForgeML.git
cd DataForgeML
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Verify everything works:

```bash
pytest
```

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
docs: add numpy docstrings to ImputationOrchestrator
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

1. CI runs pytest on Python 3.10, 3.11, and 3.12.
2. The maintainer reviews your PR — expect feedback or questions.
3. Address review comments with new commits on the same branch (do not force-push).
4. Once approved and CI is green, the maintainer merges.

---

## What not to do

- Do not open a PR without a linked issue.
- Do not combine multiple unrelated issues in one PR.
- Do not modify `pyproject.toml` version — releases are managed by the maintainer.
- Do not add dependencies without prior discussion in the issue thread.
