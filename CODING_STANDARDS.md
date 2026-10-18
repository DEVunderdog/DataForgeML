# Coding Standards

Judgement calls that no linter or test can check. The code-review step holds every
diff against each rule here. Anything mechanical belongs in a linter, a type check,
or a contract test instead, so this file never lists it.

Each rule states what the code does and why.

## Reuse the shared helpers

A new Operation reuses what its sibling Operations already share. When a helper
exists, call it. When a second Operation needs logic that so far lives in one
place, extract it into a shared module and have both call it.

Look in these modules before writing a helper:

- `_boundary/_detail.py`: cutting a result page with its `omitted` entry, and
  building the `unknown_column` error.
- `_boundary/_workspace.py`: reading and writing Handles.

**Why:** a pasted copy drifts. The paging rule, the `unknown_column` error shape,
and the Handle checks are part of the Agent-facing contract. When one copy changes
and another does not, two Operations answer the same mistake in two ways. On #598 an
Operation pasted all three helpers, and only review caught it.

**In review:** flag a function in the diff that reproduces the shape of a shared
helper, even under a new name. The fix is to call the helper, or to extract the
shared shape.

## Document the internal seams in numpy style

The internal seams that Operations call each have a numpy docstring before the task
is done (ADR-0034, rescoped by ADR-0108). The scope covers each seam, its public
methods, and the record it returns:

- the router, `dataforge_ml.imputation.route` (the package-root `route` is the
  Operation);
- `resolve_recipe` and `fit_unit`;
- the profiler, `StructuralProfiler`;
- the splitter, `DataSplitter`;
- the generic C2ST seam, `c2st` (the contract a second adapter is written
  against, ADR-0087 amended).

`_`-prefixed methods and functions are exempt.

The sections:

```python
def method(self, param: Type) -> ReturnType:
    """One-line summary of what this does.

    Longer prose only when the behaviour is not obvious.

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

- `Parameters` for every argument beyond `self`.
- `Returns` for every non-`None` return.
- `Raises` for every exception the code raises itself.
- Name the `CONTEXT.md` terms; never redefine them.

When a diff touches a file in scope, every in-scope symbol in that file gets its
docstring, not only the ones the diff changed.

Operations are out of this scope. Their docstrings follow the Agent rule in
ADR-0108, and the contract tests enforce it.

**Why:** the maintainers are coding Agents. They read these seams the way a user
reads an Operation, so the docstring is the contract they code against. A file left
half documented tells the next reader that the gaps are fine.

**In review:** flag an in-scope symbol in a touched file that lacks a docstring or
one of the sections above, and a docstring that redefines a `CONTEXT.md` term.
