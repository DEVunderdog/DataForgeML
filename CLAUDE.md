# DataForgeML — Claude Code Instructions

## Documentation rule (ADR-0034)

Every class, every public method on that class, and every exported standalone function that falls within documentation scope **must** have a numpy-style docstring before the task is considered done.

### Documentation scope

A symbol is **in scope** if it belongs to one of:
- Everything exported from `dataforge_ml.__init__`
- Phase Orchestrators (`StructuralProfiler`, `ImputationOrchestrator`, and future phase orchestrators)
- All Config dataclasses (`PipelineConfig`, `ProfileConfig`, `ImputationConfig`, all Phase Sub-Configs, `SplitConfig`)
- All standalone public functions in the Public API

`_`-prefixed (private) methods and functions are **exempt**.

### Required docstring structure

```python
def method(self, param: Type) -> ReturnType:
    """One-line summary of what this does.

    Longer prose explanation only when behaviour is non-obvious (optional).

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

- `Parameters` section: required for any argument beyond `self`.
- `Returns` section: required for any non-`None` return value.
- `Raises` section: required for any exception the method explicitly raises.
- Do not repeat domain definitions from `CONTEXT.md` — reference those terms by name only.
- Do not document `_`-prefixed helpers.

### When modifying existing in-scope files

Add missing docstrings to all in-scope symbols in that file before closing the task. Do not leave a file partially documented after touching it.
