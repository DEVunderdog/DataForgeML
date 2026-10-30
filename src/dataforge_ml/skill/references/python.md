# Python (Code Mode)

Every command is also a Python function, for glue code such as a loop over folds. The CLI and Python run the same function, so arguments, results, Handles and Error Codes are identical.

## Calling

- Import from the package root: `from dataforge_ml import load, profile, split`. The function name is the command name in snake_case (`fit-unit` → `fit_unit`).
- Pass arguments by keyword, under their snake_case names (`--unit-id` → `unit_id=`). A JSON flag takes the same structure as a dict or list: `thresholds={"missingness": {"severity_high": 0.3}}`.
- Workspace Settings come from the env vars (`DATAFORGE_ML_WORKSPACE`, `DATAFORGE_ML_LITERAL_POLICY`, `DATAFORGE_ML_DECLARED_INPUTS`, `DATAFORGE_ML_CORE_BUDGET`), then `.dataforge/config.toml`. Python has no settings flags.

## Results

A call returns a frozen Wire Record:

- Read fields as attributes: `result.profile`, `result.warnings`.
- A field the CLI prints in Table Layout is a sequence of records: `for unit in recipe.units: unit.unit_id`.
- A result keyed by column has `.get(column)`, which returns that column's record.
- `.to_dict()` returns the exact JSON object the CLI prints.

## Errors

A failure raises a Library Error, a subclass of `dataforge_ml.records.DataForgeError`, one per category: `InvalidInput`, `UnknownName`, `Conflict`, `Infeasible`, `IncompatibleArtifact` and `Unexpected`. Catch the category, then branch on `err.code`. The record's fields read off the error too: `err.param`, `err.candidates`, `err.hint`. `err.to_dict()` is the JSON the CLI prints.

## Example

```python
from dataforge_ml import (
    compose_imputer, evaluate_imputation, fit_unit, impute, load, profile,
    resolve_recipe, route, split,
)
from dataforge_ml.records import UnknownName

frame = load(path="train.csv").frame
prof = profile(frame=frame).profile
parts = split(profile=prof, scheme={"kind": "kfold", "k": 3})
resolved = resolve_recipe(routing=route(profile=prof).routing)

# One recipe for every fold: units fitted from another recipe are refused.
for fold in parts.folds:
    units = [
        fit_unit(recipe=resolved.recipe, unit_id=u.unit_id, frame=fold.train).unit
        for u in resolved.units
    ]
    imputer = compose_imputer(recipe=resolved.recipe, units=units).imputer
    filled = impute(imputer=imputer, frame=fold.validation).frame
    report = evaluate_imputation(profile=prof, imputed=filled)
    print(fold.index, report.to_dict()["census"])

try:
    profile(frame="frame_00000000")
except UnknownName as err:
    print(err.code, err.candidates)
```
