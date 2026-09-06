# Imputation

The imputation door is user-orchestrated (ADR-0060, ADR-0071): `decide()`
builds the pure plan, a caller-written `fit_unit` loop trains it one unit at a
time, and `FittedImputer.compose` rolls the trained units into the
whole-dataframe imputer.

The plan has two authors (ADR-0083): `decide()` derives it from a profile, and
`author()` takes it from a person who knows their own data. The plan either
produces is the same object, and everything downstream treats them alike.

`author()` is also how your own estimator gets onto a plan:
`author(..., estimators={"mice": my_model})` stamps the MICE block
`ModelChoice.Custom` and trains it with `my_model` itself — the object you
passed, not a clone. Two consequences: mutating it after authoring moves every
plan holding it, and it is **never serialized**, so a saved plan reloads as
`Custom` with the slot empty and raises at fit time until you supply the
estimator again.

The two authors also compose: `author(map, base=plan)` re-authors a plan you
already have, so a user who ran `decide()` and disagrees with two of forty
columns changes those two and keeps the thirty-eight the router got right.
A column the map does not name keeps its decision verbatim; sentinels,
`config_snapshot` and custom estimators carry over; the decided hyperparameter
base is gap-filled for new units and never overwritten for carried ones; and
tuning deltas ride along for units that survive the edit but are removed
outright for those it dissolved. Re-authoring a *deserialized* plan works, but
the units saved beside it were fitted under the old strategies — the result is
a plan to fit fresh, never one to pair with those units.

## The guarantee boundary

A hand-authored plan is the same object as a decided one, so almost everything
the library promises holds for it unchanged. What narrows is narrow and
specific: the promises that depend on the library having *chosen the
estimator*, and the promises that depend on the door having *seen a profile*.

These hold for every plan, whichever author wrote it:

| Guarantee | Why it still holds |
|---|---|
| Frame coverage — every column of `columns` gets a decision | By construction: `default` stamps every column the map does not name |
| Observed-Value Preservation (ADR-0078) | Enforced inside each unit's own `transform`, below the plan |
| Composition — `FittedImputer.compose`'s exact-coverage gate (ADR-0071) | Units are a pure projection of `column_decisions` |
| Execution — `fit_unit`, single-track raise, unit independence | Nothing on the fit path asks who wrote the plan |
| Persistence of the plan and of each fitted unit (ADR-0072) | Unchanged, *except* a `Custom` estimator, which is dropped and re-supplied |
| Sentinel normalisation at fit and at transform | Via `author`'s plan-level `numeric_sentinels` / `string_sentinels` |
| Dial validation — `with_hyperparameters` against a complete base | The door writes the same decided base `decide()` does, from the same table |
| `core_budget` arithmetic | For library estimators; see the row below for `Custom` |

These hold only where the library chose the estimator, or only where a profile
was read:

| Guarantee | How it narrows for a hand-authored plan |
|---|---|
| `core_budget` arithmetic for a `Custom` unit | Priced at `1` — the truthful count of cores *the library* spends. The budget cannot see parallelism you configured inside your own estimator |
| Determinism across `n_jobs` (ADR-0069's RandomForest pinning) | Holds **only** where the library built the estimator. A foreign estimator's determinism is its own |
| Routing rationale (`signals`) | **Empty**, legally: a hand-authored decision explains no routing, because there was none |
| Size-guard warnings before fit | **None** — the door has no frame to measure. A plan that cannot train raises at fit |
| Reload-and-fit without further input | **Not** for a `Custom` plan: `build_from_choice` raises until you supply the estimator again |
| Tuning of the estimator's own dials | **Not** for a foreign estimator — the library cannot enumerate dials it does not define |

One asymmetry follows from the door having no frame. A manual author gets a
hard `UnitNotTrainableError` at fit where a *forced* automatic one gets a
`FitSignals` warning: parity would mean handing a decide-time door a frame, and
decide-time doors do not touch data.

### A self-parallelising estimator

`core_budget` prices a `Custom` unit at `1` in every branch, because the
library never sets a foreign estimator's parameters. If the estimator you
supplied parallelises itself — `n_jobs`, `nthread`, `num_threads`, or an
internal pool exposed as no parameter at all — the budget is structurally blind
to it, and there is deliberately no probe: one would fire on correct plans while
missing the spellings it does not know.

The documented pattern is to fit that unit **outside your pool**, where it has
the machine to itself, and to use `total_cores` as the reservation hatch for
the rest:

```python
budget = core_budget(plan, max_workers=n_workers, total_cores=n_cores - reserved)

# The self-parallelising unit runs alone, with the whole machine.
(mice,) = plan.units_for(ImputationStrategy.MICE)
custom = fit_unit(plan, mice, df)

with ThreadPoolExecutor(max_workers=n_workers) as pool:
    rest = [
        pool.submit(fit_unit, plan, unit, df, n_jobs_inner=budget[unit.unit_id])
        for unit in plan.units
        if unit.strategy is not ImputationStrategy.MICE
    ]
    ...
```

## Decision layer

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._decision_assembler.decide
```

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._authoring.author
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._authoring.AuthoredColumn
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ImputationDecision
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ColumnImputationDecision
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ImputationUnit
   :members:
   :undoc-members:
   :show-inheritance:
```

## Execution layer

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._unit_fit.fit_unit
```

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._unit_fit.core_budget
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._unit_fit.UnitFitResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fit_signals.FitSignals
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fitted_imputer.FittedUnit
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fitted_imputer.FittedImputer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ImputationConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.NumericImputationConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ImputationStrategy
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ModelChoice
   :members:
   :undoc-members:
   :show-inheritance:
```

## Results

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ImputationResult
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ColumnImputationRecord
   :members:
   :undoc-members:
   :show-inheritance:
```

## Exceptions and warnings

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._unit_fit.UnitNotTrainableError
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fit_signals.ImputationFitWarning
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fitted_imputer.UnseenColumnError
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fitted_imputer.FittedColumnAbsentError
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._fitted_imputer.DroppedColumnAbsentWarning
   :members:
   :undoc-members:
   :show-inheritance:
```
