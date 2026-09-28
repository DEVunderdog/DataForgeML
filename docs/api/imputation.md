# Imputation

The imputation door is user-orchestrated (ADR-0071, ADR-0088): `route()`
produces the pure routing, `resolve_recipe()` resolves dials and profile estimates
into an immutable recipe, `derive_units()` materialises the executable units,
a caller-written `fit_unit` loop trains each unit, and `FittedImputer.compose` rolls
the trained units into the whole-dataframe imputer.

The routing has two authors (ADR-0083, ADR-0090): `route()` derives it from a profile,
and `author()` takes it from a person who knows their own data. The routing either
produces is the same object (`ImputationRouting`), and everything downstream treats
them alike.

`author()` reads names and types only. `profile=` supplies the column universe and
each column's semantic type. A non-`Numeric` column may be declared only
`Passthrough` or `Dropped`. The door writes no dials and no estimates: the
bimodal centres, `feature_cols` and `domain_snap_bounds` are computed by
`resolve_recipe()`, and any still missing are supplied with
`recipe.with_estimates(...)`.

`routing.with_model_choice(...)` is the one door for the MICE block's estimator,
on a routed or an authored routing alike. Pass a `ModelChoice` member to pick a
library family. Pass your own estimator instance to stamp the block
`ModelChoice.Custom` and train it with that object itself, not a clone. A bare
`ModelChoice.Custom` raises, and so does a routing with no MICE block. Two
consequences of holding the object: mutating it after the edit moves every
routing holding it, and it is **never serialized**, so a saved routing reloads as
`Custom` with the slot empty and raises at fit time until you supply the
estimator again.

The two authors also compose: `author(map, base=routing)` re-authors a routing you
already have, so a user who ran `route()` and disagrees with two of forty
columns changes those two and keeps the thirty-eight the router got right.
A column the map does not name keeps its `ColumnRouting` verbatim; a named
column is rebuilt from the map and keeps only its semantic type. MICE membership
decides the estimator: if MICE columns remain, the model choice and estimator
carry; if the edit creates the block, it is stamped `BayesianRidge`; if the edit
dissolves it, both go. The recipe resolves estimates for any changed strategies.
Re-authoring a *deserialized* routing works, but
the units saved beside it were fitted under the old strategies — the result is
a routing to fit fresh, never one to pair with those units.

## The guarantee boundary

A hand-authored routing is the same object as a routed one, so almost everything
the library promises holds for it unchanged. What narrows is narrow and
specific: the promises that depend on the library having *chosen the
estimator*, and the promises that depend on the door having *seen a profile*.

These hold for every routing, whichever author wrote it:

| Guarantee | Why it still holds |
|---|---|
| Frame coverage — every column of the profile gets a routing | By construction: `default` stamps every column the map does not name |
| Observed-Value Preservation (ADR-0078) | Enforced inside each unit's own `transform`, below the recipe |
| Composition — `FittedImputer.compose`'s exact-coverage gate (ADR-0071) | Units are a pure projection of `routing.column_routings` |
| Execution — `fit_unit`, single-track raise, unit independence | Nothing on the fit path asks who wrote the routing |
| Persistence of routing, recipe, and each fitted unit (ADR-0072, ADR-0096) | Unchanged, *except* a `Custom` estimator, which is dropped and re-supplied |
| Sentinel normalisation at fit and at transform | The recipe carries `numeric_sentinels` / `string_sentinels` off the profile, whichever author wrote the routing |
| Dial validation — `with_hyperparameters` on `ImputationRecipe` | Validated against unit hyperparameters |

These hold only where the library chose the estimator, or only where a profile
was read:

| Guarantee | How it narrows for a hand-authored routing |
|---|---|
| Determinism across `n_jobs` (ADR-0069's RandomForest pinning) | Holds **only** where the library built the estimator. A foreign estimator's determinism is its own |
| Routing rationale (`signals`) | **Empty**, legally: a hand-authored routing explains no automatic routing, because there was none |
| Size-guard warnings before fit | **None** — the door has no frame to measure. A unit that cannot train raises at fit |
| Reload-and-fit without further input | **Not** for a `Custom` routing: `build_from_choice` raises until you supply the estimator again |
| Tuning of the estimator's own dials | **Not** for a foreign estimator — the library cannot enumerate dials it does not define |

One asymmetry follows from the door having no frame. A manual author gets a
hard `UnitNotTrainableError` at fit where a *forced* automatic one gets a
`FitSignals` warning: parity would mean handing a route-time door a frame, and
route-time doors do not touch data.

### A self-parallelising estimator

The library never sets a foreign estimator's parameters. If the estimator you
supplied parallelises itself — `n_jobs`, `nthread`, `num_threads`, or an
internal pool exposed as no parameter at all — the library cannot see it, and
there is deliberately no probe: one would fire on correct plans while missing
the spellings it does not know.

The documented pattern is to fit that unit **outside your pool**, where it has
the machine to itself:

```python
# The self-parallelising unit runs alone, with the whole machine.
(mice,) = [u for u in units if u.strategy is ImputationStrategy.MICE]
custom = fit_unit(recipe, mice, df)

with ThreadPoolExecutor(max_workers=n_workers) as pool:
    rest = [
        pool.submit(fit_unit, recipe, unit, df, n_jobs_inner=1)
        for unit in units
        if unit.strategy is not ImputationStrategy.MICE
    ]
    ...
```

## Routing layer

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._router.route
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
.. autoclass:: dataforge_ml.imputation._config.ImputationRouting
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.ColumnRouting
   :members:
   :undoc-members:
   :show-inheritance:
```

## Recipe and units layer

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._recipe.resolve_recipe
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._recipe.ImputationRecipe
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._recipe.ColumnEstimates
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._units.derive_units
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
