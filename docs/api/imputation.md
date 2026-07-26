# Imputation

The imputation door is user-orchestrated (ADR-0060, ADR-0071): `decide()`
builds the pure plan, a caller-written `fit_unit` loop trains it one unit at a
time, and `FittedImputer.compose` rolls the trained units into the
whole-dataframe imputer.

## Decision layer

```{eval-rst}
.. autofunction:: dataforge_ml.imputation._decision_assembler.decide
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

## Evaluation

```{eval-rst}
.. autoclass:: dataforge_ml.imputation.evaluation.EvaluationOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.InspectionDiagnostic
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.InspectionReport
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.AccuracyDiagnostic
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.imputation._config.AccuracyReport
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
.. autoclass:: dataforge_ml.imputation._fitted_imputer.UnfittedColumnError
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
