# Evaluation

The evaluation module scores **the data, never the imputation model** (ADR-0087).
Its question is whether a table produced by a pipeline phase still carries the
distribution and dependence structure of the data behind it; the model that
produced the table is disposable and is never graded.

It is independent and opt-in. `EvaluationConfig` stands alone and is never
nested on `PipelineConfig`, because evaluation is stateless and re-configured
freely between calls while poking at results, whereas the pipeline config is
built once for a run.

The module is layered. A **generic** layer holds the metric arithmetic and knows
nothing about any one phase — its seam takes two frames and returns a
`C2STResult`. It stays private: the user drives evaluation through
`evaluate_imputation`, and the future non-imputation consumer is a sibling
adapter inside the library. Consumer sub-packages adapt a phase's artefacts to
that layer; `evaluation/imputation/` is the first of them.

## Reading a report

`evaluate_imputation` returns one record per **active** column, always —
*untestable is a result, never an absence*. A column the test could not run on
carries a `C2STOutcome` naming why and no `C2STScore` object at all, so
`record.score.mean_frame_z` raises `AttributeError` rather than reading a
stand-in that could collapse into a passing mean. A column excluded under the
Phase Active-Columns Contract is absent from the census entirely and raises
`KeyError`.

Two things the rendered report says that are worth stating here too:

- **Scalar strategies fail at close to 100%, and that is the design.** The
  column under test is itself a feature, so a Mean/Median/Mode/Constant fill's
  one-value spike is visible. A report on a median-filled table flags every such
  column; that is the *baseline* which makes a model-based number interpretable,
  not a list of defects.
- **A fired test is ambiguous under MAR.** Under a strongly MAR mechanism the
  observed and filled rows are *supposed* to differ, so a flag cannot separate
  "the imputation is bad" from "the missingness is informative". The report
  prints this caveat whenever anything is flagged, and no dial suppresses it:
  full configurability governs thresholds and behaviour, not the library's
  disclosure of a known limit of its own number.

## Entry point

```{eval-rst}
.. autofunction:: dataforge_ml.evaluation.imputation._adapter.evaluate_imputation
```

## Configuration

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation._config.EvaluationConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation._config.C2STConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation._config.EvaluationMetric
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation._c2st.C2STScheme
   :members:
   :undoc-members:
   :show-inheritance:
```

## Records

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.EvaluationReport
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STReport
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.ColumnC2STResult
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STScore
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation._c2st.C2STResult
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STProvenance
   :members:
   :show-inheritance:
```

## Outcome vocabulary

`C2STOutcome`, `C2STAnnotation` and `C2STVerdict` classify output and are never
user-supplied — a stated exception to ADR-0050, because reading the report *is*
branching on them.

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STOutcome
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STAnnotation
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml.evaluation.imputation._records.C2STVerdict
   :members:
   :show-inheritance:
```
