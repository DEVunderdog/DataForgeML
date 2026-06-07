# ADR 0021: fit_transform returns tuple[FittedImputer, ImputationResult]

**Status:** Accepted

## Context

`ImputationOrchestrator.fit_transform(train_df, profile)` is a convenience method that calls `fit().transform(train_df)` in one step. The original return type was `ImputationResult` only — the `FittedImputer` was computed internally and discarded.

The sklearn convention for `fit_transform` is: fit the model and return the transformed training data. Following that convention here means callers get the imputed train DataFrame but have no `FittedImputer` to apply to the test set. The natural next step for many callers would be to call `fit_transform(test_df)` — which re-fits on test data, a data-leakage error.

## Decision

`fit_transform` returns `tuple[FittedImputer, ImputationResult]`. The caller receives both the fitted imputer and the imputed train DataFrame in a single call.

## Rationale

- Returning only `ImputationResult` discards the `FittedImputer`, forcing callers who need to impute a test set to call `fit()` a second time — or silently re-fit on test.
- Returning the tuple makes the correct workflow self-evident: `fitted_imputer, train_result = orchestrator.fit_transform(train_df, profile)`, then `test_result = fitted_imputer.transform(test_df)`.
- The `FittedImputer` is already computed as part of `fit()`; returning it adds zero cost.
- Breaking the sklearn `fit_transform` convention is intentional: sklearn's convention hides the fitted model, which is safe for sklearn pipelines but dangerous for a standalone imputer where the user must apply it to a separate test set.

## Consequences

- Callers who previously unpacked `fit_transform` as a single `ImputationResult` receive a type error and must update to tuple unpacking. This is a breaking API change.
- The tuple signature documents the correct train/test workflow at the call site, reducing the need for docstring explanation alone.
