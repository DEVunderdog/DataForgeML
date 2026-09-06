# Diagnostics separated from fit into an opt-in, stateless Evaluation orchestrator

`ImputationOrchestrator.fit()` computed `ImputationFitDiagnostic` inline, cross-validating each model-based column with `refit_r2_cv_folds` (default 5) fold refits. That measurement costs roughly 5× the actual imputation fit and is welded into every fit, so a run that only needs imputed data pays the evaluation tax and appears hung during it. We separate the two: `fit()` / `fit_transform()` now only learn fill values and models; fit-quality measurement moves to a distinct **Evaluation** phase the user invokes by choice.

Evaluation is **its own orchestrator** (mirroring the "each phase is an orchestrator you pass `observer=` to" pattern) and is **stateless**: the caller passes the training data back in (`evaluate(train_df, profile)`) rather than the `FittedImputer` retaining it. Diagnostics are returned as their own report keyed by column, not smuggled back onto every `ColumnImputationRecord`.

## Status

superseded by ADR-0087 — the opt-in, stateless Evaluation phase survives; the orchestrator class, the diagnostics it computed and the premise that the imputation *model* is what gets scored do not.

## Considered Options

- **Stateful `FittedImputer` that hoards `train_df` so `evaluate()` needs no arguments.** Rejected: it breaks the object's serializable, stateless contract and keeps a whole DataFrame alive in memory long after fitting. Passing the data back at evaluate time is a trivial ergonomic cost and arguably a feature (evaluate against any dataset).
- **A method on `FittedImputer` rather than a separate orchestrator.** Rejected in favor of a dedicated orchestrator so the phase-orchestrator-with-`observer=` shape stays uniform across the pipeline — the same consistency the granularity and concurrency decisions rely on.
- **Keep diagnostics in `fit()` but make them a config toggle.** Rejected: the concern is not just cost but lifecycle — impute now, evaluate later by choice — which a phase boundary expresses and a boolean flag does not.

## Consequences

- `fit()` output no longer carries `diagnostic` on its records by default; consumers that want quality metrics call the Evaluation phase explicitly.
- Evaluation is itself an independent stage that honors the concurrency (ADR-0056) and Substep-progress (ADR-0055) rules — its fold work parallelizes across columns and emits Substep heartbeats.
- The cross-validated refit cost is unchanged in magnitude but now paid only on demand, on the user's clock.
