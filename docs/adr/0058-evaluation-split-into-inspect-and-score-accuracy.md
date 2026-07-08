# Evaluation split into a cheap `inspect` and an expensive `score_accuracy`, reusing fit output

ADR-0057 made Evaluation an opt-in, stateless orchestrator — but its single `evaluate(train_df, profile)` entry point **re-routes and re-fits every model from scratch** (via `NumericImputer.fit(collect_diagnostics=True)`) and throws those models away, paying the full cross-validated refit tax even when the user only wants a cheap "do the filled-in numbers look sensible?" check. We separate the two questions Evaluation was conflating, because only one of them is retrain-free.

We split `evaluate` into **two methods on `EvaluationOrchestrator`**:

- **`inspect(fitted_imputer, train_df) -> InspectionReport`** — the cheap check. It reuses the models `fit()` already produced (passed in via the `FittedImputer`) and needs **no retraining and no `profile`**. Implemented on top of `FittedImputer.transform()` — transform to fill the holes, then compare the filled cells against the observed values — so it does not duplicate model-application code. Yields the distributional/model-metadata diagnostics (`imputed_*`, `observed_*`, `variance_ratio`, `converged`, `n_iter`, `n_neighbors_used`, `k_capped`).
- **`score_accuracy(fitted_imputer, train_df, profile) -> AccuracyReport`** — the expensive check. Held-out accuracy (`r2_cv`, `rmse`, `mae`) is *irreducibly* a refit: the fitted model has already seen every cell, so scoring it in-sample is optimistically biased. It cross-validates on folds, but **reuses the strategy decisions already recorded** in `fitted_imputer.records` instead of re-routing, and needs `profile` only to rebuild the fold estimators.

`collect_diagnostics` is **removed from `NumericImputer.fit`**; the CV-fold logic (`_compute_*_diagnostics`) **moves out** of `_numeric_imputer.py` into the Evaluation module. `fit` returns to doing exactly one thing — routing and learning models. Scope stays **imputation only**: no cross-phase evaluation framework is built (deterministic phases like encoding/scaling have no held-out truth to score); the semantic-type registry remains the sole, already-present extension seam for future imputer column-types.

## Status

accepted — revises ADR-0057 (supersedes its single-`evaluate`/full-refit design and its `DiagnosticsReport`/`ImputationFitDiagnostic` return shape)

## Considered Options

- **Keep one `evaluate` method with an `include_accuracy` flag.** Rejected: a default-valued boolean re-hides the very cost surprise ADR-0057 set out to fix — `evaluate(...)` looks free and someone flips the flag without registering the ~5× retrain. Two differently-named methods make the expensive path impossible to call by accident, and their signatures already differ (the cheap one needs no `profile`).
- **Keep the single `ImputationFitDiagnostic` struct, each method fills its portion.** Rejected: calling `inspect` would return an object with `r2/rmse/mae` blank, its meaning dependent on which method produced it — the same "one thing pretending to be another" mess, relocated to the return type. Two focused reports keep every field always-meaningful. `r2_train` is renamed `r2_cv` (it was never a train score).
- **Derive accuracy cheaply from the already-fitted model (no refit).** Rejected as dishonest: scoring a model on cells it trained on is optimistically biased, not fit quality. Honest held-out accuracy *is* the refit; the cost is real and now paid only when `score_accuracy` is called explicitly.
- **A general cross-phase evaluation framework (abstract `Evaluator`, phase registry).** Rejected as premature: imputation is the only phase that guesses unknown values, so it is the only one with a held-out-truth notion. An interface carved from one example would be shaped wrong; the second real evaluator, if it ever arrives, defines it correctly then.

## Consequences

- `NumericImputer.fit` sheds its `collect_diagnostics` double life and `_NumericFitBundle.diagnostics`; routing is decided once (at fit) and reused at accuracy time rather than recomputed.
- Public API churn: `DiagnosticsReport` / `ImputationFitDiagnostic` are retired in favour of `InspectionReport` (`InspectionDiagnostic`) and `AccuracyReport` (`AccuracyDiagnostic`); `r2_train` becomes `r2_cv`.
- `inspect` reuses `FittedImputer.transform`, so it stays correct automatically as new imputer types are added.
- The concurrency (ADR-0056) and Substep-progress (ADR-0055) rules still apply to `score_accuracy`'s fold work; `inspect` is cheap enough to need neither.
