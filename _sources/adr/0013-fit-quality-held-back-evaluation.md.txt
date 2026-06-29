# Held-back evaluation for ImputationFitDiagnostic R² — k-fold CV on complete rows

When computing `r2_train` for `ImputationFitDiagnostic`, we use k-fold cross-validation (k=5, configurable via `refit_r2_cv_folds`) on the complete rows of the column's joint array. In each fold a throwaway model is trained on (k-1)/k of the complete rows and evaluated on the remaining 1/k; `r2_train` is the mean R² across all k folds. The final stored model is fitted on all of `train_df` before the diagnostic runs and is never touched by the evaluation step.

**Why k-fold CV instead of a single 80/20 holdout (original approach, superseded):**

The original approach held back a fixed 20% of complete rows, fit one throwaway model, and used that single R² as the diagnostic signal. When complete rows are few (e.g. 30–60 rows), a single split is a high-variance estimator — one unlucky draw can send a good column to Median or let a genuinely poor model through. k=5 averages five independent evaluations, reducing variance in the R² estimate without changing what the metric means or adding cost to the final stored fit.

**Why the minimum complete rows threshold is 50 (raised from 25):**

With k=5, each validation fold contains 1/k of the complete rows. At the previous floor of 25, each validation fold held ~5 rows — too few for a meaningful R² evaluation. The new floor of 50 ensures at least 10 rows per validation fold. Below 50 complete rows, `r2_train = None`.

**Architecture: the final model is separate from the diagnostic.**

The final stored model trains on all of `train_df` (IterativeImputer handles missing values internally). It is fitted inside `_fit_regression` / the KNN and MICE blocks before the diagnostic function is ever called. The k-fold throwaway models are created and discarded entirely inside `_compute_*_diagnostic`. No re-training of the final model occurs after the diagnostic.

**What does not change:**

- `variance_ratio` and `converged`/`n_iter` are read from the final model — no throwaway model is involved for either.
- k-fold makes `r2_train` a more reliable diagnostic signal; the metric definition is unchanged.
- k=5 applies uniformly to Regression, KNN, and MICE diagnostics.
- In-sample R² (testing the model on its own training rows) remains rejected: a model that memorised training data would score high even if it generalises poorly.
