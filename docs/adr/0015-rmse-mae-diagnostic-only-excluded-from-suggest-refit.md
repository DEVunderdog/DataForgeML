# ADR 0015: RMSE and MAE are diagnostic-only — excluded from suggest_refit_config rules

`suggest_refit_config` uses `r2_train` and `converged` to generate automated per-column config overrides. It does not use `rmse` or `mae`, even though both are available on `ImputationFitDiagnostic`.

The reason is unit scale. R² is dimensionless and bounded [−∞, 1] regardless of the column. A universal threshold (`refit_r2_threshold = 0.1`) means the same thing for every column: the model explains less than 10% of variance. RMSE and MAE are in the column's own units — an RMSE of 5.0 is catastrophic on a 0–10 column and irrelevant on a 0–100,000 column. There is no universal "RMSE too high" number that works across all columns, so no sensible automated rule can be written against it.

A normalised variant (`rmse / observed_std`) would restore scale-independence but is redundant: it captures the same signal as R² (both measure prediction quality relative to the column's spread) without adding new information. Adding a rule against it would complicate `suggest_refit_config` without enabling any decision that `r2_train` does not already cover.

`rmse` and `mae` are preserved as reporting fields: they give the user an interpretable, domain-specific error in the column's own units ("imputed values are off by ±2.3 kg on average") that they can judge in context. Automated decisions stay on R² and convergence.
