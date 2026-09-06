# ADR 0015: RMSE and MAE are diagnostic-only — not used for automated decisions

**Status:** Superseded by ADR-0087 (previously amended by ADR-0057 / ADR-0058) — `AccuracyDiagnostic`, `r2_cv`, `rmse` and `mae` are **deleted, not renamed**. The stance was right and ADR-0087 generalises it: the unit-scale argument against thresholding RMSE is one instance of why cell-level error is the wrong target, and C2ST's `z` is the dimensionless, cross-column-comparable number this ADR reached for and could not build from RMSE.

`ImputationFitDiagnostic` exposes `r2_train`, `rmse`, and `mae`. Only `r2_train` and `converged` are consumed by automated decision logic; `rmse` and `mae` are reporting fields only.

The reason is unit scale. R² is dimensionless and bounded [−∞, 1] regardless of the column. A universal threshold (e.g. `r2_train < 0.1`) means the same thing for every column: the model explains less than 10% of variance. RMSE and MAE are in the column's own units — an RMSE of 5.0 is catastrophic on a 0–10 column and irrelevant on a 0–100,000 column. There is no universal "RMSE too high" number that works across all columns, so no sensible automated rule can be written against it.

A normalised variant (`rmse / observed_std`) would restore scale-independence but is redundant: it captures the same signal as R² (both measure prediction quality relative to the column's spread) without adding new information.

`rmse` and `mae` are preserved as reporting fields: they give the user an interpretable, domain-specific error in the column's own units ("imputed values are off by ±2.3 kg on average") that they can judge in context. Automated decisions stay on R² and convergence.
