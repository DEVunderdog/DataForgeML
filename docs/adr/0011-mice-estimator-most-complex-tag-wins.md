# ADR 0011: MICE estimator selection uses the most complex NonlinearityTag across the block

**Status:** Accepted

`IterativeImputer` accepts a single `estimator` parameter applied uniformly across every column in every round. MICE columns may each carry a different `NonlinearityTag` (computed by `NonlinearityProfiler` in Phase 1). Three aggregation strategies were considered:

1. **Most-complex tag wins** — take the most complex `NonlinearityTag` across all MICE columns and pass it to `RegressionEstimatorFactory`. If any column is `ComplexNonlinear`, the whole block uses `GradientBoostingRegressor` or `RandomForestRegressor`.
2. **Majority vote** — whichever tag applies to the most columns wins. Introduces ambiguity on ties and adds a new aggregation code path.
3. **Per-column estimator wrapper** — assign each target column its own estimator inside the MICE round. `IterativeImputer` does not support this natively; it would require a bespoke wrapper estimator with significant complexity.

**Decision:** Option 1 (most-complex tag wins).

**Reason:** A linear estimator applied to a `ComplexNonlinear` column produces biased imputed values that propagate as noisy features into every subsequent MICE round, compounding error across the whole block. A tree-based estimator applied to a `Linear` column is merely somewhat less statistically efficient — it still produces unbiased estimates. The asymmetry in harm makes the conservative "promote to the most complex" rule strictly preferable to majority vote. Per-column wrapping is not worth the implementation cost when the conservative rule produces acceptable outcomes.

**Corollary:** If all MICE columns are `Unpredictable` (both linear and tree R² near zero), the block is skipped entirely and each column falls back to Median individually — no MICE model is fitted.
