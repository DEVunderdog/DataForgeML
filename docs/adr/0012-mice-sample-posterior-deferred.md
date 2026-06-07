# ADR 0012: sample_posterior stays False in MICE; posterior sampling deferred

**Status:** Accepted

Theoretical MICE (van Buuren 2007) draws imputed values from the posterior predictive distribution of the internal estimator at each round rather than using the point estimate. `IterativeImputer` supports this via `sample_posterior=True`, applicable only to probabilistic estimators (primarily `BayesianRidge`).

**Why not enable it now:**

1. **Estimator incompatibility.** Scope 2 auto-selects the MICE estimator from `NonlinearityTag` — the block may use `RandomForestRegressor` or `GradientBoostingRegressor`, neither of which supports `sample_posterior`. Enabling it conditionally (only when estimator is `BayesianRidge`) means imputation silently switches between deterministic and stochastic depending on data structure. A user running identical pipeline code on two datasets gets different reproducibility guarantees with no visible signal.

2. **Breaks `FittedImputer.transform()` determinism.** `FittedImputer.transform()` is a pure function: same input always produces same output. `sample_posterior=True` makes every call stochastic. This invalidates any downstream test that asserts on imputed values and requires `random_state` management across the full pipeline.

3. **Requires multiple-imputation infrastructure to be useful.** The variance captured by posterior sampling is only meaningful if `transform()` is called multiple times and results are combined via Rubin's rules. That is a separate architectural scope — closer to Scope D (imputation quality evaluation) than to hyperparameter selection.

**Decision:** `sample_posterior=False`, not user-configurable in Scope 2. Revisit when multiple-imputation (run MICE k times, combine via Rubin's rules) is scoped as a first-class feature.
