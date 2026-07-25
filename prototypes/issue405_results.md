# Issue #405 measurement spike — results

Per-column `Regression` (a) vs joint block-only chained equations (b) vs joint
full-breadth chained equations (c). RMSE/MAE on imputed cells only, 30 seeds per
scenario, estimators built by the library's own `RegressionEstimatorFactory`,
`max_iter`/`tol` mirroring `_decision_assembler`. Data: 4 missing-block columns +
8 complete predictors; missingness ~20% on block columns.

| scenario | estimator | shape | RMSE (mean ± std) | MAE | mean fit time |
|---|---|---|---|---|---|
| linear_mcar | BayesianRidge | per_column | 0.6327 ± 0.0757 | 0.4987 | 0.62 s |
| | | block_only | 1.2000 ± 0.1833 | 0.8947 | 0.11 s |
| | | full_joint | 0.6327 ± 0.0757 | 0.4987 | 0.17 s |
| linear_mar | BayesianRidge | per_column | 0.6343 ± 0.0741 | 0.5012 | 0.62 s |
| | | block_only | 1.2242 ± 0.2051 | 0.9106 | 0.10 s |
| | | full_joint | 0.6343 ± 0.0741 | 0.5012 | 0.16 s |
| nonlinear_mcar | RandomForest | per_column | 0.6839 ± 0.0573 | 0.4789 | 99.5 s |
| | | block_only | 1.1940 ± 0.0668 | 0.8936 | 5.6 s |
| | | full_joint | 0.6838 ± 0.0574 | 0.4790 | 20.2 s |
| nonlinear_mar | RandomForest | per_column | 0.6935 ± 0.0668 | 0.4824 | 97.8 s |
| | | block_only | 1.2012 ± 0.0607 | 0.8985 | 5.5 s |
| | | full_joint | 0.6938 ± 0.0666 | 0.4827 | 19.9 s |

## Pairwise, ADR-0070 style (RMSE, vs full_joint)

| scenario | comparison | wins | mean rel diff | worst case |
|---|---|---|---|---|
| linear_mcar | per_column vs full_joint | 8–18 | +0.0001% | 0.0013% |
| linear_mar | per_column vs full_joint | 8–18 | +0.0001% | 0.0005% |
| nonlinear_mcar | per_column vs full_joint | 16–14 | +0.0165% | 0.45% |
| nonlinear_mar | per_column vs full_joint | 20–10 | −0.0488% | 0.32% |
| linear_mcar | block_only vs full_joint | 0–30 | +89.7% | 145.6% |
| linear_mar | block_only vs full_joint | 0–30 | +92.9% | 156.2% |
| nonlinear_mcar | block_only vs full_joint | 0–30 | +75.2% | 95.9% |
| nonlinear_mar | block_only vs full_joint | 0–30 | +74.2% | 97.5% |

## Reading

- **Per-column ≈ joint full-breadth.** Sub-0.05% mean deltas with no consistent
  direction (a coin flip on wins) — indistinguishable from noise at ADR-0070's
  evidentiary bar. The collapse does not regress accuracy.
- **Block-only predictors lose decisively when complete columns carry signal:**
  +74–93% RMSE, 0/120 seed wins. Full breadth is the accuracy-first default for
  the predictor-breadth dial.
- **Cost:** the joint full-breadth fit is ~4–5× cheaper than N per-column fits
  (0.17 s vs 0.62 s linear; 20 s vs 99 s RandomForest), with the gap growing in N.
- Caveat: block columns here also correlate with each other via shared latent
  factors, so block-only had real signal available — its loss is not an artifact
  of independent block columns.
