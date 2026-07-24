# Issue #410 measurement spike — results

**Does ADR-0070's still-open *scalar-half* train/serve skew cost accuracy, now
that the Regression→MICE collapse (#407/#408/#409) spreads it from ex-Regression
columns to every MICE-routed column?**

The skew (ADR-0070, consequence 3): scalar-owned predictor columns are
median-filled by a *separate scalar unit before the model loop*, so the joint
`IterativeImputer` is **fit** on the raw train split (those columns still hold
their NaNs, imputed by its own internal round-robin) but **serves** off the
shared pre-model snapshot where they arrive already median-filled. Fit sees a
*modeled* fill, serve sees a *static median* → train ≠ serve. #405 never measured
this — it used `fit_transform` on one array (fit == serve, no scalar unit).

Three shapes over one dataset (same rows — the skew is a fill-*state* mismatch,
not a held-out-rows question; only that one variable changes between shapes),
RMSE on imputed target cells only, 30 seeds × 4 scenarios, library estimators,
MICE `max_iter`/`tol` mirrored from `_decision_assembler`:

- **skewed** — production post-collapse: fit on raw frame (scalar NaN), serve on
  scalar-median-filled frame. `train ≠ serve`.
- **matched** — close the skew from the fit side (fit the block against the
  scalar-filled frame; ADR-0070's rejected "chained fit" for the scalar layer).
  `train == serve`.
- **ideal** — reference only, *not shippable* under current routing: no scalar
  unit; `IterativeImputer` models the scalar columns itself. No skew, no static
  median.

Two populations scored separately: **ex_reg** (ex-Regression-shaped, leans hard
on scalar predictors — already carried this skew) and **mice**
(pre-existing-MICE-shaped, siblings + a modest scalar term — *newly exposed* by
the collapse).

## RMSE (mean over 30 seeds), imputed cells only

| scenario | estimator | shape | ex_reg | mice | mean fit time |
|---|---|---|---|---|---|
| linear_mcar | BayesianRidge | skewed | 2.2922 | 1.7371 | 0.38 s |
| | | matched | 1.0955 | 0.9562 | 0.34 s |
| | | ideal | 0.7741 | 0.7140 | 0.36 s |
| linear_mar | BayesianRidge | skewed | 1.9406 | 1.5326 | 0.39 s |
| | | matched | 1.1466 | 0.8759 | 0.35 s |
| | | ideal | 0.8719 | 0.7390 | 0.37 s |
| nonlinear_mcar | RandomForest | skewed | 1.1242 | 1.7469 | 19.1 s |
| | | matched | 1.1177 | 1.7236 | 19.2 s |
| | | ideal | 1.0468 | 1.7124 | 18.7 s |
| nonlinear_mar | RandomForest | skewed | 1.3713 | 1.9614 | 19.3 s |
| | | matched | 1.3712 | 1.9979 | 19.0 s |
| | | ideal | 1.2948 | 1.9538 | 18.7 s |

## Pairwise, ADR-0070 style (RMSE, wins over 30 seeds)

### The skew: skewed vs matched (does the train/serve skew cost accuracy?)

| scenario | population | skewed wins | matched wins | mean rel diff | worst case |
|---|---|---|---|---|---|
| linear_mcar | ex_reg | 0 | 30 | **+106.9%** | 392% |
| linear_mcar | mice | 3 | 27 | **+99.1%** | 1068% |
| linear_mar | ex_reg | 3 | 27 | **+70.9%** | 255% |
| linear_mar | mice | 5 | 25 | **+75.9%** | 798% |
| nonlinear_mcar | ex_reg | 13 | 17 | +0.45% | 15% |
| nonlinear_mcar | mice | 14 | 16 | +1.51% | 37% |
| nonlinear_mar | ex_reg | 16 | 14 | +0.27% | 13% |
| nonlinear_mar | mice | 16 | 14 | −1.84% | 36% |

### Grounding: matched vs ideal (does median-filling scalar cols at all cost?)

| scenario | population | matched wins | ideal wins | mean rel diff |
|---|---|---|---|---|
| linear_mcar | ex_reg | 1 | 29 | +54.1% |
| linear_mcar | mice | 1 | 29 | +32.3% |
| linear_mar | ex_reg | 2 | 28 | +55.9% |
| linear_mar | mice | 2 | 28 | +27.2% |
| nonlinear_mcar | ex_reg | 4 | 26 | +8.4% |
| nonlinear_mcar | mice | 11 | 19 | +2.1% |
| nonlinear_mar | ex_reg | 2 | 28 | +7.0% |
| nonlinear_mar | mice | 15 | 15 | +3.5% |

## Reading

- **The scalar-half skew is NOT noise — for linear estimators.** With
  BayesianRidge (a common routing target), the skew costs **+71% to +107%** mean
  RMSE and loses **0–5 of 30 seeds**, on *both* the ex-Regression population and
  the newly-exposed MICE population. This clears ADR-0070's sub-0.05% evidentiary
  bar by three to four orders of magnitude. Unlike the sibling-half ADR-0070
  closed (a measured coin flip), this half is a real, one-directional accuracy
  regression. The collapse cannot silently ship it under a "document and accept"
  reading.
- **The skew IS noise — for tree estimators.** With RandomForest it is a coin
  flip (±0.3–1.8% mean, ~15/15 wins). Trees split on thresholds and are largely
  indifferent to whether a skewed predictor arrives median-filled or
  round-robin-filled, as long as the value lands in the same region.
- **Why the split:** the scalar columns are right-skewed (exponential) — which is
  *why* scalar routing chose median fill in the first place. A linear model's
  coefficient multiplies the whole gap between the modeled fit-time value and the
  static serve-time median; a tree mostly absorbs it. The cost therefore scales
  with (a) how linearly a column leans on scalar predictors and (b) how skewed
  those predictors are.
- **Both populations pay it.** The newly-exposed pre-existing-MICE columns are
  hit on the same order as ex-Regression columns (+76–99% linear), confirming the
  spread #409 predicted is real, not confined to columns that already carried it.
- **Deeper grounding (out of scope to act on here):** even with the skew closed,
  static-median-filling scalar predictors still loses to letting the model impute
  them (`matched` vs `ideal`: +27–56% linear). That points past the skew to the
  scalar-routing boundary itself — explicitly out of scope on the map (KNN /
  scalar / GMM routing untouched). Recorded, not pursued.

## Consequence for the locking ADR

`#409`'s tentative "defer, maybe accept as documented noise" is **falsified for
the linear case**. Disposition is now a live decision, not a documentation line:

- Closing the skew (the `matched` shape — fit the block against the scalar-filled
  frame) recovers the full linear loss, but reintroduces a **fit-time ordering
  dependency between the scalar layer and the model layer** — exactly what
  ADR-0070 removed, and what ADR-0061 (resumable, independently-retrainable
  units) and ADR-0056 (fit-time parallelism) weigh against.
- ADR-0070 rejected the analogous "chained fit" because its benefit was noise. The
  benefit here is **not** noise for linear estimators, so that precedent does not
  transfer — the architectural cost may now be worth paying.

The measurement settles the empirical question (is it noise?). The disposition
*decision* — close it and pay the ordering cost, restrict where the skew is
allowed, or find another shape — is handed to the ADR-authoring track as a fresh,
now-sharp decision.

Data: `prototypes/issue410_results.json` · benchmark:
`prototypes/issue410_scalar_skew_benchmark.py`
