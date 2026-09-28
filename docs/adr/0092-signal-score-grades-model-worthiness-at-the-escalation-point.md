# The Signal Score grades model-worthiness at the escalation point

Routing today asks "is a model worth it over a scalar?" three different ways, each a cliff: `mcar_feature_predictability_threshold = 0.2` on max |r| (MCAR only — the MAR path never asks), the `Unpredictable` tag (`r2_rf < 0.05`) as a pre-routing guard, and `NearConstant` (mode share > 0.90) as a de-escalation cap. The profiler already fits the models that answer the question and throws the numbers away, keeping only their difference. The **Signal Score** replaces all three with one graded quantity, read at the one model-based escalation point ADR-0091 created.

## Status

accepted — resolves [What composes the weighted signal score](https://github.com/DEVunderdog/DataForgeML/issues/518) on map #528, re-ratifying its map #516 resolution with the validation loop stripped out. Amends ADR-0016 (`Unpredictable` is no longer a router predicate), ADR-0035 (the `Unpredictable` and `NearConstant` terminals leave the BoundedDiscrete sub-chain) and ADR-0091 (fills in the "let the signal score pick" hand-off). Amended by ADR-0094: the Capability Ladder puts KNN below MICE, and `Unpredictable` maps to BayesianRidge.

## Decisions

- **Four components, grouped by the question each answers.** Not a flat weighted sum: that would need three invented scale mappings (R², |r|, nats, a count), each an unmeasured threshold.
  - **Explainable Variance** = `max( max(r2_linear, r2_rf), (max|r|)² )`, negative R² clamped to 0. The R² pair is the direct reading but comes from ≤500 listwise-complete rows; max |r| is pairwise-complete over the whole column. Squaring |r| puts it on R²'s scale by construction — it *is* the R² of the best one-predictor OLS. `NumericStats` retains `r2_rf` and `r2_linear` (today only `r2_gap` survives).
  - **Latent Structure** = `1 − exp(−2 · max_MI)`, from a new max per-predictor MI on `NumericStats`. The Gaussian identity maps nats onto R²'s scale without tuning. Max, not the stored mean, because one strong predictor among thirty junk ones averages to nothing. It can only raise the score: high MI with low R² means the depth-4 probe was too weak.
  - **Signal Breadth** = `count / (count + 1)`, `count` = predictors above `mice_correlation_threshold`. A robustness reading — many sources means MICE still has predictors when some are missing in the same rows — so it keys on the absolute count. It can only add.
  - `score = clamp( base + w_breadth · breadth, min = w_latent · latent, max = 1 )`. `r2_gap` does not enter; it answers *which estimator* (#526).
- **Two weights, two tier bounds, all dials.** `w_breadth`, `w_latent` and the two tier bounds are routing gates on the imputation config (ADR-0089's split). #519 measures each as a one-dimensional sweep, and also whether the 500-row R² should be discounted for its sample. Given up: `count/(count+1)` and the Gaussian identity are shape choices; only their weights are measured.
- **Missing readings degrade; they are never invented.** When the R² pair is absent (the probe needs ≥20 rows complete across *every* numeric column, which heavy missingness breaks while the floor's Usable Rows still pass) the base is `(max|r|)²` alone and the signal names why. The score is `None` only when no predictor correlation exists; it then routes as the bottom tier. A probe fit that throws yields `None` for R² or MI, never `0.0` — the refusal shape #488 settled. Given up: `(max|r|)²` is a one-predictor lower bound, so degraded columns score conservatively.
- **Behind the existing triggers.** Severity, skew, kurtosis, outlier density and MAR still decide whether a column reaches the escalation point — they answer "does a scalar hurt here?". The score answers "can a model beat a scalar?" for columns that arrive. Given up: a Minor MCAR column with R² 0.9 still gets Mean.
- **One escalation point, both paths.** MAR, MCAR and bimodal branch 2 all reach it. The MAR asymmetry disappears as a consequence of ADR-0091, not a fix here. A MAR-suspect Severe column can now be scored down to a scalar: MAR severity stops being a licence to model.
- **Three Signal Tiers.** Bottom → a scalar fill regardless of feasibility; middle → the cheapest feasible candidate; top → the richest; the ordering of candidates is #526's capability ladder. The bottom tier's fill is **Mode** for BoundedDiscrete, **GMM Sampling** for a continuous bimodal column reaching the point through branch 2 (a median sits in the valley between the peaks), and the skew-picked central tendency otherwise. Two tiers would make the score a boolean again; four asks #519 to defend a boundary it cannot.
- **`mcar_feature_predictability_threshold` is deleted**, not retuned: it cut max |r|, and the score grades a different quantity.
- **`Unpredictable` is deleted as a router predicate** at both sites (Priority 4 and BoundedDiscrete step a). It is not moved to the floor: "trained fine, learned nothing" is not a feasibility sentence. The tag survives as a profile label and an estimator input.
- **`NearConstant` is deleted as a router predicate** at both sites (Priority 6 and BoundedDiscrete step b). Its only stated reason was cost ("fitting is wasted"), and a near-constant column whose rare values are predictable is exactly what a model should fill. The flag survives as a profile label, so a user who wants the old behaviour forces a scalar or excludes the column. Given up: some near-constant columns now train a model that learns little.
- **Computed at routing time, shown only as a signal.** A private helper beside the floor reads the profile's components and the config's dials. It runs at the escalation point and, as with the floor, for columns `per_column_strategy` forces onto MICE or KNN. The score, tier and any degradation reason go into the column's signal text. `ImputationRouting` gains no score field and no consulted flag. Given up: the number is not machine-readable off the routing, and a forced scalar on a high-scoring column goes unsaid.

## Considered Options

- **A flat weighted sum of all four components** (the question as first posed). Rejected: incommensurable scales and a five-or-six-dimensional sweep on a box where MICE OOMs at 5k rows.
- **R² alone.** Rejected: the thinnest-sampled reading.
- **`None` → bottom tier whenever the probe skips** (the #516 answer). Rejected: it wrongly assumed max |r| vanished with the probe, and pushed heavily-missing frames to scalars on a probe artifact.
- **Moving `NearConstant` under the Resource Ceiling** (the #516 answer). Rejected: a Resource Ceiling reads raw size; mode share is not size.
- **Replacing the triggers with the score.** Rejected for now: a larger change that merges two questions; left to a future effort.
- **Carrying the score on every column with a consulted flag** (the #516 answer). Rejected: its readers — the belief record, audit and recommender — are out of scope.

## Consequences

- `RegressionEstimatorFactory` maps `Unpredictable` to `None` and the fitter skips the block; a column the score sends to MICE can carry that tag. Handed to [Whether the two quantities govern the estimator pick and ADR-0062's hyperparameters](https://github.com/DEVunderdog/DataForgeML/issues/526).
- The `max_iter` signal reading `r2_gap` must tolerate `None`.
- #519's before/after must show MAR Severe and near-constant columns changing route.
