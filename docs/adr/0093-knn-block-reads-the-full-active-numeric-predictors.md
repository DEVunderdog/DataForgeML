# The KNN block reads the full active-numeric predictors

The KNN fitter measures distance over its own block's columns and nothing else. A one-column KNN block therefore has no coordinate a receiver shares with a donor, and sklearn silently falls back to the training mean: `KNNImputer(n_neighbors=1)` on `[1, 2, 10, NaN]` returns `4.33`, and one correlated predictor turns that into `10`. Routing makes such a block whenever exactly one column lands on KNN. It also makes ADR-0091's KNN Rows per Predictor (active width − 1) untrue. **The KNN block now measures distance over every active numeric column and writes back only the columns it owns**, the same shape MICE took in #417.

## Status

accepted — resolves [Whether the KNN block reads the full active-numeric predictors](https://github.com/DEVunderdog/DataForgeML/issues/541) on map #528. Amends ADR-0091 (its KNN predictor count is now true; the caveat is gone, the count is unchanged) and ADR-0070 (the MICE and KNN blocks both have inputs ⊃ targets; "a KNN block reads exactly the columns it fills" no longer holds).

## Decisions

- **Full active-numeric breadth, no dial.** The predictor set is the one MICE reads: active `Numeric` columns whose frame dtype is numeric. The block writes back only its own columns. Given up: KNN weights every scaled coordinate equally, unlike a regression, so on a wide frame unrelated columns can dilute the distance. There is no switch to turn widening off.
- **Scalar-owned predictors are filled before the fit.** They are filled to their own decided central tendency, uniformly, exactly as MICE does (#418). The serve frame already carries those fills, and `KNNImputer` keeps the training matrix as its donor pool, so a raw fit frame would match filled receivers against unfilled donors. Given up: rows with many filled cells look falsely close — the scalar fill's own cost, paid at serve time anyway.
- **The fitted KNN unit splits what it reads from what it owns**, as `FittedMICE` does. Scaling statistics cover what it reads. A predictor absent from the serve frame is an all-missing coordinate. A predictor all-missing in training must not shift the owned columns' positions in the output. The read set is required: a KNN unit saved before this change does not load (the map wants no compatibility shims).
- **`n_neighbors` and `weights` read the distance space.** Their width operand is the active numeric width and their missingness operand the mean effective-null ratio across it, both off the profile, so neither depends on block membership. Given up: dial values change for existing configs; with default bounds `k` rarely moves, but `weights` turns `"uniform"` above 30 active columns, so `"distance"` becomes rare on wide frames.
- **No width ceiling.** `knn_max_rows` stays KNN's only Resource Ceiling. Memory grows linearly with width; the row × row distance work is chunked. Reviving `knn_max_features` would contradict ADR-0091, and a column is never moved to a cheaper strategy to fit the machine. Given up: on a very wide frame even a one-column KNN block is slower.
- **Locked by argument, checked by measurement.** The one-column mean fill is a proven defect no measurement can undo, so the decision is not held for one. [Measure the floor bounds, signal weights, and tier boundaries](https://github.com/DEVunderdog/DataForgeML/issues/519) compares block-only and widened KNN on RMSE, on frames with added unrelated columns and on real frames, multi-column blocks only. **Reopen if widened loses with a consistent direction across seeds**; the fallback is correlated-only predictors, never block-only.

## Measured

[Whether widened KNN beats the block-only fit](https://github.com/DEVunderdog/DataForgeML/issues/542), split out of #519, ran the check. **The decision holds; nothing reopens.** Numbers are widened ÷ block-only masked-cell RMSE, median over 30 seeds with a 95% bootstrap CI; the reopen rule counts a loss only on a real frame.

- **Real frames: widened never loses.** ames (11-column block, width 37): 0.912 [0.908, 0.925], 30/30 seeds. life_expectancy (14-column block, width 20): 0.974 [0.935, 1.012], a tie. Titanic has a one-column block and was excluded.
- **Signal in the predictors: widening wins big.** 0.61 with no noise columns, still 0.85 with 32 added.
- **Dilution is real but bounded.** When only the block carries signal and every added column is noise, widened loses 4–8% (1.036–1.082) with a consistent direction. Equal `k` in both arms still loses 3.4%, so the dial is not the cause. Recorded as a dilution curve, not a reopen.
- **Never near a scalar fill.** In every condition both arms beat a median fill; widened's worst ratio to it is 0.952.

## Considered Options

- **Correlated-only predictors** (columns passing a correlation threshold against some KNN target, capped). Keeps unrelated columns out of the distance, but needs a new threshold, repeats the relevance gating the Signal Score and floor already did per column, and is block-level because sklearn shares one distance across a row — so the floor's per-target count would read a block quantity again, the circularity ADR-0091 escaped. A column with no correlated predictor still gets a mean fill. Kept as the fallback if measurement reopens this.
- **Stay block-only, with a routing rule for a KNN block under two columns.** Changes no working block, but makes the floor's KNN count circular (block width is what the floor decides) and fixes only the extreme case: a two-column block of unrelated columns is still near a mean fill.
