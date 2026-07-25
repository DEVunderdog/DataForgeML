# Regression collapses into a unified chained-equations MICE strategy

`ImputationStrategy.Regression` and `MICE` were always the same algorithm. Both build one `sklearn` `IterativeImputer`; the only real differences were which columns each fed the estimator as predictors and how many targets it kept. `Regression` fed a single target *all* active-numeric columns and kept one; `MICE` fit over the block's own columns alone and kept them all. Yet `Regression` was a distinct enum value gated by a fragile `severity==High and corrs and n_rows>=min` routing branch, and when many columns routed to it independently the library paid **N separate full chained-equations fits**, each re-modelling every other column to keep one target's sliver. This ADR removes the strategy. **Every column that used to route to `Regression` now routes to `MICE`, joins the single joint block, and is imputed by one `IterativeImputer` fit over the full active-numeric matrix.** The distinction that justified two strategies never existed at the algorithm level, and measurement confirmed it does not exist at the accuracy level either.

This ADR *authors the record*; it decides nothing new. Every input was settled in its own ticket ([#404](https://github.com/DEVunderdog/DataForgeML/issues/404)–[#412](https://github.com/DEVunderdog/DataForgeML/issues/412)) against the accuracy-first bar ADR-0070 established — measure across seeds, take the simpler shape when the delta is noise, keep the costlier shape only where accuracy demonstrably demands it. What follows is the locked collapse, grouped by the five axes the destination named, each citing the ticket that decided it.

## What the measurement settled

The collapse rests on two spikes, both scored as RMSE against known ground truth on imputed cells only.

**Per-column `Regression` ≈ joint full-breadth ([#405](https://github.com/DEVunderdog/DataForgeML/issues/405)).** Across 30 seeds × 4 scenarios, the mean delta between running N independent single-target fits and one joint block over the full active-numeric matrix was ≤0.05% with no consistent direction — the collapse does not regress accuracy. Block-only predictors (a joint fit that ignores complete columns) lost **0 of 120 seeds** at +74–93% RMSE, so predictor breadth is not a free dial: the narrow shape is a measured, decisive loss. The joint fit is also ~4–5× cheaper than N per-column fits — a consequence of the right accuracy design, not its goal.

**The scalar-half train/serve skew is not uniformly noise ([#410](https://github.com/DEVunderdog/DataForgeML/issues/410)).** ADR-0070 closed the *sibling* half of the model train/serve skew and left the *scalar* half open — a model trained on raw feature nulls but served on scalar-filled features. Under the collapse that skew *spreads*: every MICE-routed column now reads the full active-numeric matrix, so every one of them inherits it, not just ex-`Regression` columns. Measured, the skew costs BayesianRidge (linear) **+71–107% RMSE at 0–5/30 seed wins** — three-plus orders past ADR-0070's noise bar — while for RandomForest it is a coin flip (±0.3–1.8%). This falsified [#409](https://github.com/DEVunderdog/DataForgeML/issues/409)'s tentative accept-as-noise reading and forced the skew to be closed, not accepted.

## Locked decisions

### Routing taxonomy ([#406](https://github.com/DEVunderdog/DataForgeML/issues/406))

- **`ImputationStrategy.Regression` is removed.** Every routing branch that emitted it now emits `MICE`. The old routing signals (`severity==High`, correlation presence) survive only as **provenance strings** — descriptive record, no behavioural fork.
- **`regression_min_rows` is renamed `mice_min_rows` and applied as a uniform floor to every MICE entry path** — including `multi_mar`/`Severe`, which previously had no row floor. This is a deliberate tiny-*n* behaviour change: below-floor columns divert to KNN→Median rather than entering a chained fit on too few rows.
- **The standalone-vs-block distinction dissolves.** Ex-`Regression` columns enter the shared assembler path; block topology is a downstream grouping detail, not a strategy.

### Block shape ([#405](https://github.com/DEVunderdog/DataForgeML/issues/405), [#407](https://github.com/DEVunderdog/DataForgeML/issues/407))

- **One joint block, no partition.** All MICE-routed columns join a single `IterativeImputer` — exactly the shape [#405](https://github.com/DEVunderdog/DataForgeML/issues/405) validated.
- **No block-size cap.** `n_nearest_features` correlation-selection is the *sole* scale governor (all predictors at ≤10 columns; else the median informative count, capped at 20 — the [#404](https://github.com/DEVunderdog/DataForgeML/issues/404) dial). A column is never severed to a cheaper strategy to keep the block small.
- **Predictor participation is jointly-imputed-then-discarded.** The block fits one `IterativeImputer` over the **full active-numeric matrix** and writes back **only** the MICE-owned columns — generalizing `FittedRegression`'s `[target] + feature_columns` / `target_idx` pattern block-wide. This preserves unit independence (no cross-unit ordering) and is the shape [#405](https://github.com/DEVunderdog/DataForgeML/issues/405) measured.
- **Two implementation consequences follow for the code effort:** `FittedMICE` (`_fitted_imputer.py`) gains a write-back restriction — today it fits block-only and emits every column it saw; and `_compute_mice_n_nearest_features` (`_decision_assembler.py`) must count over the full active-numeric breadth, not just the MICE columns.

### Dials collapse to fixed behaviour ([#408](https://github.com/DEVunderdog/DataForgeML/issues/408))

- **There is no predictor-breadth dial.** Full active-numeric breadth is the only mode; block-only is a measured strict loss ([#405](https://github.com/DEVunderdog/DataForgeML/issues/405)), and the "correlation-selected subset" that looked like a third breadth mode was a category error — it is `n_nearest_features` scale governance operating *inside* full breadth, not a breadth choice.
- **There is no joint-vs-single dial.** Always one joint block; one-at-a-time is an accuracy tie at 4–5× the cost.
- **The accuracy-first vs full-configurability collision resolves toward accuracy-first on both axes.** Configurability governs thresholds and legitimate tradeoffs — not access to shapes measurement proved never-correct. The only predictor config surface is the three existing scale-governance thresholds: `mice_n_nearest_features_min_cols`, `mice_max_nearest_features`, `mice_correlation_threshold`.

### Transform frame and the scalar-half skew ([#409](https://github.com/DEVunderdog/DataForgeML/issues/409), [#410](https://github.com/DEVunderdog/DataForgeML/issues/410), [#411](https://github.com/DEVunderdog/DataForgeML/issues/411))

- **The block inherits `FittedRegression`'s frame split.** Its inputs are a strict superset of its targets, so — like the Regression unit before it — it fits on the raw train split and serves off ADR-0070's shared pre-model snapshot. Train ≠ serve, deliberately, unchanged from the Regression precedent.
- **The scalar-half skew is closed, uniformly ([#411](https://github.com/DEVunderdog/DataForgeML/issues/411), option 1).** Every unified block fits on the same scalar-filled frame it serves on, killing the +71–107% linear regression. It is **not** gated on estimator family — a "trees are skew-immune" gate was rejected as a fragile assumption bought to save one `column.median()`.
- **The block reproduces the scalar fill itself.** It reads the scalar-owned predictors' central tendency (median/mean/mode) off the plan's decided tendency and applies it over the train frame, rather than consuming sibling `FittedScalar` units. This keeps `fit_unit` standalone: the ADR-0061/0071 independence invariant and ADR-0056's parallelism hold, and the feared fit-time ordering dependency is designed out. What remains is a contained *logic*-coupling — the block's fitter duplicates the scalar central-tendency rule — with no fit-time ordering dependency.

### Migration ([#412](https://github.com/DEVunderdog/DataForgeML/issues/412))

- **`Regression` leaves zero residue.** Config: remove the `Regression` enum member (`_config.py:42`), the `force_column_strategy` override for it, and the regression-specific `max_iter` setter. `regression_min_rows` is *renamed* to `mice_min_rows`, not deleted (it survives as the MICE floor).
- **`evaluation.py`: strip the `FittedRegression` isinstance branch, the `regression:{col}` lookup, and `_score_regression_cv`.** Ex-`Regression` columns score through the MICE path.
- **Remove both `FittedRegression` (`_fitted_units.py`) and `fit_regression_unit` (`_fitters.py`).** Neither is publicly exported, and the pattern already lives block-wide in `FittedMICE` after the block-shape change above.
- **No back-compat.** Pre-release: serialized `FittedRegression` payloads and `Regression`-forcing configs hard-fail on load. The break is accepted consciously. The hard-fail is the **raw unpickling failure** — deleting the class is the whole mechanism. No named guard, no registry of removed unit types in the decode path: that would be machinery built for a population of artifacts that does not exist, and it would grow a tombstone list with every future removal. When a change genuinely does break artifacts users hold, it is announced in the docs and the changelog, not encoded as a curated error in the decode path (`_serialization.py`).

## Status

accepted

Planning-only. This ADR is the map's destination ([#403](https://github.com/DEVunderdog/DataForgeML/issues/403)); the code change is a follow-on effort.

## Considered Options

- **Keep `Regression` and `MICE` as separate strategies (status quo).** Rejected. It preserved two enum values, a fragile routing branch, and N-fits-for-N-columns cost to protect a per-column-vs-joint accuracy difference that [#405](https://github.com/DEVunderdog/DataForgeML/issues/405) measured at ≤0.05% with no consistent direction. Two shapes for one algorithm, paying real complexity and compute for a non-benefit.
- **Collapse, but keep block-only predictors (ignore complete columns).** Rejected on measurement: 0/120 seed wins at +74–93% RMSE ([#405](https://github.com/DEVunderdog/DataForgeML/issues/405)). Full active-numeric breadth is the accuracy-first default precisely because the narrow shape is a decisive loss.
- **Keep predictor-breadth and joint-vs-single as user dials.** Rejected ([#408](https://github.com/DEVunderdog/DataForgeML/issues/408)). A dial implies a legitimate tradeoff; measurement showed one setting of each is a strict loss (block-only) or a tie at 4–5× cost (single-target). Exposing them would sell configurability over a shape that is never correct. Full configurability governs thresholds, not access to measured-wrong shapes.
- **Accept the scalar-half skew as noise (following ADR-0070's sibling-half reading).** Rejected: [#410](https://github.com/DEVunderdog/DataForgeML/issues/410) falsified it — +71–107% RMSE for linear estimators, three-plus orders past ADR-0070's bar. What was noise for the sibling half is not noise for the scalar half.
- **Close the skew only for linear estimators (gate on family).** Rejected ([#411](https://github.com/DEVunderdog/DataForgeML/issues/411)). It buys back one `column.median()` per tree-based block by betting "trees are skew-immune" — a fragile assumption that couples correctness to estimator taxonomy. Closing uniformly is simpler and unconditional.
- **Reshape: model the scalar columns in-block rather than pre-filling ([#410](https://github.com/DEVunderdog/DataForgeML/issues/410)'s `ideal`).** Out of scope. It is the theoretically cleaner frame but a larger change than the destination admits, and the uniform matched-frame fill captures the measured accuracy without it. Not parked as future work, per the driving dev.
- **Consume sibling `FittedScalar` units to get the scalar fill.** Rejected ([#411](https://github.com/DEVunderdog/DataForgeML/issues/411)). It reintroduces a fit-time ordering dependency between the scalar and model layers, landing on ADR-0061/0071's independent-unit model and ADR-0056's parallelism. Reproducing the central-tendency rule inline keeps the unit standalone at the cost of a contained logic-duplication — the right trade against a re-coupled fit graph.

## Consequences

- **Tiny-*n* `multi_mar`/`Severe` columns change routing.** The uniform `mice_min_rows` floor diverts below-floor columns that previously entered a chained fit to KNN→Median. Deliberate ([#406](https://github.com/DEVunderdog/DataForgeML/issues/406)).
- **Imputation output changes for every ex-`Regression` column, and for pre-existing MICE columns.** Ex-`Regression` columns now serve through the joint block; every MICE column now reads the full active-numeric matrix and is fit on the scalar-filled frame. This is user-visible. Pre-release, no legacy engine to diverge from.
- **The `FittedRegression` unit type is gone.** Any serialized `FittedRegression` payload hard-fails on load; the migration accepts this consciously. `evaluation.py`, config, and the fitter module all lose their Regression-specific arms.
- **The scalar central-tendency rule is duplicated in two places.** The block's fitter reproduces the fill that `FittedScalar` also computes. This is a logic-coupling, not a fit-time dependency — a change to the central-tendency rule must be applied in both, and a future reader must not "de-duplicate" by having the block consume scalar units, which is the exact re-coupling [#411](https://github.com/DEVunderdog/DataForgeML/issues/411) designed out.
- **`FittedMICE` gains a write-back restriction and a full-breadth fit surface.** The code effort must teach it to fit over the full active-numeric matrix while emitting only its owned columns, and adjust `_compute_mice_n_nearest_features` to count over that breadth. Until then the block would over-write columns it does not own.
- **The independence and parallelism invariants survive intact.** ADR-0056 (thread concurrency), ADR-0061/0071 (independent, standalone unit fit), and ADR-0070 (shared pre-model snapshot) all hold unchanged — the skew fix was deliberately shaped to avoid reopening any of them.
- **One strategy fewer to route, fit, score, serialize, and document.** The routing branch, the second fitter, the second fitted-unit type, and the evaluation arm all collapse into the MICE path. This is the simplification the collapse was for; the ~4–5× fit-cost reduction rides along as a consequence.
