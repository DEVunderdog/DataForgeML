# A Fitted Unit fills holes and never edits an observed cell: transform returns every non-missing input cell bit-for-bit, original dtype included

A **Fitted Unit** fills holes. It never edits a value the user supplied. After `transform`, every cell that was not missing in the input is identical — same value, same dtype. This ADR records that invariant as domain knowledge, resolving [Fitted Units must preserve observed values and dtypes (#399)](https://github.com/DEVunderdog/DataForgeML/issues/399) via [#400](https://github.com/DEVunderdog/DataForgeML/issues/400), [#401](https://github.com/DEVunderdog/DataForgeML/issues/401), and [#402](https://github.com/DEVunderdog/DataForgeML/issues/402). (The tickets name this record ADR-0077; that number was claimed by the exclusion-enforcement ADR in the interim, so it lands here as ADR-0078.)

Three independent defects converged on the same missing guarantee: the whole-column domain snap (`FittedRegression`, `FittedMICE`, `_FittedKNN`, `FittedClusterConditional` rounded and clipped *observed* cells of a `domain_snap_bounds` column — a user's observed `37.6` came back `38.0`), silent `Int64`→`Float64` widening (`FittedClusterConditional` and `FittedGMMSampling` wrote back via `pl.Series(col, numpy_arr)`, where `to_numpy()` on a nullable integer column yields `float64`), and KNN's lossy scale/inverse-scale round trip drifting observed floats. `FittedScalar` was already correct — `fill_null` writes only null cells.

Locked decisions:

- **The invariant is bit-for-bit, dtype included — not "not meaningfully changed".** The weak form is not assertable: any tolerance invites the next drift to hide under it. Dtype is part of the guarantee, which fixes the integer widening in the same place; the consequence is that a `3.7` fill destined for an `Int64` column now rounds to `4` instead of widening the column — matching the integer-fill convention `FittedScalar` and the numpy write-back already follow.
- **Enforced inside each unit's own `transform`, as its final step, via one shared private helper** (`_preserve_observed` in the imputation utils module). The unit hands the frame it received and the frame it produced through the helper; observed cells are restored from the input, missing cells keep the fill. No mask pre-capture is needed — Polars frames are immutable, so the input frame is intact when `transform` returns.
- **Missing means the Effective Null predicate** — `is_null() OR is_nan() OR is_infinite()` for float columns, `is_null()` alone otherwise — matching exactly what the frame-to-numpy conversion hands the estimator as a hole.
- **Scope is the unit's `target_columns` only**, never the whole frame. A unit has no standing over columns it does not own.
- **The domain snap stays a whole-column expression; the helper corrects afterwards.** Correctness has one enforcement point, not two.
- **No opt-out flag, no configuration, no migration path.** The full-configurability rule (advanced-developer library) covers thresholds and strategy knobs, not correctness invariants. There is nothing legitimate to configure toward.

The verification seam is exactly one: the public `decide` → `fit_unit` → `result.fitted.transform(df)` path, parametrized over all six unit types. The helper is private and never tested directly — naming it in a test would pin the implementation. `FittedImputer.transform` is deliberately not a second seam: the guarantee is a property of the unit.

## Status

accepted

Depends on ADR-0070 (units transform off a shared pre-model snapshot and merge only their own columns — what makes per-unit enforcement compose) and ADR-0071 (the stateless door this invariant must hold on). Shipped in v2.3.0 — a minor bump, not a patch, because imputation output changes for columns carrying `domain_snap_bounds`.

## Considered Options

- **Enforcing in the orchestrator's unit loop** (`FittedImputer.transform` restores observed cells after each unit runs). Rejected: fixes all units from one place, but makes the guarantee a property of `FittedImputer` rather than of the unit — leaving the ADR-0071 stateless door unprotected, where a user holds a single Fitted Unit and calls `transform` with no `FittedImputer` involved — and contradicting what the fitted-units module claims about its own types.
- **Converting the `FittedUnit` structural protocol into a base class** with a concrete `transform` delegating to a per-unit implementation hook. Rejected: makes the guarantee impossible to forget, but restructures a public protocol into inheritance to buy one forgettable call site. The residual risk — a future unit forgetting the final helper call — is carried by the parametrized test instead.
- **A bare `is_null()` missingness predicate.** Rejected: a raw `NaN` in a float column reads as *observed* under `is_null()`, so the helper would restore `NaN` over a cell the estimator had just correctly filled — the unit would fill a hole and immediately un-fill it.
- **Rewriting each domain snap to be mask-aware** (snap only the cells that were missing). Rejected: reaches the same result but establishes a second enforcement point that a future unit could get right in one place and wrong in the other; the whole-column snap plus a single terminal correction keeps one place to be correct.

## Consequences

- **Observed-value-revising strategies are foreclosed as Fitted Units.** An imputation strategy that intends to *revise* an observed value — outlier correction, measurement-error repair — can no longer be a Fitted Unit and would need its own phase. The invariant is definitional: a unit that edits observed cells is not doing imputation.
- **The standalone-door sentinel asymmetry is accepted and documented, not fixed.** The stateless door resolves float `NaN`/`Inf` as missing with no configuration (the Effective Null predicate needs no state), but a bare unit cannot resolve sentinel-encoded **Effective Nulls**, because the declared sentinel maps live on the **Imputation Decision** (ADR-0068) and a bare unit does not hold them. A caller transforming through a bare unit owns sentinel normalisation; the composed `FittedImputer.transform` normalises off the plan's maps as before. This gap pre-exists the change.
- **Imputation output changes for columns carrying `domain_snap_bounds`**: observed cells are no longer rounded/clipped, integer columns no longer widen, and KNN's round-trip float drift on observed cells is gone. A numeric diff against a previous run is explicable by exactly this; the changelog states it plainly and the version bumps to 2.3.0.
- **`FittedScalar` is untouched** — it already satisfied the invariant — and rides in the parametrized test as the no-op control so it must keep passing unchanged.
