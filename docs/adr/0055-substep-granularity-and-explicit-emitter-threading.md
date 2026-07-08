# Substep-level progress via an explicit Emitter threaded into sub-processors

ADR-0054 set the progress floor at column-level and deferred anything finer. In practice that floor makes long inner stretches — the single joint MICE fit over all columns, and the cross-validated diagnostics — run silently for minutes after the per-column `item` events have all fired in a burst, so the run looks hung exactly when it is busiest. We amend that floor: we introduce a **Substep** — a named unit of work *below* the column (a strategy-block fit, a diagnostics fold, a per-column model fit within a set) — surfaced as a new `substep` `event_type`. A consumer drives a coarse progress bar off `item` and a live detail line off `substep`. We stop *above* solver-iteration granularity (no "MICE round N"): the library must own every emit point, and per-round reporting would require reaching into an estimator's internals.

To let deep methods emit at all — today `NumericImputer.fit()` and its helpers are observer-blind — an orchestrator builds one **Emitter** (carrying `phase` + Progress Observer, owning the Substep/index bookkeeping and the Two-Sink Rule) and **threads it explicitly** into the sub-processors and their fitting/diagnostic helpers. This is the uniform contract every current and future phase honors.

## Status

accepted — amends ADR-0054 (supersedes its "column-level is the item floor" consequence)

## Considered Options

- **Overload `item` for Substeps instead of adding `substep`.** Rejected: a consumer counting `item`s to fill a progress bar would have its "k of N" polluted by sub-column chatter. A distinct type keeps coarse and fine cleanly separable; it is an additive `StrEnum` value, so existing observers that ignore it are unaffected.
- **Ambient Emitter via `contextvars` (no signature changes).** Rejected: hidden control flow, harder to test, and a concrete footgun under our thread-based concurrency (ADR-0056) — the ambient value silently vanishes in worker threads unless the context is explicitly copied in. Explicit threading of one small object is the cost of ~6 function signatures growing an `emitter` parameter, which in a library that prizes explicitness is a feature, not a tax.
- **Solver-iteration granularity ("MICE round N").** Rejected as before: `IterativeImputer` exposes no per-round callback, so it would require wrapping or monkeypatching sklearn internals — fragile across versions and not a point the library controls.
- **Keep emission in the orchestrator, just fix the burst ordering.** Rejected: MICE fits all columns in one joint call, so the longest silent block stays silent regardless of when the column `item` fires. Only emission from inside the fit removes the silence.

## Consequences

- The `substep` value is added to the public `EventType` vocabulary; `PipelineEvent` gains no new fields (`index`/`total` already carry sub-progression like "fold 3/5").
- Under concurrency (ADR-0056), Substep events from parallel units **interleave** rather than arriving in strict order — accepted as "here's what's in flight," which is the point.
- The Emitter-threaded-from-orchestrator pattern is the standing rule for all phases; profiling has the same latent silence and is to be retrofitted to it as a tracked follow-up, not in this change.
