# Batch scheduling is user-owned; `fit_many` is removed

ADR-0071 shipped execution as a stateless `fit_unit` primitive plus two thin wrappers, one of which — `fit_many(decision, df, max_workers=...)` — drove a batch of units through a thread pool with `max_workers` as the single ADR-0069 layer switch (`n_jobs_inner = 1 if workers > 1 else -1`). This ADR removes `fit_many` and hands batch scheduling to the user. `FittedImputer.compose` and the `fit_unit` primitive are untouched; `UnitFitResult` survives as `fit_unit`'s return type.

The defect is not correctness — with `max_workers > 1` every core stays busy at the outer layer, and for batches of many similar-sized units the wrapper's schedule is near-optimal. The defect is that a single global layer switch cannot express the one schedule that matters when a batch is **dominated by a large RandomForest-routed unit**: outer-fan the units that have no inner parallelism, then run the inner-capable stragglers one at a time with `n_jobs_inner=-1`. Under `fit_many` that batch ends with idle workers watching one unit grind single-core. The raw material for the better schedule is already public — the plan carries `model_choice` per column (ADR-0062/ADR-0066), so inner-capable units are identifiable, and `fit_unit` exposes `n_jobs_inner` directly — but the wrapper's presence steers users toward its one coarse shape instead.

Locked decisions:

- **`fit_many` is deleted**, along with its exports from `dataforge_ml` and `dataforge_ml.imputation`. The public shape becomes `decide()` → edit → `fit_unit` loop (user-owned) → `FittedImputer.compose` → `transform`.
- **`fit_unit` keeps `n_jobs_inner=-1` as its default.** The sequential one-at-a-time drive is the primary shape and is outer-degree-one, which per ADR-0069 deserves wide inner parallelism. The default is not flipped to protect hand-rolled pools.
- **The pin-when-parallel rule becomes loud documentation, not enforcement** (informed consent, consistent with the killed size guard of ADR-0071). A user who parallelises `fit_unit` calls themselves must pass `n_jobs_inner=1`; failing to do so oversubscribes every core (N threads × all-core RandomForest fits) and is silently slow, not wrong. `fit_unit`'s docstring states this rule prominently.
- **The ADR-0069 determinism criterion is unaffected and stays tested.** Core-invariance belongs to `_CoreInvariantRandomForest`, not to the deleted wrapper; the concurrency suite asserts it over a hand-rolled pool of `fit_unit(n_jobs_inner=1)` against a sequential `n_jobs_inner=-1` drive.
- **Per-call normalisation cost is accepted.** `fit_many` normalised the frame once per batch; N `fit_unit` calls normalise N times. Normalisation is idempotent (a Polars null matches no sentinel rule, so a second pass is a no-op) and its scan is trivial next to model training, so no public pre-normalise hook is added and no `assume_normalized` flag exists. ADR-0074's note on the batch-vs-standalone `duration_s` skew is obsolete: every `duration_s` now spans its own normalisation.

## Status

accepted; supersedes the `fit_many` clauses of ADR-0071 (both the batch-driver locked decision and its export/consequence mentions) and the `fit_many` clauses of ADR-0074

## Considered Options

- **Keep `fit_many` as-is and document the straggler partition pattern.** Zero new surface and near-optimal for the common many-similar-units batch; rejected because the wrapper's existence is itself the steering — it presents one coarse schedule as *the* batch API while the shape it mishandles is exactly the expensive one (RandomForest is the costly route), and the escape hatch it points away from is two lines of user code.
- **Make `fit_many` partition-aware** (phase 1: outer-fan the units without inner parallelism; phase 2: inner-capable units serially wide). Rejected: this policy also loses in a shape — many *small* RandomForest units get serialised in phase 2 and finish slower than plain outer fanning — and choosing between the two policies at runtime needs per-unit cost estimates the library does not have. It swaps one blind spot for another and bakes scheduling policy into a layer ADR-0071 deliberately made dumb.
- **Remove `fit_many` and flip `fit_unit`'s default to `n_jobs_inner=1`** so naive hand-rolled pools are safe. Rejected: it breaks the primary sequential drive — single core at both layers unless the user knows to pass `-1` — recreating exactly the defect ADR-0069's fourth locked decision fixed.
- **Remove `fit_many` but ship the correct pool loop as a documented copy-paste recipe.** Rejected: that is `fit_many`'s body redistributed into every user's codebase, minus tests and fail-fast semantics — the wrapper wearing a trenchcoat.

## Consequences

- Users who want batch parallelism write the `ThreadPoolExecutor` loop themselves and own the layer choice per unit: outer-fan the non-inner-capable units pinned, drive the inner-capable ones wide, or mix — the schedules `fit_many`'s single switch could not express.
- The oversubscription failure mode is real and silent: a hand-rolled pool over default `fit_unit` thrashes rather than errors. This is the accepted price of informed consent; the defence is documentation on `fit_unit`, not a guard.
- `fit_many`-era test plumbing (conftest helpers, the concurrency suite's batch entry point, surface tests) is rewritten against `fit_unit` loops. Pre-release, so no compatibility burden.
- `_resolve_fit_workers` and the batch branch of `_unit_fit.py` delete with the wrapper; stray `fit_many` references in `_fitted_imputer.py` / `_fit_signals.py` docstrings and `docs/spec/simplified-execution-layer.md` are updated.
