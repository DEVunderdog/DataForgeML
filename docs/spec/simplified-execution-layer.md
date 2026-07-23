# Spec: Simplified user-orchestrated imputation execution layer

**Status:** locked design spec — decisions only. Hand-off for a separate implementation effort.
**Source map:** [Wayfinder map #377 — Simplified user-orchestrated execution layer](https://github.com/DEVunderdog/DataForgeML/issues/377) (all tickets closed).
**Lineage:** deliberately reverses the execution-layer half of [map #333](https://github.com/DEVunderdog/DataForgeML/issues/333). #333 is implemented; this effort retires its resumable executor (ADR-0061) and its store/identity stack (ADR-0064), retires the degradation half of ADR-0066, narrows ADR-0063, and keeps intact the pure decision layer (ADR-0060), decision-carried hyperparameters (ADR-0062), and the `FittedUnit` reconstruction contract (ADR-0063).

This document is an **index over the ADRs plus a per-file implementation checklist**. Every decision lives in its ADR; this file does not restate rationale — it points, and it enumerates the concrete code changes an implementer makes.

## The decisions (ADR index)

| ADR | Decision | Supersedes / retires |
| --- | --- | --- |
| [0071](adr/0071-user-orchestrated-unit-fit-primitive.md) | Stateless `fit_unit` primitive + `FittedImputer.compose`; untrainable raises single-track; size guard killed; `NumericImputer` deleted | Supersedes 0061; retires 0066 degradation half |
| [0072](adr/0072-collapse-persistence-to-serialize-deserialize.md) | Bare polymorphic `serialize`/`deserialize`/`inspect`; store port + identity deleted; provenance/verification dropped | Supersedes 0064; narrows 0063 |
| [0073](adr/0073-hyperparameter-override-is-a-plan-edit-merge.md) | `with_hyperparameters(unit_id, dict\|None)` per-key merge onto complete decided base; strip all fitter fallbacks | Complements 0062 |
| [0074](adr/0074-per-unit-observability-fit-signals.md) | Structured ephemeral `FitSignals` on `UnitFitResult`; warnings dual-channelled; forced-oversize warned structurally | New |
| [0075](adr/0075-batch-scheduling-is-user-owned-fit-many-removed.md) | Batch scheduling is user-owned: the batch wrapper is removed and `fit_unit` is the single training primitive; the caller-side ADR-0069 pin rule is documented, not enforced | Amends 0071 |

## The shape, before → after

**Before (ADR-0061/0064):**
```
decide(profile, n_rows, config) -> ImputationDecision
exec = ImputationExecutor(decision, train_df, config, ...)
exec.execute_all_pending(max_workers=...)      # state machine, degradation, on_fit_error
imputer = exec.build()                          # completeness-gated roll-up
exec.persist_to_store(store) / rehydrate_from_store(store)   # content-addressed
```

**After (this spec):**
```
decision = decide(profile, n_rows, config)      # unchanged
decision = decision.with_hyperparameters(unit_id, {...})     # optional plan edit
results  = {u.unit_id: fit_unit(decision, u.unit_id, df)     # the user's own loop
            for u in decision.units}                         # (ADR-0075)
imputer  = FittedImputer.compose(decision, results)          # exact-coverage
out      = imputer.transform(df)
blob     = serialize(imputer_unit_or_decision_or_profile)    # bytes; user stores it
obj      = deserialize(blob)                                 # polymorphic
meta     = inspect(blob)                                     # header, no unpickle
```

The user owns the loop between `decide` and `compose`. There is no resumable object, no state machine, no store.

## Per-file implementation checklist

### Delete outright
- `src/dataforge_ml/imputation/_executor.py` — `ImputationExecutor`, `UnitState`, degradation helpers (`_degrade_unit`, `_degradation_reason`, `_emit_degradation_warning`, `_degraded_scalar`, `_degradation_target`), `persist_to_store`, `rehydrate_from_store`, `_data_fingerprint`, `_PLAN_SCALAR_STRATEGIES`. (ADR-0071)
- `src/dataforge_ml/_store.py` — `LocalFileStore`. (ADR-0072)
- `src/dataforge_ml/_identity.py` — `fitted_unit_identity` and all content-addressing. (ADR-0072)
- `src/dataforge_ml/imputation/_numeric_imputer.py` — the duplicate `NumericImputer.fit()` engine and its `_fallback_to_*` / `_resolve_fill_value` helpers. First **relocate** the survivors: `_mice_winning_tag` → `_decision_assembler.py`, `_MODEL_BASED_STRATEGIES` → `_config.py`, `_resolve_fit_workers` → `evaluation.py` (its sole remaining caller once batch scheduling is user-owned, ADR-0075). (ADR-0071 / #378)

### New files
- `src/dataforge_ml/imputation/_unit_fit.py` — `fit_unit(decision, unit_id, df, *, random_seed=None, n_jobs_inner=-1) -> UnitFitResult`, the `UnitFitResult` type, and `UnitNotTrainableError` (relocated from `_executor.py`). Normalises the frame off the plan's sentinel maps (ADR-0068) and drives via private `_dispatch_unit_fit`. Batch scheduling is user-owned (ADR-0075): training a plan is a caller-written `fit_unit` loop, and the ADR-0069 layer rule is caller-side documentation on `fit_unit` — sequential drives take `n_jobs_inner=-1`, self-parallelised drives pin `n_jobs_inner=1` per call. (ADR-0071 / 0075)

### Modify
- `src/dataforge_ml/imputation/_fitted_imputer.py` — add `FittedImputer.compose(decision, fitted_units)` classmethod (tolerant `FittedUnit | UnitFitResult`, exact-coverage raise, inherits `build()`'s structural-column projection + whole-frame guards). Collapse internals: `records` loses `fill_value` (pure structural manifest); `models` + `model_cols` → one ordered `units` list applied in plan order (ADR-0067). Delete `FittedDegradedJoint` handling. (ADR-0071)
- `src/dataforge_ml/imputation/_fitters.py` — rename `fit_unit` → `_dispatch_unit_fit`; strip **every** fallback (`hyp.get("tol", 1e-3)`, `hyp.get("max_iter", ctx.config.base_max_iter)`, `hyp.get("initial_strategy", ...)`, `hyp.get("n_neighbors", 5)`, `hyp.get("complete_frac", 0.0)`, siblings) → `hyp["..."]` (ADR-0073); add `perf_counter` timing + structural forced-oversize check + populate a `FitSignals` (ADR-0074); change `UnitFitOutcome` to carry `FitSignals` instead of `signals: tuple[str,...]`. Receive relocated `_MODEL_BASED_STRATEGIES` import.
- `src/dataforge_ml/imputation/_fitted_units.py` — strip `FittedRegression.signals`, `FittedRegression.max_iter_used`, and sibling observability fields; add `FitSignals` frozen record + `ImputationFitWarning`; delete `FittedDegradedJoint`; `FittedScalar` loses `executed_strategy`/`degradation_reason`. (ADR-0071 / 0074)
- `src/dataforge_ml/imputation/_config.py` — `ImputationDecision` holds two hyperparameter maps (decided base + sparse override delta); `_derive_units` stamps `merged = decided ⊕ delta`; add `with_hyperparameters(unit_id, dict | None)` (rename from `with_unit_hyperparameters`), unknown-key raises at edit time; receive relocated `_MODEL_BASED_STRATEGIES`. `decide()`'s KNN base must populate `complete_frac`. Delete `_ON_FIT_ERROR_POLICIES` / `on_fit_error` config surface. (ADR-0073 / 0071)
- `src/dataforge_ml/imputation/_decision_assembler.py` — receive relocated `_mice_winning_tag`; ensure the decided-base maps are always complete per strategy (the schema the override validates against). (ADR-0073)
- `src/dataforge_ml/_serialization.py` — rewrite around `serialize(obj) -> bytes` / `deserialize(data) -> FittedUnit | ImputationDecision | StructuralProfileResult` (polymorphic, `kind` tag) / `inspect(data) -> dict` (header without unpickle); JSON envelope + single joblib tail for the unit only; keep `produced_with` six-version + SHA-256 gate pre-unpickle on the unit; light schema+`library_version` stamp on decision/profile; delete the `ArtifactStore`/`DocumentStore`/`BlobStore` protocols and `ArtifactIdentityMismatchError`/`verify_identity`. (ADR-0072)
- `src/dataforge_ml/imputation/_fitted_persistence.py` — collapse: delete `save_fitted_unit_addressed` / `load_fitted_unit_addressed`; keep the unit envelope encode/decode, now reached via `serialize`/`deserialize`. (ADR-0072)
- `src/dataforge_ml/profiling/orchestrator.py` — remove the `data_fingerprint` output and the whole-frame hash computed for keying; drop the fingerprint field from `StructuralProfileResult`. (ADR-0072)
- `src/dataforge_ml/__init__.py` and `src/dataforge_ml/imputation/__init__.py` — see Public API changes below.

## Public API changes (`__all__`)

**Remove:** `ImputationExecutor`, `UnitState`, `ArtifactStore`, `LocalFileStore`, `ArtifactIdentityMismatchError`.

**Add:** `fit_unit`, `UnitFitResult`, `serialize`, `deserialize`, `inspect`, `FitSignals`, `ImputationFitWarning`. (`FittedImputer.compose` is a method on an already-exported type. No batch wrapper is exported — batch scheduling is user-owned, ADR-0075.)

**Keep:** `decide`, `FittedUnit`, `FittedImputer`, `UnitNotTrainableError` (relocated), `IncompatibleArtifactError`, `ArtifactPythonVersionWarning`, and everything unrelated to execution/persistence.

All added public symbols are documentation-scoped (ADR-0034): numpy-style docstrings required on `fit_unit`, `UnitFitResult`, `FittedImputer.compose`, `serialize`, `deserialize`, `inspect`, `FitSignals`, `ImputationFitWarning`, and `ImputationDecision.with_hyperparameters`.

## Cross-cutting invariants (do not regress)

- **Never re-add the forced-strategy size guard** to `decide()` or `fit_unit`. Forcing past a threshold is informed consent; ADR-0074 *warns*, never blocks. (ADR-0071)
- **Untrainable is single-track:** always `UnitNotTrainableError`. No degradation, no `on_fit_error`, no two-track. (ADR-0071)
- **A missing hyperparameter key is a `decide()` bug**, surfaced as a loud `KeyError` from the stripped fitter — never a silent config re-read. (ADR-0073)
- **`FitSignals` is ephemeral** — never serialized; no fit-runtime facts persist on fitted objects. (ADR-0074)
- **Application order is plan order** (ADR-0067), preserved by `compose`'s ordered `units` list — never thread-completion order.

## Out of scope (from the map)

- Building the deferred persistence service / API (inherited from #333).
- The pure decision layer internals (`decide()`, routing, ADR-0060/0062) except the hyperparameter override channel.
- Future phases (encoding/scaling/…).
- [Hyperparameter optimization (#384)](https://github.com/DEVunderdog/DataForgeML/issues/384) — a search's apply-mechanism is `with_hyperparameters`; designing the search is its own effort.
