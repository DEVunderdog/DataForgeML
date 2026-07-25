# Declared sentinels are a decide-time fact carried on the plan, and execution normalises for itself

`ImputationOrchestrator.fit` normalises effective nulls before fitting; `ImputationExecutor` never did. It stored the frame raw and every fitter read it as-is, so a caller handing the executor a frame with a declared sentinel (`-999`) got the sentinel averaged in as legitimate data and surviving untouched into the output, where the legacy path imputed the sentinel rows to the mean of the clean values. The two engines disagreed on the same inputs. The defect was latent only because the executor has no public door yet.

The question was never *whether* to normalise — it was **where**, because copying the orchestrator's call into the executor is the answer that looks obvious and quietly re-couples execution to the profile that ADR-0060 separated.

Locked decisions:

- **`numeric_sentinels` and `string_sentinels` are fields on `ImputationDecision`.** Sentinels are a decide-time fact *about the data*, discovered by Phase 1 and fixed before anything trains, which is ADR-0060's own test for what belongs on the plan. `decide()` copies them off the profile alongside `profile_provenance`. This is ADR-0066's argument for `forced`, applied unchanged: **a plan loaded from the store must know what to normalise without the original profile object**, and that is impossible while the sentinel maps live only on a profile the store does not hold.
- **`ImputationExecutor.__init__` normalises the frame it is handed.** The executor owns its own correctness regardless of caller, and takes the maps off the plan it already holds — so its inputs stay `(plan, frame, config)` and do not widen to include the profile. Normalisation is idempotent, so the legacy path passing an already-normalised frame costs one no-op pass, not a wrong answer.
- **`data_fingerprint` stays on the frame as handed in, not on the normalised frame the units are fitted on.** Fingerprinting what was trained on reads as the more truthful choice and is wrong here on two counts: `_data_fingerprint` exists to match the *profiler's* fingerprint, which is taken over raw data; and the executor only ever `load`s from the store — whoever writes a unit holds the raw frame and must be able to compute the same key. Normalisation is a pure function of `(raw frame, plan sentinel maps)` and the plan's identity is already part of the key, so the raw fingerprint identifies the normalised data exactly as tightly while staying reproducible from the frame a caller actually has.
- **`build()` carries the plan's sentinel maps onto the composed `FittedImputer`.** It was passing `{}`, so a `FittedImputer` built by the executor would not have normalised at transform time either — the same defect on the other side of the seam, fixed by the same field.

## Status

accepted

## Considered Options

- **Normalise at the boundary and require a normalised frame.** Rejected. It keeps the executor's inputs narrowest, and it is the cheapest change, but it makes "already normalised" an unwritten precondition of the execution layer — exactly the class of assumption the Decision/Execution split exists to remove. Enforcing it is not available either: a normalised frame is not distinguishable from a raw frame that happens to contain no sentinels, so the precondition could only ever be *documented*, i.e. trusted. The whole point of the effort is that a plan plus a frame is sufficient.
- **Pass the profile into `ImputationExecutor.__init__` and normalise from it.** Rejected. It is self-contained and needs no schema change, but it widens the executor's inputs from `(plan, frame, config)` to include the profile, re-coupling execution to the object ADR-0060 separated it from — and it makes a plan loaded from the store unexecutable without also having its source profile, which is precisely what content-addressed plan storage (ADR-0064) was built to avoid.
- **Leave it and let #372's parity gate catch it.** Rejected. The gate would fail correctly but the failure would read as a fitter-extraction defect rather than a null-handling one, and #368 would meanwhile export an executor that trains on sentinels.

## Consequences

- **The plan's schema grows two fields**, which is a serialization change (ADR-0063). Pre-release, so no compatibility burden; `from_dict` defaults both to `{}`, so an older document loads as a plan with no declared sentinels — which is the correct reading of a document written before the field existed.
- **A hand-edited plan can now lie about the data.** The sentinel maps are editable like every other plan field, so a user can remove a declaration the profiler made and have the executor train on the sentinel. This is the price of the plan being the single self-sufficient input to execution, and it is the same exposure every other decide-time field already carries.
- **The maps are duplicated between profile and plan**, and a plan re-decided from a changed profile is the only thing that refreshes them. `identity()` is unaffected — it chains to the profile identity, which already covers the sentinel declarations transitively — so a plan whose sentinels disagree with its profile cannot arise from `decide()`, only from an edit.
- **The executor holds a second reference to the incoming frame** (`_raw_train_df`), purely to fingerprint against. Two frames named `train_df`-something in one object is a genuine readability cost and an invitation to fit against the wrong one; the alternative — hashing eagerly in `__init__` and keeping only the digest — was rejected because it charges every caller a full serialise-and-SHA of the frame whether or not they ever touch a store.
- **Stored unit keys are unaffected**, since the fingerprint's input did not change. Had it moved to the normalised frame, every previously persisted unit would have missed and retrained.
- **Non-numeric sentinel handling is carried but not yet exercised.** `string_sentinels` rides along for the same reason the orchestrator passes it, though Phase 2 currently routes only numeric columns.
