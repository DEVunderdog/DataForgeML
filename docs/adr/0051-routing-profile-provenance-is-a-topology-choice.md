# ADR 0051: Imputation routing-profile provenance is a topology choice, not a config flag

`ImputationOrchestrator.fit(train_df, profile)` routes strategy selection off whatever `StructuralProfileResult` it is handed and is agnostic to that profile's provenance. Whether routing sees full-dataset statistics (mild optimism) or train-only statistics (leakage-clean) is therefore decided by *how the user composes profiling and splitting* — the Split-Profiling Topology — and the library exposes **no** `route_from_train_profile`-style flag to select it. This records why that flag is deliberately absent.

## Context

The Fit/Transform Discipline says every transforming phase must learn its parameters exclusively from training data. Imputation's *parameters* (fill values, fitted models) already are train-only. But its *routing decision* — median-vs-mean, MICE-vs-KNN, is-this-bimodal — is made from the profile it receives, and in the profile-first topology that profile is computed on the full dataset. A strict reading flags this as a transforming phase making a fit-time decision informed by test rows.

The magnitude is second-order (routing is a discrete, low-bandwidth choice, not a fitted parameter; only the `r2_train` diagnostic is nudged), and the profile-stratified split makes train distributionally representative of full, so full-data and train-only routing produce near-identical decisions on any dataset large enough for the difference to matter. Train-only routing is also **not strictly better**: on small data it estimates skew / bimodality / correlation from fewer rows and can *miss* structure the full data shows clearly — and small data is exactly where leakage bites hardest. So the two stances are a genuine robustness-vs-purity trade-off, not an upgrade.

The decisive fact is architectural: the phases are independent, and the imputation router already consumes a caller-supplied profile without inspecting where it came from. That means both stances are already reachable by composition:

- **Profile-first** (smart split): full profile is mandatory (the stratified split is driven by it and must precede any train partition), and routing reuses it. Full-data routing is intrinsic here, not a choice.
- **Split-first** (plain split): profile-unaware splitter → profile `train_df` → route from the train profile. Leakage-clean, no second pass. Every plain-tier CV strategy (group, walk-forward, repeated) enables this for free.

## Decision

Do not add an imputation config flag for routing-profile provenance. The leakage-vs-robustness stance is expressed by pipeline topology (see the Split-Profiling Topology term). The router stays provenance-agnostic. This amends the Fit/Transform Discipline: routing decisions are a sanctioned exception to "learn exclusively from train" **only** in the profile-first topology, where full-dataset routing is intrinsic and the stratified split renders it near-equivalent to train-only routing.

## Considered options

- **Add `ImputationConfig.route_from_train_profile: bool`.** Rejected: it re-encodes a choice the architecture already expresses by composition, defaults to an arbitrary stance, and (when true) forces a second full profiling pass for a second-order gain the stratified split has already all but erased.
- **Make train-only routing the default (re-profile train inside `fit`).** Rejected: trades a robust default for a noisier one on small data — the regime where leakage matters most — and couples the router to a profiler it currently does not need.
- **Say nothing and keep full-data routing implicit.** Rejected: a future reader sees a transforming phase consuming a full-dataset profile and reasonably suspects a leakage bug. The deliberate no-flag decision and its topology rationale need to be on record.

## Consequences

- Users who want leakage-audited routing must choose the split-first topology (plain splitter → profile train → fit). The docs carry this guidance; the library does not warn if a user profiles full then hands that profile to `fit` after a plain split.
- The smart stratified-split path keeps full-data routing permanently. Reversing that later (forcing train-only routing) would be a fit-contract behavior change, which is why the deliberate acceptance is recorded here.
