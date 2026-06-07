# ADR 0007: Profile-stratified split uses full profile signals, not missingness only

**Status:** Accepted

## Context

When a user splits their dataset before running transforming phases, a naive random split can produce distributional imbalances that break downstream phases: a column with low missingness gets zero missing values in train (Phase 2 cannot fit an imputer for it), extreme outliers cluster in one split (Phase 3 Outlier Detection sees different tail densities), or rare categorical labels appear only in the test split (Phase 5 Encoding cannot encode unseen labels).

The question was whether to add a missingness-aware split (fixing only Phase 2's concern) or a broader profile-aware split that addresses all downstream phases simultaneously.

## Decision

`DataSplitter.profile_stratified_split(profile, test_size)` consumes a `StructuralProfileResult` and stratifies by a composite row-level score derived from three signal families: missingness density (rows with missing values), extreme value presence (rows in numeric tails), and rare label presence (rows containing low-frequency categorical values or near-constant column anomalies). No user configuration is required — signals are auto-derived from the profile.

## Rationale

- Phase 1 already computes all the signals needed: missingness per column, numeric percentiles and flags, categorical frequency distributions. The information is available at zero extra cost.
- A missingness-only split would need to be revisited and extended when Phases 3 and 5 were implemented. A single general-purpose profile-stratified split solves the problem once.
- Users with constrained split requirements (temporal ordering, domain partitions) still have `random_split` and `time_split`. The profile-stratified variant is an opt-in safe default, not a replacement.
- `SplitImbalanceWarning` from `ImputationOrchestrator.fit()` guides users who used a non-profile-stratified split toward this method when a problem is detected.

## Consequences

- `DataSplitter` gains a dependency on `StructuralProfileResult` (a Phase 1 output type). This means `DataSplitter` is only fully functional after Phase 1 has run, which is the intended workflow.
- The composite stratification score is a heuristic — it cannot guarantee perfect balance across all columns simultaneously (especially in small datasets). It reduces risk; it does not eliminate it.
- `profile_stratified_split` does not replace stratification by target class. Users who need both target stratification and profile stratification should use `profile_stratified_split` with the target column also declared in `PipelineConfig`.
