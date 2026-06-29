# ADR 0020: TestSplitImbalanceWarning lives in FittedImputer.transform(), not DataSplitter

**Status:** Superseded by ADR-0036

## Context

Scope 9 introduced a test-side split imbalance check: warn when the test split has a proportionally lower missing-value rate than the full-dataset profile for a fitted column. The question was where this check should fire.

The two candidate locations were:
1. `DataSplitter.profile_stratified_split()` — fires before fit, earlier signal.
2. `FittedImputer.transform()` — fires when the test set is actually transformed.

## Decision

The test-side check lives in `FittedImputer.transform()` and emits `TestSplitImbalanceWarning`.

## Rationale

- `DataSplitter.profile_stratified_split()` is only called by users who opted into the recommended split method. Users who used `random_split`, `time_split`, or a custom split — the group most likely to have a bad test split — never reach that code path.
- `FittedImputer.transform()` is called for every test set regardless of how the split was produced. The warning fires universally.
- The cost is storing `profile_missing_ratio` per column in `ColumnImputationRecord` so `transform()` has the profile ratio available without re-receiving the full profile.

## Consequences

- `ColumnImputationRecord` gains a `profile_missing_ratio: float` field for every column with a fitted imputation strategy.
- The check fires later (at transform time) than it would if placed in the splitter. This is acceptable — the warning is diagnostic, not preventive.
- Users who call `transform()` on a validation fold in k-fold cross-validation will receive per-fold warnings, which is correct behaviour: each fold should be evaluated independently.
