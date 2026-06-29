# ADR 0036: Remove proportional split imbalance checks

**Status:** Accepted  
**Supersedes:** ADR-0020, ADR-0022

The `TrainSplitImbalanceWarning` and `TestSplitImbalanceWarning` checks, along with `split_imbalance_ratio_threshold` and `profile_missing_ratio`, are removed entirely.

## Why

The proportional check (`partition_ratio < threshold × profile_ratio`) is a heuristic with no principled basis. The threshold has no correct default because the right tolerance depends on partition size — a deviation that is statistically impossible on a 1000-row test set is entirely expected on a 6-row one. The check fires spuriously for small partitions even when stratification worked correctly, and can silently miss bad splits when the threshold is loose.

More fundamentally, the check does not protect against a real failure mode. For mean/median/mode strategies, the fill value is computed from observed (non-null) values regardless of how many null rows training saw — the imputation output is the same. For model-based strategies (KNN, Regression, MICE), the model learns to predict from features, not from the count of null training rows. The one genuine failure — a column that was completely null-free in training — is already caught precisely and as an error (not a warning) by the Passthrough violation check in `FittedImputer.transform()`.

## Where the library draws the boundary

The library's answer to split quality is `DataSplitter.profile_stratified_split()` and `profile_stratified_kfold()`. These are the tools; using them is the user's responsibility. A warning that fires after a bad split has already been made does not prevent the bad split and produces false positives that train users to ignore warnings.

## Considered alternative

Fixing the check to be partition-size-aware (binomial confidence interval or expected-null-count floor) was evaluated. Rejected because even a corrected check guards against a failure mode that does not exist for the imputation strategies in use. Fixing the mechanism while leaving the premise broken is not worth the complexity.

## Consequences

- `TrainSplitImbalanceWarning` and `TestSplitImbalanceWarning` are removed from the public API (hard delete, no deprecation stubs — library is pre-1.0).
- `split_imbalance_ratio_threshold` removed from `NumericImputationConfig` and `FittedImputer`.
- `profile_missing_ratio` removed from `ColumnImputationRecord`.
- `_check_split_imbalance` function removed from `ImputationOrchestrator`.
- The Passthrough violation check in `FittedImputer.transform()` is retained — it is precise, actionable, and guards a real failure.
