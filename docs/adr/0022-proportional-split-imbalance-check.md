# ADR 0022: Split imbalance check is proportional, not binary

**Status:** Accepted

## Context

The original `_check_split_imbalance` fired only when `train_df[col].null_count() == 0` for a column the full-dataset profile reported as having missingness. The check was binary: train is either completely clean or not.

This misses a dangerous partial case: a column that is 20% missing in the full dataset but only 2% missing in train passes the binary check silently. The imputation model is trained on a set where missingness is drastically underrepresented — the learned fill statistics and model weights do not reflect the population.

## Decision

Replace the binary check with a proportional check. For both train and test:

- `train_ratio < split_imbalance_ratio_threshold × profile_ratio` → `TrainSplitImbalanceWarning`
- `test_ratio < split_imbalance_ratio_threshold × profile_ratio` → `TestSplitImbalanceWarning`

Default threshold: `split_imbalance_ratio_threshold = 0.5`, configurable in `NumericImputationConfig`.

## Rationale

- The binary check protects only the degenerate case (all missing rows in test). The proportional check protects against the larger class of skewed splits where missingness is representable but severely underrepresented.
- A relative threshold scales naturally: a column with 1% profile missingness warns only if train sees less than 0.5%, while a column with 30% missingness warns if train sees less than 15%.
- 0.5 as the default captures the motivating case (20% → 2%) without being so strict that mildly imperfect splits from profile-stratified splitting trigger false alarms.
- Making the threshold configurable acknowledges that teams with known distributional skew (e.g. time-based splits where recent data has higher missingness) may need to relax it.

## Consequences

- `NumericImputationConfig` gains `split_imbalance_ratio_threshold: float = 0.5`.
- The check no longer fires on the exact-zero case specifically — it fires on any ratio below the threshold, which strictly subsumes the binary case at any threshold > 0.
- Numeric sentinel columns remain exempt from the proportional check until Scope 5 ships: `_resolve_effective_nulls` does not normalise user-declared numeric sentinels, so the denominator for those columns would be incorrect.
