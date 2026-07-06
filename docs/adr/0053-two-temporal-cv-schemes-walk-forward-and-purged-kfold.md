# ADR 0053: Two temporal CV schemes ship separately — walk-forward and purged K-fold

Temporal cross-validation is served by **two distinct methods**, not one parameterised method: `time_series_cv` (forward-chaining walk-forward) and `purged_kfold` (López de Prado purged/embargoed K-fold). They have different partition geometries and, crucially, different causality guarantees. A future reader will wonder why temporal CV wasn't unified behind a single entry point; this records that the two schemes are not variants of each other.

## Context

Walk-forward is forward-chaining: train is always chronologically before val. Its only leakage surface is the single boundary just before val, handled by a `gap` (rows). There is never training data after val, so there is nothing to embargo on the far side.

Purged K-fold cuts the timeline into `n_splits` contiguous blocks and uses every *other* block as train — both before and after the val block. Training data therefore exists on both sides of val, which introduces two leakage surfaces walk-forward does not have: label windows that overlap val (handled by **purge**) and serial correlation from rows just after val (handled by **embargo**). Embargo-after is meaningful *only* because purged K-fold has train-after-val.

The two also make opposite causality trades: walk-forward never predicts the past from the future (strict causality, less data per fold, tiny early folds); purged K-fold does use future data to predict past (acceptable when only label-overlap leakage matters), buying equal-sized folds each using ~(K−1)/K of the data. Folding these into one method would force a single geometry and a single set of parameters onto two genuinely different tools.

## Decision

- `time_series_cv(time_column, n_splits, window="expanding"|"rolling", gap=0, test_size=None, max_train_size=None)` — walk-forward, train always before val.
- `purged_kfold(time_column, n_splits, embargo=0, label_end_column=None)` — purged K-fold; `embargo` is a time duration in `time_column` units; `label_end_column` (opt-in) enables true purge on label-window overlap, degrading to embargo-only when absent.

Both are plain-tier, sort by `time_column`, and never stratify.

## Considered options

- **One method with a `purged=True` flag.** Rejected: the two geometries differ in whether train can follow val, which changes the meaning of the parameters (`gap` vs `embargo`, purge only defined for the K-fold layout). A flag would gate half the parameters as no-ops in each mode.
- **Only walk-forward.** Rejected: the user explicitly needs purged K-fold, and forward-chaining alone cannot use post-val data or handle overlapping labels.
- **Embargo/purge as a fraction or row count.** Rejected for embargo: irregular sampling makes fractions and row counts silently wrong; a time duration is the only unit robust to uneven spacing.

## Consequences

- Users must know which causality assumption their task makes and pick the matching method; the library does not choose for them (consistent with ADR-0049 — temporality is user intent).
- `purged_kfold` without a `label_end_column` provides embargo-only protection, which does not catch label overlaps wider than the embargo window. This is documented, not warned.
