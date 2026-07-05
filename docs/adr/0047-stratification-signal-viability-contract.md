# ADR 0047: Stratification signals pass a viability contract; the cap ranks by importance, not rarity

`build_label_matrix` previously admitted every signal with at least one positive and, when over the cap, retained the **rarest** signals (smallest proportion of 1s). This was backwards: a signal with too few positives is exactly the one `iterstrat` cannot place into every partition, so ranking by rarity preferentially kept the least-splittable signals and could evict the target entirely. With the removal of the post-hoc train/test imbalance warnings, the label matrix is now the *only* safeguard of split quality, so we replace the rarest-first policy with an explicit four-gate viability contract.

## Context

Stratification is not profiling. Phase 1 computes every indicator because doing so is free insurance (the accuracy-over-speed principle). Stratification operates under a hard budget: `MultilabelStratifiedKFold(k)` can only put a positive of a label into every fold when that label has at least `k` positives, and every additional signal sparsifies the label-combination space, making the split noisier. The governing principle here is therefore the inverse — *the fewest signals that capture the most distributional coverage*.

## The contract

Every candidate signal passes four gates, in order:

1. **Relevant** — a downstream phase's quality depends on the property being balanced.
2. **Viable** — `min(positives, negatives) >= floor`, with `floor = k` for k-fold and `ceil(2 / min(test_size, 1 − test_size))` for shuffle-split. The floor is computed by the caller (which knows the split intent) and threaded in as a `min_positives` argument; `build_label_matrix` stays ignorant of split modes and only enforces "≥ this many 1s and ≥ this many 0s." This two-sided check subsumes both the old all-zeros drop and a symmetric all-ones drop.
3. **Non-redundant** — signals imposing the same balancing constraint are collapsed. Redundancy is measured by correlation strength, which catches both near-identical (`+1`) and mirror-image (`−1`) pairs; a naive shared-positives check misses the mirror-image case (e.g. the two flags of a binary target). Mutually-exclusive class sets (target classes, quantile buckets) additionally drop their last member dummy-style. The threshold is conservative — only near-perfect duplicates are removed, never merely-similar signals.
4. **Prioritized** — over the cap after gates 1–3, retain by importance: target (never evictable) → rare categorical → numeric extremes/skew → missingness (last). Within a family, rarer viable signals are kept first.

The cap is `min(max_stratification_signals, n_rows / rows_per_signal)`. The row-based term guards the more-signals-than-rows regime, where the splitter is handed more simultaneous constraints than the data can satisfy. Both terms are configurable (`rows_per_signal` defaults to ~10, a standard per-constraint rule of thumb). When even the target cannot be supported, the matrix is empty and the splitter falls back to a random split.

## Considered options

- **Keep rarest-first, add only a viability floor.** Rejected: rarity and importance are orthogonal; a rare noise signal would still evict the target.
- **Measure redundancy by shared positives (Jaccard).** Rejected: it treats a binary target's two mirror flags as maximally *different* (they share zero positives) and keeps both, which is the exact double-count we want to remove.

## Consequences

- Behaviour changes on any dataset that previously hit the cap or carried single-positive signals — acceptable, as the library is pre-release and correctness of the split outranks continuity.
- Un-viable rare *labels* (target classes, rare categorical values) are not simply dropped; their rows are routed per ADR-0048.
