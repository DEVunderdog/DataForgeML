# ADR 0052: Grouped splitting keeps disjointness hard and balances the target only

The smart-tier grouped splitter treats **group-disjointness as an inviolable constraint** and treats target-class balance as a best-effort layer on top (`StratifiedGroupKFold` semantics). It balances the **target only** — not the full profile-signal matrix that `profile_stratified_split` uses. A future reader will reasonably ask why the "smart" grouped splitter does *less* signal-balancing than the smart ungrouped one; this records that the asymmetry is deliberate and mathematically forced.

## Context

Groups (patient, user, session) arrive as indivisible blocks. Two desirable properties of a grouped fold conflict:

1. **Group-disjointness** — no group appears in both train and val. This is the *entire reason the method exists*: without it, a model memorises the entity and validation is optimistically biased.
2. **Balance** — each fold carries proportional shares of some signal(s).

Because a group's block has a fixed internal composition, keeping it whole can push a fold off the ideal balance; the two goals cannot always both be satisfied. Two knobs must therefore be decided: (a) which wins when they conflict, and (b) how much to balance at all.

## Decision

- **(a) Disjointness wins, always.** When keeping a group whole conflicts with balance, the group stays whole and balance absorbs the cost. A grouped splitter that leaked a group to improve balance would betray its only guarantee.
- **(b) Balance the target only.** Grouped splitting balances the target class distribution under the group constraint and does **not** attempt to force the full stratification signal matrix (rare categories, missingness density, extreme-value rows, bimodal clusters, …) through as well. Whole-group constraints plus dozens of simultaneous balance constraints is so over-constrained it frequently admits no valid partition; target-only balance is robust and always solvable.

This means the smart grouped splitter is `StratifiedGroupKFold` when a target is present (group-disjoint, target-balanced) and plain group-disjoint k-fold when no target is declared.

## Considered options

- **Full profile-signal matrix under the group constraint.** Rejected: over-constrained and frequently infeasible; the multilabel-stratified-group problem has no robust general solver, so it would fail or emit degenerate folds on realistic data.
- **Soft disjointness (allow rare group leakage to improve balance).** Rejected: sacrifices the one guarantee the method exists to provide.

## Consequences

- Grouped splits carry a weaker distributional guarantee than `profile_stratified_split`: only the target is balanced, and even that best-effort. This is documented, not warned at runtime, consistent with the library's no-post-hoc-imbalance-check stance.
- A user needing both whole groups and full-matrix balance has no library path — it is deliberately out of scope, not an oversight.
