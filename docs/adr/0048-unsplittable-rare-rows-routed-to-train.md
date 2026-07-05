# ADR 0048: Rows of rare labels too sparse to stratify are routed into the training split

A target class or rare categorical value with fewer rows than the viability floor (ADR-0047) cannot appear in both partitions — a single-row label physically cannot be split. Rather than let `iterstrat` place such a row arbitrarily, its rows are deterministically routed into the **training** partition.

## Context

The rare categorical signal's purpose is per-label presence in both splits so Phase 5 encoding never meets an unseen category at transform time. For a label with enough rows this is achieved by stratification. For a label too sparse to stratify, no algorithm can put it on both sides, so the only remaining decision is *which* side, and whether to surface it.

## Decision

Route the sparse label's rows into **train**. Then the label is guaranteed learned at fit time, and the worst case — a category present in test but never seen in training, which crashes or degrades Phase 5 `transform()` — becomes impossible. This applies uniformly to unsplittable **target classes** and **rare categorical values** (an earlier draft handled the target with a keep-and-warn rule; the routing rule is strictly better and is used for both).

The routing is a split-side action applied after `iterstrat` decides the rest of the partition, and it is performed **silently** — consistent with the library's stance of doing no post-hoc split-imbalance checking (the two former imbalance warnings were removed as noise). The behaviour is documented so users are aware, matching the pattern already chosen for imbalance handling: the library handles it, the docs explain it, no runtime warning fires.

## Consequences

- The cost is evaluation-side only: an ultra-rare label will not appear in the test set, so the split does not *evaluate* the model's handling of it. This is unavoidable — a label with one total row cannot be held out regardless of method.
- There is no model-side or overfitting cost. The forced rows number a handful at most (only sub-floor labels qualify); a few extra training rows of a genuinely-occurring class is coverage, not memorised noise.
- The split is no longer produced purely by `iterstrat`; a deterministic post-step reassigns a small, well-defined set of rows. This is the deliberate deviation a future reader would otherwise question.
