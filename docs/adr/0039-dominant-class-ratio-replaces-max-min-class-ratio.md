# ADR 0039: Dominant-class ratio (top-2) replaces max/min class ratio in ImbalanceMetrics

The `class_ratio` field on `ImbalanceMetrics` is redefined from `max_freq / min_freq` to `max_freq / second_max_freq`. The industry-standard max/min formulation is contaminated by rare tail entries and is made redundant by the normalized Gini and normalized Shannon entropy fields added in the same scope.

## Context

`ImbalanceMetrics` carries three imbalance signals computed by `CategoricalProfiler`: `class_ratio`, `shannon_entropy`, and `gini_impurity`. The original `class_ratio` is `max_freq / min_freq` — the most frequent category's probability divided by the least frequent category's probability.

This formulation has a structural defect: `min_freq` is dominated by whatever rare category happens to appear least often. A single category with two occurrences in a 10,000-row dataset produces a `class_ratio` of 5,000 even when the remaining distribution is balanced. `RareCategoryStats` already tracks rare categories explicitly; having `class_ratio` silently amplify the same signal produces noise, not independent information.

The same scope also adds `normalized_shannon_entropy` and `normalized_gini` — both of which capture full-distribution uniformity on a `[0, 1]` scale. Once those fields exist, `max_freq / min_freq` provides no independent signal: it duplicates the tail-concentration information already in `RareCategoryStats` and the distribution uniformity already in the normalized metrics.

## Decision

Replace `class_ratio = max_freq / min_freq` with `dominant_class_ratio = max_freq / second_max_freq`.

`dominant_class_ratio` answers a distinct, stable question: does the most common label overwhelm even the runner-up? A value of 10 means the dominant category appears 10× more than the second-most-common — a direct signal for encoding strategy and downstream model decisions. It is unaffected by rare tail entries because it never references `min_freq`.

`dominant_class_ratio` is `null` when `cardinality < 2` (no second category exists to compare against).

## Considered options

**Keep max/min, add normalized metrics alongside it.** Rejected — the three resulting metrics are not independent. `max_freq / min_freq` and `normalized_gini` both measure distribution imbalance; their correlation makes the third field uninformative noise rather than signal.

**Drop class_ratio entirely.** Considered, but `dominant_class_ratio` answers a question neither normalized metric surfaces directly: whether a single category dominates even the second-most-common. That specific signal is load-bearing for encoding and imputation routing.

## Consequences

- `ImbalanceMetrics.class_ratio` is renamed to `ImbalanceMetrics.dominant_class_ratio`. Existing consumers of the JSON output that key on `"class_ratio"` must update their field reference.
- The field is `Optional[float]` (`null` when `cardinality < 2`), whereas the previous field defaulted to `0.0`. Consumers must handle `null`.
- `CategoricalProfiler._compute_value_distribution` reads `second_max_freq` from the sorted value-count frame (already computed; `vc["count"][1]` after descending sort).
