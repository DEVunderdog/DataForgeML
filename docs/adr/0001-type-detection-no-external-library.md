# ADR 0001: Type detection without an external type-inference library

**Status:** Accepted

## Context

The type detector (`_type_detector.py`) was misclassifying high-cardinality string columns like Titanic's `Name` as `Categorical` instead of `Text`. This caused downstream failures: Cramér's V division-by-zero, ANOVA on single-sample groups, and MI on 0-row arrays — all because `Name` was being passed into the correlation pipeline as a categorical with ~891 unique values.

The fix required improving the free-text detection heuristics. The question was whether to delegate type inference to an established library (`visions`, used by ydata-profiling) rather than maintaining in-house heuristics.

## Decision

Fix the detection logic in-house. No external type-inference library is introduced.

## Rationale

- `visions` classifies string columns as `URL`, `EmailAddress`, `IPAddress`, `UUID`, `Path`, etc. It has no concept of **Text vs Categorical** — the distinction an ML pipeline needs. We would still need our own heuristics on top of it.
- The actual bug was two threshold constants being too conservative and one detection path being missing — a 20-line fix, not a structural gap.
- Adding `visions` would introduce a pandas-centric dependency into a Polars-native codebase and a type vocabulary that requires translation at every call site.

## Consequences

- The free-text detection thresholds (`_FREE_TEXT_MEDIAN_CHARS`, `_FREE_TEXT_AVG_WORDS`, `_FREE_TEXT_P90_CHARS`) and the new `_FREE_TEXT_HIGH_UNIQUE_WITH_SPACES` constant are the authoritative knobs for Text/Categorical boundary tuning.
- If detection accuracy becomes a recurring maintenance burden, revisit `visions` or a dedicated text-detection approach at that point — not preemptively.
