# ADR 0026: `UnseenColumnError` fires on any column absent from `self.records`, regardless of missingness

**Status:** Accepted

## Context

After ADR 0024, `self.records` is a complete train schema manifest. A column present in `transform()` input but absent from `self.records` is unambiguously a new column — never seen during fit. The question was whether to raise immediately (any unknown column) or lazily (only when the unknown column has missing values, matching the existing `UnfittedColumnError` pattern).

## Decision

`UnseenColumnError` fires immediately at the start of `transform()` for any column in the input that has no entry in `self.records`, regardless of whether that column has missing values.

## Rationale

- **Unknown columns have no pipeline contract at all**: no profile, no imputation parameters, no normalization parameters, nothing downstream knows about them. The absence of missing values does not make passthrough safe — the same column may have missing values in a later batch, producing inconsistent errors at inconsistent times.
- **Eager failure is cheaper**: detecting schema drift before any processing begins produces a clearer error and avoids partial DataFrame mutations.
- **Consistency with schema enforcement intent**: `UnfittedColumnError` (Passthrough + missing values) and `FittedColumnAbsentError` (active strategy column absent) both enforce schema correctness. Lazy checking for unknown columns would create an inconsistent gap where schema violations are sometimes silent and sometimes loud depending on missingness.
- **The tradeoff is acceptable**: the only callers who break are those adding derived features between fit and transform — a workflow that is structurally wrong. Derived features must be consistent across splits. Adding columns after fit bypasses the entire feature contract.

## Consequences

- `UnseenColumnError` fires before `_resolve_effective_nulls` and before any fill logic — at the earliest possible point in `transform()`.
- All unknown columns are collected and named in a single raise, not one per column.
- `UnseenColumnError` is distinct from `UnfittedColumnError`: the former means "never seen during fit"; the latter means "seen during fit with no missingness, now has missing values on test."
