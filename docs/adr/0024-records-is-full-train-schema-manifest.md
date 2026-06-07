# ADR 0024: `self.records` is a complete manifest of the train schema

**Status:** Accepted

## Context

Before this decision, `FittedImputer.records` only contained columns that had a registered imputer — in practice, only `SemanticType.Numeric` columns. Text, Identifier, Categorical, Boolean, and Datetime columns produced no records because their semantic types have no entry in `_IMPUTATION_REGISTRY`.

This made `records` structurally ambiguous as a schema manifest: a column absent from `records` could mean either "intentionally skipped (Text/Identifier)" or "never seen during fit (new column introduced after the split)." `transform()` had no way to distinguish the two cases, so it treated both identically — silent passthrough — regardless of whether the column had missing values.

## Decision

`fit()` writes a `ColumnImputationRecord` with `strategy=Passthrough` for every column present in `train_df`, regardless of semantic type. After `fit()` returns, `records` is a complete manifest of the full train schema. Any column present in a `transform()` input but absent from `records` is unambiguously new — never seen during fit — and raises `UnseenColumnError`.

## Rationale

- **Single source of truth**: one data structure answers "was this column seen during fit?" — no parallel `seen_columns` set required.
- **Unambiguous invariant**: "no record = never seen during fit" is now always true. `transform()` can enforce schema correctness without special-casing semantic types.
- **Audit completeness**: callers inspecting `records` after fit see the full output schema — including columns the library intentionally skips — without having to know which types are registered.
- The alternative (a separate `seen_columns: set[str]`) introduces two data structures that must both be kept in sync and both serialised, for no semantic gain.

## Consequences

- `records` grows to include Text, Identifier, Categorical, Boolean, and Datetime columns as Passthrough entries.
- Callers that consume `records` downstream (e.g. `suggest_refit_config`) must guard against non-Numeric Passthrough records — these carry no diagnostic and no fill value.
- The invariant "no record = never seen" is enforceable in `transform()` and is the foundation for `UnseenColumnError` (see ADR 0026).
