# ADR 0025: Indicator column records are written into `self.records` at fit time

**Status:** Accepted

## Context

`FittedImputer.transform()` appends `{col}_missing` columns for every column where `indicator_added=True`. These produced columns appeared in the output DataFrame with no representation in `self.records`. Downstream phases and callers had no way to know what indicator columns `transform()` would produce before calling it. Indicator columns passed silently into downstream phases (Normalization, Encoding, Scaling) with no instruction to skip them.

The question was when to write indicator column records: at fit time (pre-populating records for not-yet-existing columns) or at transform time (mutating records when the columns are produced).

## Decision

`fit()` pre-populates a `ColumnImputationRecord` with `strategy=ImputationStrategy.Indicator` and `semantic_type=SemanticType.Boolean` for every `{col}_missing` column that will be produced during `transform()`. These records are written into `self.records` before `fit()` returns. `transform()` does not mutate `self.records`.

## Rationale

- **Immutability is load-bearing**: `FittedImputer` is designed to be stateless and serializable (ADR 0006). If `transform()` mutated `self.records`, the serialized form of `FittedImputer` would depend on whether `transform()` had been called. `to_dict()` / `from_dict()` round-trips would be inconsistent. Calling `transform()` twice would produce different records state.
- **Output schema inspectability**: callers can read `self.records` after `fit()` to understand the complete output schema — including indicator columns — without calling `transform()` first.
- **Indicator columns are fully determined at fit time**: `indicator_added=True` is set during `fit()`. The indicator column names are known before any transform occurs, making pre-population unambiguous.
- The slight unintuitive aspect — records referencing columns that don't exist in any DataFrame yet — is a worthwhile tradeoff for the immutability guarantee.

## Consequences

- `self.records` contains entries for `{col}_missing` columns that do not exist in any DataFrame at fit time.
- `ImputationStrategy.Indicator` is a new enum value. It is not a fill strategy — it marks a produced column in the audit log.
- The absent-column error check in `transform()` must skip `ImputationStrategy.Indicator` entries: they are produced by `transform()`, not consumed as inputs.
- `apply_exclusions(config)` is extended to register all `ImputationStrategy.Indicator` columns as Soft Exclusions for Phases 3–6, so downstream phases skip them automatically.
- `SemanticType.Boolean` is assigned to indicator records because `{col}_missing` is always a binary Int8 `{0, 1}` column by construction.
