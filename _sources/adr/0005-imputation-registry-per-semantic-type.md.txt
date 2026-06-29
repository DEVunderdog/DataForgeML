# ADR 0005: Imputation sub-processors registered per SemanticType

**Status:** Accepted

## Context

Phase 2 (Imputation) must handle missing values across all SemanticTypes: Numeric, Categorical, Boolean, and Datetime. Each type requires a different strategy family — Numeric uses Mean/Median/KNN/MICE etc., Categorical uses Mode/constant, Boolean uses Mode, Datetime requires temporal-aware filling. The question was whether to write one unified imputer with branching logic or to have separate sub-processors per type.

## Decision

Phase 2 uses a `_IMPUTATION_REGISTRY: dict[SemanticType, ImputationSubProcessor]` pattern, identical to Phase 1's `_COLUMN_PROFILER_REGISTRY`. The `ImputationOrchestrator` routes columns to the registered sub-processor for their SemanticType. `SemanticType.Text` and `SemanticType.Identifier` have no registered imputer and are skipped silently.

## Rationale

- Mirrors the established Phase 1 pattern — no new architectural concept introduced.
- Each sub-processor is independently testable and holds no cross-type logic.
- New SemanticType imputers (e.g. `DatetimeImputer`) are added by registering a new entry — no changes to the orchestrator.
- A single unified imputer with branching would couple all type-specific logic into one class and make it increasingly hard to extend as more types are added.

## Consequences

- `NumericImputer` is the first sub-processor, implemented in Phase 2 step 1.
- `CategoricalImputer`, `BooleanImputer`, and `DatetimeImputer` are added in subsequent steps.
- The `ImputationOrchestrator` reads `SemanticType` from `StructuralProfileResult.columns[col].semantic_type` — Phase 1 must have run before Phase 2.
- `SemanticType.Text` columns with missing values are left untouched and pass through to downstream phases as-is.
