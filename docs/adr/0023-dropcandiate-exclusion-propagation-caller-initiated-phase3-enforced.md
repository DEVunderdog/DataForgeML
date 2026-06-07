# ADR 0023: DropCandidate exclusion propagation is caller-initiated and Phase-3-enforced

**Status:** Accepted

## Context

When Phase 2 drops a column (>50% missing, `MissingnessFlag.DropCandidate`), the column is physically removed from the returned `dataframe` and listed in `ImputationResult.dropped_columns`. But `PipelineConfig.exclude_columns` — the source of truth for Hard Exclusions consumed by every downstream phase orchestrator via `resolve_active_columns` — is not updated. A future Phase 3 orchestrator calling `resolve_active_columns` would treat the dropped column as active.

The natural fix would be to have `ImputationOrchestrator.fit()` mutate `PipelineConfig` automatically — but this requires passing `PipelineConfig` into `fit()` or binding it in the orchestrator's constructor.

## Decision

Propagation is caller-initiated, not automatic inside `fit()`.

- `FittedImputer.apply_exclusions(config)` is the explicit call the caller makes after `fit()` and before Phase 3. It reads all `ImputationStrategy.Dropped` columns from `self.records` and calls `config.add_exclusions(dropped_cols)`, which deduplicates and extends `PipelineConfig.exclude_columns`.
- `FittedImputer` carries an internal `_exclusions_applied: bool` flag (not serialised) that `apply_exclusions` sets to `True`. `transform()` stamps this value onto `ImputationResult.exclusions_applied`.
- Enforcement lives in Phase 3's orchestrator, not Phase 2: Phase 3 raises if it receives an `ImputationResult` where `exclusions_applied` is `False`.

## Rationale

Making propagation automatic inside `fit()` would give `fit()` a side effect on `PipelineConfig`. This is problematic for two reasons: (1) every re-fit call would accumulate exclusions in the same config object, making re-fitting non-idempotent with respect to the config state; (2) it binds `ImputationOrchestrator` to `PipelineConfig`, a dependency it currently does not have, just to handle a pipeline-chaining concern.

Placing enforcement in Phase 3 rather than Phase 2 avoids penalising users who run Phase 2 standalone (no chained pipeline). Phase 2 has no visibility into whether Phase 3 exists; Phase 3 does.

## Consequences

- `PipelineConfig` gains `add_exclusions(cols: list[str])`.
- `FittedImputer` gains `apply_exclusions(config: PipelineConfig)` and an internal `_exclusions_applied: bool` flag.
- `ImputationResult` gains `exclusions_applied: bool` (default `False`).
- After `FittedImputer.from_dict()`, `_exclusions_applied` resets to `False`. Callers who chain phases must call `apply_exclusions` again after deserialising — this is documented on `apply_exclusions`, not enforced at runtime.
- Phase 3 (not yet implemented) is responsible for the enforcement check.
