# ADR 0006: fit() returns a stateless FittedImputer rather than mutating the orchestrator

**Status:** Accepted

## Context

Transforming phases must support fit/transform discipline: learn parameters from training data, apply them to any split. The question was whether the `ImputationOrchestrator` should be stateful (sklearn-style: `fit()` mutates internal state, `transform()` reads that state), or whether `fit()` should return a separate `FittedImputer` object.

## Decision

`ImputationOrchestrator.fit()` returns a `FittedImputer` dataclass. The orchestrator itself remains stateless. `FittedImputer.transform()` applies learned parameters. `FittedImputer` supports `to_dict()` / `from_dict()` for serialization.

## Rationale

- **Serialization is clean**: `FittedImputer` contains only learned parameters (scalar fill values, serialized model bytes). Serializing a stateful orchestrator would bundle config, routing logic, and learned state together — harder to version and inspect.
- **Production use case**: a pipeline trained on Monday's data must be applicable to Tuesday's new rows without re-fitting. A serializable `FittedImputer` is the natural unit for this.
- **Separation of concerns**: the orchestrator owns strategy selection logic; `FittedImputer` owns learned state. These are different responsibilities and should not live in the same object.
- The sklearn stateful convention is familiar but couples config and learned state, which creates awkwardness when saving partial pipeline state.

## Consequences

- `ImputationOrchestrator` is always safe to reuse across multiple `fit()` calls — no shared mutable state.
- Callers must hold a reference to the returned `FittedImputer` to call `transform()`. There is no implicit state on the orchestrator to fall back on.
- All transforming phases added in future (Phases 3–6) should follow the same pattern: stateless orchestrator, serializable fitted object returned from `fit()`.
