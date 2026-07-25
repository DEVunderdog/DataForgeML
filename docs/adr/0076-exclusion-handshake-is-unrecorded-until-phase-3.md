# ADR 0076: The exclusion handshake is unrecorded until Phase 3 exists

**Status:** Accepted. Amends ADR-0023.

## Context

ADR-0023 gave `FittedImputer` an internal `_exclusions_applied: bool` flag, set by
`apply_exclusions()` and stamped onto `ImputationResult.exclusions_applied` by
`transform()`. Its stated consumer was Phase 3's orchestrator, which would raise on
receiving a result where the flag was `False` — the caller having forgotten to
propagate dropped columns into `PipelineConfig`.

Phase 3 does not exist and is not scheduled. `PipelinePhase.OutlierDetection` is an
enum member used as a soft-exclusion key; there is no orchestrator behind it. The flag
has therefore been write-only for its entire life: nothing in `src/` ever read
`ImputationResult.exclusions_applied`, and the only assertions on it were tests
pinning the flag's own mechanics.

Two costs followed. The state read as a bug to anyone tracing it — a mutation with no
guard anywhere — and cost a re-derivation each time. And the mechanism could not do
the job it was designed for: because the flag records only *that*
`apply_exclusions` was called and not *which config* received the exclusions,
`apply_exclusions(config_a)` followed by handing Phase 3 `config_b` sets the flag
`True` and passes the check, producing exactly the mismatch the check exists to catch.

## Decision

Remove `_exclusions_applied` and `ImputationResult.exclusions_applied`. Keep
`apply_exclusions()` — it performs the real config mutation and has real callers.

The invariant Phase 3 must enforce is recorded here rather than in code: **before a
downstream phase consumes a `PipelineConfig` alongside a `FittedImputer`, every column
recorded with `ImputationStrategy.Dropped` must be present in
`config.exclude_columns`.** Phase 3 should verify this by direct comparison against
the imputer's records, not by trusting a boolean.

## Rationale

The flag would be deleted when Phase 3 arrived regardless, since the direct comparison
is the check that actually holds. Keeping it until then buys nothing and charges the
comprehension cost on every read. This also matches the direction of the surrounding
work, which has removed `ImputationOrchestrator`, `_executor.py`, `_store.py`,
`_identity.py`, degradation reasons, and provenance stamps — machinery built ahead of
a consumer and held in place by tests rather than by use.

Keeping enforcement out of Phase 2 remains correct for ADR-0023's original reason:
running imputation standalone without ever calling `apply_exclusions` is legitimate,
and Phase 2 cannot see whether a Phase 3 is coming.

## Consequences

- `FittedImputer.__post_init__` is gone; it existed only to initialise the flag.
- `ImputationResult` loses a public field. Nothing in `src/` read it; callers who read
  it for their own bookkeeping must track the call themselves.
- The invariant now lives only in this ADR and in the `apply_exclusions` docstring.
  Phase 3's author must read one of the two — a real risk, accepted because the flag
  never enforced anything, so what is lost is a hint rather than a guarantee.
- Phase 3 will need the `FittedImputer` (or its dropped-column list) to run the direct
  comparison, not just the `ImputationResult`. That is a heavier coupling between
  phases than passing a result object forward, and is the price of a check that
  actually detects the wrong-config case.
