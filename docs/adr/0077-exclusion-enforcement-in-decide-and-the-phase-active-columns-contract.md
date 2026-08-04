# ADR 0077: Exclusion enforcement lives in `decide()`, and every phase entry point resolves its own active columns

**Status:** Accepted.

## Context

Column exclusions were only enforced in Phase 1. A user who declared a Soft
Exclusion for the Imputation phase (`add_phase_exclusion(PipelinePhase.Imputation,
col)`) found it silently ignored — the column was routed, fitted, and imputed
anyway. A Hard Exclusion added after profiling was equally ignored: the imputation
door trusted whatever columns appeared in the Phase 1 profile and never consulted
the exclusion configuration it was handed.

This is the same removal-gap class as Scope 368. The Phase 2 orchestrator's
definition carried "owns all column routing, exclusion, and sequencing decisions";
when the Decision/Execution split deleted `ImputationOrchestrator` (ADR-0060,
ADR-0071), the exclusion-enforcement responsibility it carried was never re-homed.
The accidental protection that existed — Profiling-soft-excluded placeholder
profiles carry no semantic type and are skipped — is a coincidence of placeholder
shape, not enforcement.

## Decision

Five decisions, locked together, plus one generalized contract.

### 1. Enforcement point: `decide()`

`decide()` resolves the active column set for the Imputation phase via
`PipelineConfig.resolve_active_columns(PipelinePhase.Imputation, profile.columns)`,
filtered over the profiled column set. The entire plan — targets, predictors,
blocks, shape, hyperparameter signals — is computed over active columns only.
Purity is preserved: the config was already an input to `decide()`; it simply
reads more of it. `fit_unit` / `fit_many` remain trusting executors and gain no
exclusion logic.

*Gain:* one enforcement point at the single decide-time entry, so an exclusion
declaration is honored everywhere the plan reaches; execution primitives stay
plan-trusting, so informed-consent plan edits remain possible. *Give up:* an
advanced user can no longer get a decision for an excluded column without first
removing the exclusion — no plan edit can resurrect a column the plan omitted.
(This originally named `with_strategy` as the path that could not do it;
ADR-0082 deleted that method, which strengthens the point rather than changing
it — re-routing is now a fresh `decide`, where the exclusion is enforced.)

### 2. Plan representation: Passthrough-with-signal / omission

Imputation-soft-excluded numeric columns receive a `ColumnImputationDecision`
with strategy Passthrough plus the signal string `"soft-excluded for Imputation
phase"` (a shared constant, `_EXCLUSION_SIGNAL`). Hard-excluded columns are
omitted from the plan entirely; the plan's `config_snapshot` retains the
exclusion lists for recoverability.

The asymmetry mirrors the two exclusion kinds' semantics: a soft-excluded column
stays in the dataset, so the plan describes it (and the fitted manifest keeps a
record for it, preserving the "no record = never seen during fit" invariant); a
hard-excluded column will not exist downstream, so the plan describes only the
columns downstream phases will ever see.

One transform-side consequence: `FittedImputer.transform`'s Passthrough-violation
check (`UnfittedColumnError` on Passthrough columns carrying nulls) exempts
records whose signals carry `_EXCLUSION_SIGNAL`. A column that is Passthrough by
declaration — not because fit saw no missingness — must ride its missing values
through untouched rather than raising.

*Gain:* a plan inspector can distinguish "excluded by my declaration" from
"passed through by type"; hard exclusion leaves no per-column ghost. *Give up:*
recovering *which* columns were hard-excluded requires reading the config
snapshot, not the per-column map; and the exclusion signal string becomes a
load-bearing constant shared between the assembler and transform.

### 3. Full predictor ban

Excluded columns are removed from the numeric block wholesale, not just as
targets. Feature counts, MICE/KNN block membership, multi-MAR candidate
counting, regression feature-column lists, correlated-feature lists for
Cluster-Conditional, and all block-level hyperparameter signals (missingness
fractions, pairwise correlations, nearest-feature counts) are computed over
active columns only. Mechanically, the profile's pairwise `feature_correlation`
view is filtered once to the active numeric block, so no routing branch or
hyperparameter signal can count an excluded column as a feature. The plan's
`decided_for_shape` reflects the active set.

*Gain:* no estimator trains on a column carrying raw NaNs that imputation will
never fill, and the plan's recorded shape honestly reflects what execution
faces. *Give up:* an excluded column with real predictive signal is unavailable
to sibling imputations — the user's exclusion is taken at full strength, and
routing outcomes can change (a column may lose enough correlated features to
fall from a model-based branch to a scalar fill).

### 4. Plan-time contradiction raise

`decide()` raises `ValueError` when a hard- or Imputation-soft-excluded column
appears in `mnar_columns`, `per_column_strategy`, `per_column_constant_fill`, or
the indicator-column declarations — every offending column reported in one
raise, before any routing.

This is a deliberate asymmetry with Phase 1, which keeps its silent-ignore of
overrides referencing excluded columns. The asymmetry is documented, not
unified: changing Phase 1's behavior would churn settled semantics for no user
request, while Phase 2's overrides are precise per-column declarations whose
silent loss is exactly the "one side silently wins" failure this scope exists
to remove.

*Gain:* contradictory config the user typed surfaces before any fitting.
*Give up:* cross-phase consistency — the two phases now respond differently to
the same shape of contradiction, and the asymmetry is a fact a user must learn.

### 5. Transform contract: user-owned dropping, raise as backstop

Hard Exclusion never physically drops a column from any DataFrame, at fit or
transform time. The user drops hard-excluded columns from their frames; the
existing strict unknown-column raise at `FittedImputer.transform`
(`UnseenColumnError` — a hard-excluded column has no record in the manifest,
because it was omitted from the plan) is the enforcement backstop.
`FittedImputer` remains config-free.

*Gain:* the library never mutates the user's data shape behind their back, and
`FittedImputer` keeps its serializable, config-free contract. *Give up:* the
failure is late (transform, not fit) and arrives as a generic unseen-column
error rather than a purpose-built "you forgot to drop an excluded column"
message; the user carries the dropping responsibility.

### The generalized phase contract

**Every phase entry point — orchestrator or door — must resolve its own active
columns for its own phase, via `PipelineConfig.resolve_active_columns`; no phase
may rely on an upstream phase having filtered for it.** This binds the future
entry points of Phases 3–6, whatever shape they take. The contract exists so
that deleting or restructuring a phase's entry point can never orphan exclusion
enforcement again: the responsibility attaches to *being an entry point*, not to
a particular class that might be removed.

*Gain:* exclusion enforcement survives any future re-architecture of a phase's
entry seam. *Give up:* redundant resolution work when phases are chained — each
entry point re-resolves rather than trusting its predecessor — accepted because
the resolution is a set operation over column names, trivial next to any
phase's real work.

## Consequences

- Imputation-phase Soft Exclusions and post-profiling Hard Exclusions are now
  honored; previously both were dead config for Phase 2.
- A config with no exclusions produces a plan identical to before — the change
  is invisible unless exclusions are declared (guarded by a regression test).
- `_EXCLUSION_SIGNAL` lives in `imputation/_config.py` and is read by both the
  decision assembler (writer) and `FittedImputer.transform` (exemption check).
- Profiling's silent-ignore of overrides for excluded columns is unchanged; the
  Phase 1/Phase 2 asymmetry is recorded here rather than unified.
- The Profiling-phase soft-exclusion placeholder behavior (placeholder profiles
  carry no semantic type and are skipped by `decide()`) keeps working unchanged;
  Profiling-phase and Imputation-phase exclusion semantics stay independent.
- Phases 3–6 inherit the phase contract; their specs must show where
  `resolve_active_columns` is called before their entry points are accepted.
- CONTEXT.md's Hard Exclusion, Soft Exclusion, and Phase Orchestrator entries
  are updated for the new enforcement point (same change as this ADR).
