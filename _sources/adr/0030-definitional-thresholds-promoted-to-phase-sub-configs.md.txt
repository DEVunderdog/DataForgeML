# ADR-0030: Definitional thresholds promoted to user-configurable Phase Sub-Configs

## Status
Accepted — Scope 15

## Context

Approximately 40 threshold constants were hard-coded as private module-level constants
(e.g. `_SEVERITY_MINOR = 0.01`, `_MAR_CORRELATION_THRESHOLD = 0.60`, `_SKEW_NORMAL = 0.5`)
scattered across eight Phase 1 profiling modules and the splitting module. Users could not
override any of them without patching source code. For datasets whose domain norms differ
from the defaults — medical data where 20% missingness is routine, financial data with
extreme skew bands — the fixed values produced wrong labels and therefore wrong downstream
routing decisions.

Three alternatives were considered:

1. **Keep as module-level constants** — rejected: no user override path.
2. **Move all ~40 fields flat onto `ProfileConfig`** — rejected: a 40-field flat dataclass
   with no internal grouping is unnavigable. Breaks the precedent set by
   `ImputationConfig.numeric`.
3. **Dataset-adaptive auto-computation** — deferred to a future scope. Computing thresholds
   from the data itself requires threshold estimation to be a fit-time operation (train-only),
   which breaks the current model where Phase 1 is non-transforming and may run on the full
   dataset without leakage risk.

## Decision

Promote all threshold constants to user-configurable fields in purpose-built **Phase
Sub-Config** dataclasses, one per sub-processor, nested as named fields on `ProfileConfig`.
Sub-processors receive their sub-config in their constructor. All module-level threshold
constants are deleted; the sub-config default value is the single source of truth.

### Phase Sub-Configs added to `ProfileConfig`

| Field | Class | File |
|---|---|---|
| `ProfileConfig.missingness` | `MissingnessProfileConfig` | `_missingness_config.py` |
| `ProfileConfig.numeric` | `NumericProfileConfig` | `_numeric_config.py` |
| `ProfileConfig.type_detection` | `TypeDetectionConfig` | `_type_detection_config.py` |
| `ProfileConfig.categorical` | `CategoricalProfileConfig` | `_categorical_config.py` |
| `ProfileConfig.correlation` | `CorrelationProfileConfig` | `_correlation_config.py` |
| `ProfileConfig.datetime_` | `DatetimeProfileConfig` | `_datetime_config.py` |

### Top-level orchestrator field added to `ProfileConfig`

`row_drop_threshold: float = 0.50` — used by `StructuralProfiler` (the orchestrator
directly, not any sub-processor). Placed alongside the existing orchestrator-level fields
`memory_threshold_mb` and `chunk_size`, not inside any sub-config.

### SplitConfig added for the splitting module

`SplitConfig` is a new dataclass in `src/dataforge_ml/splitting/_config.py` with two fields:
`max_stratification_signals: int = 50` and `boolean_minority_threshold: float = 0.05`.
Added to `PipelineConfig` as `split: SplitConfig`. `DataSplitter.__init__` accepts an
optional `SplitConfig` (defaulting to `SplitConfig()`), consistent with how other phase
components receive their config at construction time.

### ADR-0002 clarification

ADR-0002 prohibits sub-processors from accessing a "shared config object." That prohibition
targets `ProfileConfig` and `PipelineConfig` — objects that carry routing state, exclusions,
and cross-phase signals. A Phase Sub-Config carrying only computation parameters (threshold
constants) for a single sub-processor is not a shared config object. Passing it to the
sub-processor's constructor does not give the sub-processor access to routing authority.
ADR-0002's intent is preserved; its language is clarified.

### Threshold ownership rules

- **Definitional thresholds** — what a label means (e.g. what fraction of missingness
  qualifies as `MissingSeverity.High`) — live in the Phase Sub-Config of the sub-processor
  that *produces* the label.
- **Operational thresholds** — what action to take given a label (e.g. `knn_max_rows`) —
  live in the Phase Config of the phase that *consumes* the label. Already true for
  `NumericImputationConfig`; unchanged by this ADR.
- **Orchestrator thresholds** — thresholds used by the Phase Orchestrator itself, not any
  sub-processor — live as top-level fields on the Phase Config directly.
- **Duplicate threshold names** (e.g. `near_constant_threshold` appears in both
  `NumericProfileConfig` and `CategoricalProfileConfig`) — kept as separate independent
  fields with the same default. Users may tune them independently; the library does not
  enforce consistency between them.

## Consequences

**Gains:**
- Every threshold is user-configurable; domain experts can tune without patching source code.
- Consistent with the existing `ImputationConfig.numeric` nested sub-config pattern.
- Sub-configs are independently instantiable and independently testable.
- Each new threshold added in the future requires a one-field change to the sub-config only,
  with no constructor explosion at the sub-processor or orchestrator callsite.

**Costs:**
- Six new config classes must be maintained, serialised (`to_dict` / `from_dict`), and tested.
- `ProfileConfig` construction is more verbose when overriding nested fields
  (`config.missingness.severity_high = 0.30` vs a flat field).
- CONTEXT.md reference `ProfileConfig.mar_correlation_threshold` updated to
  `ProfileConfig.missingness.mar_correlation_threshold`.
