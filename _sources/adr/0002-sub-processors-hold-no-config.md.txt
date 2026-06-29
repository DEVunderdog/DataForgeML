# ADR 0002: Sub-processors hold no configuration — Phase Orchestrators own all routing and exclusion decisions

**Status:** Accepted

## Context

The original implementation gave every sub-profiler (e.g. `NumericProfiler`, `CategoricalProfiler`, `MissingnessProfiler`, `TabularProfiler`) a reference to `ProfileConfig` via the `Profiling` base class. Each type profiler had its own `_eligible` method that read `config.column_overrides` to decide whether to process a column. `MissingnessProfiler` additionally read `column_overrides` to suppress sentinel-string detection for columns overridden to Numeric, Datetime, Boolean, or Identifier types. `TabularProfiler` filtered `config.exclude_columns` independently to build its own column scope.

This fragmented exclusion and routing logic across multiple levels of the call stack. `StructuralProfiler` (the Phase Orchestrator) already resolved the active column set via `resolve_active_columns` and routed columns by `SemanticType` in step 6 of its pipeline — meaning the `_eligible` checks in sub-profilers were redundant re-implementations of decisions the orchestrator had already made. The `MissingnessProfiler` sentinel suppression logic was found to be incorrect: the comment in the code explicitly stated that treating `"NA"` as a sentinel is correct for a string column overridden to Numeric, yet the code suppressed it. `TabularProfiler`'s independent filter caused dataset-level stats (column count, sparsity) to diverge from the raw input depending on pipeline configuration, making it unsuitable as a general-purpose data catalog tool.

## Decision

Sub-processors hold no configuration reference. Specifically:

- The `Profiling` base class `__init__` takes no config argument.
- All `_eligible` checks are removed from type profilers (`NumericProfiler`, `CategoricalProfiler`, `DatetimeProfiler`, `BooleanProfiler`, `TextProfiler`). They receive `(DataFrame, list[str])` and profile exactly those columns.
- `MissingnessProfiler` sentinel detection is purely dtype-driven. The `column_overrides` lookup and `_SENTINEL_SUPPRESSING_SEMANTICS` suppression are removed. `_sentinel_eligible` takes only `dtype`.
- `TabularProfiler` receives the full raw DataFrame and applies no exclusion filter. Its stats describe the data as it arrived, independent of pipeline configuration.
- Null-detection primitives (`_sentinel_eligible`, `_inf_eligible`, `_SENTINEL_STRINGS`) are extracted to a shared `_null_detection.py` module to avoid cross-module private imports.
- If a sub-processor needs a computation parameter in the future, it declares it explicitly in its own constructor — not through an inherited config object.

## Rationale

- The `_eligible` checks were dead logic: `StructuralProfiler` already guaranteed that `NumericProfiler` only received columns resolved to `SemanticType.Numeric` before delegation. Duplicating that check downstream invited the two to drift.
- Centralising all exclusion decisions at the Phase Orchestrator layer makes the column lifecycle easy to trace: one call to `resolve_active_columns`, one routing step, then dumb batch processors.
- The sentinel suppression via `column_overrides` conflated two unrelated concerns — type routing and null-detection strategy. Effective null detection is a property of the raw data's dtype, not of the user's intended semantic type.
- `TabularProfiler` as a pipeline-agnostic data catalog is more reusable and predictable. Its stats should not change based on what columns the user chose to exclude from ML processing.

## Consequences

- All column routing and exclusion decisions for Phase 1 live exclusively in `StructuralProfiler`.
- Sub-processors are simpler and fully testable in isolation — no config fixture needed, just a DataFrame and a column list.
- Future Phase Orchestrators (Phases 2–6) have a clear contract: they call `resolve_active_columns`, decide what goes where, and pass pre-decided column lists to their sub-processors. Sub-processors are never consulted on routing.
- Effective null counting is now consistent between `MissingnessProfiler` and `StructuralProfiler._compute_row_distribution` — both use the same dtype-driven logic from `_null_detection.py`.
