# The Public API is defined by user-supply, not by reachability or type kind

The package root (`dataforge_ml.__init__`) had grown ad-hoc: `StructuralProfiler` was exported but its sibling phase orchestrator `ImputationOrchestrator` was not; `DataSplitter` was exported but `SplitConfig` was not; the seven Profile Sub-Configs and both Imputation configs were absent; and the only imputation symbol at root was `ImputationFitDiagnostic` — a deeply nested leaf. There was never a rule for what to export, so the surface drifted.

We adopt a single governing rule: **a symbol is Public API if and only if the user must reference it to configure, drive, or handle the pipeline.** That promotes every config object and Phase Sub-Config, every input enum, every entry point, the result/nested-record types users hold, user-facing fit-quality diagnostics, and every exception the user catches — bringing the root to ~31 names. The full inventory lives in the CONTEXT.md **Public API** glossary entry.

## Consequences

- **The test is user-supply, not reachability.** `NumericKind` becomes Public because the user passes it to `set_numeric_kind`, even though it is also reachable as computed output on `ColumnProfile`. `TypeFlag` stays internal because it only ever appears as output and is never user-supplied. This corrects an earlier CONTEXT.md statement that listed `NumericKind` as an internal, non-public type.
- **"Diagnostic" was an overloaded word.** Output-only internal refinements (`TypeFlag`, per-modality stats) are excluded, but user-facing fit-quality diagnostics (`ImputationFitDiagnostic`) are Public — the pipeline computes them expressly for the user to inspect imputation quality.
- **The root namespace is flat and wide (~31 names).** A tiered API (top-level configs at root, sub-configs reachable only via `dataforge_ml.profiling`) was rejected: it reintroduces the "which submodule is this in?" hunt that motivated the change and cuts against the one-stop-solution intent.
- **Every promoted name is now a frozen compatibility surface.** Renaming a config, an enum member, or an exception class is henceforth a breaking change. For exceptions this is the point — a name you cannot rely on is not catchable in practice.
