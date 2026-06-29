# PipelineConfig lives at the package root, not inside the profiling module

`PipelineConfig` and `PipelinePhase` are cross-phase concerns — they know nothing about profiling specifically — but they were originally placed in `profiling/config.py` because profiling was the first phase implemented. This creates a layering violation: external callers importing `PipelineConfig` were reaching into a phase module to get a pipeline-level object, and the types were missing from the top-level `__init__.py` entirely.

We moved `PipelineConfig`, `PipelinePhase`, and the shared enums `SemanticType` and `Modality` to a top-level `dataforge_ml/config.py`. Phase-specific types (`ProfileConfig`, result dataclasses, `TypeFlag`, `NumericKind`) remain in `profiling/config.py`.

A `pipeline/` sub-package was considered but rejected — we have no orchestrator or runner code yet, and the extra namespace layer buys nothing today. This can be revisited when pipeline orchestration is implemented.
