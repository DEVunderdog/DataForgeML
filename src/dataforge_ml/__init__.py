"""DataForgeML public API.

The names re-exported here constitute the Public API. A symbol is exported at
the package root iff the user must reference it to configure, drive, or handle
the pipeline (ADR-0050). Output-only internal types (e.g. ``TypeFlag``,
per-modality stats dataclasses) remain accessible only via submodule imports.

Every name is imported on first use, not when the package is imported. The
``dataforge`` CLI imports this package, and ``dataforge --help`` and
``dataforge schema`` must answer without importing pandas, polars or
scikit-learn (ADR-0112).
"""

from __future__ import annotations

import importlib
from typing import Any

#: Each exported name, grouped by the module that defines it.
_EXPORTS: dict[str, tuple[str, ...]] = {
    # --- Bare-bytes persistence: serialize / deserialize / inspect (ADR-0072)
    "._serialization": ("ArtifactPythonVersionWarning", "deserialize", "inspect", "serialize"),
    # --- Pipeline-level config and shared enums
    ".config": ("Modality", "PipelineConfig", "PipelinePhase", "SemanticType"),
    # --- Evaluation: c2st is public; the umbrella is gone (ADR-0087 amended).
    # ``C2STOutcome`` / ``C2STAnnotation`` / ``C2STVerdict`` classify output and
    # are never user-supplied, a stated exception to ADR-0050 (ADR-0087):
    # reading the report *is* branching on them, so the handling half of the
    # export rule reaches them.
    ".evaluation": (
        "C2STAnnotation",
        "C2STConfig",
        "C2STOutcome",
        "C2STProvenance",
        "C2STReport",
        "C2STResult",
        "C2STScheme",
        "C2STScore",
        "C2STVerdict",
        "ColumnC2STResult",
        "c2st",
        "imputation_score_c2st",
    ),
    # --- The layered imputation door (ADR-0088/0089/0090)
    ".imputation": (
        "AuthoredColumn",
        "ColumnEstimates",
        "ColumnImputationRecord",
        "ColumnRouting",
        "DroppedColumnAbsentWarning",
        "FitSignals",
        "FittedImputer",
        "FittedUnit",
        "ImputationConfig",
        "ImputationRecipe",
        "ImputationResult",
        "ImputationRouting",
        "ImputationStrategy",
        "ImputationUnit",
        "ModelChoice",
        "NumericImputationConfig",
        "RoutingReason",
        "UnitFitResult",
        "author",
        "derive_units",
        "fit_unit",
    ),
    # --- Observability: Pipeline Event stream + observers
    ".observability": ("EventType", "PipelineEvent", "stderr_observer"),
    # --- Operations: the Agent-facing surface (ADR-0099). Every other name
    # here is today's door, and becomes internal as the Operations land.
    ".operations.describe_columns": ("describe_columns",),
    ".operations.describe_target": ("describe_target",),
    ".operations.explain_routing": ("explain_routing",),
    ".operations.export_frame": ("export_frame",),
    ".operations.list_handles": ("list_handles",),
    ".operations.load": ("load",),
    ".operations.missingness": ("missingness",),
    ".operations.profile": ("profile",),
    ".operations.resolve_recipe": ("resolve_recipe",),
    ".operations.route": ("route",),
    ".operations.split": ("split",),
    ".operations.top_associations": ("top_associations",),
    ".operations.value_summary": ("value_summary",),
    # --- Config objects, Phase Sub-Configs, input enums and result records
    ".profiling": (
        "CategoricalProfileConfig",
        "ColumnProfile",
        "CorrelationProfileConfig",
        "DatasetStats",
        "DatetimeProfileConfig",
        "MissingnessProfileConfig",
        "NonlinearityProfileConfig",
        "NumericKind",
        "NumericProfileConfig",
        "ProfileConfig",
        "StructuralProfileResult",
        "TypeDetectionConfig",
    ),
    # --- Entry points
    ".profiling.orchestrator": ("StructuralProfiler",),
    ".splitting": ("DataSplitter", "FoldResult", "HoldoutCVResult", "SplitConfig", "SplitResult"),
    ".utils.data_loader": ("DataLoader",),
}

_MODULE_OF: dict[str, str] = {
    name: module for module, names in _EXPORTS.items() for name in names
}

__all__ = [
    "ArtifactPythonVersionWarning",
    "AuthoredColumn",
    "C2STAnnotation",
    "C2STConfig",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STResult",
    "C2STScheme",
    "C2STScore",
    "C2STVerdict",
    "CategoricalProfileConfig",
    "ColumnC2STResult",
    "ColumnEstimates",
    "ColumnImputationRecord",
    "ColumnProfile",
    "ColumnRouting",
    "CorrelationProfileConfig",
    "DataLoader",
    "DataSplitter",
    "DatasetStats",
    "DatetimeProfileConfig",
    "DroppedColumnAbsentWarning",
    "EventType",
    "FitSignals",
    "FittedImputer",
    "FittedUnit",
    "FoldResult",
    "HoldoutCVResult",
    "ImputationConfig",
    "ImputationRecipe",
    "ImputationResult",
    "ImputationRouting",
    "ImputationStrategy",
    "ImputationUnit",
    "MissingnessProfileConfig",
    "Modality",
    "ModelChoice",
    "NonlinearityProfileConfig",
    "NumericImputationConfig",
    "NumericKind",
    "NumericProfileConfig",
    "PipelineConfig",
    "PipelineEvent",
    "PipelinePhase",
    "ProfileConfig",
    "RoutingReason",
    "SemanticType",
    "SplitConfig",
    "SplitResult",
    "StructuralProfileResult",
    "StructuralProfiler",
    "TypeDetectionConfig",
    "UnitFitResult",
    "author",
    "c2st",
    "derive_units",
    "describe_columns",
    "describe_target",
    "deserialize",
    "explain_routing",
    "export_frame",
    "fit_unit",
    "imputation_score_c2st",
    "inspect",
    "list_handles",
    "load",
    "missingness",
    "profile",
    "resolve_recipe",
    "route",
    "serialize",
    "split",
    "stderr_observer",
    "top_associations",
    "value_summary",
]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
