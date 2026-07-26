"""DataForgeML public API.

The names re-exported here constitute the Public API. A symbol is exported at
the package root iff the user must reference it to configure, drive, or handle
the pipeline (ADR-0050). Output-only internal types (e.g. ``TypeFlag``,
per-modality stats dataclasses) remain accessible only via submodule imports.
"""

# --- Pipeline-level config and shared enums -------------------------------
# --- Bare-bytes persistence: serialize / deserialize / inspect (ADR-0072) ---
from ._serialization import (
    ArtifactPythonVersionWarning,
    IncompatibleArtifactError,
    deserialize,
    inspect,
    serialize,
)
from .config import Modality, PipelineConfig, PipelinePhase, SemanticType

# --- User-facing fit-quality diagnostics -----------------------------------
# --- Exceptions the user catches -------------------------------------------
from .imputation import (
    AccuracyDiagnostic,
    AccuracyReport,
    ColumnImputationDecision,
    ColumnImputationRecord,
    DroppedColumnAbsentWarning,
    EvaluationOrchestrator,
    FitSignals,
    FittedColumnAbsentError,
    FittedImputer,
    FittedUnit,
    ImputationConfig,
    ImputationDecision,
    ImputationFitWarning,
    ImputationResult,
    ImputationStrategy,
    InspectionDiagnostic,
    InspectionReport,
    ModelChoice,
    NumericImputationConfig,
    UnfittedColumnError,
    UnitFitResult,
    UnitNotTrainableError,
    UnseenColumnError,
    decide,
    fit_unit,
)

# --- Observability: Pipeline Event stream + observers ----------------------
from .observability import EventType, PipelineEvent, stderr_observer

# --- Config objects and Phase Sub-Configs ----------------------------------
# --- Input enums (user-supplied via setter or field) -----------------------
# --- Result and nested-record types ----------------------------------------
from .profiling import (
    CategoricalProfileConfig,
    ColumnProfile,
    CorrelationProfileConfig,
    DatasetStats,
    DatetimeProfileConfig,
    MissingnessProfileConfig,
    NonlinearityProfileConfig,
    NumericKind,
    NumericProfileConfig,
    OverrideCoercionError,
    ProfileConfig,
    StructuralProfileResult,
    TypeDetectionConfig,
)

# --- Entry points ----------------------------------------------------------
from .profiling.orchestrator import StructuralProfiler
from .splitting import (
    DataSplitter,
    FoldResult,
    HoldoutCVResult,
    SplitConfig,
    SplitResult,
)
from .utils.data_loader import DataLoader, UnsupportedFormatError

__all__ = [
    # Pipeline-level config and shared enums
    "PipelineConfig",
    "PipelinePhase",
    "SemanticType",
    "Modality",
    # Entry points
    "StructuralProfiler",
    "EvaluationOrchestrator",
    "FittedImputer",
    # Imputation is user-orchestrated: decide() plans, the caller's fit_unit()
    # loop trains (batch scheduling is user-owned, ADR-0075), and
    # FittedImputer.compose() aggregates. There is no fused entry point
    # (ADR-0060, ADR-0071).
    "decide",
    "fit_unit",
    "UnitFitResult",
    "FitSignals",
    "UnitNotTrainableError",
    "FittedUnit",
    "DataSplitter",
    "DataLoader",
    # Config objects and Phase Sub-Configs
    "ProfileConfig",
    "MissingnessProfileConfig",
    "NumericProfileConfig",
    "NonlinearityProfileConfig",
    "TypeDetectionConfig",
    "CategoricalProfileConfig",
    "CorrelationProfileConfig",
    "DatetimeProfileConfig",
    "ImputationConfig",
    "NumericImputationConfig",
    "SplitConfig",
    # Input enums
    "NumericKind",
    "ImputationStrategy",
    "ModelChoice",
    # Observability
    "PipelineEvent",
    "EventType",
    "stderr_observer",
    # Result and nested-record types
    "StructuralProfileResult",
    "ColumnProfile",
    "DatasetStats",
    "ImputationResult",
    "ColumnImputationRecord",
    "ColumnImputationDecision",
    "ImputationDecision",
    "SplitResult",
    "FoldResult",
    "HoldoutCVResult",
    # User-facing fit-quality diagnostics
    "InspectionDiagnostic",
    "InspectionReport",
    "AccuracyDiagnostic",
    "AccuracyReport",
    # Exceptions the user catches
    "UnseenColumnError",
    "FittedColumnAbsentError",
    "UnfittedColumnError",
    "OverrideCoercionError",
    "UnsupportedFormatError",
    "IncompatibleArtifactError",
    # Warnings the user filters
    "ArtifactPythonVersionWarning",
    "ImputationFitWarning",
    "DroppedColumnAbsentWarning",
    # Bare-bytes persistence (ADR-0072)
    "serialize",
    "deserialize",
    "inspect",
]
