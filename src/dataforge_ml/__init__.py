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
    AuthoredColumn,
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
    ImputationUnit,
    InspectionDiagnostic,
    InspectionReport,
    ModelChoice,
    NumericImputationConfig,
    UnitFitResult,
    UnitNotTrainableError,
    UnseenColumnError,
    author,
    core_budget,
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
    "PipelineConfig",
    "PipelinePhase",
    "SemanticType",
    "Modality",

    "StructuralProfiler",
    "EvaluationOrchestrator",
    "FittedImputer",
    
    "decide",
    "author",
    "AuthoredColumn",
    "fit_unit",
    "ImputationUnit",
    "core_budget",
    "UnitFitResult",
    "FitSignals",
    "UnitNotTrainableError",
    "FittedUnit",
    "DataSplitter",
    "DataLoader",
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
    "NumericKind",
    "ImputationStrategy",
    "ModelChoice",
    "PipelineEvent",
    "EventType",
    "stderr_observer",
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
    "InspectionDiagnostic",
    "InspectionReport",
    "AccuracyDiagnostic",
    "AccuracyReport",
    "UnseenColumnError",
    "FittedColumnAbsentError",
    "OverrideCoercionError",
    "UnsupportedFormatError",
    "IncompatibleArtifactError",
    "ArtifactPythonVersionWarning",
    "ImputationFitWarning",
    "DroppedColumnAbsentWarning",
    "serialize",
    "deserialize",
    "inspect",
]
