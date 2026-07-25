"""DataForgeML public API.

The names re-exported here constitute the Public API. A symbol is exported at
the package root iff the user must reference it to configure, drive, or handle
the pipeline (ADR-0050). Output-only internal types (e.g. ``TypeFlag``,
per-modality stats dataclasses) remain accessible only via submodule imports.
"""

# --- Pipeline-level config and shared enums -------------------------------
from .config import PipelineConfig, PipelinePhase, SemanticType, Modality

# --- Entry points ----------------------------------------------------------
from .profiling.orchestrator import StructuralProfiler
from .imputation import (
    EvaluationOrchestrator,
    FitSignals,
    FittedImputer,
    FittedUnit,
    ImputationFitWarning,
    UnitFitResult,
    UnitNotTrainableError,
    decide,
    fit_unit,
)
from .splitting import DataSplitter
from .utils.data_loader import DataLoader

# --- Config objects and Phase Sub-Configs ----------------------------------
from .profiling import (
    ProfileConfig,
    MissingnessProfileConfig,
    NumericProfileConfig,
    NonlinearityProfileConfig,
    TypeDetectionConfig,
    CategoricalProfileConfig,
    CorrelationProfileConfig,
    DatetimeProfileConfig,
)
from .imputation import ImputationConfig, NumericImputationConfig
from .splitting import SplitConfig

# --- Input enums (user-supplied via setter or field) -----------------------
from .profiling import NumericKind
from .imputation import ImputationStrategy, ModelChoice

# --- Observability: Pipeline Event stream + observers ----------------------
from .observability import PipelineEvent, EventType, stderr_observer

# --- Result and nested-record types ----------------------------------------
from .profiling import StructuralProfileResult, ColumnProfile, DatasetStats
from .imputation import (
    ImputationResult,
    ColumnImputationRecord,
    ColumnImputationDecision,
    ImputationDecision,
)
from .splitting import SplitResult, FoldResult, HoldoutCVResult

# --- User-facing fit-quality diagnostics -----------------------------------
from .imputation import (
    AccuracyDiagnostic,
    AccuracyReport,
    InspectionDiagnostic,
    InspectionReport,
)

# --- Exceptions the user catches -------------------------------------------
from .imputation import (
    UnseenColumnError,
    FittedColumnAbsentError,
    UnfittedColumnError,
    DroppedColumnAbsentWarning,
)
from .profiling import OverrideCoercionError
from .utils.data_loader import UnsupportedFormatError
from ._serialization import (
    ArtifactPythonVersionWarning,
    IncompatibleArtifactError,
)

# --- Bare-bytes persistence: serialize / deserialize / inspect (ADR-0072) ---
from ._serialization import serialize, deserialize, inspect

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
