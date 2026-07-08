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
from .imputation import EvaluationOrchestrator, ImputationOrchestrator, FittedImputer
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
from .imputation import ImputationStrategy

# --- Observability: Pipeline Event stream + observers ----------------------
from .observability import PipelineEvent, EventType, stderr_observer

# --- Result and nested-record types ----------------------------------------
from .profiling import StructuralProfileResult, ColumnProfile, DatasetStats
from .imputation import ImputationResult, ColumnImputationRecord
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
)
from .profiling import OverrideCoercionError
from .utils.data_loader import UnsupportedFormatError

__all__ = [
    # Pipeline-level config and shared enums
    "PipelineConfig",
    "PipelinePhase",
    "SemanticType",
    "Modality",
    # Entry points
    "StructuralProfiler",
    "ImputationOrchestrator",
    "EvaluationOrchestrator",
    "FittedImputer",
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
]
