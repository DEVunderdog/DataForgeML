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

# --- Evaluation: c2st is public; the umbrella is gone (ADR-0087 amended) ---
from .evaluation import (
    C2STAnnotation,
    C2STConfig,
    C2STDtypeError,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STResult,
    C2STSampleFloorError,
    C2STScheme,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
    c2st,
    imputation_score_c2st,
)

# --- The layered imputation door (ADR-0088/0089/0090) ----------------------
from .imputation import (
    AuthoredColumn,
    ColumnEstimates,
    ColumnImputationRecord,
    ColumnRouting,
    DroppedColumnAbsentWarning,
    FitSignals,
    FittedColumnAbsentError,
    FittedImputer,
    FittedUnit,
    ImputationConfig,
    ImputationFitWarning,
    ImputationRecipe,
    ImputationResult,
    ImputationRouting,
    ImputationStrategy,
    ImputationUnit,
    ModelChoice,
    NumericImputationConfig,
    UnitFitResult,
    UnitNotTrainableError,
    UnseenColumnError,
    author,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
)

# --- Observability: Pipeline Event stream + observers ----------------------
from .observability import EventType, PipelineEvent, stderr_observer

# --- Operations: the Agent-facing surface (ADR-0099). Every other name here
# is today's door, and becomes internal as the Operations land. ------------
from .operations.export_frame import export_frame
from .operations.list_handles import list_handles
from .operations.load import load

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
    "ArtifactPythonVersionWarning",
    "AuthoredColumn",
    "C2STAnnotation",
    "C2STConfig",
    "C2STDtypeError",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STResult",
    "C2STSampleFloorError",
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
    "FittedColumnAbsentError",
    "FittedImputer",
    "FittedUnit",
    "FoldResult",
    "HoldoutCVResult",
    "ImputationConfig",
    "ImputationFitWarning",
    "ImputationRecipe",
    "ImputationResult",
    "ImputationRouting",
    "ImputationStrategy",
    "ImputationUnit",
    "IncompatibleArtifactError",
    "MissingnessProfileConfig",
    "Modality",
    "ModelChoice",
    "NonlinearityProfileConfig",
    "NumericImputationConfig",
    "NumericKind",
    "NumericProfileConfig",
    "OverrideCoercionError",
    "PipelineConfig",
    "PipelineEvent",
    "PipelinePhase",
    "ProfileConfig",
    "SemanticType",
    "SplitConfig",
    "SplitResult",
    "StructuralProfileResult",
    "StructuralProfiler",
    "TypeDetectionConfig",
    "UnitFitResult",
    "UnitNotTrainableError",
    "UnseenColumnError",
    "UnsupportedFormatError",
    "author",
    "c2st",
    "derive_units",
    "deserialize",
    "fit_unit",
    # --- Evaluation ---------------------------------------------------------
    # ``C2STOutcome`` / ``C2STAnnotation`` / ``C2STVerdict`` classify output and
    # are never user-supplied, a stated exception to ADR-0050 (ADR-0087):
    # reading the report *is* branching on them, so the handling half of the
    # export rule reaches them. ``c2st`` is public as of ADR-0087's amendment
    # (#537): the umbrella that justified privacy is gone.
    "imputation_score_c2st",
    "inspect",
    # --- Operations ---------------------------------------------------------
    "export_frame",
    "list_handles",
    "load",
    "resolve_recipe",
    "route",
    "serialize",
    "stderr_observer",
]
