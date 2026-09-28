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
    "FittedImputer",

    "route",
    "author",
    "AuthoredColumn",
    "resolve_recipe",
    "ImputationRouting",
    "ImputationRecipe",
    "ColumnRouting",
    "ColumnEstimates",
    "derive_units",
    "fit_unit",
    "ImputationUnit",
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
    "SplitResult",
    "FoldResult",
    "HoldoutCVResult",
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

    # --- Evaluation ---------------------------------------------------------
    # ``C2STOutcome`` / ``C2STAnnotation`` / ``C2STVerdict`` classify output and
    # are never user-supplied, a stated exception to ADR-0050 (ADR-0087):
    # reading the report *is* branching on them, so the handling half of the
    # export rule reaches them. ``c2st`` is public as of ADR-0087's amendment
    # (#537): the umbrella that justified privacy is gone.
    "imputation_score_c2st",
    "c2st",
    "C2STConfig",
    "C2STScheme",
    "C2STReport",
    "ColumnC2STResult",
    "C2STScore",
    "C2STResult",
    "C2STProvenance",
    "C2STOutcome",
    "C2STAnnotation",
    "C2STVerdict",
    "C2STDtypeError",
    "C2STSampleFloorError",
]
