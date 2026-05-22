from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dataforge_ml.profiling.config import ProfileConfig


class SemanticType(StrEnum):
    Numeric = "numeric"
    Categorical = "categorical"
    Datetime = "datetime"
    Boolean = "boolean"
    Text = "text"
    Identifier = "identifier"


class Modality(StrEnum):
    Tabular = "tabular"


class PipelinePhase(StrEnum):
    Profiling = "profiling"
    Imputation = "imputation"
    OutlierDetection = "outlier_detection"
    Normalization = "normalization"
    Encoding = "encoding"
    Scaling = "scaling"


def _default_profile_config() -> ProfileConfig:
    from dataforge_ml.profiling.config import ProfileConfig
    return ProfileConfig()


@dataclass
class PipelineConfig:
    """
    Master configuration for the full 6-phase feature engineering pipeline.

    Parameters
    ----------
    exclude_columns : list[str]
        Hard exclusions — columns dropped globally from every phase.
    phase_exclusions : dict[PipelinePhase, list[str]]
        Soft exclusions — columns bypassed for a specific phase but retained
        in the dataset.
    column_overrides : dict[str, SemanticType]
        Explicit semantic type assignments respected by all downstream phases.
    profiling : ProfileConfig
        Phase 1-specific parameters (correlation, chunking, memory threshold).
    """

    exclude_columns: list[str] = field(default_factory=list)
    phase_exclusions: dict[PipelinePhase, list[str]] = field(default_factory=dict)
    column_overrides: dict[str, SemanticType] = field(default_factory=dict)
    profiling: ProfileConfig = field(default_factory=_default_profile_config)

    def resolve_active_columns(
        self, phase: PipelinePhase, available_columns: list[str]
    ) -> list[str]:
        """
        Return the columns the given phase should operate on.

        Hard exclusions are applied first, then phase-specific soft exclusions.
        Columns absent from available_columns are silently ignored in both lists.
        """
        hard_set = set(self.exclude_columns)
        soft_set = set(self.phase_exclusions.get(phase, []))
        excluded = hard_set | soft_set
        return [c for c in available_columns if c not in excluded]

    def set_column_type(
        self, column: str, semantic_type: Union[str, SemanticType]
    ) -> None:
        """Explicitly set the semantic type for a column, overriding auto-detection."""
        if isinstance(semantic_type, str):
            try:
                semantic_type = SemanticType(semantic_type)
            except ValueError:
                valid = [e.value for e in SemanticType]
                raise ValueError(
                    f"Unknown semantic type {semantic_type!r}. "
                    f"Valid values: {valid}"
                )
        self.column_overrides[column] = semantic_type

    def set_columns_type(
        self, columns: list[str], semantic_type: Union[str, SemanticType]
    ) -> None:
        """Assign the same semantic type to every column in the list."""
        for column in columns:
            self.set_column_type(column, semantic_type)

    def to_dict(self) -> dict:
        return {
            "exclude_columns": list(self.exclude_columns),
            "phase_exclusions": {
                str(phase): list(cols)
                for phase, cols in self.phase_exclusions.items()
            },
            "column_overrides": {
                col: str(sem_type)
                for col, sem_type in self.column_overrides.items()
            },
            "profiling": self.profiling.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        from dataforge_ml.profiling.config import ProfileConfig
        return cls(
            exclude_columns=list(data.get("exclude_columns", [])),
            phase_exclusions={
                PipelinePhase(phase_str): list(cols)
                for phase_str, cols in data.get("phase_exclusions", {}).items()
            },
            column_overrides={
                col: SemanticType(sem_str)
                for col, sem_str in data.get("column_overrides", {}).items()
            },
            profiling=ProfileConfig.from_dict(data.get("profiling", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineConfig:
        return cls.from_dict(json.loads(json_str))
