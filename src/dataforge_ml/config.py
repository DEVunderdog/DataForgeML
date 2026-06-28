from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from dataforge_ml.profiling._config import ProfileConfig, NumericKind
    from dataforge_ml.imputation._config import ImputationConfig
    from dataforge_ml.splitting._config import SplitConfig


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
    from dataforge_ml.profiling._config import ProfileConfig
    return ProfileConfig()


def _default_imputation_config() -> ImputationConfig:
    from dataforge_ml.imputation._config import ImputationConfig
    return ImputationConfig()


def _default_split_config() -> SplitConfig:
    from dataforge_ml.splitting._config import SplitConfig
    return SplitConfig()


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
    numeric_kind_overrides : dict[str, NumericKind]
        Explicit ``NumericKind`` assignments for individual columns, applied
        after auto-detection in Phase 1. Only valid for columns whose final
        ``SemanticType`` is ``Numeric``; raises at orchestrator time otherwise.
    profiling : ProfileConfig
        Phase 1-specific parameters (correlation, chunking, memory threshold).
    imputation : ImputationConfig
        Phase 2-specific parameters (strategy thresholds, size guards).
    split : SplitConfig
        Splitting thresholds (stratification signal cap, boolean minority bar).
    random_seed : int, optional
        Single seed for all stochastic pipeline operations, including GMM
        Sampling during bimodal imputation. None produces non-deterministic
        output.
    """

    exclude_columns: list[str] = field(default_factory=list)
    phase_exclusions: dict[PipelinePhase, list[str]] = field(default_factory=dict)
    column_overrides: dict[str, SemanticType] = field(default_factory=dict)
    numeric_kind_overrides: dict[str, NumericKind] = field(default_factory=dict)
    profiling: ProfileConfig = field(default_factory=_default_profile_config)
    imputation: ImputationConfig = field(default_factory=_default_imputation_config)
    split: SplitConfig = field(default_factory=_default_split_config)
    random_seed: Optional[int] = None

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

    def add_exclusions(self, cols: list[str]) -> None:
        """Add columns to the hard exclusion set, deduplicating automatically.

        Columns already present in ``exclude_columns`` and duplicate entries
        within ``cols`` are silently ignored. Calling with an empty list is a
        no-op.

        Parameters
        ----------
        cols : list[str]
            Column names to register as hard exclusions. Deduplication is
            handled here; callers do not need to pre-deduplicate.
        """
        existing = set(self.exclude_columns)
        for col in cols:
            if col not in existing:
                self.exclude_columns.append(col)
                existing.add(col)

    def set_column_type(
        self, column: str, semantic_type: Union[str, SemanticType]
    ) -> None:
        """Explicitly set the semantic type for a column, overriding auto-detection.

        Parameters
        ----------
        column : str
            Name of the column to override.
        semantic_type : str or SemanticType
            The desired semantic type. Accepts enum values or their string
            equivalents (e.g. ``"numeric"``, ``"categorical"``).

        Raises
        ------
        ValueError
            When ``semantic_type`` is a string that does not match any
            ``SemanticType`` value.
        """
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
        """Assign the same semantic type to every column in the list.

        Parameters
        ----------
        columns : list[str]
            Column names to override.
        semantic_type : str or SemanticType
            The desired semantic type applied to every column in the list.
        """
        for column in columns:
            self.set_column_type(column, semantic_type)

    def set_numeric_kind(
        self, column: str, kind: Union[str, NumericKind]
    ) -> None:
        """Explicitly set the ``NumericKind`` for a single column.

        Parameters
        ----------
        column : str
            Name of the column to override.
        kind : str or NumericKind
            The desired numeric kind. Accepts enum values or their string
            equivalents (``"continuous"``, ``"bounded_discrete"``).

        Raises
        ------
        ValueError
            When ``kind`` is a string that does not match any ``NumericKind``
            value.
        """
        from dataforge_ml.profiling._config import NumericKind as _NumericKind
        if isinstance(kind, str):
            try:
                kind = _NumericKind(kind)
            except ValueError:
                valid = [e.value for e in _NumericKind]
                raise ValueError(
                    f"Unknown NumericKind {kind!r}. Valid values: {valid}"
                )
        self.numeric_kind_overrides[column] = kind

    def set_columns_numeric_kind(
        self, columns: list[str], kind: Union[str, NumericKind]
    ) -> None:
        """Assign the same ``NumericKind`` to every column in the list.

        Parameters
        ----------
        columns : list[str]
            Column names to override.
        kind : str or NumericKind
            The desired numeric kind applied to every column in the list.
        """
        for column in columns:
            self.set_numeric_kind(column, kind)

    def to_dict(self) -> dict:
        """Serialise the pipeline configuration to a plain dictionary.

        Returns
        -------
        dict
            All fields serialised to JSON-compatible types; nested configs are
            recursively serialised via their own ``to_dict`` methods.
        """
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
            "numeric_kind_overrides": {
                col: str(kind)
                for col, kind in self.numeric_kind_overrides.items()
            },
            "profiling": self.profiling.to_dict(),
            "imputation": self.imputation.to_dict(),
            "split": self.split.to_dict(),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        """Reconstruct a ``PipelineConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by ``to_dict()``.

        Returns
        -------
        PipelineConfig
            Fully populated configuration instance with all nested sub-configs
            restored.
        """
        from dataforge_ml.profiling._config import ProfileConfig, NumericKind as _NumericKind
        from dataforge_ml.imputation._config import ImputationConfig
        from dataforge_ml.splitting._config import SplitConfig
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
            numeric_kind_overrides={
                col: _NumericKind(kind_str)
                for col, kind_str in data.get("numeric_kind_overrides", {}).items()
            },
            profiling=ProfileConfig.from_dict(data.get("profiling", {})),
            imputation=ImputationConfig.from_dict(data.get("imputation", {})),
            split=SplitConfig.from_dict(data.get("split", {})),
            random_seed=data.get("random_seed"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise the pipeline configuration to a JSON string.

        Parameters
        ----------
        indent : int
            Number of spaces used for JSON indentation.

        Returns
        -------
        str
            JSON representation of ``to_dict()``.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineConfig:
        """Reconstruct a ``PipelineConfig`` from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string as produced by ``to_json()``.

        Returns
        -------
        PipelineConfig
            Fully populated configuration instance.
        """
        return cls.from_dict(json.loads(json_str))
