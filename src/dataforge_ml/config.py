from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Union, Optional
from types import MappingProxyType

if TYPE_CHECKING:
    from dataforge_ml.profiling._config import ProfileConfig, NumericKind
    from dataforge_ml.imputation._config import ImputationConfig
    from dataforge_ml.splitting._config import SplitConfig


class SemanticType(StrEnum):
    """The ML-level interpretation assigned to a column by the type detector.

    Used throughout the pipeline to route columns to the correct sub-processors
    and to determine which statistical operations apply. See CONTEXT.md §SemanticType
    for the full type taxonomy and the Text vs Categorical distinction.
    """

    Numeric = "numeric"
    Categorical = "categorical"
    Datetime = "datetime"
    Boolean = "boolean"
    Text = "text"
    Identifier = "identifier"


class Modality(StrEnum):
    """The data modality the pipeline operates on.

    Currently only ``Tabular`` is supported. Reserved for future expansion to
    additional modalities (time-series, image, etc.).
    """

    Tabular = "tabular"


class PipelinePhase(StrEnum):
    """The six sequential phases of the DataForgeML feature engineering pipeline.

    Phase Orchestrators call ``PipelineConfig.resolve_active_columns`` with one
    of these values to obtain the column set for that phase after Hard and Soft
    Exclusions are applied.
    """

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

    Attributes
    ----------
    exclude_columns : tuple[str, ...]
        Hard exclusions — columns dropped globally from every phase.
    phase_exclusions : MappingProxyType[PipelinePhase, tuple[str, ...]]
        Soft exclusions — columns bypassed for a specific phase but retained
        in the dataset.
    column_overrides : MappingProxyType[str, SemanticType]
        Explicit semantic type assignments respected by all downstream phases.
    numeric_kind_overrides : MappingProxyType[str, NumericKind]
        Explicit ``NumericKind`` assignments for individual columns, applied
        after auto-detection in Phase 1. Only valid for columns whose final
        ``SemanticType`` is ``Numeric``; raises at orchestrator time otherwise.
    """

    _exclude_columns: list[str] = field(default_factory=list, init=False)
    _phase_exclusions: dict[PipelinePhase, list[str]] = field(default_factory=dict, init=False)
    _column_overrides: dict[str, SemanticType] = field(default_factory=dict, init=False)
    _numeric_kind_overrides: dict[str, NumericKind] = field(default_factory=dict, init=False)
    profiling: ProfileConfig = field(default_factory=_default_profile_config)
    imputation: ImputationConfig = field(default_factory=_default_imputation_config)
    split: SplitConfig = field(default_factory=_default_split_config)
    random_seed: Optional[int] = None

    @property
    def exclude_columns(self) -> tuple[str, ...]:
        """Hard exclusions — columns dropped globally from every phase.

        Returns
        -------
        tuple[str, ...]
            A snapshot tuple of columns registered as hard exclusions.
        """
        return tuple(self._exclude_columns)

    @property
    def phase_exclusions(self) -> "MappingProxyType[PipelinePhase, tuple[str, ...]]":
        """Soft exclusions — columns bypassed for a specific phase but retained in the dataset.

        Returns
        -------
        MappingProxyType[PipelinePhase, tuple[str, ...]]
            A read-only view mapping each phase to a tuple of excluded columns.
        """
        from types import MappingProxyType
        return MappingProxyType({k: tuple(v) for k, v in self._phase_exclusions.items()})

    @property
    def column_overrides(self) -> "MappingProxyType[str, SemanticType]":
        """Explicit semantic type assignments respected by all downstream phases.

        Returns
        -------
        MappingProxyType[str, SemanticType]
            A read-only view mapping columns to their explicitly assigned SemanticType.
        """
        from types import MappingProxyType
        return MappingProxyType(self._column_overrides)

    @property
    def numeric_kind_overrides(self) -> "MappingProxyType[str, NumericKind]":
        """Explicit NumericKind assignments for individual columns, applied after auto-detection in Phase 1.

        Returns
        -------
        MappingProxyType[str, NumericKind]
            A read-only view mapping columns to their explicitly assigned NumericKind.
        """
        from types import MappingProxyType
        return MappingProxyType(self._numeric_kind_overrides)

    def resolve_active_columns(
        self, phase: PipelinePhase, available_columns: list[str]
    ) -> list[str]:
        """Return the columns the given phase should operate on.

        Hard Exclusions are applied first, then phase-specific Soft Exclusions.
        Columns absent from ``available_columns`` are silently ignored in both
        exclusion lists.

        Parameters
        ----------
        phase : PipelinePhase
            The pipeline phase requesting the active column set.
        available_columns : list[str]
            The full list of columns currently present in the DataFrame.

        Returns
        -------
        list[str]
            Columns from ``available_columns`` that are not excluded by either
            Hard or Soft Exclusion rules for the given phase, preserving the
            original order.
        """
        hard_set = set(self.exclude_columns)
        soft_set = set(self.phase_exclusions.get(phase, ()))
        excluded = hard_set | soft_set
        return [c for c in available_columns if c not in excluded]

    def add_exclusion(self, column: Union[str, list[str]]) -> None:
        """Add columns to the hard exclusion set, deduplicating automatically.

        Columns already present in the exclusion list and duplicate entries
        within the input are silently ignored. Calling with an empty list is a
        no-op.

        Parameters
        ----------
        column : str or list[str]
            Column name(s) to register as hard exclusions. Deduplication is
            handled here; callers do not need to pre-deduplicate.
        """
        cols = [column] if isinstance(column, str) else column
        existing = set(self._exclude_columns)
        for col in cols:
            if col not in existing:
                self._exclude_columns.append(col)
                existing.add(col)

    def add_phase_exclusion(self, phase: Union[PipelinePhase, str], column: Union[str, list[str]]) -> None:
        """Add columns to the soft exclusion set for a specific phase.

        Parameters
        ----------
        phase : PipelinePhase or str
            The phase for which to exclude the column(s).
        column : str or list[str]
            Column name(s) to register as soft exclusions for this phase.
            Deduplication is handled automatically.
        """
        if isinstance(phase, str):
            phase = PipelinePhase(phase)
            
        cols = [column] if isinstance(column, str) else column
        phase_list = self._phase_exclusions.setdefault(phase, [])
        existing = set(phase_list)
        for col in cols:
            if col not in existing:
                phase_list.append(col)
                existing.add(col)

    def set_column_type(
        self, column: Union[str, list[str]], semantic_type: Union[str, SemanticType]
    ) -> None:
        """Explicitly set the semantic type for one or more columns, overriding auto-detection.

        Parameters
        ----------
        column : str or list[str]
            Name of the column(s) to override.
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
        cols = [column] if isinstance(column, str) else column
        for col in cols:
            self._column_overrides[col] = semantic_type

    def set_numeric_kind(
        self, column: Union[str, list[str]], kind: Union[str, NumericKind]
    ) -> None:
        """Explicitly set the ``NumericKind`` for one or more columns.

        Parameters
        ----------
        column : str or list[str]
            Name of the column(s) to override.
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
        cols = [column] if isinstance(column, str) else column
        for col in cols:
            self._numeric_kind_overrides[col] = kind

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
        from dataforge_ml.profiling._config import ProfileConfig
        from dataforge_ml.imputation._config import ImputationConfig
        from dataforge_ml.splitting._config import SplitConfig
        cfg = cls(
            profiling=ProfileConfig.from_dict(data.get("profiling", {})),
            imputation=ImputationConfig.from_dict(data.get("imputation", {})),
            split=SplitConfig.from_dict(data.get("split", {})),
            random_seed=data.get("random_seed"),
        )
        
        cfg.add_exclusion(data.get("exclude_columns", []))
        
        for phase_str, cols in data.get("phase_exclusions", {}).items():
            cfg.add_phase_exclusion(phase_str, cols)
            
        for col, sem_str in data.get("column_overrides", {}).items():
            cfg.set_column_type(col, sem_str)
            
        for col, kind_str in data.get("numeric_kind_overrides", {}).items():
            cfg.set_numeric_kind(col, kind_str)
            
        return cfg

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
