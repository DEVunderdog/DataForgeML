"""
Unit tests for PipelineConfig.resolve_active_columns and serialisation.

All tests are pure — no DataFrames, no StructuralProfiler.
"""

import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.profiling._config import ProfileConfig

_ALL_PHASES = list(PipelinePhase)


# ---------------------------------------------------------------------------
# resolve_active_columns — hard exclusions
# ---------------------------------------------------------------------------


def test_hard_excluded_column_absent_from_every_phase():
    available = ["a", "b", "c", "d"]
    cfg = PipelineConfig(exclude_columns=["b"])

    for phase in _ALL_PHASES:
        result = cfg.resolve_active_columns(phase, available)
        assert "b" not in result, f"hard-excluded 'b' must be absent for {phase}"
        assert set(result) == {"a", "c", "d"}


def test_hard_exclusion_on_nonexistent_column_is_silently_ignored():
    available = ["x", "y"]
    cfg = PipelineConfig(exclude_columns=["z"])

    for phase in _ALL_PHASES:
        assert cfg.resolve_active_columns(phase, available) == ["x", "y"]


# ---------------------------------------------------------------------------
# resolve_active_columns — soft exclusions
# ---------------------------------------------------------------------------


def test_soft_excluded_column_absent_from_its_phase():
    available = ["a", "b", "c"]
    cfg = PipelineConfig(phase_exclusions={PipelinePhase.Scaling: ["b"]})

    result = cfg.resolve_active_columns(PipelinePhase.Scaling, available)
    assert "b" not in result


def test_soft_excluded_column_present_in_all_other_phases():
    available = ["a", "b", "c"]
    cfg = PipelineConfig(phase_exclusions={PipelinePhase.Scaling: ["b"]})

    for phase in _ALL_PHASES:
        if phase == PipelinePhase.Scaling:
            continue
        result = cfg.resolve_active_columns(phase, available)
        assert "b" in result, f"soft-excluded 'b' must be present for {phase}"


# ---------------------------------------------------------------------------
# resolve_active_columns — hard takes precedence over soft
# ---------------------------------------------------------------------------


def test_column_in_both_hard_and_soft_always_absent():
    available = ["a", "b", "c"]
    cfg = PipelineConfig(
        exclude_columns=["b"],
        phase_exclusions={PipelinePhase.Profiling: ["b"]},
    )

    for phase in _ALL_PHASES:
        result = cfg.resolve_active_columns(phase, available)
        assert "b" not in result, (
            f"column in both hard and soft exclusions must be absent for {phase}"
        )


# ---------------------------------------------------------------------------
# resolve_active_columns — no exclusions
# ---------------------------------------------------------------------------


def test_no_exclusions_returns_available_columns_unchanged():
    available = ["x", "y", "z"]
    cfg = PipelineConfig()

    for phase in _ALL_PHASES:
        assert cfg.resolve_active_columns(phase, available) == available


def test_empty_available_columns_returns_empty():
    cfg = PipelineConfig(exclude_columns=["a"], phase_exclusions={PipelinePhase.Encoding: ["b"]})

    for phase in _ALL_PHASES:
        assert cfg.resolve_active_columns(phase, []) == []


# ---------------------------------------------------------------------------
# resolve_active_columns — ordering preserved
# ---------------------------------------------------------------------------


def test_active_columns_preserve_input_order():
    available = ["c", "a", "b", "d"]
    cfg = PipelineConfig(exclude_columns=["b"])

    result = cfg.resolve_active_columns(PipelinePhase.Profiling, available)
    assert result == ["c", "a", "d"]


# ---------------------------------------------------------------------------
# Serialisation — to_dict / from_dict
# ---------------------------------------------------------------------------


def test_to_dict_serialises_phase_exclusion_keys_as_strings():
    cfg = PipelineConfig(
        phase_exclusions={PipelinePhase.Scaling: ["age"]},
    )
    d = cfg.to_dict()
    assert all(isinstance(k, str) for k in d["phase_exclusions"])
    assert "scaling" in d["phase_exclusions"]


def test_to_dict_serialises_column_override_values_as_strings():
    cfg = PipelineConfig(column_overrides={"score": SemanticType.Categorical})
    d = cfg.to_dict()
    assert d["column_overrides"]["score"] == "categorical"
    assert isinstance(d["column_overrides"]["score"], str)


def test_from_dict_restores_pipeline_phase_enum_keys():
    d = {
        "exclude_columns": [],
        "phase_exclusions": {"outlier_detection": ["postcode"]},
        "column_overrides": {},
        "profiling": {},
    }
    cfg = PipelineConfig.from_dict(d)
    assert PipelinePhase.OutlierDetection in cfg.phase_exclusions
    assert cfg.phase_exclusions[PipelinePhase.OutlierDetection] == ["postcode"]


def test_from_dict_restores_semantic_type_enum_values():
    d = {
        "exclude_columns": [],
        "phase_exclusions": {},
        "column_overrides": {"score": "categorical"},
        "profiling": {},
    }
    cfg = PipelineConfig.from_dict(d)
    assert isinstance(cfg.column_overrides["score"], SemanticType)
    assert cfg.column_overrides["score"] == SemanticType.Categorical


def test_from_dict_reconstructs_nested_profile_config():
    d = {
        "exclude_columns": [],
        "phase_exclusions": {},
        "column_overrides": {},
        "profiling": {
            "compute_correlation": True,
            "target_columns": ["label"],
            "memory_threshold_mb": 256.0,
            "chunk_size": 50000,
        },
    }
    cfg = PipelineConfig.from_dict(d)
    assert isinstance(cfg.profiling, ProfileConfig)
    assert cfg.profiling.compute_correlation is True
    assert cfg.profiling.target_columns == ["label"]
    assert cfg.profiling.memory_threshold_mb == 256.0
    assert cfg.profiling.chunk_size == 50000


# ---------------------------------------------------------------------------
# Serialisation — full round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_complex_config():
    original = PipelineConfig(
        exclude_columns=["id", "internal_key"],
        phase_exclusions={
            PipelinePhase.Scaling: ["year_of_birth"],
            PipelinePhase.OutlierDetection: ["postcode"],
        },
        column_overrides={"score": SemanticType.Categorical},
        profiling=ProfileConfig(
            compute_correlation=True,
            target_columns=["label"],
            memory_threshold_mb=256.0,
            chunk_size=50_000,
        ),
    )

    restored = PipelineConfig.from_json(original.to_json())

    assert restored.exclude_columns == original.exclude_columns
    assert restored.phase_exclusions == original.phase_exclusions
    assert restored.column_overrides == original.column_overrides
    assert isinstance(restored.column_overrides["score"], SemanticType)
    assert PipelinePhase.Scaling in restored.phase_exclusions
    assert PipelinePhase.OutlierDetection in restored.phase_exclusions
    assert restored.profiling.compute_correlation == original.profiling.compute_correlation
    assert restored.profiling.target_columns == original.profiling.target_columns
    assert restored.profiling.memory_threshold_mb == original.profiling.memory_threshold_mb
    assert restored.profiling.chunk_size == original.profiling.chunk_size


def test_round_trip_empty_config():
    original = PipelineConfig()
    restored = PipelineConfig.from_json(original.to_json())

    assert restored.exclude_columns == []
    assert restored.phase_exclusions == {}
    assert restored.column_overrides == {}
