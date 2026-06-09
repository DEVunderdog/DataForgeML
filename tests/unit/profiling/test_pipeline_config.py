"""
Unit tests for PipelineConfig.resolve_active_columns and serialisation.

All tests are pure — no DataFrames, no StructuralProfiler.
"""

import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.profiling._config import ProfileConfig
from dataforge_ml.profiling._missingness_config import MissingnessProfileConfig
from dataforge_ml.profiling._numeric_config import NumericProfileConfig
from dataforge_ml.profiling._type_detection_config import TypeDetectionConfig
from dataforge_ml.profiling._categorical_config import CategoricalProfileConfig
from dataforge_ml.profiling._correlation_config import CorrelationProfileConfig
from dataforge_ml.profiling._datetime_config import DatetimeProfileConfig

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


# ---------------------------------------------------------------------------
# ProfileConfig sub-config round-trip
# ---------------------------------------------------------------------------


def test_profile_config_sub_config_round_trip():
    original = ProfileConfig(
        compute_correlation=True,
        row_drop_threshold=0.30,
        missingness=MissingnessProfileConfig(
            severity_minor=0.02,
            severity_moderate=0.08,
            severity_high=0.25,
            mar_correlation_threshold=0.70,
            col_drop_threshold=0.60,
        ),
        numeric=NumericProfileConfig(
            skew_normal=0.3,
            skew_moderate=0.8,
            skew_high=1.5,
            kurt_platykurtic_upper=-2.0,
            kurt_leptokurtic_lower=4.0,
            near_constant_threshold=0.85,
            scale_orders_of_magnitude=4,
            discrete_max_unique=25,
        ),
        type_detection=TypeDetectionConfig(
            numeric_coerce_threshold=0.99,
            datetime_coerce_threshold=0.90,
            encoded_category_max_unique=20,
            identifier_unique_ratio=0.995,
        ),
        categorical=CategoricalProfileConfig(
            rare_threshold_pct=0.02,
            stratification_rare_threshold_pct=0.10,
            near_constant_threshold=0.85,
        ),
        correlation=CorrelationProfileConfig(
            near_redundant_pearson_threshold=0.85,
            near_redundant_cramer_v_threshold=0.70,
            near_redundant_eta_squared_threshold=0.40,
            mi_min_rows=20,
        ),
        datetime_=DatetimeProfileConfig(
            mnar_null_ratio_threshold=0.02,
            high_gap_cv_threshold=0.8,
            recent_window_fraction=0.15,
        ),
    )

    restored = ProfileConfig.from_dict(original.to_dict())

    assert restored.row_drop_threshold == original.row_drop_threshold

    assert restored.missingness.severity_minor == original.missingness.severity_minor
    assert restored.missingness.severity_moderate == original.missingness.severity_moderate
    assert restored.missingness.severity_high == original.missingness.severity_high
    assert restored.missingness.mar_correlation_threshold == original.missingness.mar_correlation_threshold
    assert restored.missingness.col_drop_threshold == original.missingness.col_drop_threshold

    assert restored.numeric.skew_normal == original.numeric.skew_normal
    assert restored.numeric.skew_moderate == original.numeric.skew_moderate
    assert restored.numeric.skew_high == original.numeric.skew_high
    assert restored.numeric.kurt_platykurtic_upper == original.numeric.kurt_platykurtic_upper
    assert restored.numeric.kurt_leptokurtic_lower == original.numeric.kurt_leptokurtic_lower
    assert restored.numeric.near_constant_threshold == original.numeric.near_constant_threshold
    assert restored.numeric.scale_orders_of_magnitude == original.numeric.scale_orders_of_magnitude
    assert restored.numeric.discrete_max_unique == original.numeric.discrete_max_unique

    assert restored.type_detection.numeric_coerce_threshold == original.type_detection.numeric_coerce_threshold
    assert restored.type_detection.datetime_coerce_threshold == original.type_detection.datetime_coerce_threshold
    assert restored.type_detection.encoded_category_max_unique == original.type_detection.encoded_category_max_unique
    assert restored.type_detection.identifier_unique_ratio == original.type_detection.identifier_unique_ratio

    assert restored.categorical.rare_threshold_pct == original.categorical.rare_threshold_pct
    assert restored.categorical.stratification_rare_threshold_pct == original.categorical.stratification_rare_threshold_pct
    assert restored.categorical.near_constant_threshold == original.categorical.near_constant_threshold

    assert restored.correlation.near_redundant_pearson_threshold == original.correlation.near_redundant_pearson_threshold
    assert restored.correlation.near_redundant_cramer_v_threshold == original.correlation.near_redundant_cramer_v_threshold
    assert restored.correlation.near_redundant_eta_squared_threshold == original.correlation.near_redundant_eta_squared_threshold
    assert restored.correlation.mi_min_rows == original.correlation.mi_min_rows

    assert restored.datetime_.mnar_null_ratio_threshold == original.datetime_.mnar_null_ratio_threshold
    assert restored.datetime_.high_gap_cv_threshold == original.datetime_.high_gap_cv_threshold
    assert restored.datetime_.recent_window_fraction == original.datetime_.recent_window_fraction


def test_profile_config_defaults_unchanged():
    cfg = ProfileConfig()

    assert cfg.row_drop_threshold == 0.50
    assert cfg.missingness.severity_minor == 0.01
    assert cfg.missingness.severity_moderate == 0.05
    assert cfg.missingness.severity_high == 0.20
    assert cfg.missingness.mar_correlation_threshold == 0.60
    assert cfg.missingness.col_drop_threshold == 0.50
    assert cfg.numeric.skew_normal == 0.5
    assert cfg.numeric.skew_moderate == 1.0
    assert cfg.numeric.skew_high == 2.0
    assert cfg.numeric.near_constant_threshold == 0.90
    assert cfg.type_detection.numeric_coerce_threshold == 0.95
    assert cfg.type_detection.identifier_unique_ratio == 0.99
    assert cfg.categorical.rare_threshold_pct == 0.01
    assert cfg.categorical.near_constant_threshold == 0.90
    assert cfg.correlation.near_redundant_pearson_threshold == 0.95
    assert cfg.datetime_.mnar_null_ratio_threshold == 0.05
