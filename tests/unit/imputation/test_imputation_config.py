"""
Unit tests for ImputationConfig, NumericImputationConfig, and PipelineConfig
imputation wiring.

All tests are pure — no DataFrames, no StructuralProfiler.
"""

import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.imputation import (
    ColumnImputationRecord,
    ImputationConfig,
    ImputationResult,
    ImputationStrategy,
    NumericImputationConfig,
)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_all_types_importable_from_imputation_module():
    from dataforge_ml.imputation import (  # noqa: F401
        ColumnImputationRecord,
        ImputationConfig,
        ImputationResult,
        ImputationStrategy,
        NumericImputationConfig,
    )


def test_imputation_strategy_has_expected_values():
    assert ImputationStrategy.Mean == "mean"
    assert ImputationStrategy.Median == "median"
    assert ImputationStrategy.Mode == "mode"
    assert ImputationStrategy.KNN == "knn"
    assert ImputationStrategy.Regression == "regression"
    assert ImputationStrategy.MICE == "mice"
    assert ImputationStrategy.Constant == "constant"
    assert ImputationStrategy.Dropped == "dropped"
    assert ImputationStrategy.Passthrough == "passthrough"


# ---------------------------------------------------------------------------
# NumericImputationConfig — defaults
# ---------------------------------------------------------------------------


def test_numeric_imputation_config_default_knn_max_rows():
    cfg = NumericImputationConfig()
    assert cfg.knn_max_rows == 50_000


def test_numeric_imputation_config_default_knn_max_features():
    cfg = NumericImputationConfig()
    assert cfg.knn_max_features == 50


def test_numeric_imputation_config_default_regression_min_rows():
    cfg = NumericImputationConfig()
    assert cfg.regression_min_rows == 500


def test_numeric_imputation_config_default_mnar_constant_fill():
    cfg = NumericImputationConfig()
    assert cfg.mnar_constant_fill == -1


# ---------------------------------------------------------------------------
# NumericImputationConfig — to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_numeric_config_to_dict_contains_all_keys():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {"knn_max_rows", "knn_max_features", "regression_min_rows", "mnar_constant_fill"}


def test_numeric_config_round_trip_default_values():
    original = NumericImputationConfig()
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == original.knn_max_rows
    assert restored.knn_max_features == original.knn_max_features
    assert restored.regression_min_rows == original.regression_min_rows
    assert restored.mnar_constant_fill == original.mnar_constant_fill


def test_numeric_config_round_trip_non_default_values():
    original = NumericImputationConfig(
        knn_max_rows=10_000,
        knn_max_features=20,
        regression_min_rows=1_000,
        mnar_constant_fill=-999,
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == 10_000
    assert restored.knn_max_features == 20
    assert restored.regression_min_rows == 1_000
    assert restored.mnar_constant_fill == -999


def test_numeric_config_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.knn_max_rows == 50_000
    assert cfg.knn_max_features == 50
    assert cfg.regression_min_rows == 500
    assert cfg.mnar_constant_fill == -1


# ---------------------------------------------------------------------------
# ImputationConfig — defaults
# ---------------------------------------------------------------------------


def test_imputation_config_default_numeric_is_numeric_imputation_config():
    cfg = ImputationConfig()
    assert isinstance(cfg.numeric, NumericImputationConfig)


def test_imputation_config_default_mnar_columns_empty():
    cfg = ImputationConfig()
    assert cfg.mnar_columns == []


def test_imputation_config_default_add_indicator_columns_empty():
    cfg = ImputationConfig()
    assert cfg.add_indicator_columns == []


# ---------------------------------------------------------------------------
# ImputationConfig — to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_imputation_config_to_dict_contains_expected_keys():
    cfg = ImputationConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {"numeric", "mnar_columns", "add_indicator_columns"}


def test_imputation_config_to_dict_numeric_is_nested_dict():
    cfg = ImputationConfig()
    d = cfg.to_dict()
    assert isinstance(d["numeric"], dict)
    assert "knn_max_rows" in d["numeric"]


def test_imputation_config_round_trip_default_values():
    original = ImputationConfig()
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == []
    assert restored.add_indicator_columns == []
    assert restored.numeric.knn_max_rows == 50_000


def test_imputation_config_round_trip_non_default_values():
    original = ImputationConfig(
        numeric=NumericImputationConfig(knn_max_rows=5_000, regression_min_rows=200),
        mnar_columns=["income", "age"],
        add_indicator_columns=["score"],
    )
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == ["income", "age"]
    assert restored.add_indicator_columns == ["score"]
    assert restored.numeric.knn_max_rows == 5_000
    assert restored.numeric.regression_min_rows == 200


def test_imputation_config_from_dict_empty_uses_defaults():
    cfg = ImputationConfig.from_dict({})
    assert isinstance(cfg.numeric, NumericImputationConfig)
    assert cfg.mnar_columns == []
    assert cfg.add_indicator_columns == []


def test_imputation_config_mnar_columns_are_independent_copies():
    original = ImputationConfig(mnar_columns=["a", "b"])
    d = original.to_dict()
    d["mnar_columns"].append("c")
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == ["a", "b"]


# ---------------------------------------------------------------------------
# PipelineConfig — imputation field wiring
# ---------------------------------------------------------------------------


def test_pipeline_config_has_imputation_field():
    cfg = PipelineConfig()
    assert hasattr(cfg, "imputation")


def test_pipeline_config_imputation_default_is_imputation_config():
    cfg = PipelineConfig()
    assert isinstance(cfg.imputation, ImputationConfig)


def test_pipeline_config_imputation_default_has_correct_thresholds():
    cfg = PipelineConfig()
    assert cfg.imputation.numeric.knn_max_rows == 50_000
    assert cfg.imputation.numeric.knn_max_features == 50
    assert cfg.imputation.numeric.regression_min_rows == 500
    assert cfg.imputation.numeric.mnar_constant_fill == -1


def test_pipeline_config_two_instances_have_independent_imputation_configs():
    cfg1 = PipelineConfig()
    cfg2 = PipelineConfig()
    cfg1.imputation.mnar_columns.append("x")
    assert cfg2.imputation.mnar_columns == []


# ---------------------------------------------------------------------------
# PipelineConfig — to_dict includes imputation
# ---------------------------------------------------------------------------


def test_pipeline_config_to_dict_includes_imputation_key():
    cfg = PipelineConfig()
    d = cfg.to_dict()
    assert "imputation" in d


def test_pipeline_config_to_dict_imputation_contains_numeric():
    cfg = PipelineConfig()
    d = cfg.to_dict()
    assert "numeric" in d["imputation"]
    assert d["imputation"]["numeric"]["knn_max_rows"] == 50_000


def test_pipeline_config_to_dict_imputation_contains_mnar_columns():
    cfg = PipelineConfig(imputation=ImputationConfig(mnar_columns=["col_a"]))
    d = cfg.to_dict()
    assert d["imputation"]["mnar_columns"] == ["col_a"]


# ---------------------------------------------------------------------------
# PipelineConfig — from_dict reconstructs imputation
# ---------------------------------------------------------------------------


def test_pipeline_config_from_dict_reconstructs_imputation_config():
    d = {
        "exclude_columns": [],
        "phase_exclusions": {},
        "column_overrides": {},
        "profiling": {},
        "imputation": {
            "numeric": {"knn_max_rows": 20_000, "knn_max_features": 30},
            "mnar_columns": ["revenue"],
            "add_indicator_columns": [],
        },
    }
    cfg = PipelineConfig.from_dict(d)
    assert isinstance(cfg.imputation, ImputationConfig)
    assert cfg.imputation.numeric.knn_max_rows == 20_000
    assert cfg.imputation.numeric.knn_max_features == 30
    assert cfg.imputation.mnar_columns == ["revenue"]


def test_pipeline_config_from_dict_without_imputation_key_uses_defaults():
    d = {
        "exclude_columns": [],
        "phase_exclusions": {},
        "column_overrides": {},
        "profiling": {},
    }
    cfg = PipelineConfig.from_dict(d)
    assert isinstance(cfg.imputation, ImputationConfig)
    assert cfg.imputation.numeric.knn_max_rows == 50_000


# ---------------------------------------------------------------------------
# PipelineConfig — full round-trip with imputation
# ---------------------------------------------------------------------------


def test_pipeline_config_round_trip_includes_imputation():
    original = PipelineConfig(
        exclude_columns=["id"],
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=15_000,
                knn_max_features=25,
                regression_min_rows=300,
                mnar_constant_fill=-99,
            ),
            mnar_columns=["salary", "age"],
            add_indicator_columns=["credit_score"],
        ),
    )
    restored = PipelineConfig.from_json(original.to_json())

    assert restored.exclude_columns == ["id"]
    assert isinstance(restored.imputation, ImputationConfig)
    assert restored.imputation.numeric.knn_max_rows == 15_000
    assert restored.imputation.numeric.knn_max_features == 25
    assert restored.imputation.numeric.regression_min_rows == 300
    assert restored.imputation.numeric.mnar_constant_fill == -99
    assert restored.imputation.mnar_columns == ["salary", "age"]
    assert restored.imputation.add_indicator_columns == ["credit_score"]


def test_pipeline_config_round_trip_empty_config_imputation_defaults():
    original = PipelineConfig()
    restored = PipelineConfig.from_json(original.to_json())

    assert restored.imputation.mnar_columns == []
    assert restored.imputation.add_indicator_columns == []
    assert restored.imputation.numeric.knn_max_rows == 50_000
