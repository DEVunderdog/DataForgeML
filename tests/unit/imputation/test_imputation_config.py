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


def test_numeric_imputation_config_default_gradient_boost_min_rows():
    cfg = NumericImputationConfig()
    assert cfg.gradient_boost_min_rows == 10_000


# ---------------------------------------------------------------------------
# NumericImputationConfig — to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_numeric_config_to_dict_contains_all_keys():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "knn_max_rows",
        "knn_max_features",
        "regression_min_rows",
        "mnar_constant_fill",
        "gradient_boost_min_rows",
        "regression_base_max_iter",
        "knn_min_neighbors",
        "knn_max_neighbors",
        "knn_distance_weight_max_null_ratio",
        "knn_distance_weight_max_features",
    }


def test_numeric_config_round_trip_default_values():
    original = NumericImputationConfig()
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == original.knn_max_rows
    assert restored.knn_max_features == original.knn_max_features
    assert restored.regression_min_rows == original.regression_min_rows
    assert restored.mnar_constant_fill == original.mnar_constant_fill
    assert restored.gradient_boost_min_rows == original.gradient_boost_min_rows
    assert restored.regression_base_max_iter == original.regression_base_max_iter


def test_numeric_config_round_trip_non_default_values():
    original = NumericImputationConfig(
        knn_max_rows=10_000,
        knn_max_features=20,
        regression_min_rows=1_000,
        mnar_constant_fill=-999,
        gradient_boost_min_rows=25_000,
        regression_base_max_iter=20,
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == 10_000
    assert restored.knn_max_features == 20
    assert restored.regression_min_rows == 1_000
    assert restored.mnar_constant_fill == -999
    assert restored.gradient_boost_min_rows == 25_000
    assert restored.regression_base_max_iter == 20


def test_numeric_config_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.knn_max_rows == 50_000
    assert cfg.knn_max_features == 50
    assert cfg.regression_min_rows == 500
    assert cfg.mnar_constant_fill == -1
    assert cfg.gradient_boost_min_rows == 10_000
    assert cfg.regression_base_max_iter == 10


def test_numeric_config_default_regression_base_max_iter():
    cfg = NumericImputationConfig()
    assert cfg.regression_base_max_iter == 10


def test_numeric_config_custom_regression_base_max_iter():
    cfg = NumericImputationConfig(regression_base_max_iter=20)
    assert cfg.regression_base_max_iter == 20


# ---------------------------------------------------------------------------
# NumericImputationConfig — KNN tuning fields (defaults)
# ---------------------------------------------------------------------------


def test_numeric_config_default_knn_min_neighbors():
    cfg = NumericImputationConfig()
    assert cfg.knn_min_neighbors == 5


def test_numeric_config_default_knn_max_neighbors():
    cfg = NumericImputationConfig()
    assert cfg.knn_max_neighbors == 25


def test_numeric_config_default_knn_distance_weight_max_null_ratio():
    cfg = NumericImputationConfig()
    assert cfg.knn_distance_weight_max_null_ratio == 0.15


def test_numeric_config_default_knn_distance_weight_max_features():
    cfg = NumericImputationConfig()
    assert cfg.knn_distance_weight_max_features == 30


# ---------------------------------------------------------------------------
# NumericImputationConfig — KNN tuning fields round-trip
# ---------------------------------------------------------------------------


def test_numeric_config_knn_tuning_fields_in_to_dict():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert d["knn_min_neighbors"] == 5
    assert d["knn_max_neighbors"] == 25
    assert d["knn_distance_weight_max_null_ratio"] == 0.15
    assert d["knn_distance_weight_max_features"] == 30


def test_numeric_config_knn_tuning_fields_round_trip_default():
    original = NumericImputationConfig()
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_min_neighbors == original.knn_min_neighbors
    assert restored.knn_max_neighbors == original.knn_max_neighbors
    assert restored.knn_distance_weight_max_null_ratio == original.knn_distance_weight_max_null_ratio
    assert restored.knn_distance_weight_max_features == original.knn_distance_weight_max_features


def test_numeric_config_knn_tuning_fields_round_trip_non_default():
    original = NumericImputationConfig(
        knn_min_neighbors=3,
        knn_max_neighbors=50,
        knn_distance_weight_max_null_ratio=0.20,
        knn_distance_weight_max_features=20,
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_min_neighbors == 3
    assert restored.knn_max_neighbors == 50
    assert restored.knn_distance_weight_max_null_ratio == 0.20
    assert restored.knn_distance_weight_max_features == 20


def test_numeric_config_knn_tuning_fields_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.knn_min_neighbors == 5
    assert cfg.knn_max_neighbors == 25
    assert cfg.knn_distance_weight_max_null_ratio == 0.15
    assert cfg.knn_distance_weight_max_features == 30


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


# ---------------------------------------------------------------------------
# ColumnImputationRecord — domain_snap_bounds round-trip
# ---------------------------------------------------------------------------


def test_column_imputation_record_domain_snap_bounds_round_trips():
    from dataforge_ml.imputation._fitted_imputer import FittedImputer

    record = ColumnImputationRecord(
        column="rating",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        domain_snap_bounds=(1.0, 5.0),
    )
    d = record.to_dict()
    assert d["domain_snap_bounds"] == [1.0, 5.0]

    fi = FittedImputer(records={"rating": record})
    restored = FittedImputer.from_dict(fi.to_dict())
    assert restored.records["rating"].domain_snap_bounds == (1.0, 5.0)


def test_column_imputation_record_domain_snap_bounds_none_by_default():
    record = ColumnImputationRecord(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Mean,
    )
    d = record.to_dict()
    assert d["domain_snap_bounds"] is None


def test_column_imputation_record_missing_domain_snap_bounds_deserialises_to_none():
    from dataforge_ml.imputation._fitted_imputer import FittedImputer

    legacy_dict = {
        "records": {
            "age": {
                "column": "age",
                "semantic_type": "numeric",
                "strategy": "mean",
                "fill_value": 30.0,
                "indicator_added": False,
                "signals": [],
                # no domain_snap_bounds key — simulates an old serialised record
            }
        },
        "models": {},
        "model_cols": {},
    }
    fi = FittedImputer.from_dict(legacy_dict)
    assert fi.records["age"].domain_snap_bounds is None
