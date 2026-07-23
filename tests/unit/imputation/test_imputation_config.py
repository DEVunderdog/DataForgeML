"""
Unit tests for ImputationConfig, NumericImputationConfig, and PipelineConfig
imputation wiring.

All tests are pure — no DataFrames, no StructuralProfiler.
"""

import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.imputation import (
    ColumnImputationDecision,
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
    from dataforge_ml.imputation import (
        ColumnImputationRecord,  # noqa: F401
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
    assert ImputationStrategy.MNAR == "mnar"
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


def test_numeric_imputation_config_mnar_constant_fill_removed():
    with pytest.raises(TypeError):
        NumericImputationConfig(mnar_constant_fill=-1)


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
        "gradient_boost_min_rows",
        "base_max_iter",
        "knn_min_neighbors",
        "knn_max_neighbors",
        "knn_distance_weight_max_null_ratio",
        "knn_distance_weight_max_features",
        "mice_n_nearest_features_min_cols",
        "mice_max_nearest_features",
        "mice_correlation_threshold",
        "mcar_feature_predictability_threshold",
        "per_column_strategy",
        "per_column_constant_fill",
        "per_column_max_iter",
        "knn_n_neighbors",
        "mice_max_iter",
        "refit_r2_min_complete_rows",
        "refit_r2_cv_folds",
        "bimodal_grouping_variables",
        "bimodal_min_correlated_features",
        "bimodal_correlation_threshold",
        "max_workers",
    }


def test_numeric_config_round_trip_default_values():
    original = NumericImputationConfig()
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == original.knn_max_rows
    assert restored.knn_max_features == original.knn_max_features
    assert restored.regression_min_rows == original.regression_min_rows
    assert restored.gradient_boost_min_rows == original.gradient_boost_min_rows
    assert restored.base_max_iter == original.base_max_iter


def test_numeric_config_round_trip_non_default_values():
    original = NumericImputationConfig(
        knn_max_rows=10_000,
        knn_max_features=20,
        regression_min_rows=1_000,
        gradient_boost_min_rows=25_000,
        base_max_iter=20,
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_max_rows == 10_000
    assert restored.knn_max_features == 20
    assert restored.regression_min_rows == 1_000
    assert restored.gradient_boost_min_rows == 25_000
    assert restored.base_max_iter == 20


def test_numeric_config_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.knn_max_rows == 50_000
    assert cfg.knn_max_features == 50
    assert cfg.regression_min_rows == 500
    assert cfg.gradient_boost_min_rows == 10_000
    assert cfg.base_max_iter == 10
    assert cfg.bimodal_correlation_threshold == 0.2


def test_numeric_config_round_trip_bimodal_correlation_threshold():
    original = NumericImputationConfig(bimodal_correlation_threshold=0.45)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.bimodal_correlation_threshold == 0.45


def test_numeric_config_from_dict_ignores_legacy_mnar_constant_fill():
    cfg = NumericImputationConfig.from_dict({"mnar_constant_fill": -9999, "knn_max_rows": 1_000})
    assert cfg.knn_max_rows == 1_000


def test_numeric_config_round_trip_per_column_strategy_non_empty():
    original = NumericImputationConfig(
        _per_column_strategy={
            "sensor": ImputationStrategy.Median,
            "income": ImputationStrategy.Regression,
        },
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.per_column_strategy == {
        "sensor": ImputationStrategy.Median,
        "income": ImputationStrategy.Regression,
    }


def test_numeric_config_round_trip_per_column_constant_fill_non_empty():
    original = NumericImputationConfig(
        _per_column_constant_fill={"tx_count": 0.0},
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.per_column_constant_fill["tx_count"] == pytest.approx(0.0)
    assert "tx_count" not in restored.per_column_strategy


def test_numeric_config_default_regression_base_max_iter():
    cfg = NumericImputationConfig()
    assert cfg.base_max_iter == 10


def test_numeric_config_custom_regression_base_max_iter():
    cfg = NumericImputationConfig(base_max_iter=20)
    assert cfg.base_max_iter == 20


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
    assert (
        restored.knn_distance_weight_max_null_ratio
        == original.knn_distance_weight_max_null_ratio
    )
    assert (
        restored.knn_distance_weight_max_features
        == original.knn_distance_weight_max_features
    )


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
# NumericImputationConfig — MICE n_nearest_features fields (defaults)
# ---------------------------------------------------------------------------


def test_numeric_config_default_mice_n_nearest_features_min_cols():
    cfg = NumericImputationConfig()
    assert cfg.mice_n_nearest_features_min_cols == 10


def test_numeric_config_default_mice_max_nearest_features():
    cfg = NumericImputationConfig()
    assert cfg.mice_max_nearest_features == 20


def test_numeric_config_default_mice_correlation_threshold():
    cfg = NumericImputationConfig()
    assert cfg.mice_correlation_threshold == 0.1


# ---------------------------------------------------------------------------
# NumericImputationConfig — MICE n_nearest_features fields round-trip
# ---------------------------------------------------------------------------


def test_numeric_config_mice_fields_in_to_dict():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert d["mice_n_nearest_features_min_cols"] == 10
    assert d["mice_max_nearest_features"] == 20
    assert d["mice_correlation_threshold"] == 0.1


def test_numeric_config_mice_fields_round_trip_default():
    original = NumericImputationConfig()
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert (
        restored.mice_n_nearest_features_min_cols
        == original.mice_n_nearest_features_min_cols
    )
    assert restored.mice_max_nearest_features == original.mice_max_nearest_features
    assert restored.mice_correlation_threshold == original.mice_correlation_threshold


def test_numeric_config_mice_fields_round_trip_non_default():
    original = NumericImputationConfig(
        mice_n_nearest_features_min_cols=5,
        mice_max_nearest_features=15,
        mice_correlation_threshold=0.25,
    )
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.mice_n_nearest_features_min_cols == 5
    assert restored.mice_max_nearest_features == 15
    assert restored.mice_correlation_threshold == 0.25


def test_numeric_config_mice_fields_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.mice_n_nearest_features_min_cols == 10
    assert cfg.mice_max_nearest_features == 20
    assert cfg.mice_correlation_threshold == 0.1


# ---------------------------------------------------------------------------
# NumericImputationConfig — mcar_feature_predictability_threshold
# ---------------------------------------------------------------------------


def test_numeric_config_default_mcar_feature_predictability_threshold():
    cfg = NumericImputationConfig()
    assert cfg.mcar_feature_predictability_threshold == 0.2


def test_numeric_config_mcar_feature_predictability_threshold_in_to_dict():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert d["mcar_feature_predictability_threshold"] == 0.2


def test_numeric_config_mcar_feature_predictability_threshold_round_trip_non_default():
    original = NumericImputationConfig(mcar_feature_predictability_threshold=0.35)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.mcar_feature_predictability_threshold == 0.35


def test_numeric_config_mcar_feature_predictability_threshold_from_dict_empty_uses_default():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.mcar_feature_predictability_threshold == 0.2


# ---------------------------------------------------------------------------
# ImputationConfig — defaults
# ---------------------------------------------------------------------------


def test_imputation_config_default_numeric_is_numeric_imputation_config():
    cfg = ImputationConfig()
    assert isinstance(cfg.numeric, NumericImputationConfig)


def test_imputation_config_default_mnar_columns_empty():
    cfg = ImputationConfig()
    assert cfg.mnar_columns == ()


def test_imputation_config_default_add_indicator_columns_empty():
    cfg = ImputationConfig()
    assert cfg.add_indicator_columns == ()


# ---------------------------------------------------------------------------
# ImputationConfig — to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_imputation_config_to_dict_contains_expected_keys():
    cfg = ImputationConfig()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "numeric",
        "mnar_columns",
        "add_indicator_columns",
    }


def test_imputation_config_to_dict_numeric_is_nested_dict():
    cfg = ImputationConfig()
    d = cfg.to_dict()
    assert isinstance(d["numeric"], dict)
    assert "knn_max_rows" in d["numeric"]


def test_imputation_config_round_trip_default_values():
    original = ImputationConfig()
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == ()
    assert restored.add_indicator_columns == ()
    assert restored.numeric.knn_max_rows == 50_000


def test_imputation_config_round_trip_non_default_values():
    original = ImputationConfig(
        numeric=NumericImputationConfig(knn_max_rows=5_000, regression_min_rows=200),
    )
    original.add_mnar_column(["income", "age"])
    original.add_indicator_column(["score"])
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == ("income", "age")
    assert restored.add_indicator_columns == ("score",)
    assert restored.numeric.knn_max_rows == 5_000
    assert restored.numeric.regression_min_rows == 200


def test_imputation_config_from_dict_empty_uses_defaults():
    cfg = ImputationConfig.from_dict({})
    assert isinstance(cfg.numeric, NumericImputationConfig)
    assert cfg.mnar_columns == ()
    assert cfg.add_indicator_columns == ()


def test_imputation_config_mnar_columns_are_independent_copies():
    original = ImputationConfig()
    original.add_mnar_column(["a", "b"])
    d = original.to_dict()
    d["mnar_columns"].append("c")
    restored = ImputationConfig.from_dict(original.to_dict())
    assert restored.mnar_columns == ("a", "b")
def test_imputation_config_from_dict_raises_on_invalid_entry():
    d = {
        "numeric": {"per_column_strategy": {"income": "median"}},
        "mnar_columns": ["income"],
    }
    with pytest.raises(ValueError, match="mutually exclusive: 'income'"):
        ImputationConfig.from_dict(d)


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


def test_pipeline_config_two_instances_have_independent_imputation_configs():
    cfg1 = PipelineConfig()
    cfg2 = PipelineConfig()
    cfg1.imputation.add_mnar_column("x")
    assert cfg2.imputation.mnar_columns == ()


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
    cfg = PipelineConfig()
    cfg.imputation = ImputationConfig()
    cfg.imputation.add_mnar_column(["col_a"])
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
    assert cfg.imputation.mnar_columns == ("revenue",)


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
    original = PipelineConfig()
    original.add_exclusion("id")
    original.imputation = ImputationConfig(
        numeric=NumericImputationConfig(
            knn_max_rows=15_000,
            knn_max_features=25,
            regression_min_rows=300,
        ),
    )
    original.imputation.add_mnar_column(["salary", "age"])
    original.imputation.add_indicator_column(["credit_score"])
    restored = PipelineConfig.from_json(original.to_json())

    assert restored.exclude_columns == ("id",)
    assert isinstance(restored.imputation, ImputationConfig)
    assert restored.imputation.numeric.knn_max_rows == 15_000
    assert restored.imputation.numeric.knn_max_features == 25
    assert restored.imputation.numeric.regression_min_rows == 300
    assert restored.imputation.mnar_columns == ("salary", "age")
    assert restored.imputation.add_indicator_columns == ("credit_score",)


def test_pipeline_config_round_trip_empty_config_imputation_defaults():
    original = PipelineConfig()
    restored = PipelineConfig.from_json(original.to_json())

    assert restored.imputation.mnar_columns == ()
    assert restored.imputation.add_indicator_columns == ()
    assert restored.imputation.numeric.knn_max_rows == 50_000


# ---------------------------------------------------------------------------
# ColumnImputationRecord — domain_snap_bounds round-trip
# ---------------------------------------------------------------------------


def test_column_imputation_record_domain_snap_bounds_round_trips(round_trip):
    from dataforge_ml.imputation._fitted_imputer import FittedImputer

    record = ColumnImputationRecord(decision=ColumnImputationDecision(column="rating", semantic_type=SemanticType.Numeric, strategy=ImputationStrategy.Regression, domain_snap_bounds=(1.0, 5.0)))
    d = record.to_dict()
    assert d["domain_snap_bounds"] == [1.0, 5.0]

    fi = FittedImputer(records={"rating": record})
    restored = round_trip(fi)
    assert restored.records["rating"].decision.domain_snap_bounds == (1.0, 5.0)


def test_column_imputation_record_domain_snap_bounds_none_by_default():
    record = ColumnImputationRecord(decision=ColumnImputationDecision(column="age", semantic_type=SemanticType.Numeric, strategy=ImputationStrategy.Mean))
    d = record.to_dict()
    assert d["domain_snap_bounds"] is None


# ---------------------------------------------------------------------------
# NumericImputationConfig — per_column_strategy construction-time validation
# ---------------------------------------------------------------------------


def test_per_column_strategy_rejects_passthrough():
    with pytest.raises(ValueError, match="internal-only"):
        NumericImputationConfig(
            _per_column_strategy={"col_a": ImputationStrategy.Passthrough}
        )


def test_per_column_strategy_rejects_indicator():
    with pytest.raises(ValueError, match="internal-only"):
        NumericImputationConfig(
            _per_column_strategy={"col_a": ImputationStrategy.Indicator}
        )


def test_per_column_strategy_rejects_dropped_names_exclude_columns():
    with pytest.raises(ValueError, match="PipelineConfig.exclude_columns"):
        NumericImputationConfig(
            _per_column_strategy={"col_a": ImputationStrategy.Dropped}
        )


def test_per_column_strategy_rejects_mnar_names_mnar_columns():
    with pytest.raises(ValueError, match="mnar_columns"):
        NumericImputationConfig(
            _per_column_strategy={"col_a": ImputationStrategy.MNAR}
        )


def test_per_column_strategy_constant_without_fill_raises():
    with pytest.raises(ValueError, match="per_column_constant_fill"):
        NumericImputationConfig(
            _per_column_strategy={"tx_count": ImputationStrategy.Constant}
        )


def test_per_column_strategy_constant_without_fill_names_column():
    with pytest.raises(ValueError, match="'tx_count'"):
        NumericImputationConfig(
            _per_column_strategy={"tx_count": ImputationStrategy.Constant}
        )


def test_per_column_strategy_constant_with_fill_constructs():
    cfg = NumericImputationConfig(
        _per_column_strategy={"tx_count": ImputationStrategy.Constant},
        _per_column_constant_fill={"tx_count": -1.0},
    )
    assert cfg.per_column_strategy["tx_count"] == ImputationStrategy.Constant
    assert cfg.per_column_constant_fill["tx_count"] == -1.0


def test_per_column_constant_fill_alone_constructs():
    cfg = NumericImputationConfig(
        _per_column_constant_fill={"tx_count": 0.0},
    )
    assert cfg.per_column_constant_fill["tx_count"] == 0.0
    assert "tx_count" not in cfg.per_column_strategy


def test_per_column_strategy_all_allowed_strategies_construct():
    allowed = {
        "col_mean": ImputationStrategy.Mean,
        "col_median": ImputationStrategy.Median,
        "col_mode": ImputationStrategy.Mode,
        "col_knn": ImputationStrategy.KNN,
        "col_reg": ImputationStrategy.Regression,
        "col_mice": ImputationStrategy.MICE,
    }
    cfg = NumericImputationConfig(_per_column_strategy=allowed)
    assert len(cfg.per_column_strategy) == 6


def test_per_column_strategy_error_names_the_column():
    with pytest.raises(ValueError, match="'income'"):
        NumericImputationConfig(
            _per_column_strategy={"income": ImputationStrategy.Dropped}
        )



def test_per_column_strategy_empty_dict_default_constructs():
    cfg = NumericImputationConfig()
    assert cfg.per_column_strategy == {}
    assert cfg.per_column_constant_fill == {}


# ---------------------------------------------------------------------------
# ImputationConfig — MNAR conflict check
# ---------------------------------------------------------------------------


def test_imputation_config_mnar_conflict_raises_when_column_in_both():
    with pytest.raises(ValueError):
        cfg = ImputationConfig(
            numeric=NumericImputationConfig(
                _per_column_strategy={"income": ImputationStrategy.Median}
            ),
        )
        cfg.add_mnar_column("income")


def test_imputation_config_mnar_conflict_message_names_all_conflicting_columns():
    with pytest.raises(ValueError, match="'income'") as exc_info:
        cfg = ImputationConfig(
            numeric=NumericImputationConfig(
                _per_column_strategy={
                    "income": ImputationStrategy.Median,
                    "age": ImputationStrategy.Mean,
                }
            ),
        )
        cfg.add_mnar_column(["income", "age"])
    assert "'age'" in str(exc_info.value)


def test_imputation_config_mnar_conflict_no_error_when_disjoint():
    cfg = ImputationConfig(
        numeric=NumericImputationConfig(
            _per_column_strategy={"sensor": ImputationStrategy.Median}
        ),
    )
    cfg.add_mnar_column("income")
    assert cfg.mnar_columns == ("income",)


def test_imputation_config_mnar_conflict_no_error_when_mnar_empty():
    cfg = ImputationConfig(
        numeric=NumericImputationConfig(
            _per_column_strategy={"sensor": ImputationStrategy.Median}
        ),
    )
    cfg.add_mnar_column([])
    assert cfg.numeric.per_column_strategy["sensor"] == ImputationStrategy.Median


def test_imputation_config_mnar_conflict_no_error_when_per_column_strategy_empty():
    cfg = ImputationConfig(
        numeric=NumericImputationConfig(),
    )
    cfg.add_mnar_column("income")
    assert cfg.mnar_columns == ("income",)


def test_imputation_config_mnar_conflict_no_error_when_both_empty():
    cfg = ImputationConfig()
    assert cfg.mnar_columns == ()
    assert cfg.numeric.per_column_strategy == {}


# ---------------------------------------------------------------------------


def test_column_imputation_record_missing_domain_snap_bounds_deserialises_to_none():
    record = ColumnImputationRecord.from_dict({
        "column": "age",
        "semantic_type": "Numeric",
        "strategy": "Mean",
        "fill_value": 30.0,
        "indicator_added": False,
        "signals": [],
        # no domain_snap_bounds key — a record serialised before the field existed
    })
    assert record.decision.domain_snap_bounds is None


def test_column_imputation_record_carries_no_diagnostic_field():
    """Diagnostics moved to the Evaluation reports; records dropped the field (ADR-0058)."""
    record = ColumnImputationRecord(decision=ColumnImputationDecision(column="age", semantic_type=SemanticType.Numeric, strategy=ImputationStrategy.Mean), fill_value=30.0)
    assert not hasattr(record, "diagnostic")


def test_column_imputation_record_to_dict_excludes_diagnostic_key():
    record = ColumnImputationRecord(decision=ColumnImputationDecision(column="age", semantic_type=SemanticType.Numeric, strategy=ImputationStrategy.Mean), fill_value=30.0)
    assert "diagnostic" not in record.to_dict()


# ---------------------------------------------------------------------------
# NumericImputationConfig — per_column_max_iter / knn_n_neighbors / mice_max_iter defaults
# ---------------------------------------------------------------------------


def test_numeric_config_default_per_column_max_iter():
    cfg = NumericImputationConfig()
    assert cfg.per_column_max_iter == {}


def test_numeric_config_default_knn_n_neighbors():
    cfg = NumericImputationConfig()
    assert cfg.knn_n_neighbors is None


def test_numeric_config_default_mice_max_iter():
    cfg = NumericImputationConfig()
    assert cfg.mice_max_iter is None


def test_numeric_config_default_refit_r2_min_complete_rows():
    cfg = NumericImputationConfig()
    assert cfg.refit_r2_min_complete_rows == 50


# ---------------------------------------------------------------------------
# NumericImputationConfig — six new fields in to_dict
# ---------------------------------------------------------------------------


def test_numeric_config_per_column_max_iter_in_to_dict():
    cfg = NumericImputationConfig(_per_column_max_iter={"income": 20})
    d = cfg.to_dict()
    assert d["per_column_max_iter"] == {"income": 20}


def test_numeric_config_knn_n_neighbors_in_to_dict():
    cfg = NumericImputationConfig(knn_n_neighbors=15)
    d = cfg.to_dict()
    assert d["knn_n_neighbors"] == 15


def test_numeric_config_mice_max_iter_in_to_dict():
    cfg = NumericImputationConfig(mice_max_iter=100)
    d = cfg.to_dict()
    assert d["mice_max_iter"] == 100


def test_numeric_config_refit_fields_in_to_dict():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert d["refit_r2_min_complete_rows"] == 50


# ---------------------------------------------------------------------------
# NumericImputationConfig — from_dict({}) produces correct defaults
# ---------------------------------------------------------------------------


def test_numeric_config_new_fields_from_dict_empty_uses_defaults():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.per_column_max_iter == {}
    assert cfg.knn_n_neighbors is None
    assert cfg.mice_max_iter is None
    assert cfg.refit_r2_min_complete_rows == 50
    assert cfg.refit_r2_cv_folds == 5


# ---------------------------------------------------------------------------
# NumericImputationConfig — from_dict round-trips for non-default values
# ---------------------------------------------------------------------------


def test_numeric_config_per_column_max_iter_round_trip():
    original = NumericImputationConfig(_per_column_max_iter={"age": 30, "income": 50})
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.per_column_max_iter == {"age": 30, "income": 50}


def test_numeric_config_knn_n_neighbors_round_trip():
    original = NumericImputationConfig(knn_n_neighbors=7)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.knn_n_neighbors == 7


def test_numeric_config_mice_max_iter_round_trip():
    original = NumericImputationConfig(mice_max_iter=100)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.mice_max_iter == 100


def test_numeric_config_refit_r2_min_complete_rows_round_trip():
    original = NumericImputationConfig(refit_r2_min_complete_rows=50)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.refit_r2_min_complete_rows == 50


# ---------------------------------------------------------------------------
# NumericImputationConfig — refit_r2_cv_folds
# ---------------------------------------------------------------------------


def test_numeric_config_default_refit_r2_cv_folds():
    cfg = NumericImputationConfig()
    assert cfg.refit_r2_cv_folds == 5


def test_numeric_config_refit_r2_cv_folds_in_to_dict():
    cfg = NumericImputationConfig()
    d = cfg.to_dict()
    assert d["refit_r2_cv_folds"] == 5


def test_numeric_config_refit_r2_cv_folds_round_trip():
    original = NumericImputationConfig(refit_r2_cv_folds=10)
    restored = NumericImputationConfig.from_dict(original.to_dict())
    assert restored.refit_r2_cv_folds == 10


def test_numeric_config_refit_r2_cv_folds_from_dict_empty_uses_default():
    cfg = NumericImputationConfig.from_dict({})
    assert cfg.refit_r2_cv_folds == 5


# ---------------------------------------------------------------------------
# NumericImputationConfig — Setter Validation & Encapsulation Lockdown
# ---------------------------------------------------------------------------


def test_numeric_config_direct_writes_raise_type_error():
    cfg = NumericImputationConfig()
    
    with pytest.raises(TypeError):
        cfg.per_column_strategy["age"] = ImputationStrategy.Median
        
    with pytest.raises(TypeError):
        cfg.per_column_constant_fill["age"] = 1.0
        
    with pytest.raises(TypeError):
        cfg.per_column_max_iter["age"] = 10
        
    with pytest.raises(TypeError):
        cfg.bimodal_grouping_variables["age"] = "group"


def test_numeric_config_setter_validation_strategy_blocked():
    cfg = NumericImputationConfig()
    with pytest.raises(ValueError, match="'Dropped' cannot be used"):
        cfg.set_per_column_strategy("col", ImputationStrategy.Dropped)
        
    with pytest.raises(ValueError, match="'MNAR' cannot be used"):
        cfg.set_per_column_strategy("col", ImputationStrategy.MNAR)
        
    with pytest.raises(ValueError, match="internal-only"):
        cfg.set_per_column_strategy("col", ImputationStrategy.Passthrough)
        
    with pytest.raises(ValueError, match="internal-only"):
        cfg.set_per_column_strategy("col", ImputationStrategy.Indicator)


def test_numeric_config_setter_validation_constant_without_fill():
    cfg = NumericImputationConfig()
    with pytest.raises(ValueError, match="strategy is 'Constant' but no fill value was provided"):
        cfg.set_per_column_strategy("col", ImputationStrategy.Constant)


def test_numeric_config_setter_validation_constant_with_fill():
    cfg = NumericImputationConfig()
    cfg.set_per_column_constant_fill("col", 42.0)
    cfg.set_per_column_strategy("col", ImputationStrategy.Constant)
    assert cfg.per_column_strategy["col"] == ImputationStrategy.Constant


def test_numeric_config_setter_validation_constant_fill_nan_inf():
    cfg = NumericImputationConfig()
    with pytest.raises(ValueError, match="NaN or infinity"):
        cfg.set_per_column_constant_fill("col", float("nan"))
    with pytest.raises(ValueError, match="NaN or infinity"):
        cfg.set_per_column_constant_fill("col", float("inf"))


def test_numeric_config_setter_validation_max_iter():
    cfg = NumericImputationConfig()
    with pytest.raises(ValueError, match="must be > 0"):
        cfg.set_per_column_max_iter("col", 0)
    with pytest.raises(ValueError, match="must be > 0"):
        cfg.set_per_column_max_iter("col", -5)


def test_numeric_config_setter_validation_grouping_variable():
    cfg = NumericImputationConfig()
    with pytest.raises(ValueError, match="empty or purely whitespace"):
        cfg.set_bimodal_grouping_variable("col", "")
    with pytest.raises(ValueError, match="empty or purely whitespace"):
        cfg.set_bimodal_grouping_variable("col", "   ")


def test_numeric_config_from_dict_raises_on_invalid_entry():
    with pytest.raises(ValueError, match="must be > 0"):
        NumericImputationConfig.from_dict({"per_column_max_iter": {"col": -1}})
        
    with pytest.raises(ValueError, match="NaN or infinity"):
        NumericImputationConfig.from_dict({"per_column_constant_fill": {"col": float("inf")}})
        
    with pytest.raises(ValueError, match="'Dropped' cannot be used"):
        NumericImputationConfig.from_dict({"per_column_strategy": {"col": "dropped"}})
        
    with pytest.raises(ValueError, match="empty or purely whitespace"):
        NumericImputationConfig.from_dict({"bimodal_grouping_variables": {"col": "  "}})


def test_numeric_config_setters_accept_lists():
    cfg = NumericImputationConfig()
    cfg.set_per_column_constant_fill(["c1", "c2"], 0.0)
    cfg.set_per_column_strategy(["c1", "c2"], ImputationStrategy.Constant)
    cfg.set_per_column_max_iter(["c1", "c2"], 100)
    cfg.set_bimodal_grouping_variable(["c1", "c2"], "group")
    
    assert cfg.per_column_constant_fill == {"c1": 0.0, "c2": 0.0}
    assert cfg.per_column_strategy == {"c1": ImputationStrategy.Constant, "c2": ImputationStrategy.Constant}
    assert cfg.per_column_max_iter == {"c1": 100, "c2": 100}
    assert cfg.bimodal_grouping_variables == {"c1": "group", "c2": "group"}
