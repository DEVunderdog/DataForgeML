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


def test_numeric_config_from_dict_ignores_legacy_mnar_constant_fill():
    cfg = NumericImputationConfig.from_dict({"mnar_constant_fill": -9999, "knn_max_rows": 1_000})
    assert cfg.knn_max_rows == 1_000


def test_numeric_config_round_trip_per_column_strategy_non_empty():
    original = NumericImputationConfig(
        per_column_strategy={
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
        per_column_constant_fill={"tx_count": 0.0},
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


# ---------------------------------------------------------------------------
# NumericImputationConfig — per_column_strategy construction-time validation
# ---------------------------------------------------------------------------


def test_per_column_strategy_rejects_passthrough():
    with pytest.raises(ValueError, match="internal-only"):
        NumericImputationConfig(
            per_column_strategy={"col_a": ImputationStrategy.Passthrough}
        )


def test_per_column_strategy_rejects_indicator():
    with pytest.raises(ValueError, match="internal-only"):
        NumericImputationConfig(
            per_column_strategy={"col_a": ImputationStrategy.Indicator}
        )


def test_per_column_strategy_rejects_dropped_names_exclude_columns():
    with pytest.raises(ValueError, match="PipelineConfig.exclude_columns"):
        NumericImputationConfig(
            per_column_strategy={"col_a": ImputationStrategy.Dropped}
        )


def test_per_column_strategy_rejects_mnar_names_mnar_columns():
    with pytest.raises(ValueError, match="mnar_columns"):
        NumericImputationConfig(
            per_column_strategy={"col_a": ImputationStrategy.MNAR}
        )


def test_per_column_strategy_constant_without_fill_raises():
    with pytest.raises(ValueError, match="per_column_constant_fill"):
        NumericImputationConfig(
            per_column_strategy={"tx_count": ImputationStrategy.Constant}
        )


def test_per_column_strategy_constant_without_fill_names_column():
    with pytest.raises(ValueError, match="'tx_count'"):
        NumericImputationConfig(
            per_column_strategy={"tx_count": ImputationStrategy.Constant}
        )


def test_per_column_strategy_constant_with_fill_constructs():
    cfg = NumericImputationConfig(
        per_column_strategy={"tx_count": ImputationStrategy.Constant},
        per_column_constant_fill={"tx_count": -1.0},
    )
    assert cfg.per_column_strategy["tx_count"] == ImputationStrategy.Constant
    assert cfg.per_column_constant_fill["tx_count"] == -1.0


def test_per_column_constant_fill_alone_constructs():
    cfg = NumericImputationConfig(
        per_column_constant_fill={"tx_count": 0.0},
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
    cfg = NumericImputationConfig(per_column_strategy=allowed)
    assert len(cfg.per_column_strategy) == 6


def test_per_column_strategy_error_names_the_column():
    with pytest.raises(ValueError, match="'income'"):
        NumericImputationConfig(
            per_column_strategy={"income": ImputationStrategy.Dropped}
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
        ImputationConfig(
            mnar_columns=["income"],
            numeric=NumericImputationConfig(
                per_column_strategy={"income": ImputationStrategy.Median}
            ),
        )


def test_imputation_config_mnar_conflict_message_names_all_conflicting_columns():
    with pytest.raises(ValueError, match="'income'") as exc_info:
        ImputationConfig(
            mnar_columns=["income", "age"],
            numeric=NumericImputationConfig(
                per_column_strategy={
                    "income": ImputationStrategy.Median,
                    "age": ImputationStrategy.Mean,
                }
            ),
        )
    assert "'age'" in str(exc_info.value)


def test_imputation_config_mnar_conflict_no_error_when_disjoint():
    cfg = ImputationConfig(
        mnar_columns=["income"],
        numeric=NumericImputationConfig(
            per_column_strategy={"sensor": ImputationStrategy.Median}
        ),
    )
    assert cfg.mnar_columns == ["income"]


def test_imputation_config_mnar_conflict_no_error_when_mnar_empty():
    cfg = ImputationConfig(
        mnar_columns=[],
        numeric=NumericImputationConfig(
            per_column_strategy={"sensor": ImputationStrategy.Median}
        ),
    )
    assert cfg.numeric.per_column_strategy["sensor"] == ImputationStrategy.Median


def test_imputation_config_mnar_conflict_no_error_when_per_column_strategy_empty():
    cfg = ImputationConfig(
        mnar_columns=["income"],
        numeric=NumericImputationConfig(),
    )
    assert cfg.mnar_columns == ["income"]


def test_imputation_config_mnar_conflict_no_error_when_both_empty():
    cfg = ImputationConfig()
    assert cfg.mnar_columns == []
    assert cfg.numeric.per_column_strategy == {}


# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ImputationFitDiagnostic — dataclass, to_dict / from_dict, ColumnImputationRecord wiring
# ---------------------------------------------------------------------------


def _make_diagnostic(**overrides):
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    defaults = {
        "r2_train": 0.75,
        "converged": True,
        "n_iter": 5,
        "imputed_mean": 42.0,
        "imputed_std": 3.5,
        "observed_mean": 40.0,
        "observed_std": 4.0,
        "variance_ratio": 0.875,
        "n_neighbors_used": None,
        "k_capped": None,
    }
    defaults.update(overrides)
    return ImputationFitDiagnostic(**defaults)


def test_imputation_fit_diagnostic_importable_from_imputation_module():
    from dataforge_ml.imputation import ImputationFitDiagnostic  # noqa: F401


def test_imputation_fit_diagnostic_importable_from_dataforge_ml():
    from dataforge_ml import ImputationFitDiagnostic  # noqa: F401


def test_imputation_fit_diagnostic_fields_typed_correctly():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = _make_diagnostic(r2_train=0.5, converged=False, n_iter=10)
    assert isinstance(diag.r2_train, float)
    assert isinstance(diag.converged, bool)
    assert isinstance(diag.n_iter, int)
    assert isinstance(diag.imputed_mean, float)
    assert isinstance(diag.imputed_std, float)
    assert isinstance(diag.observed_mean, float)
    assert isinstance(diag.observed_std, float)
    assert isinstance(diag.variance_ratio, float)


def test_imputation_fit_diagnostic_optional_fields_accept_none():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = ImputationFitDiagnostic(
        r2_train=None,
        converged=None,
        n_iter=None,
        imputed_mean=1.0,
        imputed_std=0.0,
        observed_mean=1.0,
        observed_std=0.5,
        variance_ratio=0.0,
    )
    assert diag.r2_train is None
    assert diag.converged is None
    assert diag.n_iter is None


def test_imputation_fit_diagnostic_to_dict_has_all_ten_keys():
    diag = _make_diagnostic()
    d = diag.to_dict()
    assert set(d.keys()) == {
        "r2_train", "converged", "n_iter",
        "imputed_mean", "imputed_std",
        "observed_mean", "observed_std",
        "variance_ratio",
        "n_neighbors_used", "k_capped",
    }


def test_imputation_fit_diagnostic_to_dict_preserves_values():
    diag = _make_diagnostic(r2_train=0.9, converged=True, n_iter=7, variance_ratio=0.6)
    d = diag.to_dict()
    assert d["r2_train"] == pytest.approx(0.9)
    assert d["converged"] is True
    assert d["n_iter"] == 7
    assert d["variance_ratio"] == pytest.approx(0.6)


def test_imputation_fit_diagnostic_to_dict_with_none_fields():
    diag = _make_diagnostic(r2_train=None, converged=None, n_iter=None)
    d = diag.to_dict()
    assert d["r2_train"] is None
    assert d["converged"] is None
    assert d["n_iter"] is None


def test_imputation_fit_diagnostic_from_dict_round_trip():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = _make_diagnostic(r2_train=0.55, converged=False, n_iter=10, variance_ratio=0.2)
    restored = ImputationFitDiagnostic.from_dict(diag.to_dict())
    assert restored.r2_train == pytest.approx(0.55)
    assert restored.converged is False
    assert restored.n_iter == 10
    assert restored.variance_ratio == pytest.approx(0.2)
    assert restored.imputed_mean == pytest.approx(diag.imputed_mean)
    assert restored.imputed_std == pytest.approx(diag.imputed_std)
    assert restored.observed_mean == pytest.approx(diag.observed_mean)
    assert restored.observed_std == pytest.approx(diag.observed_std)


def test_imputation_fit_diagnostic_from_dict_none_optional_fields():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    d = {
        "r2_train": None,
        "converged": None,
        "n_iter": None,
        "imputed_mean": 5.0,
        "imputed_std": 1.0,
        "observed_mean": 5.5,
        "observed_std": 2.0,
        "variance_ratio": 0.5,
    }
    restored = ImputationFitDiagnostic.from_dict(d)
    assert restored.r2_train is None
    assert restored.converged is None
    assert restored.n_iter is None


# KNN-specific fields: n_neighbors_used and k_capped


def test_imputation_fit_diagnostic_knn_fields_default_to_none():
    diag = _make_diagnostic()
    assert diag.n_neighbors_used is None
    assert diag.k_capped is None


def test_imputation_fit_diagnostic_knn_fields_accept_values():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = ImputationFitDiagnostic(
        r2_train=0.8,
        converged=None,
        n_iter=None,
        imputed_mean=1.0,
        imputed_std=0.5,
        observed_mean=1.0,
        observed_std=0.6,
        variance_ratio=0.83,
        n_neighbors_used=7,
        k_capped=False,
    )
    assert diag.n_neighbors_used == 7
    assert diag.k_capped is False


def test_imputation_fit_diagnostic_to_dict_includes_knn_fields():
    diag = _make_diagnostic(n_neighbors_used=5, k_capped=True)
    d = diag.to_dict()
    assert d["n_neighbors_used"] == 5
    assert d["k_capped"] is True


def test_imputation_fit_diagnostic_to_dict_knn_fields_none_by_default():
    diag = _make_diagnostic()
    d = diag.to_dict()
    assert d["n_neighbors_used"] is None
    assert d["k_capped"] is None


def test_imputation_fit_diagnostic_from_dict_round_trips_knn_fields():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    diag = _make_diagnostic(n_neighbors_used=9, k_capped=False)
    restored = ImputationFitDiagnostic.from_dict(diag.to_dict())
    assert restored.n_neighbors_used == 9
    assert restored.k_capped is False


def test_imputation_fit_diagnostic_from_dict_missing_knn_fields_default_to_none():
    from dataforge_ml.imputation._config import ImputationFitDiagnostic

    d = {
        "r2_train": None,
        "converged": None,
        "n_iter": None,
        "imputed_mean": 1.0,
        "imputed_std": 0.5,
        "observed_mean": 1.0,
        "observed_std": 0.5,
        "variance_ratio": 1.0,
        # n_neighbors_used and k_capped intentionally absent (old serialised format)
    }
    restored = ImputationFitDiagnostic.from_dict(d)
    assert restored.n_neighbors_used is None
    assert restored.k_capped is None


def test_imputation_fit_diagnostic_regression_mice_knn_fields_remain_none():
    diag = _make_diagnostic(converged=True, n_iter=3)
    assert diag.n_neighbors_used is None
    assert diag.k_capped is None


def test_column_imputation_record_diagnostic_defaults_to_none():
    record = ColumnImputationRecord(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Mean,
        fill_value=30.0,
    )
    assert record.diagnostic is None


def test_column_imputation_record_diagnostic_field_accepted():
    record = ColumnImputationRecord(
        column="score",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        diagnostic=_make_diagnostic(),
    )
    assert record.diagnostic is not None
    assert record.diagnostic.r2_train == pytest.approx(0.75)


def test_column_imputation_record_to_dict_includes_diagnostic_key():
    record = ColumnImputationRecord(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Mean,
        fill_value=30.0,
    )
    d = record.to_dict()
    assert "diagnostic" in d


def test_column_imputation_record_to_dict_diagnostic_is_none_for_scalar_strategy():
    record = ColumnImputationRecord(
        column="age",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Mean,
        fill_value=30.0,
    )
    d = record.to_dict()
    assert d["diagnostic"] is None


def test_column_imputation_record_to_dict_diagnostic_is_nested_dict_when_set():
    diag = _make_diagnostic()
    record = ColumnImputationRecord(
        column="score",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        diagnostic=diag,
    )
    d = record.to_dict()
    assert isinstance(d["diagnostic"], dict)
    assert set(d["diagnostic"].keys()) == {
        "r2_train", "converged", "n_iter",
        "imputed_mean", "imputed_std",
        "observed_mean", "observed_std",
        "variance_ratio",
        "n_neighbors_used", "k_capped",
    }


def test_column_imputation_record_to_dict_diagnostic_values_correct():
    diag = _make_diagnostic(r2_train=0.8, n_iter=3)
    record = ColumnImputationRecord(
        column="income",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.KNN,
        diagnostic=diag,
    )
    d = record.to_dict()
    assert d["diagnostic"]["r2_train"] == pytest.approx(0.8)
    assert d["diagnostic"]["n_iter"] == 3


def test_fitted_imputer_from_dict_round_trips_diagnostic():
    from dataforge_ml.imputation._fitted_imputer import FittedImputer

    diag = _make_diagnostic(r2_train=0.6, converged=True, n_iter=8, variance_ratio=0.7)
    record = ColumnImputationRecord(
        column="income",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Regression,
        diagnostic=diag,
    )
    fi = FittedImputer(records={"income": record})
    restored = FittedImputer.from_dict(fi.to_dict())

    restored_diag = restored.records["income"].diagnostic
    assert restored_diag is not None
    assert restored_diag.r2_train == pytest.approx(0.6)
    assert restored_diag.converged is True
    assert restored_diag.n_iter == 8
    assert restored_diag.variance_ratio == pytest.approx(0.7)
    assert restored_diag.imputed_mean == pytest.approx(diag.imputed_mean)
    assert restored_diag.imputed_std == pytest.approx(diag.imputed_std)
    assert restored_diag.observed_mean == pytest.approx(diag.observed_mean)
    assert restored_diag.observed_std == pytest.approx(diag.observed_std)


def test_fitted_imputer_from_dict_payload_without_diagnostic_key_gives_none():
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
                "domain_snap_bounds": None,
                # no "diagnostic" key — simulates a pre-Scope-3 serialised record
            }
        },
        "models": {},
        "model_cols": {},
    }
    fi = FittedImputer.from_dict(legacy_dict)
    assert fi.records["age"].diagnostic is None


def test_fitted_imputer_from_dict_payload_with_null_diagnostic_gives_none():
    from dataforge_ml.imputation._fitted_imputer import FittedImputer

    payload = {
        "records": {
            "age": {
                "column": "age",
                "semantic_type": "numeric",
                "strategy": "mean",
                "fill_value": 30.0,
                "indicator_added": False,
                "signals": [],
                "domain_snap_bounds": None,
                "diagnostic": None,
            }
        },
        "models": {},
        "model_cols": {},
    }
    fi = FittedImputer.from_dict(payload)
    assert fi.records["age"].diagnostic is None


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
    cfg = NumericImputationConfig(per_column_max_iter={"income": 20})
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
    original = NumericImputationConfig(per_column_max_iter={"age": 30, "income": 50})
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
